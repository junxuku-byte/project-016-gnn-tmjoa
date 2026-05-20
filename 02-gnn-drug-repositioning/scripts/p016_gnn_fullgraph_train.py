#!/usr/bin/env python3
"""
Project-016 Phase 2c — Full-Graph Training + Top-K Prediction
Train on ALL edges, score all unseen drug-disease pairs.
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter
from pathlib import Path

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.5
PATIENCE = 15
POS_WEIGHT = 3.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
TRAIN_FILE = DATA_DIR / "p016_train_v5_0.json"
DRUG_FEAT_FILE = DATA_DIR / "drug_features_chembl.json"
DISEASE_FEAT_FILE = DATA_DIR / "disease_features_domain.json"
TOPO_FILE = DATA_DIR / "drug_topological_embeddings.json"
CKPT_FILE = DATA_DIR / "models" / "p016_gnn_fullgraph.pt"
OUT_FILE = DATA_DIR / "p016_top_predictions_phase2c.json"


def load_all_features():
    with open(DRUG_FEAT_FILE) as f:
        drug_data = json.load(f)
    with open(DISEASE_FEAT_FILE) as f:
        disease_data = json.load(f)
    with open(TOPO_FILE) as f:
        topo_data = json.load(f)

    drugs = drug_data["drugs"]
    diseases = disease_data["diseases"]
    drug2idx = {d: i for i, d in enumerate(drugs)}
    disease2idx = {d: i for i, d in enumerate(diseases)}

    chembl = torch.tensor([drug_data["features"][d] for d in drugs], dtype=torch.float)
    topo = torch.tensor([topo_data["embeddings"][d] for d in drugs], dtype=torch.float)
    drug_feats = torch.cat([chembl, topo], dim=1)
    disease_feats = torch.tensor([disease_data["features"][d] for d in diseases], dtype=torch.float)

    return drugs, diseases, drug2idx, disease2idx, drug_feats, disease_feats


class DrugRepositionGNN(nn.Module):
    def __init__(self, hidden_channels=HIDDEN_DIM, out_channels=EMB_DIM, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("drug", "treats", "disease"): SAGEConv((-1, -1), hidden_channels),
                ("disease", "treated_by", "drug"): SAGEConv((-1, -1), hidden_channels),
            }, aggr="mean")
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training) for key, x in x_dict.items()}
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict

    def predict_edge(self, drug_emb, disease_emb):
        x = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.predictor(x).squeeze(-1)


def build_graph(samples, drug2idx, disease2idx, drug_feats, disease_feats):
    data = HeteroData()
    data["drug"].x = drug_feats.clone()
    data["disease"].x = disease_feats.clone()

    drug_idx = [drug2idx[s["drug"]] for s in samples]
    disease_idx = [disease2idx[s["disease"]] for s in samples]
    labels = [s["label"] for s in samples]
    weights = []
    for s in samples:
        h = s.get("hardness", "soft")
        hw = {"hard": 1.5, "medium": 1.2, "soft": 1.0, "positive": 1.0}.get(h, 1.0)
        weights.append(hw)

    edge_index = torch.tensor([drug_idx, disease_idx], dtype=torch.long)
    data["drug", "treats", "disease"].edge_index = edge_index
    data["drug", "treats", "disease"].edge_label = torch.tensor(labels, dtype=torch.float)
    data["drug", "treats", "disease"].edge_weight = torch.tensor(weights, dtype=torch.float)
    data["disease", "treated_by", "drug"].edge_index = edge_index.flip(0)

    return data


def train_fullgraph(model, data, optimizer):
    """Train on all edges with internal train/val split for early stopping."""
    n_edges = data["drug", "treats", "disease"].edge_index.size(1)
    # 90% train, 10% val (random split for monitoring)
    perm = torch.randperm(n_edges)
    train_idx = perm[:int(0.9 * n_edges)]
    val_idx = perm[int(0.9 * n_edges):]

    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    val_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    best_val_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        x_dict = model(data.x_dict, data.edge_index_dict)
        ei = data["drug", "treats", "disease"].edge_index[:, train_mask]
        el = data["drug", "treats", "disease"].edge_label[train_mask]
        ew = data["drug", "treats", "disease"].edge_weight[train_mask]

        pred = model.predict_edge(x_dict["drug"][ei[0]], x_dict["disease"][ei[1]])

        pos_mask = el > 0.5
        weight = torch.ones_like(el)
        weight[pos_mask] = POS_WEIGHT
        weight = weight * ew

        loss = F.binary_cross_entropy_with_logits(pred, el, weight=weight)
        l2_reg = 0.001 * sum(p.pow(2).sum() for p in model.lin.parameters())
        loss = loss + l2_reg
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                x_dict = model(data.x_dict, data.edge_index_dict)
                ei_val = data["drug", "treats", "disease"].edge_index[:, val_mask]
                el_val = data["drug", "treats", "disease"].edge_label[val_mask]
                pred_val = torch.sigmoid(model.predict_edge(x_dict["drug"][ei_val[0]], x_dict["disease"][ei_val[1]]))
                pred_np = pred_val.cpu().numpy()
                label_np = el_val.cpu().numpy()
                binary = (label_np >= 0.5).astype(int)
                val_auc = roc_auc_score(binary, pred_np) if len(np.unique(binary)) >= 2 else 0.0

            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_auc


def score_all_pairs(model, data, drugs, diseases, drug2idx, disease2idx, existing_pairs):
    """Score all unseen drug-disease pairs."""
    model.eval()

    candidates = []
    for d in drugs:
        for dis in diseases:
            if (d, dis) not in existing_pairs:
                candidates.append((d, dis))

    print(f"\nScoring {len(candidates)} candidate pairs...")

    batch_size = 1024
    all_scores = []

    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            drug_idx = torch.tensor([drug2idx[d] for d, _ in batch], dtype=torch.long, device=DEVICE)
            disease_idx = torch.tensor([disease2idx[dis] for _, dis in batch], dtype=torch.long, device=DEVICE)

            drug_emb = x_dict["drug"][drug_idx]
            disease_emb = x_dict["disease"][disease_idx]
            scores = torch.sigmoid(model.predict_edge(drug_emb, disease_emb))
            all_scores.extend(scores.cpu().tolist())

    scored = list(zip(candidates, all_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def main():
    print("=" * 60)
    print("Project-016 Phase 2c — Full-Graph Training + Top-K Prediction")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    with open(TRAIN_FILE) as f:
        data_json = json.load(f)
    all_samples = data_json["splits"]["train"]

    drugs, diseases, drug2idx, disease2idx, drug_feats, disease_feats = load_all_features()
    print(f"\nDrugs: {len(drugs)} | Diseases: {len(diseases)}")

    existing_pairs = set((s["drug"], s["disease"]) for s in all_samples)

    # Build graph
    data = build_graph(all_samples, drug2idx, disease2idx, drug_feats, disease_feats)
    data = data.to(DEVICE)

    # Train
    print(f"\n[1] Training on full graph ({len(all_samples)} edges)...")
    model = DrugRepositionGNN(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, best_val_auc = train_fullgraph(model, data, optimizer)
    print(f"\n  Best Val AUC during training: {best_val_auc:.4f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "drug2idx": drug2idx,
        "disease2idx": disease2idx,
        "drugs": drugs,
        "diseases": diseases,
        "val_auc": best_val_auc,
    }, CKPT_FILE)
    print(f"  Saved: {CKPT_FILE}")

    # Score all pairs
    print(f"\n[2] Generating Top-K predictions...")
    scored = score_all_pairs(model, data, drugs, diseases, drug2idx, disease2idx, existing_pairs)

    # Output
    print(f"\n[3] Top-20 Overall:")
    for i, ((d, dis), score) in enumerate(scored[:20]):
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")

    print(f"\n[4] Top-10 for TMJOA:")
    tmjoa = [p for p in scored if p[0][1] == "TMJOA"]
    for i, ((d, dis), score) in enumerate(tmjoa[:10]):
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")

    print(f"\n[5] Top-10 for TMJ:")
    tmj = [p for p in scored if p[0][1] == "TMJ"]
    for i, ((d, dis), score) in enumerate(tmj[:10]):
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")

    # Save
    output = {
        "metadata": {
            "version": "phase2c_fullgraph",
            "checkpoint": str(CKPT_FILE),
            "total_candidates": len(scored),
            "val_auc": best_val_auc,
        },
        "top_20_overall": [{"rank": i+1, "drug": d, "disease": dis, "score": round(score, 4)} for i, ((d, dis), score) in enumerate(scored[:20])],
        "top_10_tmjoa": [{"rank": i+1, "drug": d, "disease": dis, "score": round(score, 4)} for i, ((d, dis), score) in enumerate(tmjoa[:10])],
        "top_10_tmj": [{"rank": i+1, "drug": d, "disease": dis, "score": round(score, 4)} for i, ((d, dis), score) in enumerate(tmj[:10])],
        "all_predictions": [{"drug": d, "disease": dis, "score": round(score, 4)} for (d, dis), score in scored],
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved: {OUT_FILE}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
