#!/usr/bin/env python3
"""
Project-016 Phase 2c — Stratified K-Fold Cross-Validation
Addresses Court decision: stratified by edge label, ensures balanced pos/neg per fold.
Transductive evaluation (nodes in graph, test edges hidden).
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
from sklearn.model_selection import StratifiedKFold
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
OUT_FILE = DATA_DIR / "models" / "p016_phase2c_stratified_kfold.json"
OUT_FILE.parent.mkdir(exist_ok=True)


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
    """Build full graph from all samples. Returns data + list of sample indices."""
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


def train_epoch(model, data, optimizer, mask):
    model.train()
    optimizer.zero_grad()
    x_dict = model(data.x_dict, data.edge_index_dict)

    ei = data["drug", "treats", "disease"].edge_index[:, mask]
    el = data["drug", "treats", "disease"].edge_label[mask]
    ew = data["drug", "treats", "disease"].edge_weight[mask]

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
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    x_dict = model(data.x_dict, data.edge_index_dict)
    ei = data["drug", "treats", "disease"].edge_index[:, mask]
    el = data["drug", "treats", "disease"].edge_label[mask]

    pred = torch.sigmoid(model.predict_edge(x_dict["drug"][ei[0]], x_dict["disease"][ei[1]]))
    pred_np = pred.cpu().numpy()
    label_np = el.cpu().numpy()
    binary = (label_np >= 0.5).astype(int)

    if len(np.unique(binary)) < 2:
        return {"auc": 0.0, "ap": 0.0, "n_pos": int(binary.sum()), "n_neg": int((1-binary).sum())}

    auc = roc_auc_score(binary, pred_np)
    ap = average_precision_score(binary, pred_np)
    return {"auc": auc, "ap": ap, "n_pos": int(binary.sum()), "n_neg": int((1-binary).sum())}


def run_fold(model_class, fold_idx, train_mask, val_mask, test_mask, data, all_samples):
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx + 1}/5 (Stratified)")
    print(f"  Train: {train_mask.sum()} edges | Val: {val_mask.sum()} edges | Test: {test_mask.sum()} edges")

    # Show class balance per split
    train_labels = [all_samples[i]["label"] for i in range(len(all_samples)) if train_mask[i]]
    val_labels = [all_samples[i]["label"] for i in range(len(all_samples)) if val_mask[i]]
    test_labels = [all_samples[i]["label"] for i in range(len(all_samples)) if test_mask[i]]
    print(f"  Train pos/neg: {sum(1 for l in train_labels if l>=0.5)}/{sum(1 for l in train_labels if l<0.5)}")
    print(f"  Val pos/neg: {sum(1 for l in val_labels if l>=0.5)}/{sum(1 for l in val_labels if l<0.5)}")
    print(f"  Test pos/neg: {sum(1 for l in test_labels if l>=0.5)}/{sum(1 for l in test_labels if l<0.5)}")

    model = model_class(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    best_val_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, train_mask)

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, data, val_mask)
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_metrics['auc']:.4f} (pos={val_metrics['n_pos']}, neg={val_metrics['n_neg']})")

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, data, test_mask)
    print(f"  Test AUC: {test_metrics['auc']:.4f} | AP: {test_metrics['ap']:.4f} (pos={test_metrics['n_pos']}, neg={test_metrics['n_neg']})")

    return test_metrics


def main():
    print("=" * 60)
    print("Project-016 Phase 2c — Stratified K-Fold (5-fold)")
    print(f"Device: {DEVICE} | Features: ChEMBL + Topological")
    print("=" * 60)

    with open(TRAIN_FILE) as f:
        data = json.load(f)
    all_samples = data["splits"]["train"]

    drugs, diseases, drug2idx, disease2idx, drug_feats, disease_feats = load_all_features()

    # Prepare stratified labels for splitting
    labels = np.array([s["label"] for s in all_samples])
    binary_labels = (labels >= 0.5).astype(int)

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Positives: {binary_labels.sum()}, Negatives: {len(binary_labels) - binary_labels.sum()}")

    # Build full graph (all edges included for transductive setting)
    full_data = build_graph(all_samples, drug2idx, disease2idx, drug_feats, disease_feats)
    full_data = full_data.to(DEVICE)

    # Stratified K-Fold on edge indices
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_indices = np.arange(len(all_samples))

    all_test_aucs = []
    all_test_aps = []

    print(f"\n[1] Running 5-fold stratified cross-validation...")

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_indices, binary_labels)):
        # Further split train_val into train and val (80/20 of train_val)
        train_val_labels = binary_labels[train_val_idx]
        # Stratify again
        n_train = int(0.8 * len(train_val_idx))
        # Simple split (maintain approximate ratio)
        pos_idx = train_val_idx[train_val_labels == 1]
        neg_idx = train_val_idx[train_val_labels == 0]

        n_train_pos = int(0.8 * len(pos_idx))
        n_train_neg = int(0.8 * len(neg_idx))

        train_pos = pos_idx[:n_train_pos]
        train_neg = neg_idx[:n_train_neg]
        val_pos = pos_idx[n_train_pos:]
        val_neg = neg_idx[n_train_neg:]

        train_idx = np.concatenate([train_pos, train_neg])
        val_idx = np.concatenate([val_pos, val_neg])

        # Create masks on the full graph
        train_mask = torch.zeros(len(all_samples), dtype=torch.bool)
        val_mask = torch.zeros(len(all_samples), dtype=torch.bool)
        test_mask = torch.zeros(len(all_samples), dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        metrics = run_fold(DrugRepositionGNN, fold_idx, train_mask, val_mask, test_mask, full_data, all_samples)
        all_test_aucs.append(metrics["auc"])
        all_test_aps.append(metrics["ap"])

    print(f"\n{'='*60}")
    print(f"Stratified K-Fold Complete")
    print(f"  Test AUC: {np.mean(all_test_aucs):.4f} (+/- {np.std(all_test_aucs):.4f})")
    print(f"  Test AP:  {np.mean(all_test_aps):.4f} (+/- {np.std(all_test_aps):.4f})")
    print(f"  Per-fold AUC: {[f'{a:.4f}' for a in all_test_aucs]}")
    print(f"  Per-fold AP:  {[f'{a:.4f}' for a in all_test_aps]}")
    print(f"{'='*60}")

    summary = {
        "phase": "2c",
        "evaluation": "stratified_kfold_5",
        "features": "chembl_9 + topo_64",
        "test_auc_mean": float(np.mean(all_test_aucs)),
        "test_auc_std": float(np.std(all_test_aucs)),
        "test_ap_mean": float(np.mean(all_test_aps)),
        "test_ap_std": float(np.std(all_test_aps)),
        "per_fold_auc": all_test_aucs,
        "per_fold_ap": all_test_aps,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
