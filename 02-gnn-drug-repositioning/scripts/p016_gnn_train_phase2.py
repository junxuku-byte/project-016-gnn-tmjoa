#!/usr/bin/env python3
"""
Project-016 GNN Phase 2 — Inductive Training with Real Features
Leave-drug-out cross-validation + ChEMBL drug features + domain disease features

Inductive setup: test drugs are held out during training, their edges unseen.
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

# ========== Config ==========
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

# ========== Paths ==========
DATA_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
OUT_DIR = DATA_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "p016_train_v5_0.json"
DRUG_FEAT_FILE = DATA_DIR / "drug_features_chembl.json"
DISEASE_FEAT_FILE = DATA_DIR / "disease_features_domain.json"


# ========== Load Features ==========

def load_features():
    with open(DRUG_FEAT_FILE) as f:
        drug_data = json.load(f)
    with open(DISEASE_FEAT_FILE) as f:
        disease_data = json.load(f)
    with open(DATA_DIR / "drug_topological_embeddings.json") as f:
        topo_data = json.load(f)

    drugs = drug_data["drugs"]
    diseases = disease_data["diseases"]

    drug2idx = {d: i for i, d in enumerate(drugs)}
    disease2idx = {d: i for i, d in enumerate(diseases)}

    # Drug features: ChEMBL 9-dim + topological 64-dim = 73-dim
    chembl_feats = torch.tensor([drug_data["features"][d] for d in drugs], dtype=torch.float)
    topo_feats = torch.tensor([topo_data["embeddings"][d] for d in drugs], dtype=torch.float)
    drug_feats = torch.cat([chembl_feats, topo_feats], dim=1)

    # Disease features: domain 6-dim
    disease_feats = torch.tensor([disease_data["features"][d] for d in diseases], dtype=torch.float)

    print(f"  Drug feature dim: {drug_feats.size(1)} (ChEMBL {chembl_feats.size(1)} + Topo {topo_feats.size(1)})")

    return drugs, diseases, drug2idx, disease2idx, drug_feats, disease_feats


# ========== Build Graph ==========

def build_hetero_graph(samples, drug2idx, disease2idx, drug_feats, disease_feats, test_drugs=None):
    """
    Build graph. If test_drugs provided, mask out their edges from training graph.
    """
    data = HeteroData()

    num_drugs = len(drug2idx)
    num_diseases = len(disease2idx)

    # Project raw features to EMB_DIM
    drug_proj = nn.Linear(drug_feats.size(1), EMB_DIM)
    disease_proj = nn.Linear(disease_feats.size(1), EMB_DIM)

    with torch.no_grad():
        data["drug"].x = drug_proj(drug_feats)
        data["disease"].x = disease_proj(disease_feats)

    test_drug_set = set(test_drugs) if test_drugs else set()

    drug_indices = []
    disease_indices = []
    labels = []
    weights = []
    hardness_scores = []

    for s in samples:
        drug = s["drug"]
        disease = s["disease"]

        # Skip test drugs in training graph
        if drug in test_drug_set:
            continue

        drug_idx = drug2idx[drug]
        disease_idx = disease2idx[disease]

        drug_indices.append(drug_idx)
        disease_indices.append(disease_idx)
        labels.append(s["label"])

        hardness = s.get("hardness", "soft")
        hw = {"hard": 1.5, "medium": 1.2, "soft": 1.0, "positive": 1.0}.get(hardness, 1.0)
        weights.append(hw)

        hardness_map = {"positive": 2.0, "hard": 1.5, "medium": 1.0, "soft": 0.5}
        hardness_scores.append(hardness_map.get(hardness, 0.5))

    if len(drug_indices) == 0:
        return None  # No training edges

    edge_index = torch.tensor([drug_indices, disease_indices], dtype=torch.long)
    edge_label = torch.tensor(labels, dtype=torch.float)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    edge_hardness = torch.tensor(hardness_scores, dtype=torch.float)

    data["drug", "treats", "disease"].edge_index = edge_index
    data["drug", "treats", "disease"].edge_label = edge_label
    data["drug", "treats", "disease"].edge_weight = edge_weight
    data["drug", "treats", "disease"].edge_hardness = edge_hardness

    # Reverse edges
    data["disease", "treated_by", "drug"].edge_index = edge_index.flip(0)

    return data


# ========== Model ==========

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
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                      for key, x in x_dict.items()}

        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict

    def predict_edge(self, drug_emb, disease_emb):
        x = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.predictor(x).squeeze(-1)


# ========== Training ==========

def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()

    x_dict = model(data.x_dict, data.edge_index_dict)

    edge_index = data["drug", "treats", "disease"].edge_index[:, train_mask]
    edge_label = data["drug", "treats", "disease"].edge_label[train_mask]
    edge_weight = data["drug", "treats", "disease"].edge_weight[train_mask]

    drug_emb = x_dict["drug"][edge_index[0]]
    disease_emb = x_dict["disease"][edge_index[1]]

    pred = model.predict_edge(drug_emb, disease_emb)

    pos_mask = edge_label > 0.5
    weight = torch.ones_like(edge_label)
    weight[pos_mask] = POS_WEIGHT
    weight = weight * edge_weight

    loss = F.binary_cross_entropy_with_logits(pred, edge_label, weight=weight)

    l2_reg = 0.001 * sum(p.pow(2).sum() for p in model.lin.parameters())
    loss = loss + l2_reg

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    x_dict = model(data.x_dict, data.edge_index_dict)

    edge_index = data["drug", "treats", "disease"].edge_index[:, mask]
    edge_label = data["drug", "treats", "disease"].edge_label[mask]

    drug_emb = x_dict["drug"][edge_index[0]]
    disease_emb = x_dict["disease"][edge_index[1]]

    pred = torch.sigmoid(model.predict_edge(drug_emb, disease_emb))
    pred_np = pred.cpu().numpy()
    label_np = edge_label.cpu().numpy()

    binary_label = (label_np >= 0.5).astype(int)
    if len(np.unique(binary_label)) < 2:
        return {"auc": 0.0, "ap": 0.0}

    auc = roc_auc_score(binary_label, pred_np)
    ap = average_precision_score(binary_label, pred_np)
    return {"auc": auc, "ap": ap}


# ========== Leave-Drug-Out Cross-Validation ==========

def leave_drug_out_cv(samples, drugs, drug2idx, disease2idx, drug_feats, disease_feats, n_folds=5):
    """
    Stratified leave-drug-out: each fold holds out ~20% of drugs.
    For each fold:
      - Train on drugs not in test set
      - Validate on a subset of train drugs
      - Test on held-out drugs (inductive)
    """
    # Group samples by drug
    drug_samples = {d: [] for d in drugs}
    for s in samples:
        drug_samples[s["drug"]].append(s)

    # Stratified: ensure positive drugs are spread across folds
    pos_drugs = [d for d in drugs if any(s["label"] >= 0.5 for s in drug_samples[d])]
    neg_drugs = [d for d in drugs if d not in pos_drugs]

    random.shuffle(pos_drugs)
    random.shuffle(neg_drugs)

    # Create folds
    fold_size_pos = max(1, len(pos_drugs) // n_folds)
    fold_size_neg = max(1, len(neg_drugs) // n_folds)

    folds = []
    for i in range(n_folds):
        test_pos = pos_drugs[i * fold_size_pos:(i + 1) * fold_size_pos] if i < n_folds - 1 else pos_drugs[i * fold_size_pos:]
        test_neg = neg_drugs[i * fold_size_neg:(i + 1) * fold_size_neg] if i < n_folds - 1 else neg_drugs[i * fold_size_neg:]
        test_drugs = test_pos + test_neg
        train_drugs = [d for d in drugs if d not in test_drugs]

        # Validation: 10% of train drugs
        random.shuffle(train_drugs)
        val_split = int(0.9 * len(train_drugs))
        train_drugs_final = train_drugs[:val_split]
        val_drugs = train_drugs[val_split:]

        folds.append({
            "train_drugs": train_drugs_final,
            "val_drugs": val_drugs,
            "test_drugs": test_drugs,
        })

    return folds


def run_fold(model_class, fold_idx, fold, samples, drug2idx, disease2idx, drug_feats, disease_feats):
    """Train and evaluate one fold."""
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx + 1}/5")
    print(f"  Train drugs: {len(fold['train_drugs'])}, Val drugs: {len(fold['val_drugs'])}, Test drugs: {len(fold['test_drugs'])}")

    # Build train graph (without test drugs)
    train_samples = [s for s in samples if s["drug"] in fold["train_drugs"]]
    val_samples = [s for s in samples if s["drug"] in fold["val_drugs"]]
    test_samples = [s for s in samples if s["drug"] in fold["test_drugs"]]

    print(f"  Train edges: {len(train_samples)}, Val edges: {len(val_samples)}, Test edges: {len(test_samples)}")

    # Build graph with all edges except test drugs
    all_train_samples = [s for s in samples if s["drug"] not in fold["test_drugs"]]
    data = build_hetero_graph(all_train_samples, drug2idx, disease2idx, drug_feats, disease_feats, test_drugs=fold["test_drugs"])

    if data is None:
        print("  ERROR: No training data")
        return None

    data = data.to(DEVICE)

    # Create masks within the graph
    all_pairs_in_graph = [(s["drug"], s["disease"]) for s in all_train_samples]

    train_set = set((s["drug"], s["disease"]) for s in train_samples)
    val_set = set((s["drug"], s["disease"]) for s in val_samples)

    train_mask = torch.tensor([((d, dis) in train_set) for d, dis in all_pairs_in_graph], dtype=torch.bool)
    val_mask = torch.tensor([((d, dis) in val_set) for d, dis in all_pairs_in_graph], dtype=torch.bool)

    # Build separate test data (inductive: test drug edges not in training graph)
    test_data = HeteroData()
    with torch.no_grad():
        test_data["drug"].x = data["drug"].x.clone()
        test_data["disease"].x = data["disease"].x.clone()

    test_drug_idx = [drug2idx[s["drug"]] for s in test_samples]
    test_disease_idx = [disease2idx[s["disease"]] for s in test_samples]
    test_labels = [s["label"] for s in test_samples]

    test_edge_index = torch.tensor([test_drug_idx, test_disease_idx], dtype=torch.long)
    test_data["drug", "treats", "disease"].edge_index = test_edge_index
    test_data["drug", "treats", "disease"].edge_label = torch.tensor(test_labels, dtype=torch.float)
    test_data["disease", "treated_by", "drug"].edge_index = test_edge_index.flip(0)
    test_data = test_data.to(DEVICE)

    # Model
    model = model_class(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    # Training
    best_val_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, train_mask)

        if epoch % 5 == 0 or epoch == 1:
            train_metrics = evaluate(model, data, train_mask)
            val_metrics = evaluate(model, data, val_mask)

            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best and test (INDUCTIVE)
    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_data, torch.ones(len(test_samples), dtype=torch.bool))
    print(f"  Test AUC (inductive): {test_metrics['auc']:.4f} | AP: {test_metrics['ap']:.4f}")

    return test_metrics


# ========== Main ==========

def main():
    print("=" * 60)
    print("Project-016 GNN Phase 2 — Inductive Leave-Drug-Out Training")
    print(f"Device: {DEVICE} | Real features: ChEMBL + Domain")
    print("=" * 60)

    # Load
    with open(TRAIN_FILE) as f:
        data = json.load(f)
    samples = data["splits"]["train"]

    drugs, diseases, drug2idx, disease2idx, drug_feats, disease_feats = load_features()
    print(f"\nDrugs: {len(drugs)} | Diseases: {len(diseases)}")
    print(f"Drug features: {drug_feats.size(1)}-dim | Disease features: {disease_feats.size(1)}-dim")

    # Cross-validation
    print("\n[1] Building leave-drug-out folds...")
    folds = leave_drug_out_cv(samples, drugs, drug2idx, disease2idx, drug_feats, disease_feats, n_folds=5)

    print(f"\n[2] Running {len(folds)}-fold cross-validation...")
    all_test_aucs = []
    all_test_aps = []

    for i, fold in enumerate(folds):
        metrics = run_fold(DrugRepositionGNN, i, fold, samples, drug2idx, disease2idx, drug_feats, disease_feats)
        if metrics:
            all_test_aucs.append(metrics["auc"])
            all_test_aps.append(metrics["ap"])

    print(f"\n{'='*60}")
    print(f"Cross-validation complete")
    print(f"  Test AUC: {np.mean(all_test_aucs):.4f} (+/- {np.std(all_test_aucs):.4f})")
    print(f"  Test AP:  {np.mean(all_test_aps):.4f} (+/- {np.std(all_test_aps):.4f})")
    print(f"  Per-fold AUC: {[f'{a:.4f}' for a in all_test_aucs]}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "phase": "2",
        "features": "chembl_drug + domain_disease",
        "evaluation": "leave-drug-out",
        "n_folds": 5,
        "test_auc_mean": float(np.mean(all_test_aucs)),
        "test_auc_std": float(np.std(all_test_aucs)),
        "test_ap_mean": float(np.mean(all_test_aps)),
        "test_ap_std": float(np.std(all_test_aps)),
        "per_fold_auc": all_test_aucs,
        "per_fold_ap": all_test_aps,
    }

    with open(OUT_DIR / "p016_phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
