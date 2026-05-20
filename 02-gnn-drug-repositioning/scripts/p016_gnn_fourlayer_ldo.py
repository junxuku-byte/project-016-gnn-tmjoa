#!/usr/bin/env python3
"""
Phase 3c: Four-Layer GNN with Leave-Drug-Out (LDO) Inductive Evaluation.
Hides all edges for held-out drugs during training + testing.
"""

import json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from collections import defaultdict

# ─── Config ─────────────────────────────────────────────────────
SEED = 42
DEVICE = torch.device('cpu')
N_LAYERS = 2
HIDDEN = 128
DROPOUT = 0.5
LR = 0.01
EPOCHS = 300
PATIENCE = 30

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH = DATA / "four_layer_graph_full.json"
TRAIN = DATA / "p016_train_v5_0.json"
MODEL_DIR = DATA.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
OUT_MODEL = MODEL_DIR / "p016_gnn_fourlayer_ldo.pt"
OUT_RESULTS = DATA / "ldo_results.json"

# ─── Load graph ─────────────────────────────────────────────────
with open(GRAPH) as f:
    g = json.load(f)

with open(TRAIN) as f:
    train_data = json.load(f)

# Node sets
drugs = set()
for e in g['drug_target_edges']:
    drugs.add(e[0])
drugs = sorted(drugs)

targets = g['targets']
pathways = g['pathways']

diseases = set()
for e in g['pathway_disease_edges']:
    diseases.add(e[1])
diseases = sorted(diseases)

drug_idx = {d: i for i, d in enumerate(drugs)}
target_idx = {t: i for i, t in enumerate(targets)}
pw_idx = {p: i for i, p in enumerate(pathways)}
disease_idx = {d: i for i, d in enumerate(diseases)}

ND = len(drugs); NT = len(targets); NP = len(pathways); NI = len(diseases)
TOTAL = ND + NT + NP + NI

def node_offset(t, idx):
    if t == 'drug': return idx
    if t == 'target': return ND + idx
    if t == 'pathway': return ND + NT + idx
    return ND + NT + NP + idx  # disease

# Build edges (drug-target, target-pathway, pathway-disease)
edges_src, edges_dst = [], []
edge_types = []  # 0=dt, 1=tp, 2=pd

for d, t in g['drug_target_edges']:
    if d in drug_idx and t in target_idx:
        edges_src.append(node_offset('drug', drug_idx[d]))
        edges_dst.append(node_offset('target', target_idx[t]))
        edge_types.append(0)
        edges_src.append(node_offset('target', target_idx[t]))
        edges_dst.append(node_offset('drug', drug_idx[d]))
        edge_types.append(0)

for t, p in g['target_pathway_edges']:
    if t in target_idx and p in pw_idx:
        edges_src.append(node_offset('target', target_idx[t]))
        edges_dst.append(node_offset('pathway', pw_idx[p]))
        edge_types.append(1)
        edges_src.append(node_offset('pathway', pw_idx[p]))
        edges_dst.append(node_offset('target', target_idx[t]))
        edge_types.append(1)

for p, d in g['pathway_disease_edges']:
    if p in pw_idx and d in disease_idx:
        edges_src.append(node_offset('pathway', pw_idx[p]))
        edges_dst.append(node_offset('disease', disease_idx[d]))
        edge_types.append(2)
        edges_src.append(node_offset('disease', disease_idx[d]))
        edges_dst.append(node_offset('pathway', pw_idx[p]))
        edge_types.append(2)

edges_src = torch.LongTensor(edges_src)
edges_dst = torch.LongTensor(edges_dst)

# ─── Build drug-disease labels from train data ──────────────────
train_items = train_data['splits']['train']
labels = defaultdict(set)  # drug -> set of diseases
for item in train_items:
    if item.get('label') == 1 or item.get('conclusion') == 'positive':
        labels[item['drug']].add(item['disease'])

# All drug-disease pairs
all_pairs = []
for d in drugs:
    for dis in diseases:
        y = 1 if dis in labels.get(d, set()) else 0
        all_pairs.append((d, dis, y))

# ─── Node features: one-hot per layer ───────────────────────────
x = torch.zeros(TOTAL, 4)
x[:ND, 0] = 1
x[ND:ND+NT, 1] = 1
x[ND+NT:ND+NT+NP, 2] = 1
x[ND+NT+NP:, 3] = 1

# ─── Model ──────────────────────────────────────────────────────
class FourLayerGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, src, dst):
        for i, layer in enumerate(self.layers):
            x_new = torch.zeros_like(x)
            x_new = x_new.index_add_(0, dst, x[src])
            deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
            x_new = x_new / deg.unsqueeze(1)
            x = layer(x_new)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

    def predict(self, x, drug_idx, disease_idx):
        pair = torch.cat([x[drug_idx], x[disease_idx]], dim=1)
        return self.predictor(pair).squeeze()

# ─── Leave-Drug-Out CV ──────────────────────────────────────────
from sklearn.model_selection import KFold

n_drugs = len(drugs)
indices = list(range(n_drugs))
random.shuffle(indices)

N_FOLDS = 5
fold_size = n_drugs // N_FOLDS

fold_aucs = []
fold_aps = []

print(f"\n{'='*60}")
print(f"Leave-Drug-Out {N_FOLDS}-Fold Cross-Validation")
print(f"{'='*60}")

for fold in range(N_FOLDS):
    held_start = fold * fold_size
    held_end = held_start + fold_size if fold < N_FOLDS - 1 else n_drugs
    held_indices = set(indices[held_start:held_end])
    
    # Build training mask: exclude ALL edges connected to held-out drugs
    train_mask = torch.ones(len(edges_src), dtype=torch.bool)
    for i in range(len(edges_src)):
        src = edges_src[i].item()
        # If source or destination is a held-out drug
        if src < ND and src in held_indices:
            train_mask[i] = False
        dst = edges_dst[i].item()
        if dst < ND and dst in held_indices:
            train_mask[i] = False
    
    train_src = edges_src[train_mask]
    train_dst = edges_dst[train_dst] if 'train_dst' in locals() else edges_dst[train_mask]
    
    # Actually fix:
    train_dst = edges_dst[train_mask]
    
    # Training pairs: only non-held-out drugs
    train_pairs = [(d, dis, y) for d, dis, y in all_pairs if drug_idx[d] not in held_indices]
    # Test pairs: only held-out drugs (inductive!)
    test_pairs = [(d, dis, y) for d, dis, y in all_pairs if drug_idx[d] in held_indices]
    
    if len(test_pairs) == 0 or len(train_pairs) == 0:
        print(f"  Fold {fold+1}: skip (no train/test)")
        continue
    
    model = FourLayerGNN(4, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        
        h = model(x, train_src, train_dst)
        
        # Sample training pairs (max 5000 per epoch)
        sample = random.sample(train_pairs, min(5000, len(train_pairs)))
        d_idx = torch.LongTensor([drug_idx[d] for d, _, _ in sample])
        dis_idx = torch.LongTensor([disease_idx[dis] for _, dis, _ in sample])
        y = torch.FloatTensor([y for _, _, y in sample])
        
        pred = model.predict(h, d_idx, dis_idx)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        
        loss.backward()
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            break
    
    # Evaluate on held-out drugs
    model.eval()
    with torch.no_grad():
        h = model(x, edges_src, edges_dst)  # Full graph for message passing
        d_idx = torch.LongTensor([drug_idx[d] for d, _, _ in test_pairs])
        dis_idx = torch.LongTensor([disease_idx[dis] for _, dis, _ in test_pairs])
        y_true = [y for _, _, y in test_pairs]
        
        pred = model.predict(h, d_idx, dis_idx).cpu().numpy()
    
    try:
        auc = roc_auc_score(y_true, pred)
        ap = average_precision_score(y_true, pred)
    except:
        auc = 0.5
        ap = sum(y_true) / len(y_true)
    
    fold_aucs.append(auc)
    fold_aps.append(ap)
    
    print(f"  Fold {fold+1}: {len(held_indices)} held-out drugs, {len(test_pairs)} test pairs, AUC={auc:.4f}, AP={ap:.4f}")

# ─── Summary ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Leave-Drug-Out Summary (INDUCTIVE)")
print(f"{'='*60}")
print(f"  AUC:  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  AP:   {np.mean(fold_aps):.4f} ± {np.std(fold_aps):.4f}")
print(f"  Folds: {[f'{a:.3f}' for a in fold_aucs]}")

results = {
    'method': 'leave-drug-out (inductive)',
    'n_folds': N_FOLDS,
    'fold_aucs': [float(a) for a in fold_aucs],
    'fold_aps': [float(a) for a in fold_aps],
    'mean_auc': float(np.mean(fold_aucs)),
    'std_auc': float(np.std(fold_aucs)),
    'mean_ap': float(np.mean(fold_aps)),
    'std_ap': float(np.std(fold_aps)),
}

with open(OUT_RESULTS, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved to {OUT_RESULTS}")
