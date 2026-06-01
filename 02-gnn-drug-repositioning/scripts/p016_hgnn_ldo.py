#!/usr/bin/env python3
"""
Phase 3d Strategy A: Heterogeneous Message Passing GNN (HGNNConv).
Separate weight matrices for DT, TP, PD edge types + type-specific aggregation.

# ── Paths (portable) ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
"""

import json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

SEED = 42; DEVICE = torch.device('cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA = Path("DATA_DIR")
GRAPH = DATA / "four_layer_graph_full.json"
TRAIN = DATA / "p016_train_v5_0.json"
OUT = DATA / "hgnn_ldo_results.json"

HIDDEN = 128; N_LAYERS = 2; DROPOUT = 0.4
LR = 0.005; EPOCHS = 400; PATIENCE = 40
N_FOLDS = 5

# ─── Load ───────────────────────────────────────────────────────
with open(GRAPH) as f: g = json.load(f)
with open(TRAIN) as f: train = json.load(f)

drugs = sorted(set(e[0] for e in g['drug_target_edges']))
targets = g['targets']
pathways = g['pathways']
diseases = sorted(set(e[1] for e in g['pathway_disease_edges']))

drug_idx = {d:i for i,d in enumerate(drugs)}
targ_idx = {t:i for i,t in enumerate(targets)}
pw_idx   = {p:i for i,p in enumerate(pathways)}
dis_idx  = {d:i for i,d in enumerate(diseases)}

ND = len(drugs); NT = len(targets); NP = len(pathways); NI = len(diseases)
TOTAL = ND + NT + NP + NI

def off(t, i):
    return {'drug':i, 'target':ND+i, 'pathway':ND+NT+i, 'disease':ND+NT+NP+i}[t]

# ─── Build typed edges ──────────────────────────────────────────
def build_typed_edges(g):
    """Returns dict: type -> (src, dst) tensors"""
    typed = {}
    for etype, edges in [
        (0, g['drug_target_edges']), (1, g['target_pathway_edges']), (2, g['pathway_disease_edges'])
    ]:
        srcs, dsts = [], []
        for s, t in edges:
            if etype==0:
                if s not in drug_idx or t not in targ_idx: continue
                srcs.append(off('drug', drug_idx[s])); dsts.append(off('target', targ_idx[t]))
                srcs.append(off('target', targ_idx[t])); dsts.append(off('drug', drug_idx[s]))
            elif etype==1:
                if s not in targ_idx or t not in pw_idx: continue
                srcs.append(off('target', targ_idx[s])); dsts.append(off('pathway', pw_idx[t]))
                srcs.append(off('pathway', pw_idx[t])); dsts.append(off('target', targ_idx[s]))
            else:
                if s not in pw_idx or t not in dis_idx: continue
                srcs.append(off('pathway', pw_idx[s])); dsts.append(off('disease', dis_idx[t]))
                srcs.append(off('disease', dis_idx[t])); dsts.append(off('pathway', pw_idx[s]))
        if srcs:
            typed[etype] = (torch.LongTensor(srcs), torch.LongTensor(dsts))
    return typed

typed_edges = build_typed_edges(g)
print(f"  Typed edges: DT={len(typed_edges[0][0])}, TP={len(typed_edges[1][0])}, PD={len(typed_edges[2][0])}")

# ─── Labels ─────────────────────────────────────────────────────
train_items = train['splits']['train']
pos_labels = defaultdict(set)
for it in train_items:
    if it.get('label')==1 or it.get('conclusion')=='positive':
        pos_labels[it['drug']].add(it['disease'])

# all_pairs: only drugs+diseases in graph
all_pairs = [(d,di, 1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# ─── Node features: one-hot layer type ─────────────────────────
x = torch.zeros(TOTAL, 4)
x[:ND,0]=1; x[ND:ND+NT,1]=1; x[ND+NT:ND+NT+NP,2]=1; x[ND+NT+NP:,3]=1

# ─── Heterogeneous GNN ─────────────────────────────────────────
class HeteroGNNLayer(nn.Module):
    """One heterogeneous conv layer: per-edge-type weights + type-aware aggregation."""
    def __init__(self, in_dim, out_dim, n_types=3, dropout=0.3):
        super().__init__()
        self.W_msg = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_types)])
        self.W_self = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(4)])  # per node type
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, typed_edges):
        """Aggregate messages per edge type, then sum + self-loop."""
        out = torch.zeros(x.size(0), self.W_msg[0].out_features)
        deg_total = torch.zeros(x.size(0))

        for etype, (src, dst) in typed_edges.items():
            msg = self.W_msg[etype](x[src])
            out = out.index_add(0, dst, msg)
            deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
            deg_total += deg

        # Normalise + self-loop
        deg_total = deg_total.clamp(min=1)
        out = out / deg_total.unsqueeze(1)

        # Self-loop: per node-type weight
        self_loop = torch.zeros_like(out)
        self_loop[:ND] = self.W_self[0](x[:ND])
        self_loop[ND:ND+NT] = self.W_self[1](x[ND:ND+NT])
        self_loop[ND+NT:ND+NT+NP] = self.W_self[2](x[ND+NT:ND+NT+NP])
        self_loop[ND+NT+NP:] = self.W_self[3](x[ND+NT+NP:])

        return F.relu(self.dropout(out + self_loop))

class HeteroGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGNNLayer(in_dim, hidden, dropout=dropout))
        for _ in range(n_layers-1):
            self.layers.append(HeteroGNNLayer(hidden, hidden, dropout=dropout))
        self.predictor = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,1)
        )

    def forward(self, x, typed_edges):
        for layer in self.layers:
            x = layer(x, typed_edges)
        return x

    def predict(self, h, d_idx, dis_idx):
        return self.predictor(torch.cat([h[d_idx], h[dis_idx]], dim=1)).squeeze()

# ─── LDO CV ─────────────────────────────────────────────────────
n_drugs = len(drugs)
indices = list(range(n_drugs))
random.shuffle(indices)
fold_size = n_drugs // N_FOLDS

fold_aucs, fold_aps = [], []

print(f"\n{'='*60}")
print(f"HeteroGNN Leave-Drug-Out {N_FOLDS}-Fold (Strategy A)")
print(f"{'='*60}")

for fold in range(N_FOLDS):
    hs, he = fold*fold_size, (fold+1)*fold_size if fold<N_FOLDS-1 else n_drugs
    held = set(indices[hs:he])

    # Mask edges: remove any edge involving held-out drug
    fold_typed = {}
    for etype, (src, dst) in typed_edges.items():
        m = torch.ones(len(src), dtype=torch.bool)
        for i in range(len(src)):
            if (src[i]<ND and src[i].item() in held) or (dst[i]<ND and dst[i].item() in held):
                m[i]=False
        if m.sum()>0:
            fold_typed[etype] = (src[m], dst[m])

    train_pairs = [(d,di,y) for d,di,y in all_pairs if drug_idx[d] not in held]
    test_pairs  = [(d,di,y) for d,di,y in all_pairs if drug_idx[d] in held]

    if len(test_pairs)==0 or len(train_pairs)==0:
        print(f"  Fold {fold+1}: skip"); continue

    model = HeteroGNN(4, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss, pat = float('inf'), 0

    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        h = model(x, fold_typed if fold_typed else typed_edges)

        samp = random.sample(train_pairs, min(8000, len(train_pairs)))
        d_idx = torch.LongTensor([drug_idx[d] for d,_,_ in samp])
        di_idx= torch.LongTensor([dis_idx[di] for _,di,_ in samp])
        y = torch.FloatTensor([y for _,_,y in samp])

        pred = model.predict(h, d_idx, di_idx)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward(); opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item(); pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    model.eval()
    with torch.no_grad():
        h = model(x, typed_edges)
        d_idx = torch.LongTensor([drug_idx[d] for d,_,_ in test_pairs])
        di_idx= torch.LongTensor([dis_idx[di] for _,di,_ in test_pairs])
        yt = [y for _,_,y in test_pairs]
        pred = model.predict(h, d_idx, di_idx).cpu().numpy()

    try:
        auc = roc_auc_score(yt, pred); ap = average_precision_score(yt, pred)
    except:
        auc, ap = 0.5, sum(yt)/len(yt)

    fold_aucs.append(auc); fold_aps.append(ap)
    print(f"  Fold {fold+1}: {len(held)} held, {len(test_pairs)} test, AUC={auc:.4f}, AP={ap:.4f}")

# ─── Summary ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"HeteroGNN LDO Summary")
print(f"{'='*60}")
print(f"  AUC:  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  AP:   {np.mean(fold_aps):.4f} ± {np.std(fold_aps):.4f}")
print(f"  Folds: {[f'{a:.3f}' for a in fold_aucs]}")

result = {
    'strategy': 'A (heterogeneous message passing)',
    'method': 'leave-drug-out (inductive)',
    'n_folds': N_FOLDS, 'n_drugs': ND, 'n_targets': NT, 'n_pathways': NP, 'n_diseases': NI,
    'fold_aucs': [float(a) for a in fold_aucs],
    'fold_aps': [float(a) for a in fold_aps],
    'mean_auc': float(np.mean(fold_aucs)), 'std_auc': float(np.std(fold_aucs)),
    'mean_ap': float(np.mean(fold_aps)), 'std_ap': float(np.std(fold_aps)),
}
with open(OUT, 'w') as f: json.dump(result, f, indent=2)
print(f"  → {OUT}")
