#!/usr/bin/env python3
"""
P1-1: Graph Ablation Experiments.
Isolate the contribution of each mechanism graph component:
1. Remove pathway layer (Drug→Target→Disease)
2. Remove disease diversity (single disease)
3. Randomize TP edges (break target-pathway specificity)
4. Randomize PD edges (break pathway-disease specificity)
5. Feature ablation (degree-only, betweenness-only, one-hot-only)
6. Degree-rewired control (preserve degree, randomize connections)
"""

import json, random, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

SEED = 42; DEVICE = torch.device('cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH = DATA / "four_layer_graph_full_v2.json"
TRAIN = DATA / "p016_train_v5_0.json"

N_LAYERS = 2; HIDDEN = 128; DROPOUT = 0.4
LR = 0.005; EPOCHS = 400; PATIENCE = 40
N_FOLDS = 5; POS_WEIGHT = 3.0

with open(GRAPH) as f: g = json.load(f)
with open(TRAIN) as f: train = json.load(f)

drugs = sorted(set(e[0] for e in g['drug_target_edges']))
targets = g['targets']
pathways = g['pathways']
diseases = sorted(set(e[1] for e in g['pathway_disease_edges']))

drug_idx = {d:i for i,d in enumerate(drugs)}
targ_idx = {t:i for i,t in enumerate(targets)}
pw_idx = {p:i for i,p in enumerate(pathways)}
dis_idx = {d:i for i,d in enumerate(diseases)}

ND = len(drugs); NT = len(targets); NP = len(pathways); NI = len(diseases)

def off(t, i):
    return {'drug':i, 'target':ND+i, 'pathway':ND+NT+i, 'disease':ND+NT+NP+i}[t]

def build_typed_edges(g):
    typed = {}
    for etype, edges in [(0, g['drug_target_edges']), (1, g['target_pathway_edges']), (2, g['pathway_disease_edges'])]:
        srcs, dsts = [], []
        for s, t in edges:
            if etype==0:
                if s not in drug_idx or t not in targ_idx: continue
                src = off('drug', drug_idx[s]); dst = off('target', targ_idx[t])
                srcs.extend([src, dst]); dsts.extend([dst, src])
            elif etype==1:
                if s not in targ_idx or t not in pw_idx: continue
                src = off('target', targ_idx[s]); dst = off('pathway', pw_idx[t])
                srcs.extend([src, dst]); dsts.extend([dst, src])
            else:
                if s not in pw_idx or t not in dis_idx: continue
                src = off('pathway', pw_idx[s]); dst = off('disease', dis_idx[t])
                srcs.extend([src, dst]); dsts.extend([dst, src])
        if srcs:
            typed[etype] = (torch.LongTensor(srcs), torch.LongTensor(dsts))
    return typed

original_typed = build_typed_edges(g)

# Labels
train_items = train['splits']['train']
pos_labels = defaultdict(set)
for it in train_items:
    if it.get('label')==1 or it.get('conclusion')=='positive':
        pos_labels[it['drug']].add(it['disease'])
all_pairs = [(d,di, 1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# ─── HeteroGNN model (reused from p016_hgnn_ldo.py) ────────────
class HeteroGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_types=3, dropout=0.3):
        super().__init__()
        self.W_msg = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_types)])
        self.W_self = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, typed_edges):
        out = torch.zeros(x.size(0), self.W_msg[0].out_features)
        deg_total = torch.zeros(x.size(0))
        for etype, (src, dst) in typed_edges.items():
            msg = self.W_msg[etype](x[src])
            out = out.index_add(0, dst, msg)
            deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
            deg_total += deg
        deg_total = deg_total.clamp(min=1)
        out = out / deg_total.unsqueeze(1)
        sl = torch.zeros_like(out)
        sl[:ND] = self.W_self[0](x[:ND]); sl[ND:ND+NT] = self.W_self[1](x[ND:ND+NT])
        sl[ND+NT:ND+NT+NP] = self.W_self[2](x[ND+NT:ND+NT+NP]); sl[ND+NT+NP:] = self.W_self[3](x[ND+NT+NP:])
        return F.relu(self.dropout(out + sl))

class HeteroGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([HeteroGNNLayer(in_dim, hidden, dropout=dropout)])
        for _ in range(n_layers-1): self.layers.append(HeteroGNNLayer(hidden, hidden, dropout=dropout))
        self.predictor = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,1))
    def forward(self, x, te):
        for l in self.layers: x = l(x, te)
        return x
    def predict(self, h, di, ds):
        return self.predictor(torch.cat([h[di], h[ds]], dim=1)).squeeze()

# ─── Feature builders ───────────────────────────────────────────
def build_features(typed_edges, feats=['layer','degree','betweenness']):
    TOTAL = ND + NT + NP + NI
    feats_list = []

    if 'layer' in feats:
        lh = torch.zeros(TOTAL, 4)
        lh[:ND,0]=1; lh[ND:ND+NT,1]=1; lh[ND+NT:ND+NT+NP,2]=1; lh[ND+NT+NP:,3]=1
        feats_list.append(lh)

    if 'degree' in feats:
        all_src = []; all_dst = []
        for _, (s,d) in typed_edges.items(): all_src.extend(s.tolist()); all_dst.extend(d.tolist())
        deg = torch.bincount(torch.LongTensor(all_src+all_dst), minlength=TOTAL).float()
        deg_norm = torch.log1p(deg) / torch.log1p(deg).max().clamp(min=1)
        feats_list.append(deg_norm.unsqueeze(1))

    if 'betweenness' in feats:
        hit = torch.zeros(TOTAL)
        adj = torch.zeros(TOTAL,TOTAL)
        for _, (s,d) in typed_edges.items():
            if len(s)>0: adj[s,d]=1; adj[d,s]=1
        for _ in range(100):
            cur = random.randrange(TOTAL)
            for _ in range(10):
                nbrs = (adj[cur]>0).nonzero(as_tuple=True)[0]
                if len(nbrs)==0: break
                cur = nbrs[random.randrange(len(nbrs))].item()
                hit[cur] += 1
        btw = hit / hit.max().clamp(min=1)
        feats_list.append(btw.unsqueeze(1))

    x = torch.cat(feats_list, dim=1) if feats_list else torch.ones(TOTAL, 1)
    return x

# ─── LDO runner ─────────────────────────────────────────────────
def run_ldo(name, typed_edges, x, overwrite_ND=None):
    """Run leave-drug-out CV with given graph and features."""
    nd = overwrite_ND if overwrite_ND else ND
    n = nd; indices = list(range(n)); random.shuffle(indices)
    fs = n // N_FOLDS
    fa, fp = [], []

    for fold in range(N_FOLDS):
        hs, he = fold*fs, (fold+1)*fs if fold<N_FOLDS-1 else n
        held = set(indices[hs:he])

        ft = {}
        for et, (src, dst) in typed_edges.items():
            m = torch.ones(len(src), dtype=torch.bool)
            for i in range(len(src)):
                if (src[i]<nd and src[i].item() in held) or (dst[i]<nd and dst[i].item() in held):
                    m[i] = False
            if m.sum()>0: ft[et] = (src[m], dst[m])

        train_pairs = [(d,di,y) for d,di,y in all_pairs if drug_idx[d] not in held]
        test_pairs = [(d,di,y) for d,di,y in all_pairs if drug_idx[d] in held]
        if not test_pairs or not train_pairs: continue

        model = HeteroGNN(x.size(1), HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best_loss, pat = float('inf'), 0

        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            h = model(x, ft if ft else typed_edges)
            samp = random.sample(train_pairs, min(8000, len(train_pairs)))
            di = torch.LongTensor([drug_idx[d] for d,_,_ in samp])
            ds = torch.LongTensor([dis_idx[di] for _,di,_ in samp])
            y = torch.FloatTensor([y for _,_,y in samp])
            pred = model.predict(h, di, ds)
            lp = F.binary_cross_entropy_with_logits(pred[y==1], y[y==1], reduction='sum')
            ln = F.binary_cross_entropy_with_logits(pred[y==0], y[y==0], reduction='sum')
            loss = (lp*POS_WEIGHT + ln)/y.numel()
            loss.backward(); opt.step()
            if loss.item() < best_loss: best_loss = loss.item(); pat = 0
            else: pat += 1
            if pat >= PATIENCE: break

        model.eval()
        with torch.no_grad():
            h = model(x, typed_edges)
            di = torch.LongTensor([drug_idx[d] for d,_,_ in test_pairs])
            ds = torch.LongTensor([dis_idx[di] for _,di,_ in test_pairs])
            yt = [y for _,_,y in test_pairs]
            scores = model.predict(h, di, ds).cpu().numpy()

        try:
            auc = roc_auc_score(yt, scores); ap = average_precision_score(yt, scores)
        except:
            auc, ap = 0.5, sum(yt)/len(yt)
        fa.append(auc); fp.append(ap)
        print(f"  Fold {fold+1}: AUC={auc:.4f}, AP={ap:.4f}")

    mean_auc = np.mean(fa); std_auc = np.std(fa)
    mean_ap = np.mean(fp); std_ap = np.std(fp)
    return {'mean_auc': float(mean_auc), 'std_auc': float(std_auc),
            'mean_ap': float(mean_ap), 'std_ap': float(std_ap),
            'fold_aucs': [float(a) for a in fa], 'fold_aps': [float(a) for a in fp]}

# ─── Ablation 1: Remove Pathway Layer (Drug→Target→Disease) ────
def ablation_remove_pathway():
    """Direct drug→target→disease edges, skip pathway layer."""
    print(f"\n{'='*60}")
    print("Ablation 1: Remove Pathway Layer (3-layer: Drug→Target→Disease)")
    print(f"{'='*60}")

    # Drug→Target edges (keep original)
    # New: Target→Disease edges (construct from target→pathway→disease pairs)
    td_edges = []
    tp_set = set((t,p) for t,p in g['target_pathway_edges'])
    pd_set = set((p,d) for p,d in g['pathway_disease_edges'])

    for t, pw in tp_set:
        for p, d in pd_set:
            if p == pw:
                td_edges.append((t, d))

    tedges = {}
    # type 0: drug-target (keep)
    srcs0, dsts0 = [], []
    for d,t in g['drug_target_edges']:
        if d in drug_idx and t in targ_idx:
            src = off('drug', drug_idx[d]); dst = off('target', targ_idx[t])
            srcs0.extend([src, dst]); dsts0.extend([dst, src])
    if srcs0: tedges[0] = (torch.LongTensor(srcs0), torch.LongTensor(dsts0))

    # type 1: target-disease (new, skip pathway)
    srcs1, dsts1 = [], []
    td_seen = set()
    for t,d in td_edges:
        if t in targ_idx and d in dis_idx:
            key = (t,d)
            if key in td_seen: continue
            td_seen.add(key)
            src = off('target', targ_idx[t]); dst = off('disease', dis_idx[d])
            srcs1.extend([src, dst]); dsts1.extend([dst, src])
    if srcs1: tedges[1] = (torch.LongTensor(srcs1), torch.LongTensor(dsts1))

    x = torch.zeros(ND+NT+NP+NI, 4)
    x[:ND,0]=1; x[ND:ND+NT,1]=1; x[ND+NT:ND+NT+NP,2]=1; x[ND+NT+NP:,3]=1

    return run_ldo("No Pathway Layer", tedges, x)

# ─── Ablation 2: Randomize TP Edges ─────────────────────────────
def ablation_randomize_tp():
    """Randomly reassign target-pathway edges (preserving graph density)."""
    print(f"\n{'='*60}")
    print("Ablation 2: Randomize Target-Pathway Edges")
    print(f"{'='*60}")

    tedges = build_typed_edges(g)  # fresh copy

    # Randomize TP edges
    tp_src_orig = tedges[1][0].clone(); tp_dst_orig = tedges[1][1].clone()
    perm = torch.randperm(len(tp_src_orig))
    tedges[1] = (tp_src_orig, tp_dst_orig[perm])

    x = build_features(tedges, feats=['layer','degree','betweenness'])
    return run_ldo("Random TP Edges", tedges, x)

# ─── Ablation 3: Randomize PD Edges ────────────────────────────
def ablation_randomize_pd():
    """Randomly reassign pathway-disease edges."""
    print(f"\n{'='*60}")
    print("Ablation 3: Randomize Pathway-Disease Edges")
    print(f"{'='*60}")

    tedges = build_typed_edges(g)
    pd_src = tedges[2][0].clone(); pd_dst = tedges[2][1].clone()
    perm = torch.randperm(len(pd_src))
    tedges[2] = (pd_src, pd_dst[perm])

    x = build_features(tedges, feats=['layer','degree','betweenness'])
    return run_ldo("Random PD Edges", tedges, x)

# ─── Ablation 4: Feature Ablation ───────────────────────────────
def ablation_features():
    """Test different feature subsets."""
    print(f"\n{'='*60}")
    print("Ablation 4: Feature Ablation")
    print(f"{'='*60}")

    tedges = build_typed_edges(g)
    results = {}

    for feats, fname in [
        (['layer','degree','betweenness'], 'Full (layer+deg+btw)'),
        (['layer','degree'], 'Layer + Degree'),
        (['layer'], 'Layer One-Hot Only'),
        (['degree','betweenness'], 'Degree + Betweenness Only'),
        ([], 'No Features (ones)'),
    ]:
        x = build_features(tedges, feats=feats)
        print(f"\n  {fname}:")
        results[fname] = run_ldo(fname, tedges, x)
        print(f"    AUC={results[fname]['mean_auc']:.4f} ± {results[fname]['std_auc']:.4f}")

    return results

# ─── Ablation 5: Degree-Preserving Rewiring ────────────────────
def ablation_rewire():
    """Rewire edges while preserving degree distribution."""
    print(f"\n{'='*60}")
    print("Ablation 5: Degree-Preserving Edge Rewiring")
    print(f"{'='*60}")

    tedges = build_typed_edges(g)
    for etype in range(3):
        if etype not in tedges: continue
        src, dst = tedges[etype]
        # Swap target endpoint while preserving source degree
        new_dst = dst[torch.randperm(len(dst))]
        tedges[etype] = (src, new_dst)

    x = build_features(tedges, feats=['layer','degree','betweenness'])
    return run_ldo("Rewired Graph", tedges, x)

# ─── Run all ablations ──────────────────────────────────────────
results = {}

# Control: original
print(f"\n{'='*60}")
print("Control: Original 4-Layer Graph")
print(f"{'='*60}")
x_orig = build_features(original_typed, feats=['layer','degree','betweenness'])
results['original'] = run_ldo("Original", original_typed, x_orig)
print(f"  AUC={results['original']['mean_auc']:.4f} ± {results['original']['std_auc']:.4f}")

results['remove_pathway'] = ablation_remove_pathway()
results['randomize_tp'] = ablation_randomize_tp()
results['randomize_pd'] = ablation_randomize_pd()
results['rewire'] = ablation_rewire()
results['feature_ablation'] = ablation_features()

# ─── Summary ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Ablation Summary")
print(f"{'='*60}")
for name, r in results.items():
    if name == 'feature_ablation':
        print(f"\n  Feature Ablation:")
        for fname, fr in r.items():
            print(f"    {fname:30s}: AUC={fr['mean_auc']:.4f} ± {fr['std_auc']:.4f}, AP={fr['mean_ap']:.4f}")
    else:
        print(f"  {name:25s}: AUC={r['mean_auc']:.4f} ± {r['std_auc']:.4f}, AP={r['mean_ap']:.4f}")

out_path = DATA / "graph_ablation_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  → {out_path}")
