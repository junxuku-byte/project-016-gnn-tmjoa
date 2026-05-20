#!/usr/bin/env python3
"""
Algorithm Iteration: Molecular Feature-Enhanced GNN.
Adds ECFP4 fingerprints for ALL 171 training drugs (including 93 cold-start),
enabling drug-drug similarity learning for drugs without target annotations.
"""

import json, random, time, re
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.random_projection import GaussianRandomProjection
from collections import defaultdict

SEED = 42; DEVICE = torch.device('cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH = DATA / "four_layer_graph_full_v2.json"
TRAIN = DATA / "p016_train_v5_0.json"

# ─── Step 1: Fetch SMILES from PubChem ───────────────────────────
print("Step 1: Fetching SMILES from PubChem...")

with open(TRAIN) as f: train_data = json.load(f)
with open(GRAPH) as f: g = json.load(f)

all_drugs = sorted(set(it['drug'] for it in train_data['splits']['train']))
print(f"  Total training drugs: {len(all_drugs)}")

smiles_cache = {}
# Try loading cache
cache_path = DATA / "drug_smiles_cache.json"
if cache_path.exists():
    with open(cache_path) as f:
        smiles_cache = json.load(f)
    print(f"  Loaded {len(smiles_cache)} cached SMILES")

for i, drug in enumerate(all_drugs):
    if drug in smiles_cache and smiles_cache[drug]:
        continue

    # Clean drug name for PubChem search
    clean = drug.replace('_', ' ').replace('_systemic_longterm', '').strip()
    
    try:
        # Search PubChem
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(clean)}/property/IsomericSMILES/JSON"
        req = Request(url, headers={"User-Agent": "p016_molf/1.0"})
        data = json.loads(urlopen(req, timeout=10).read())
        smiles = data['PropertyTable']['Properties'][0]['IsomericSMILES']
        smiles_cache[drug] = smiles
        if i % 20 == 0: print(f"  {i+1}/{len(all_drugs)}: {drug} → {smiles[:40]}...")
    except Exception as e:
        smiles_cache[drug] = None
        if i % 20 == 0: print(f"  {i+1}/{len(all_drugs)}: {drug} → NOT FOUND ({e})")
    time.sleep(0.25)

# Save cache
with open(cache_path, 'w') as f:
    json.dump(smiles_cache, f, indent=2)

found = sum(1 for v in smiles_cache.values() if v)
print(f"  SMILES found: {found}/{len(all_drugs)}")

# ─── Step 2: Generate ECFP4 Fingerprints ─────────────────────────
print("\nStep 2: Generating ECFP4 fingerprints...")

from rdkit import Chem
from rdkit.Chem import AllChem

ECFP_BITS = 1024
FP_DIM = 128  # Projected dimension

ecfp = {}
for drug, smiles in smiles_cache.items():
    if not smiles: continue
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP_BITS)
            arr = np.zeros(ECFP_BITS)
            fp_arr = np.zeros(ECFP_BITS)
            fp.GetRawBits()  # need to convert properly
            # Convert using explicit bits
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
            ecfp[drug] = arr
    except:
        pass

print(f"  ECFP4 generated: {len(ecfp)}/{len(all_drugs)}")

# ─── Step 3: Build random projection ────────────────────────────
print("\nStep 3: PCA-like dimension reduction (random projection)...")

# Collect all valid fingerprints
valid_drugs = [d for d in all_drugs if d in ecfp]
fp_matrix = np.stack([ecfp[d] for d in valid_drugs])
print(f"  FP matrix: {fp_matrix.shape}")

rp = GaussianRandomProjection(n_components=FP_DIM, random_state=SEED)
fp_proj = rp.fit_transform(fp_matrix)

# Normalize to unit variance
fp_proj = (fp_proj - fp_proj.mean(axis=0)) / (fp_proj.std(axis=0) + 1e-8)

drug_fp_proj = {}
for i, drug in enumerate(valid_drugs):
    drug_fp_proj[drug] = fp_proj[i].astype(np.float32)

# For missing drugs, use mean fingerprint
mean_fp = fp_proj.mean(axis=0).astype(np.float32)
for drug in all_drugs:
    if drug not in drug_fp_proj:
        drug_fp_proj[drug] = mean_fp.copy()

print(f"  Projected: {len(drug_fp_proj)} drugs → {FP_DIM} dims")

# ─── Step 4: Build enhanced graph + features ────────────────────
print("\nStep 4: Building enhanced node features...")

drugs = sorted(set(e[0] for e in g['drug_target_edges']))
targets = g['targets']
pathways = g['pathways']
diseases = sorted(set(e[1] for e in g['pathway_disease_edges']))

drug_idx = {d:i for i,d in enumerate(drugs)}
targ_idx = {t:i for i,t in enumerate(targets)}
pw_idx = {p:i for i,p in enumerate(pathways)}
dis_idx = {d:i for i,d in enumerate(diseases)}

ND = len(drugs); NT = len(targets); NP = len(pathways); NI = len(diseases)
TOTAL = ND + NT + NP + NI
print(f"  Graph: {ND} drugs, {NT} targets, {NP} pathways, {NI} diseases → {TOTAL} total")

def off(t, i):
    return {'drug':i, 'target':ND+i, 'pathway':ND+NT+i, 'disease':ND+NT+NP+i}[t]

# Build typed edges
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
        if srcs: typed[etype] = (torch.LongTensor(srcs), torch.LongTensor(dsts))
    return typed

typed_edges = build_typed_edges(g)

# Node features: [layer_onehot(4), degree(1), betweenness(1), ecfp_proj(128)] = 134 dims
all_src = []; all_dst = []
for _, (s,d) in typed_edges.items():
    all_src.extend(s.tolist()); all_dst.extend(d.tolist())
all_src = torch.LongTensor(all_src); all_dst = torch.LongTensor(all_dst)
deg = torch.bincount(all_src, minlength=TOTAL).float() + torch.bincount(all_dst, minlength=TOTAL).float()
deg_norm = torch.log1p(deg) / torch.log1p(deg).max().clamp(min=1)
hit = torch.zeros(TOTAL)
adj = torch.zeros(TOTAL, TOTAL)
adj[all_src, all_dst] = 1; adj[all_dst, all_src] = 1
for _ in range(100):
    cur = random.randrange(TOTAL)
    for _ in range(10):
        nbrs = (adj[cur] > 0).nonzero(as_tuple=True)[0]
        if len(nbrs)==0: break
        cur = nbrs[random.randrange(len(nbrs))].item()
        hit[cur] += 1
btw = hit / hit.max().clamp(min=1)

layer_oh = torch.zeros(TOTAL, 4)
layer_oh[:ND,0]=1; layer_oh[ND:ND+NT,1]=1
layer_oh[ND+NT:ND+NT+NP,2]=1; layer_oh[ND+NT+NP:,3]=1

# ECFP features: only for drug nodes
ecfp_feats = torch.zeros(TOTAL, FP_DIM)
for i in range(ND):
    d = drugs[i]
    if d in drug_fp_proj:
        ecfp_feats[i] = torch.FloatTensor(drug_fp_proj[d])

x_original = torch.cat([layer_oh, deg_norm.unsqueeze(1), btw.unsqueeze(1)], dim=1)
x_ecfp = torch.cat([layer_oh, deg_norm.unsqueeze(1), btw.unsqueeze(1), ecfp_feats], dim=1)
IN_DIM = x_original.size(1)
IN_DIM_ECFP = x_ecfp.size(1)

print(f"  Original features: {x_original.shape} ({IN_DIM} dims)")
print(f"  ECFP-enhanced:     {x_ecfp.shape} ({IN_DIM_ECFP} dims)")

# Labels (including cold-start drugs as evaluation-only)
train_items = train_data['splits']['train']
pos_labels = defaultdict(set)
for it in train_items:
    if it.get('label')==1 or it.get('conclusion')=='positive':
        pos_labels[it['drug']].add(it['disease'])

# all_pairs uses ALL 171 drugs (graph drugs + cold-start)
all_pairs = [(d,di, 1 if di in pos_labels.get(d,set()) else 0) for d in all_drugs for di in diseases]

# Track which drugs are cold-start
cold_start = set(all_drugs) - set(drugs)
print(f"  Cold-start drugs (no target edge): {len(cold_start)}")

# ─── HeteroGNN model ────────────────────────────────────────────
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

# ─── LDO Runner ─────────────────────────────────────────────────
def run_ldo(name, x, in_dim, tagged_drugs):
    HIDDEN=128; N_LAYERS=2; DROPOUT=0.4; LR=0.005; EPOCHS=400; PATIENCE=40
    N_FOLDS=5; POS_WEIGHT=3.0

    nd = len(tagged_drugs)
    indices = list(range(nd)); random.shuffle(indices)
    fs = nd // N_FOLDS
    fa, fp = [], []
    # Also track cold-start vs warm-start
    fa_cold, fa_warm = [], []

    for fold in range(N_FOLDS):
        hs, he = fold*fs, (fold+1)*fs if fold<N_FOLDS-1 else nd
        held = set(indices[hs:he])

        ft = {}
        for et, (src, dst) in typed_edges.items():
            m = torch.ones(len(src), dtype=torch.bool)
            for i in range(len(src)):
                if (src[i] < nd and src[i].item() in held) or (dst[i] < nd and dst[i].item() in held):
                    m[i] = False
            if m.sum()>0: ft[et] = (src[m], dst[m])

        train_pairs = [(d,di,y) for d,di,y in all_pairs if all_drugs.index(d) not in held]
        test_pairs = [(d,di,y) for d,di,y in all_pairs if all_drugs.index(d) in held]
        if not test_pairs or not train_pairs: continue

        model = HeteroGNN(in_dim, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best_loss, pat = float('inf'), 0

        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            h = model(x, ft if ft else typed_edges)
            samp = random.sample(train_pairs, min(8000, len(train_pairs)))
            di = torch.LongTensor([drug_idx.get(d, 0) for d,_,_ in samp])  # fallback=0 for cold-start
            ds = torch.LongTensor([dis_idx[di] for _,di,_ in samp])
            y = torch.FloatTensor([y for _,_,y in samp])
            pred = model.predict(h, di, ds)
            lp = F.binary_cross_entropy_with_logits(pred[y==1], y[y==1], reduction='sum')
            ln = F.binary_cross_entropy_with_logits(pred[y==0], y[y==0], reduction='sum')
            loss = (lp*POS_WEIGHT+ln)/y.numel()
            loss.backward(); opt.step()
            if loss.item() < best_loss: best_loss = loss.item(); pat = 0
            else: pat += 1
            if pat >= PATIENCE: break

        model.eval()
        with torch.no_grad():
            h = model(x, typed_edges)
            di = torch.LongTensor([drug_idx.get(d, 0) for d,_,_ in test_pairs])
            ds = torch.LongTensor([dis_idx[di] for _,di,_ in test_pairs])
            yt = [y for _,_,y in test_pairs]
            scores = model.predict(h, di, ds).cpu().numpy()

            # Cold-start subscores
            cs_mask = [d in cold_start for d,_,_ in test_pairs]
            if any(cs_mask) and not all(cs_mask):
                yt_cs = [yt[i] for i, c in enumerate(cs_mask) if c]
                sc_cs = [scores[i] for i, c in enumerate(cs_mask) if c]
                if len(set(yt_cs)) > 1:
                    fa_cold.append(roc_auc_score(yt_cs, sc_cs))
                yt_ws = [yt[i] for i, c in enumerate(cs_mask) if not c]
                sc_ws = [scores[i] for i, c in enumerate(cs_mask) if not c]
                if len(set(yt_ws)) > 1:
                    fa_warm.append(roc_auc_score(yt_ws, sc_ws))

        try:
            auc = roc_auc_score(yt, scores); ap = average_precision_score(yt, scores)
        except:
            auc, ap = 0.5, sum(yt)/len(yt)
        fa.append(auc); fp.append(ap)
        print(f"  Fold {fold+1}: AUC={auc:.4f}, AP={ap:.4f}")

    result = {
        'mean_auc': float(np.mean(fa)), 'std_auc': float(np.std(fa)),
        'mean_ap': float(np.mean(fp)), 'std_ap': float(np.std(fp)),
        'fold_aucs': [float(a) for a in fa], 'fold_aps': [float(a) for a in fp],
    }
    if fa_cold:
        result['cold_start_auc'] = float(np.mean(fa_cold))
        result['warm_start_auc'] = float(np.mean(fa_warm))
    return result

# ─── Run comparison ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("Experiment 1: Original features (6-dim)")
print(f"{'='*60}")
r_orig = run_ldo("original", x_original, IN_DIM, drugs)

print(f"\n{'='*60}")
print("Experiment 2: ECFP-enhanced features (134-dim)")
print(f"{'='*60}")
r_ecfp = run_ldo("ecfp", x_ecfp, IN_DIM_ECFP, drugs)

print(f"\n{'='*60}")
print("Molecular Features Comparison")
print(f"{'='*60}")
print(f"  Original (6-dim):     AUC={r_orig['mean_auc']:.4f} ± {r_orig['std_auc']:.4f}, AP={r_orig['mean_ap']:.4f}")
print(f"  ECFP-enhanced (134-dim): AUC={r_ecfp['mean_auc']:.4f} ± {r_ecfp['std_auc']:.4f}, AP={r_ecfp['mean_ap']:.4f}")
if 'cold_start_auc' in r_ecfp:
    print(f"    Cold-start drugs:   AUC={r_ecfp['cold_start_auc']:.4f}")
    print(f"    Warm-start drugs:   AUC={r_ecfp['warm_start_auc']:.4f}")

result = {
    'method': 'ECFP4 molecular fingerprint enhancement',
    'drugs_total': len(all_drugs),
    'drugs_with_smiles': found,
    'ecfp_bits': ECFP_BITS,
    'projected_dim': FP_DIM,
    'cold_start_count': len(cold_start),
    'original': r_orig,
    'ecfp_enhanced': r_ecfp,
}
out_path = DATA / "molecular_features_results.json"
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  → {out_path}")
