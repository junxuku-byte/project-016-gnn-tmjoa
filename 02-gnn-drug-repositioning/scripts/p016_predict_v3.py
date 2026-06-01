#!/usr/bin/env python3
"""
Project-016: Drug Repositioning Predictions on v3 Graph
HeteroGNN with 5-fold leave-drug-out cross-validation
Graph: four_layer_graph_full_v3.json (225 drugs, 176 targets, 486 DT, 379 TP, 200 PD edges)
"""

import json
import random
import time
import signal
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
GRAPH_FILE = BASE / "data" / "four_layer_graph_full_v3.json"
TRAIN_FILE = BASE / "data" / "p016_train_v5_1.json"
OUT_FILE   = BASE / "data" / "p016_predictions_v3.json"

# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEED     = 42
HIDDEN   = 128
N_LAYERS = 2
DROPOUT  = 0.4
LR       = 0.005
EPOCHS   = 400
PATIENCE = 40
N_FOLDS  = 5
DEVICE   = torch.device('cpu')

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

START_TIME = time.time()
MAX_RUNTIME = 1700  # seconds (save partial results before 30min timeout)

# ─── Partial results saver (signal + time-based) ─────────────────────────────
partial_results = {}

def save_partial(signum=None, frame=None):
    if partial_results:
        out_path = OUT_FILE.with_stem(OUT_FILE.stem + "_partial")
        with open(out_path, 'w') as f:
            json.dump(partial_results, f, indent=2)
        print(f"\n[PARTIAL] Saved to {out_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, save_partial)
signal.signal(signal.SIGINT, save_partial)

# ─── Load data ───────────────────────────────────────────────────────────────
print("=" * 70)
print("Project-016 Drug Repositioning v3 - HeteroGNN LDO CV")
print("=" * 70)

with open(GRAPH_FILE) as f: g = json.load(f)
with open(TRAIN_FILE) as f: train_data = json.load(f)

# Node lists
drugs    = sorted(set(e[0] for e in g['drug_target_edges']))
targets  = g['targets']
pathways = g['pathways']
diseases = sorted(set(e[1] for e in g['pathway_disease_edges']))

drug_idx = {d: i for i, d in enumerate(drugs)}
targ_idx = {t: i for i, t in enumerate(targets)}
pw_idx   = {p: i for i, p in enumerate(pathways)}
dis_idx  = {d: i for i, d in enumerate(diseases)}

ND = len(drugs); NT = len(targets); NP = len(pathways); NI = len(diseases)
TOTAL = ND + NT + NP + NI

print(f"\nGraph v3 stats:")
print(f"  Drugs: {ND}, Targets: {NT}, Pathways: {NP}, Diseases: {NI}")
print(f"  DT edges: {len(g['drug_target_edges'])}")
print(f"  TP edges: {len(g['target_pathway_edges'])}")
print(f"  PD edges: {len(g['pathway_disease_edges'])}")
print(f"  Total nodes: {TOTAL}")

# ─── Node offset helper ───────────────────────────────────────────────────────
def off(t, i):
    return {'drug': i, 'target': ND+i, 'pathway': ND+NT+i, 'disease': ND+NT+NP+i}[t]

# ─── Build typed edges ────────────────────────────────────────────────────────
def build_typed_edges(exclude_drug_indices=None):
    """Build typed edge tensors. Optionally exclude drug nodes from DT edges."""
    typed = {}
    excl = set(exclude_drug_indices) if exclude_drug_indices else set()
    
    # Type 0: Drug-Target (bidirectional)
    srcs, dsts = [], []
    for s, t in g['drug_target_edges']:
        if s not in drug_idx or t not in targ_idx: continue
        di = drug_idx[s]
        if di in excl: continue
        srcs.append(off('drug', di));        dsts.append(off('target', targ_idx[t]))
        srcs.append(off('target', targ_idx[t])); dsts.append(off('drug', di))
    if srcs:
        typed[0] = (torch.LongTensor(srcs), torch.LongTensor(dsts))
    
    # Type 1: Target-Pathway (bidirectional)
    srcs, dsts = [], []
    for s, t in g['target_pathway_edges']:
        if s not in targ_idx or t not in pw_idx: continue
        srcs.append(off('target', targ_idx[s])); dsts.append(off('pathway', pw_idx[t]))
        srcs.append(off('pathway', pw_idx[t])); dsts.append(off('target', targ_idx[s]))
    if srcs:
        typed[1] = (torch.LongTensor(srcs), torch.LongTensor(dsts))
    
    # Type 2: Pathway-Disease (bidirectional)
    srcs, dsts = [], []
    for s, t in g['pathway_disease_edges']:
        if s not in pw_idx or t not in dis_idx: continue
        srcs.append(off('pathway', pw_idx[s])); dsts.append(off('disease', dis_idx[t]))
        srcs.append(off('disease', dis_idx[t])); dsts.append(off('pathway', pw_idx[s]))
    if srcs:
        typed[2] = (torch.LongTensor(srcs), torch.LongTensor(dsts))
    
    return typed

# Precompute full graph edges
full_edges = build_typed_edges()
print(f"\nEdge counts (bidirectional):")
for k, (s, d) in full_edges.items():
    print(f"  Type {k}: {len(s)}")

# ─── Ground truth labels from training data ───────────────────────────────────
# Collect from training set
if 'splits' in train_data and 'train' in train_data['splits']:
    train_items = train_data['splits']['train']
elif isinstance(train_data, list):
    train_items = train_data
else:
    # Try all items
    train_items = []
    for key in ['train', 'all', 'items']:
        if key in train_data:
            train_items = train_data[key]
            break
    if not train_items:
        # flat list of objects
        for v in train_data.values():
            if isinstance(v, list):
                train_items = v
                break

pos_labels = defaultdict(set)
for it in train_items:
    lbl = it.get('label', 0)
    conc = it.get('conclusion', '')
    if lbl == 1.0 or conc == 'positive':
        d, di = it.get('drug'), it.get('disease')
        if d and di:
            pos_labels[d].add(di)

print(f"\nPositive labels: {sum(len(v) for v in pos_labels.values())} (drug-disease pairs)")
print(f"Drugs with positives: {len(pos_labels)}")

# All pairs for prediction
all_pairs = [
    (d, di, 1 if di in pos_labels.get(d, set()) else 0)
    for d in drugs for di in diseases
]
print(f"Total prediction pairs: {len(all_pairs)} ({ND} drugs × {NI} diseases)")

# ─── Node features: one-hot layer type (4-dim) ───────────────────────────────
x_base = torch.zeros(TOTAL, 4)
x_base[:ND, 0] = 1
x_base[ND:ND+NT, 1] = 1
x_base[ND+NT:ND+NT+NP, 2] = 1
x_base[ND+NT+NP:, 3] = 1

# ─── HeteroGNN Model ─────────────────────────────────────────────────────────
class HeteroGNNLayer(nn.Module):
    """One heterogeneous conv layer: per-edge-type weights + type-aware aggregation."""
    def __init__(self, in_dim, out_dim, n_types=3, dropout=0.3):
        super().__init__()
        self.out_dim = out_dim
        self.W_msg  = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_types)])
        self.W_self = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(4)])  # per node type
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.LayerNorm(out_dim)

    def forward(self, x, typed_edges):
        out = torch.zeros(x.size(0), self.out_dim)
        deg_total = torch.zeros(x.size(0))

        for etype, (src, dst) in typed_edges.items():
            msg = self.W_msg[etype](x[src])
            out = out.index_add(0, dst, msg)
            deg = torch.bincount(dst, minlength=x.size(0)).float()
            deg_total += deg

        deg_total = deg_total.clamp(min=1)
        out = out / deg_total.unsqueeze(1)

        # Self-loop per node type
        self_loop = torch.zeros(x.size(0), self.out_dim)
        self_loop[:ND] = self.W_self[0](x[:ND])
        self_loop[ND:ND+NT] = self.W_self[1](x[ND:ND+NT])
        self_loop[ND+NT:ND+NT+NP] = self.W_self[2](x[ND+NT:ND+NT+NP])
        self_loop[ND+NT+NP:] = self.W_self[3](x[ND+NT+NP:])

        h = out + self_loop
        h = self.bn(h)
        return F.relu(self.dropout(h))


class HeteroGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGNNLayer(in_dim, hidden, dropout=dropout))
        for _ in range(n_layers - 1):
            self.layers.append(HeteroGNNLayer(hidden, hidden, dropout=dropout))
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, typed_edges):
        for layer in self.layers:
            x = layer(x, typed_edges)
        return x

    def predict(self, h, d_indices, di_indices):
        return self.predictor(torch.cat([h[d_indices], h[di_indices]], dim=1)).squeeze(-1)


# ─── Mechanism paths ─────────────────────────────────────────────────────────
def compute_mechanism_paths(drug, disease_list=None):
    """Count drug→target→pathway→disease walks for each disease."""
    targets_of_drug = set()
    for s, t in g['drug_target_edges']:
        if s == drug and t in targ_idx:
            targets_of_drug.add(t)
    
    path_counts = defaultdict(int)
    for tgt in targets_of_drug:
        pathways_of_target = set()
        for s, t in g['target_pathway_edges']:
            if s == tgt and t in pw_idx:
                pathways_of_target.add(t)
        for pw in pathways_of_target:
            for s, t in g['pathway_disease_edges']:
                if s == pw and t in dis_idx:
                    if disease_list is None or t in disease_list:
                        path_counts[t] += 1
    
    return dict(path_counts)


# ─── Drug coverage ────────────────────────────────────────────────────────────
def compute_drug_coverage():
    coverage = {}
    for drug in drugs:
        n_tgt = len(set(t for s, t in g['drug_target_edges'] if s == drug))
        connected_pathways = set()
        for s, t in g['drug_target_edges']:
            if s == drug:
                for s2, t2 in g['target_pathway_edges']:
                    if s2 == t:
                        connected_pathways.add(t2)
        coverage[drug] = {'n_targets': n_tgt, 'n_pathways': len(connected_pathways)}
    return coverage


# ─── 5-Fold Leave-Drug-Out CV ─────────────────────────────────────────────────
drug_indices = list(range(ND))
random.shuffle(drug_indices)
folds = [drug_indices[i::N_FOLDS] for i in range(N_FOLDS)]

# Accumulate predictions across folds
all_drug_disease_scores = defaultdict(lambda: defaultdict(list))  # drug -> disease -> [scores]

fold_metrics = []

print(f"\n{'='*70}")
print(f"Starting 5-Fold Leave-Drug-Out CV  (EPOCHS={EPOCHS}, PATIENCE={PATIENCE})")
print(f"{'='*70}\n")

for fold_idx, test_drug_indices in enumerate(folds):
    fold_start = time.time()
    
    if time.time() - START_TIME > MAX_RUNTIME:
        print(f"\n[TIME LIMIT] Reached {MAX_RUNTIME}s, saving partial results...")
        break
    
    test_drugs = set(drugs[i] for i in test_drug_indices)
    train_drug_indices = [i for i in drug_indices if i not in set(test_drug_indices)]
    
    print(f"Fold {fold_idx+1}/{N_FOLDS}: {len(test_drugs)} test drugs, {len(train_drug_indices)} train drugs")
    
    # Build training edges (exclude test drug→target connections)
    train_edges = build_typed_edges(exclude_drug_indices=test_drug_indices)
    
    # Prepare training labels
    train_pairs_fold = [
        (d, di, lbl) for (d, di, lbl) in all_pairs
        if d not in test_drugs
    ]
    
    pos_pairs = [(d, di) for (d, di, lbl) in train_pairs_fold if lbl == 1]
    neg_pairs = [(d, di) for (d, di, lbl) in train_pairs_fold if lbl == 0]
    
    # Balance: sample negatives to 3× positives
    n_pos = len(pos_pairs)
    n_neg = min(len(neg_pairs), n_pos * 3)
    random.shuffle(neg_pairs)
    neg_pairs = neg_pairs[:n_neg]
    
    if n_pos == 0:
        print(f"  Skipping fold {fold_idx+1}: no positive training examples")
        continue
    
    balanced_pairs = pos_pairs + neg_pairs
    labels_train = [1.0] * n_pos + [0.0] * n_neg
    random.seed(SEED + fold_idx)
    combined = list(zip(balanced_pairs, labels_train))
    random.shuffle(combined)
    balanced_pairs, labels_train = zip(*combined)
    
    d_train = torch.LongTensor([off('drug', drug_idx[d]) for (d, di) in balanced_pairs])
    di_train = torch.LongTensor([off('disease', dis_idx[di]) for (d, di) in balanced_pairs])
    y_train = torch.FloatTensor(labels_train)
    
    print(f"  Train: {n_pos} pos + {n_neg} neg = {len(balanced_pairs)} samples")
    
    # Initialize model
    model = HeteroGNN(in_dim=4, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        h = model(x_base, train_edges)
        logits = model.predict(h, d_train, di_train)
        loss = criterion(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                h_val = model(x_base, train_edges)
                val_logits = model.predict(h_val, d_train, di_train)
                val_loss = criterion(val_logits, y_train).item()
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE // 10:
                print(f"  Early stop at epoch {epoch+1} (loss={best_val_loss:.4f})")
                break
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss={loss.item():.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Get embeddings with FULL graph (including test drug structure via TP/PD)
    model.eval()
    with torch.no_grad():
        # Use full edges for inference (test drugs can still propagate through TP/PD)
        h_infer = model(x_base, full_edges)
    
    # Compute AUC on test fold
    test_pairs_fold = [(d, di, lbl) for (d, di, lbl) in all_pairs if d in test_drugs]
    
    if test_pairs_fold:
        d_test_idx  = torch.LongTensor([off('drug', drug_idx[d]) for (d, di, lbl) in test_pairs_fold])
        di_test_idx = torch.LongTensor([off('disease', dis_idx[di]) for (d, di, lbl) in test_pairs_fold])
        y_test = [lbl for (_, _, lbl) in test_pairs_fold]
        
        with torch.no_grad():
            test_logits = model.predict(h_infer, d_test_idx, di_test_idx)
            test_scores = torch.sigmoid(test_logits).numpy()
        
        try:
            auc = roc_auc_score(y_test, test_scores) if len(set(y_test)) > 1 else 0.5
            ap  = average_precision_score(y_test, test_scores) if sum(y_test) > 0 else 0.0
        except Exception:
            auc, ap = 0.5, 0.0
        
        print(f"  Fold {fold_idx+1} AUC={auc:.4f}  AP={ap:.4f}  [{time.time()-fold_start:.1f}s]")
        fold_metrics.append({'fold': fold_idx+1, 'auc': auc, 'ap': ap})
        
        # Store predictions for this fold
        for i, (drug, disease, lbl) in enumerate(test_pairs_fold):
            all_drug_disease_scores[drug][disease].append(float(test_scores[i]))
    
    # Update partial results
    partial_results['fold_metrics'] = fold_metrics
    partial_results['n_folds_done'] = fold_idx + 1

print(f"\n{'='*70}")
if fold_metrics:
    mean_auc = np.mean([m['auc'] for m in fold_metrics])
    mean_ap  = np.mean([m['ap']  for m in fold_metrics])
    print(f"CV Summary: AUC={mean_auc:.4f}±{np.std([m['auc'] for m in fold_metrics]):.4f}  AP={mean_ap:.4f}")

# ─── Aggregate predictions (mean score across folds) ─────────────────────────
print("\n[Aggregating predictions...]")
all_predictions = []
for drug in drugs:
    for disease in diseases:
        scores_list = all_drug_disease_scores[drug][disease]
        if scores_list:
            score = float(np.mean(scores_list))
        else:
            score = 0.0
        all_predictions.append({'drug': drug, 'disease': disease, 'score': score})

all_predictions.sort(key=lambda x: -x['score'])

# ─── Top 20 per disease ───────────────────────────────────────────────────────
print("[Computing mechanism paths and top-20 per disease...]")
top_20_per_disease = {}

# Pre-compute mechanism paths for all drugs
mech_cache = {}
for drug in drugs:
    mech_cache[drug] = compute_mechanism_paths(drug)

for disease in diseases:
    disease_preds = [p for p in all_predictions if p['disease'] == disease]
    disease_preds.sort(key=lambda x: -x['score'])
    top_20 = disease_preds[:20]
    
    result_list = []
    for p in top_20:
        drug = p['drug']
        paths = mech_cache.get(drug, {})
        result_list.append({
            'drug': drug,
            'score': p['score'],
            'mechanism_paths': paths.get(disease, 0)
        })
    top_20_per_disease[disease] = result_list

# ─── Per-drug coverage ────────────────────────────────────────────────────────
print("[Computing per-drug coverage...]")
per_drug_coverage = compute_drug_coverage()

# ─── Build final output ───────────────────────────────────────────────────────
output = {
    "metadata": {
        "graph_version": "v3",
        "graph_file": str(GRAPH_FILE.name),
        "train_file": str(TRAIN_FILE.name),
        "model": "HeteroGNN",
        "n_drugs": ND,
        "n_targets": NT,
        "n_pathways": NP,
        "n_diseases": NI,
        "dt_edges": len(g['drug_target_edges']),
        "tp_edges": len(g['target_pathway_edges']),
        "pd_edges": len(g['pathway_disease_edges']),
        "cv_folds": N_FOLDS,
        "hidden_dim": HIDDEN,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
        "lr": LR,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "fold_metrics": fold_metrics,
        "mean_auc": float(np.mean([m['auc'] for m in fold_metrics])) if fold_metrics else 0.0,
        "mean_ap": float(np.mean([m['ap'] for m in fold_metrics])) if fold_metrics else 0.0,
        "runtime_seconds": time.time() - START_TIME
    },
    "top_20_per_disease": top_20_per_disease,
    "all_predictions": all_predictions,
    "per_drug_coverage": per_drug_coverage
}

with open(OUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[OUTPUT] Saved {len(all_predictions)} predictions → {OUT_FILE}")
print(f"[RUNTIME] {time.time()-START_TIME:.1f} seconds")

# ─── Summary Report ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

target_diseases = {
    'TMJOA': ['TMJOA', 'temporomandibular joint osteoarthritis', 'TMJ OA'],
    'Osteoporosis': ['Osteoporosis', 'bone loss'],
    'Osteoarthritis': ['Osteoarthritis', 'OA', 'Knee OA', 'Hip OA'],
    'Rheumatoid arthritis': ['Rheumatoid arthritis']
}

# Canonical name → disease names mapping for top-20
canonical_map = {}
for canonical, aliases in target_diseases.items():
    for alias in aliases:
        if alias in top_20_per_disease:
            if canonical not in canonical_map:
                canonical_map[canonical] = alias

# Build top-10 per canonical disease (aggregate scores across aliases)
drug_scores_by_canonical = defaultdict(lambda: defaultdict(list))
for canonical, aliases in target_diseases.items():
    for alias in aliases:
        if alias in top_20_per_disease:
            for entry in top_20_per_disease[alias]:
                drug_scores_by_canonical[canonical][entry['drug']].append(entry['score'])

# Also from all_predictions
for canonical, aliases in target_diseases.items():
    for p in all_predictions:
        if p['disease'] in aliases:
            drug_scores_by_canonical[canonical][p['drug']].append(p['score'])

top10_by_canonical = {}
for canonical in target_diseases:
    scores = {d: max(v) for d, v in drug_scores_by_canonical[canonical].items() if v}
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:10]
    top10_by_canonical[canonical] = ranked
    
    print(f"\nTop 10 drugs for {canonical}:")
    print(f"  {'Rank':<6} {'Drug':<30} {'Score':<8} {'Paths'}")
    print(f"  {'-'*60}")
    for rank, (drug, score) in enumerate(ranked, 1):
        paths = sum(mech_cache.get(drug, {}).get(d, 0) for d in target_diseases[canonical] if d in dis_idx)
        print(f"  {rank:<6} {drug:<30} {score:.4f}   {paths}")

# Pan-disease candidates (appear in top-10 for ≥3 diseases)
print(f"\n{'='*70}")
print("PAN-DISEASE CANDIDATES (top-10 in ≥3 diseases):")
print("="*70)

drug_top10_count = defaultdict(list)
for canonical, ranked in top10_by_canonical.items():
    for drug, score in ranked:
        drug_top10_count[drug].append(canonical)

pan_disease = {d: diseases_list for d, diseases_list in drug_top10_count.items()
               if len(diseases_list) >= 3}

if pan_disease:
    pan_sorted = sorted(pan_disease.items(), key=lambda x: -len(x[1]))
    for drug, disease_list in pan_sorted:
        scores = [s for _, ranked in top10_by_canonical.items()
                  for d, s in ranked if d == drug]
        mean_s = np.mean(scores) if scores else 0
        print(f"  {drug:<35} ({len(disease_list)} diseases, mean score={mean_s:.4f})")
        print(f"    → {', '.join(disease_list)}")
else:
    print("  (No drugs found in top-10 of ≥3 diseases)")
    # Show drugs in top-10 of 2 diseases
    pan2 = {d: dl for d, dl in drug_top10_count.items() if len(dl) >= 2}
    if pan2:
        print("\n  Drugs in top-10 of ≥2 diseases:")
        for drug, disease_list in sorted(pan2.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"    {drug:<35} → {', '.join(disease_list)}")

print(f"\n{'='*70}")
print(f"Total runtime: {time.time()-START_TIME:.1f}s")
print(f"Output: {OUT_FILE}")
print("Done.")
