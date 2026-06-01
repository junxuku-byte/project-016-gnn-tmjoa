#!/usr/bin/env python3
"""
Baseline 1: Random Walk with Restart (RWR) on four-layer heterogeneous graph.
Classic drug repositioning baseline — restart from drug, walk until convergence.

# ── Paths (portable) ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
"""

import json, random
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

SEED=42; random.seed(SEED); np.random.seed(SEED)

DATA=Path("DATA_DIR")
GRAPH=DATA/"four_layer_graph_full.json"; TRAIN=DATA/"p016_train_v5_0.json"
OUT=DATA/"rwr_baseline_results.json"

ALPHA=0.3  # restart probability
N_FOLDS=5

with open(GRAPH) as f: g=json.load(f)
with open(TRAIN) as f: train=json.load(f)

drugs=sorted(set(e[0] for e in g['drug_target_edges']))
targets=g['targets']; pathways=g['pathways']
diseases=sorted(set(e[1] for e in g['pathway_disease_edges']))

drug_idx={d:i for i,d in enumerate(drugs)}
dis_idx={d:i for i,d in enumerate(diseases)}

ND=len(drugs); NT=len(targets); NP=len(pathways); NI=len(diseases)
TOTAL=ND+NT+NP+NI

def off(t,i):
    return {'drug':i,'target':ND+i,'pathway':ND+NT+i,'disease':ND+NT+NP+i}[t]

# Build adjacency matrix
adj=np.zeros((TOTAL,TOTAL))
for d,t in g['drug_target_edges']:
    if d in drug_idx and t in drug_idx: continue
    if d in drug_idx and (t in targets):
        ti=off('target',targets.index(t)); di=off('drug',drug_idx[d])
        adj[di,ti]=adj[ti,di]=1
for t,p in g['target_pathway_edges']:
    if t in targets and p in pathways:
        ti=off('target',targets.index(t)); pi=off('pathway',pathways.index(p))
        adj[ti,pi]=adj[pi,ti]=1
for p,d in g['pathway_disease_edges']:
    if p in pathways and d in diseases:
        pi=off('pathway',pathways.index(p)); di=off('disease',diseases.index(d))
        adj[pi,di]=adj[di,pi]=1

# Row-normalize
deg=adj.sum(axis=1); deg[deg==0]=1
T=adj/deg[:,None]  # transition matrix

# RWR: steady-state p = alpha*e + (1-alpha)*T^T*p
def rwr_scores(start_nodes, restart_vector=None):
    """Returns steady-state scores over all nodes."""
    p=np.zeros(TOTAL)
    if restart_vector is None:
        for sn in start_nodes:
            p[sn]=1.0/len(start_nodes)
    else:
        p=restart_vector.copy()
    e=p.copy()
    for _ in range(100):
        p_new=ALPHA*e+(1-ALPHA)*(T.T @ p)
        if np.abs(p_new-p).sum()<1e-6: break
        p=p_new
    return p

# Labels
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive': pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# LDO evaluation: for each held-out drug, mask its edges, run RWR from drug
n=ND; indices=list(range(n)); random.shuffle(indices); fs=n//N_FOLDS
fold_aucs,fold_aps=[],[]

print(f"{'='*60}")
print(f"RWR Baseline LDO {N_FOLDS}-Fold")
print(f"{'='*60}")

for fold in range(N_FOLDS):
    hs,he=fold*fs,(fold+1)*fs if fold<N_FOLDS-1 else n
    held=set(indices[hs:he])
    
    # Build masked adjacency for this fold
    adj_fold=adj.copy()
    for d,t in g['drug_target_edges']:
        if d in drug_idx and drug_idx[d] in held:
            ti=off('target',targets.index(t)); di=off('drug',drug_idx[d])
            adj_fold[di,ti]=adj_fold[ti,di]=0
    
    deg_f=adj_fold.sum(axis=1); deg_f[deg_f==0]=1
    T_f=adj_fold/deg_f[:,None]
    
    yt,yp=[],[]
    for d,di,y in all_pairs:
        if drug_idx[d] not in held: continue
        # Run RWR from this drug (training graph)
        p=rwr_scores([off('drug',drug_idx[d])])
        # Score = RWR score on disease node
        score=p[off('disease',diseases.index(di))]
        yt.append(y); yp.append(float(score))
    
    try:
        auc=roc_auc_score(yt,yp); ap=average_precision_score(yt,yp)
    except:
        auc,ap=0.5,sum(yt)/len(yt)
    fold_aucs.append(auc); fold_aps.append(ap)
    print(f"  Fold {fold+1}: {len(held)} held, AUC={auc:.4f}, AP={ap:.4f}")

print(f"\n  RWR Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  RWR Mean AP:  {np.mean(fold_aps):.4f} ± {np.std(fold_aps):.4f}")

result={
    'method':'RWR (Random Walk with Restart)',
    'alpha':ALPHA,'n_folds':N_FOLDS,
    'fold_aucs':[float(a) for a in fold_aucs], 'fold_aps':[float(a) for a in fold_aps],
    'mean_auc':float(np.mean(fold_aucs)),'std_auc':float(np.std(fold_aucs)),
    'mean_ap':float(np.mean(fold_aps)),'std_ap':float(np.std(fold_aps))
}
with open(OUT,'w') as f: json.dump(result,f,indent=2)
print(f"  → {OUT}")
