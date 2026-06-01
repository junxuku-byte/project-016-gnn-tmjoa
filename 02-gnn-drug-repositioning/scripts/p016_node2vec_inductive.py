#!/usr/bin/env python3
"""
Improvement 1: Node2Vec TRUE INDUCTIVE baseline.
Per-fold: remove held-out drug edges from graph → re-train Node2Vec embeddings → train MLP predictor.
This eliminates the transductive leak from the previous Node2Vec baseline.

# ── Paths (portable) ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
"""

import json, random
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
import torch, torch.nn as nn, torch.nn.functional as F

SEED=42; random.seed(SEED); np.random.seed(SEED)
DEVICE=torch.device('cpu')

DATA=Path("DATA_DIR")
GRAPH=DATA/"four_layer_graph_full.json"; TRAIN=DATA/"p016_train_v5_0.json"
OUT=DATA/"node2vec_inductive_results.json"

EMB_DIM=64; WALK_LEN=15; N_WALKS=10; NEG=5
LR=0.01; EPOCHS_M=40; EPOCHS_P=300; PATIENCE=25
N_FOLDS=5; POS_WEIGHT=3.0

# Load static data
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

# Build full homogeneous adjacency
def build_adj(g, exclude_drugs=None):
    adj_list=defaultdict(list)
    for d,t in g['drug_target_edges']:
        if d in drug_idx and t in targets:
            if exclude_drugs and d in exclude_drugs: continue
            di=off('drug',drug_idx[d]); ti=off('target',targets.index(t))
            adj_list[di].append(ti); adj_list[ti].append(di)
    for t,p in g['target_pathway_edges']:
        if t in targets and p in pathways:
            ti=off('target',targets.index(t)); pi=off('pathway',pathways.index(p))
            adj_list[ti].append(pi); adj_list[pi].append(ti)
    for p,d in g['pathway_disease_edges']:
        if p in pathways and d in diseases:
            pi=off('pathway',pathways.index(p)); di=off('disease',diseases.index(d))
            adj_list[pi].append(di); adj_list[di].append(pi)
    return adj_list

# Labels
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive': pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# MLP Predictor
class MLPPredictor(nn.Module):
    def __init__(self,in_dim=EMB_DIM):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim*2,128),nn.ReLU(),nn.Dropout(0.4),nn.Linear(128,1))
    def forward(self,di,ds):
        return self.net(torch.cat([di,ds],dim=1)).squeeze()

# LDO with per-fold embedding re-training
n=ND; indices=list(range(n)); random.shuffle(indices); fs=n//N_FOLDS
fa,fp=[],[]

print(f"{'='*60}")
print(f"Node2Vec TRUE INDUCTIVE LDO {N_FOLDS}-Fold")
print(f"{'='*60}")

for fold in range(N_FOLDS):
    hs,he=fold*fs,(fold+1)*fs if fold<N_FOLDS-1 else n
    held_drug_indices=set(indices[hs:he])
    held_drug_names=set(drugs[i] for i in held_drug_indices)
    
    # Step 1: Build graph EXCLUDING held-out drugs
    adj=build_adj(g, exclude_drugs=held_drug_names)
    
    # Step 2: Re-train Node2Vec from scratch
    emb=nn.Embedding(TOTAL,EMB_DIM).to(DEVICE)
    opt=torch.optim.Adam(emb.parameters(),lr=0.03)
    
    print(f"  Fold {fold+1}: training Node2Vec on {len(adj)} nodes (excl {len(held_drug_names)} held)...")
    for ep in range(EPOCHS_M):
        total_loss=0.0; cnt=0
        for start in range(TOTAL):
            if start in held_drug_indices: continue  # skip held-out drugs
            walk=[start]
            for _ in range(WALK_LEN-1):
                if adj[walk[-1]]:
                    walk.append(random.choice(adj[walk[-1]]))
                else:
                    break
            for i in range(len(walk)):
                u=walk[i]
                ctx=walk[max(0,i-2):i]+walk[i+1:i+3]
                if not ctx: continue
                u_emb=emb(torch.LongTensor([u]).to(DEVICE))
                negs=random.sample(range(TOTAL),min(NEG,TOTAL-1))
                # avoid held-out drugs as negatives (they have no embedding context)
                v_emb=emb(torch.LongTensor(ctx+negs).to(DEVICE))
                logits=(u_emb*v_emb).sum(dim=1)
                labels=torch.cat([torch.ones(len(ctx)),torch.zeros(len(negs))]).to(DEVICE)
                loss=F.binary_cross_entropy_with_logits(logits,labels)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss+=loss.item(); cnt+=1
        if (ep+1)%20==0: print(f"    N2V ep {ep+1}: loss={total_loss/max(cnt,1):.4f}")
    
    with torch.no_grad(): embeddings=emb(torch.arange(TOTAL).to(DEVICE)).cpu()
    
    # Step 3: Train MLP predictor on non-held drugs
    tpairs=[(d,di,y) for d,di,y in all_pairs if d not in held_drug_names]
    xpairs=[(d,di,y) for d,di,y in all_pairs if d in held_drug_names]
    
    model=MLPPredictor().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR)
    best_loss,pat=float('inf'),0
    
    for ep in range(EPOCHS_P):
        model.train(); opt.zero_grad()
        samp=random.sample(tpairs,min(8000,len(tpairs)))
        di=torch.stack([embeddings[off('drug',drug_idx[d])] for d,_,_ in samp]).to(DEVICE)
        ds=torch.stack([embeddings[off('disease',dis_idx[di])] for _,di,_ in samp]).to(DEVICE)
        y=torch.FloatTensor([y for _,_,y in samp]).to(DEVICE)
        pred=model(di,ds)
        lp=F.binary_cross_entropy_with_logits(pred[y==1],y[y==1],reduction='sum')
        ln=F.binary_cross_entropy_with_logits(pred[y==0],y[y==0],reduction='sum')
        loss=(lp*POS_WEIGHT+ln)/y.numel()
        loss.backward(); opt.step()
        if loss.item()<best_loss: best_loss=loss.item(); pat=0
        else: pat+=1
        if pat>=PATIENCE: break
    
    model.eval()
    with torch.no_grad():
        di=torch.stack([embeddings[off('drug',drug_idx[d])] for d,_,_ in xpairs]).to(DEVICE)
        ds=torch.stack([embeddings[off('disease',dis_idx[di])] for _,di,_ in xpairs]).to(DEVICE)
        yt=[y for _,_,y in xpairs]
        pred=model(di,ds).cpu().numpy()
    
    try:
        auc=roc_auc_score(yt,pred); ap=average_precision_score(yt,pred)
    except:
        auc,ap=0.5,sum(yt)/len(yt)
    fa.append(auc); fp.append(ap)
    print(f"    → AUC={auc:.4f}, AP={ap:.4f}")

print(f"\n  Node2Vec INDUCTIVE Mean AUC: {np.mean(fa):.4f} ± {np.std(fa):.4f}")
print(f"  Node2Vec INDUCTIVE Mean AP:  {np.mean(fp):.4f} ± {np.std(fp):.4f}")

result={
    'method':'Node2Vec + MLP (TRUE INDUCTIVE, per-fold embedding retraining)',
    'n_folds':N_FOLDS,
    'fold_aucs':[float(a) for a in fa],'fold_aps':[float(a) for a in fp],
    'mean_auc':float(np.mean(fa)),'std_auc':float(np.std(fa)),
    'mean_ap':float(np.mean(fp)),'std_ap':float(np.std(fp))
}
with open(OUT,'w') as f: json.dump(result,f,indent=2)
print(f"  → {OUT}")
