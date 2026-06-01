#!/usr/bin/env python3
"""
Item 1: TRUE homogeneous GNN baseline — no layer type encoding, single shared W_msg.
Verifies that the ablation baseline (0.68) is not an artifact of transductive leak.

# ── Paths (portable) ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
"""
import json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cpu')

DATA=Path("DATA_DIR")
GRAPH=DATA/"four_layer_graph_full_v3.json"  # v3 superset (225 drugs), v2 at four_layer_graph_full_v2.json
TRAIN=DATA/"p016_train_v5_0.json"
OUT=DATA/"true_homogeneous_baseline.json"

HIDDEN=128; N_LAYERS=2; DROPOUT=0.4; LR=0.005
EPOCHS=400; PATIENCE=40; N_FOLDS=5; POS_WEIGHT=3.0

with open(GRAPH) as f: g=json.load(f)
with open(TRAIN) as f: train=json.load(f)

drugs=sorted(set(e[0] for e in g['drug_target_edges']))
targets=g['targets']; pathways=g['pathways']
diseases=sorted(set(e[1] for e in g['pathway_disease_edges']))

drug_idx={d:i for i,d in enumerate(drugs)}
targ_idx={t:i for i,t in enumerate(targets)}
pw_idx={p:i for i,p in enumerate(pathways)}
dis_idx={d:i for i,d in enumerate(diseases)}
ND=len(drugs); NT=len(targets); NP=len(pathways); NI=len(diseases)
TOTAL=ND+NT+NP+NI

def off(t,i):
    return {'drug':i,'target':ND+i,'pathway':ND+NT+i,'disease':ND+NT+NP+i}[t]

# Build fully homogeneous undirected edges (no type distinction)
edges_src=[]; edges_dst=[]

for d,t in g['drug_target_edges']:
    if d in drug_idx and t in targ_idx:
        di=off('drug',drug_idx[d]); ti=off('target',targ_idx[t])
        edges_src.append(di); edges_dst.append(ti)
        edges_src.append(ti); edges_dst.append(di)
for t,p in g['target_pathway_edges']:
    if t in targ_idx and p in pw_idx:
        ti=off('target',targ_idx[t]); pi=off('pathway',pw_idx[p])
        edges_src.append(ti); edges_dst.append(pi)
        edges_src.append(pi); edges_dst.append(ti)
for p,d in g['pathway_disease_edges']:
    if p in pw_idx and d in dis_idx:
        pi=off('pathway',pw_idx[p]); di=off('disease',dis_idx[d])
        edges_src.append(pi); edges_dst.append(di)
        edges_src.append(di); edges_dst.append(pi)

edges_src=torch.LongTensor(edges_src)
edges_dst=torch.LongTensor(edges_dst)
print(f"Total edges: {len(edges_src)}")

# TRUE homogeneous features: just degree + betweenness, NO layer one-hot
deg=torch.bincount(edges_src,minlength=TOTAL).float()+torch.bincount(edges_dst,minlength=TOTAL).float()
deg_norm=torch.log1p(deg)/torch.log1p(deg).max()
hit_counts=torch.zeros(TOTAL)
adj=torch.zeros(TOTAL,TOTAL); adj[edges_src,edges_dst]=1; adj[edges_dst,edges_src]=1
for _ in range(100):
    cur=random.randrange(TOTAL)
    for _ in range(10):
        nbrs=(adj[cur]>0).nonzero(as_tuple=True)[0]
        if len(nbrs)==0: break
        cur=nbrs[random.randrange(len(nbrs))].item()
        hit_counts[cur]+=1
btw=hit_counts/hit_counts.max().clamp(min=1)
x=torch.cat([deg_norm.unsqueeze(1),btw.unsqueeze(1)],dim=1)  # 2 dims only, NO layer encoding
IN_DIM=x.size(1)
print(f"Feature dim: {IN_DIM} (degree + betweenness, NO layer one-hot)")

# TRUE homogeneous GNN: single W_msg, no type distinction
class TrueHomogGNNLayer(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.3):
        super().__init__()
        self.W_msg=nn.Linear(in_dim,out_dim)
        self.W_self=nn.Linear(in_dim,out_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,src,dst):
        msg=self.W_msg(x[src])
        out=torch.zeros(x.size(0),self.W_msg.out_features)
        out=out.index_add(0,dst,msg)
        deg=torch.bincount(dst,minlength=x.size(0)).float().clamp(min=1)
        out=out/deg.unsqueeze(1)
        return F.relu(self.dropout(out+self.W_self(x)))

class TrueHomogGNN(nn.Module):
    def __init__(self,in_dim,hidden,n_layers,dropout):
        super().__init__()
        self.layers=nn.ModuleList([TrueHomogGNNLayer(in_dim,hidden,dropout)])
        for _ in range(n_layers-1): self.layers.append(TrueHomogGNNLayer(hidden,hidden,dropout))
        self.predictor=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x,src,dst):
        for l in self.layers: x=l(x,src,dst)
        return x
    def predict(self,h,di,ds):
        return self.predictor(torch.cat([h[di],h[ds]],dim=1)).squeeze()

# Labels
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive': pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# LDO
n=ND; indices=list(range(n)); random.shuffle(indices); fs=n//N_FOLDS
fa,fp=[],[]

print(f"\n{'='*60}")
print(f"TRUE Homogeneous GNN LDO {N_FOLDS}-Fold")
print(f"{'='*60}")

for fold in range(N_FOLDS):
    hs,he=fold*fs,(fold+1)*fs if fold<N_FOLDS-1 else n
    held=set(indices[hs:he])
    
    m=torch.ones(len(edges_src),dtype=torch.bool)
    for i in range(len(edges_src)):
        if (edges_src[i]<ND and edges_src[i].item() in held) or (edges_dst[i]<ND and edges_dst[i].item() in held):
            m[i]=False
    train_src=edges_src[m]; train_dst=edges_dst[m]
    
    tpairs=[(d,di,y) for d,di,y in all_pairs if drug_idx[d] not in held]
    xpairs=[(d,di,y) for d,di,y in all_pairs if drug_idx[d] in held]
    if not xpairs or not tpairs: continue

    model=TrueHomogGNN(IN_DIM,HIDDEN,N_LAYERS,DROPOUT).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    best_loss,pat=float('inf'),0

    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        h=model(x,train_src,train_dst)
        samp=random.sample(tpairs,min(8000,len(tpairs)))
        di=torch.LongTensor([drug_idx[d] for d,_,_ in samp])
        ds=torch.LongTensor([dis_idx[di] for _,di,_ in samp])
        y=torch.FloatTensor([y for _,_,y in samp])
        pred=model.predict(h,di,ds)
        lp=F.binary_cross_entropy_with_logits(pred[y==1],y[y==1],reduction='sum')
        ln=F.binary_cross_entropy_with_logits(pred[y==0],y[y==0],reduction='sum')
        loss=(lp*POS_WEIGHT+ln)/y.numel()
        loss.backward(); opt.step()
        if loss.item()<best_loss: best_loss=loss.item(); pat=0
        else: pat+=1
        if pat>=PATIENCE: break

    model.eval()
    with torch.no_grad():
        h=model(x,edges_src,edges_dst)
        di=torch.LongTensor([drug_idx[d] for d,_,_ in xpairs])
        ds=torch.LongTensor([dis_idx[di] for _,di,_ in xpairs])
        yt=[y for _,_,y in xpairs]
        pred=model.predict(h,di,ds).cpu().numpy()

    try:
        auc=roc_auc_score(yt,pred); ap=average_precision_score(yt,pred)
    except:
        auc,ap=0.5,sum(yt)/len(yt)
    fa.append(auc); fp.append(ap)
    print(f"  Fold {fold+1}: AUC={auc:.4f}, AP={ap:.4f}")

mean_auc=float(np.mean(fa)); std_auc=float(np.std(fa))
print(f"\n  TRUE Homogeneous Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  TRUE Homogeneous Mean AP:  {float(np.mean(fp)):.4f}")

# Comparison with original "homogeneous" baseline (which used layer one-hot)
print(f"\n  Comparison:")
print(f"    Previous 'homogeneous' (with layer one-hot): AUC=0.6847")
print(f"    TRUE homogeneous (no layer encoding):     AUC={mean_auc:.4f}")
print(f"    AttnHeteroGNN:                            AUC=0.8513")
print(f"    Gap to AttnHeteroGNN:                     +{0.8513-mean_auc:.4f}")

result={
    'method':'TRUE homogeneous GNN (no layer type encoding, single W_msg, purely homogeneous)',
    'n_folds':N_FOLDS,
    'fold_aucs':[float(a) for a in fa],'fold_aps':[float(a) for a in fp],
    'mean_auc':mean_auc,'std_auc':std_auc,
    'comparison':{
        'orig_homogeneous_with_layer_encoding':0.6847,
        'true_homogeneous':mean_auc,
        'attn_hetero_gnn':0.8513
    }
}
with open(OUT,'w') as f: json.dump(result,f,indent=2)
print(f"  → {OUT}")
