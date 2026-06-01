#!/usr/bin/env python3
"""
Improvement 3: Type-level Attention HeteroGNN (HAN-style).
Learns attention weights per edge type instead of simple sum aggregation.

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
OUT=DATA/"attn_hgnn_ldo_results.json"

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

# Build typed edges
typed_edges={}
for etype,edges in [(0,g['drug_target_edges']),(1,g['target_pathway_edges']),(2,g['pathway_disease_edges'])]:
    srcs,dsts=[],[]
    for s,t in edges:
        if etype==0:
            if s not in drug_idx or t not in targ_idx: continue
            srcs+=[off('drug',drug_idx[s]),off('target',targ_idx[t])]
            dsts+=[off('target',targ_idx[t]),off('drug',drug_idx[s])]
        elif etype==1:
            if s not in targ_idx or t not in pw_idx: continue
            srcs+=[off('target',targ_idx[s]),off('pathway',pw_idx[t])]
            dsts+=[off('pathway',pw_idx[t]),off('target',targ_idx[s])]
        else:
            if s not in pw_idx or t not in dis_idx: continue
            srcs+=[off('pathway',pw_idx[s]),off('disease',dis_idx[t])]
            dsts+=[off('disease',dis_idx[t]),off('pathway',pw_idx[s])]
    if srcs: typed_edges[etype]=(torch.LongTensor(srcs),torch.LongTensor(dsts))

# Topological features
all_src=[]; all_dst=[]
for _, (s,d) in typed_edges.items(): all_src.extend(s.tolist()); all_dst.extend(d.tolist())
all_src=torch.LongTensor(all_src); all_dst=torch.LongTensor(all_dst)
deg=torch.bincount(all_src,minlength=TOTAL).float()+torch.bincount(all_dst,minlength=TOTAL).float()
deg_norm=torch.log1p(deg)/torch.log1p(deg).max()
hit_counts=torch.zeros(TOTAL)
adj_mat=torch.zeros(TOTAL,TOTAL); adj_mat[all_src,all_dst]=1; adj_mat[all_dst,all_src]=1
for _ in range(100):
    cur=random.randrange(TOTAL)
    for _ in range(10):
        nbrs=(adj_mat[cur]>0).nonzero(as_tuple=True)[0]
        if len(nbrs)==0: break
        cur=nbrs[random.randrange(len(nbrs))].item()
        hit_counts[cur]+=1
btw=hit_counts/hit_counts.max().clamp(min=1)
layer_onehot=torch.zeros(TOTAL,4)
layer_onehot[:ND,0]=1; layer_onehot[ND:ND+NT,1]=1
layer_onehot[ND+NT:ND+NT+NP,2]=1; layer_onehot[ND+NT+NP:,3]=1
x=torch.cat([layer_onehot,deg_norm.unsqueeze(1),btw.unsqueeze(1)],dim=1)
IN_DIM=x.size(1)

# Labels
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive': pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# ─── Attention HeteroGNN ──────────────────────────────────────
class AttnHeteroGNNLayer(nn.Module):
    """HAN-style: per-type message passing + type-level attention aggregation."""
    def __init__(self,in_dim,out_dim,n_types=3,dropout=0.3):
        super().__init__()
        self.W_msg=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(n_types)])
        self.W_self=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(4)])
        # Type-level attention vector q
        self.q=nn.Parameter(torch.randn(out_dim))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,typed_edges):
        type_msgs=[]  # list of (TOTAL, out_dim) per edge type
        for etype,(src,dst) in typed_edges.items():
            msg=self.W_msg[etype](x[src])
            agg=torch.zeros(x.size(0),self.W_msg[0].out_features)
            agg=agg.index_add(0,dst,msg)
            deg=torch.bincount(dst,minlength=x.size(0)).float().clamp(min=1)
            agg=agg/deg.unsqueeze(1)
            type_msgs.append(agg)

        if not type_msgs:
            # No edges — just self loop
            sl=torch.zeros(x.size(0),self.W_msg[0].out_features)
            sl[:ND]=self.W_self[0](x[:ND]); sl[ND:ND+NT]=self.W_self[1](x[ND:ND+NT])
            sl[ND+NT:ND+NT+NP]=self.W_self[2](x[ND+NT:ND+NT+NP]); sl[ND+NT+NP:]=self.W_self[3](x[ND+NT+NP:])
            return F.relu(self.dropout(sl))

        # Type-level attention
        stacked=torch.stack(type_msgs,dim=1)  # (TOTAL, n_types, out_dim)
        attn_scores=torch.einsum('tnd,d->tn',stacked,self.q)  # (TOTAL, n_types)
        attn_scores=attn_scores/np.sqrt(self.q.size(0))
        attn=F.softmax(attn_scores,dim=1).unsqueeze(-1)  # (TOTAL, n_types, 1)

        # Weighted sum
        out=(stacked*attn).sum(dim=1)  # (TOTAL, out_dim)

        # Self-loop
        sl=torch.zeros_like(out)
        sl[:ND]=self.W_self[0](x[:ND]); sl[ND:ND+NT]=self.W_self[1](x[ND:ND+NT])
        sl[ND+NT:ND+NT+NP]=self.W_self[2](x[ND+NT:ND+NT+NP]); sl[ND+NT+NP:]=self.W_self[3](x[ND+NT+NP:])

        return F.relu(self.dropout(out+sl))

class AttnHeteroGNN(nn.Module):
    def __init__(self,in_dim,hidden,n_layers,dropout):
        super().__init__()
        self.layers=nn.ModuleList([AttnHeteroGNNLayer(in_dim,hidden,dropout=dropout)])
        for _ in range(n_layers-1): self.layers.append(AttnHeteroGNNLayer(hidden,hidden,dropout=dropout))
        self.predictor=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x,te):
        for l in self.layers: x=l(x,te)
        return x
    def predict(self,h,di,ds):
        return self.predictor(torch.cat([h[di],h[ds]],dim=1)).squeeze()

# ─── LDO ──────────────────────────────────────────────────────
n=ND; indices=list(range(n)); random.shuffle(indices); fs=n//N_FOLDS
fa,fp=[],[]

print(f"\n{'='*60}")
print(f"Attention HeteroGNN (HAN-style) LDO {N_FOLDS}-Fold")
print(f"{'='*60}")

for fold in range(N_FOLDS):
    hs,he=fold*fs,(fold+1)*fs if fold<N_FOLDS-1 else n
    held=set(indices[hs:he])
    ft={}
    for et,(src,dst) in typed_edges.items():
        m=torch.ones(len(src),dtype=torch.bool)
        for i in range(len(src)):
            if (src[i]<ND and src[i].item() in held) or (dst[i]<ND and dst[i].item() in held): m[i]=False
        if m.sum()>0: ft[et]=(src[m],dst[m])
    tpairs=[(d,di,y) for d,di,y in all_pairs if drug_idx[d] not in held]
    xpairs=[(d,di,y) for d,di,y in all_pairs if drug_idx[d] in held]
    if not xpairs or not tpairs: continue

    model=AttnHeteroGNN(IN_DIM,HIDDEN,N_LAYERS,DROPOUT).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    best_loss,pat=float('inf'),0

    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        h=model(x,ft if ft else typed_edges)
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
        h=model(x,typed_edges)
        di=torch.LongTensor([drug_idx[d] for d,_,_ in xpairs])
        ds=torch.LongTensor([dis_idx[di] for _,di,_ in xpairs])
        yt=[y for _,_,y in xpairs]
        pred=model.predict(h,di,ds).cpu().numpy()

    try:
        auc=roc_auc_score(yt,pred); ap=average_precision_score(yt,pred)
    except:
        auc,ap=0.5,sum(yt)/len(yt)
    fa.append(auc); fp.append(ap)
    print(f"  Fold {fold+1}: {len(held)} held, {len(xpairs)} test, AUC={auc:.4f}, AP={ap:.4f}")

print(f"\n  AttnHeteroGNN Mean AUC: {np.mean(fa):.4f} ± {np.std(fa):.4f}")
print(f"  AttnHeteroGNN Mean AP:  {np.mean(fp):.4f} ± {np.std(fp):.4f}")

result={
    'strategy':'A+B+C + HAN-style type-level attention',
    'method':'leave-drug-out (inductive)',
    'n_folds':N_FOLDS,'n_drugs':ND,
    'fold_aucs':[float(a) for a in fa],'fold_aps':[float(a) for a in fp],
    'mean_auc':float(np.mean(fa)),'std_auc':float(np.std(fa)),
    'mean_ap':float(np.mean(fp)),'std_ap':float(np.std(fp))
}
with open(OUT,'w') as f: json.dump(result,f,indent=2)
print(f"  → {OUT}")
