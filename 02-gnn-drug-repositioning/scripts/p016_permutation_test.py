#!/usr/bin/env python3

# ── Paths (portable) ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
"""Phase 3d: Permutation test for A+B+C HeteroGNN LDO AUC significance."""

import json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA=Path("DATA_DIR")
GRAPH=DATA/"four_layer_graph_full.json"; TRAIN=DATA/"p016_train_v5_0.json"
OUT=DATA/"permutation_test_results.json"

HIDDEN=128; N_LAYERS=2; DROPOUT=0.4; LR=0.005; EPOCHS=200; PATIENCE=20
N_PERM=500  # 500 permutations (enough for p<0.01 resolution)
POS_WEIGHT=3.0

# ─── Load data once ─────────────────────────────────────────────
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

# ─── Build typed edges ─────────────────────────────────────────
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

# ─── Topological features ──────────────────────────────────────
all_src=[]; all_dst=[]
for _, (s,d) in typed_edges.items(): all_src.extend(s.tolist()); all_dst.extend(d.tolist())
all_src=torch.LongTensor(all_src); all_dst=torch.LongTensor(all_dst)
deg=torch.bincount(all_src,minlength=TOTAL).float()+torch.bincount(all_dst,minlength=TOTAL).float()
deg_norm=torch.log1p(deg)/torch.log1p(deg).max()
hit_counts=torch.zeros(TOTAL)
adj=torch.zeros(TOTAL,TOTAL); adj[all_src,all_dst]=1; adj[all_dst,all_src]=1
for _ in range(100):
    cur=random.randrange(TOTAL)
    for _ in range(10):
        nbrs=(adj[cur]>0).nonzero(as_tuple=True)[0]
        if len(nbrs)==0: break
        cur=nbrs[random.randrange(len(nbrs))].item()
        hit_counts[cur]+=1
btw=hit_counts/hit_counts.max().clamp(min=1)
layer_onehot=torch.zeros(TOTAL,4)
layer_onehot[:ND,0]=1; layer_onehot[ND:ND+NT,1]=1
layer_onehot[ND+NT:ND+NT+NP,2]=1; layer_onehot[ND+NT+NP:,3]=1
x=torch.cat([layer_onehot, deg_norm.unsqueeze(1), btw.unsqueeze(1)], dim=1)
IN_DIM=x.size(1)

# ─── Labels ─────────────────────────────────────────────────────
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive': pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# ─── Model ─────────────────────────────────────────────────────
class HeteroGNNLayer(nn.Module):
    def __init__(self,in_dim,out_dim,n_types=3,dropout=0.3):
        super().__init__()
        self.W_msg=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(n_types)])
        self.W_self=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(4)])
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,te):
        out=torch.zeros(x.size(0),self.W_msg[0].out_features)
        dt=torch.zeros(x.size(0))
        for et,(src,dst) in te.items():
            out=out.index_add(0,dst,self.W_msg[et](x[src]))
            dt+=torch.bincount(dst,minlength=x.size(0)).float().clamp(min=1)
        dt=dt.clamp(min=1); out=out/dt.unsqueeze(1)
        sl=torch.zeros_like(out)
        sl[:ND]=self.W_self[0](x[:ND]); sl[ND:ND+NT]=self.W_self[1](x[ND:ND+NT])
        sl[ND+NT:ND+NT+NP]=self.W_self[2](x[ND+NT:ND+NT+NP]); sl[ND+NT+NP:]=self.W_self[3](x[ND+NT+NP:])
        return F.relu(self.dropout(out+sl))

class HeteroGNN(nn.Module):
    def __init__(self,in_dim,hidden,n_layers,dropout):
        super().__init__()
        self.layers=nn.ModuleList([HeteroGNNLayer(in_dim,hidden,dropout=dropout)])
        for _ in range(n_layers-1): self.layers.append(HeteroGNNLayer(hidden,hidden,dropout=dropout))
        self.predictor=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x,te):
        for l in self.layers: x=l(x,te)
        return x
    def predict(self,h,di,ds):
        return self.predictor(torch.cat([h[di],h[ds]],dim=1)).squeeze()

# ─── Train & eval one fold ──────────────────────────────────────
def run_fold(train_pairs, test_pairs, fold_typed):
    model=HeteroGNN(IN_DIM,HIDDEN,N_LAYERS,DROPOUT)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    best_loss,pat=float('inf'),0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        h=model(x,fold_typed if fold_typed else typed_edges)
        samp=random.sample(train_pairs,min(8000,len(train_pairs)))
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
        di=torch.LongTensor([drug_idx[d] for d,_,_ in test_pairs])
        ds=torch.LongTensor([dis_idx[di] for _,di,_ in test_pairs])
        yt=[y for _,_,y in test_pairs]
        pred=model.predict(h,di,ds).cpu().numpy()
    try: return roc_auc_score(yt,pred)
    except: return 0.5

# ─── Observed AUC (5-fold LDO) ─────────────────────────────────
print("Computing observed AUC...")
n=ND; indices=list(range(n)); random.shuffle(indices); fs=n//5
observed_aucs=[]

for fold in range(5):
    hs,he=fold*fs,(fold+1)*fs if fold<4 else n
    held=set(indices[hs:he])
    ft={}
    for et,(src,dst) in typed_edges.items():
        m=torch.ones(len(src),dtype=torch.bool)
        for i in range(len(src)):
            if (src[i]<ND and src[i].item() in held) or (dst[i]<ND and dst[i].item() in held): m[i]=False
        if m.sum()>0: ft[et]=(src[m],dst[m])
    tpairs=[(d,di,y) for d,di,y in all_pairs if drug_idx[d] not in held]
    xpairs=[(d,di,y) for d,di,y in all_pairs if drug_idx[d] in held]
    auc=run_fold(tpairs,xpairs,ft)
    observed_aucs.append(auc)
    print(f"  Observed Fold {fold+1}: AUC={auc:.4f}")

obs_mean=float(np.mean(observed_aucs))
obs_std=float(np.std(observed_aucs))
print(f"\nObserved: AUC = {obs_mean:.4f} ± {obs_std:.4f}")

# ─── Permutation test ──────────────────────────────────────────
print(f"\nRunning {N_PERM} permutations...")
null_aucs=[]

for perm_idx in range(N_PERM):
    # Shuffle labels across all pairs (but keep drugs structure)
    ys=[y for _,_,y in all_pairs]
    random.shuffle(ys)
    perm_pairs=[(d,di,ys[i]) for i,(d,di,_) in enumerate(all_pairs)]
    
    fold_aucs=[]
    for fold in range(5):
        hs,he=fold*fs,(fold+1)*fs if fold<4 else n
        held=set(indices[hs:he])
        tpairs=[(d,di,y) for d,di,y in perm_pairs if drug_idx[d] not in held]
        xpairs=[(d,di,y) for d,di,y in perm_pairs if drug_idx[d] in held]
        if not xpairs or not tpairs: continue
        ft={}
        for et,(src,dst) in typed_edges.items():
            m=torch.ones(len(src),dtype=torch.bool)
            for i in range(len(src)):
                if (src[i]<ND and src[i].item() in held) or (dst[i]<ND and dst[i].item() in held): m[i]=False
            if m.sum()>0: ft[et]=(src[m],dst[m])
        auc=run_fold(tpairs,xpairs,ft)
        fold_aucs.append(auc)
    
    null_mean=np.mean(fold_aucs)
    null_aucs.append(null_mean)
    
    if (perm_idx+1) % 50 == 0:
        p_tmp=sum(1 for v in null_aucs if v>=obs_mean)/len(null_aucs)
        print(f"  Perm {perm_idx+1}/{N_PERM}: null_mean={null_mean:.4f}, running_p={p_tmp:.3f}")

# ─── Compute p-value ────────────────────────────────────────────
p_value = sum(1 for v in null_aucs if v >= obs_mean) / len(null_aucs)

print(f"\n{'='*60}")
print(f"Permutation Test Results")
print(f"{'='*60}")
print(f"  Observed AUC:  {obs_mean:.4f} ± {obs_std:.4f}")
print(f"  Null mean:     {np.mean(null_aucs):.4f} ± {np.std(null_aucs):.4f}")
print(f"  Null median:   {np.median(null_aucs):.4f}")
print(f"  Null [95% CI]: [{np.percentile(null_aucs,2.5):.4f}, {np.percentile(null_aucs,97.5):.4f}]")
print(f"  Permutations:  {N_PERM}")
print(f"  p-value:       {p_value:.4f} {'***' if p_value<0.001 else '**' if p_value<0.01 else '*' if p_value<0.05 else 'ns'}")

result={
    'test':'permutation_test',
    'n_permutations':N_PERM,
    'observed':{'mean_auc':obs_mean,'std_auc':obs_std,'fold_aucs':observed_aucs},
    'null':{
        'mean':float(np.mean(null_aucs)),
        'std':float(np.std(null_aucs)),
        'median':float(np.median(null_aucs)),
        'ci95':[float(np.percentile(null_aucs,2.5)),float(np.percentile(null_aucs,97.5))]
    },
    'p_value':p_value,
    'significant':p_value<0.05
}
with open(OUT,'w') as f: json.dump(result,f,indent=2)
print(f"\n  → {OUT}")
