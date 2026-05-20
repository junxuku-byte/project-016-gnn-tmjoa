#!/usr/bin/env python3
"""ECFP4 fingerprint feature enhancement + LDO comparison."""
import json, random, time, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.random_projection import GaussianRandomProjection
from collections import defaultdict

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cpu')
DATA=Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

# ── Load data ──
with open(DATA/'four_layer_graph_full_v2.json') as f: g=json.load(f)
with open(DATA/'p016_train_v5_0.json') as f: train=json.load(f)
with open(DATA/'drug_smiles_cache.json') as f: sc=json.load(f)

all_drugs = sorted(set(it['drug'] for it in train['splits']['train']))
drugs=sorted(set(e[0] for e in g['drug_target_edges']))
targets=g['targets']; pathways=g['pathways']
diseases=sorted(set(e[1] for e in g['pathway_disease_edges']))
drug_idx={d:i for i,d in enumerate(drugs)}
targ_idx={t:i for i,t in enumerate(targets)}
pw_idx={p:i for i,p in enumerate(pathways)}
dis_idx={d:i for i,d in enumerate(diseases)}
ND=len(drugs); NT=len(targets); NP=len(pathways); NI=len(diseases)
TOTAL=ND+NT+NP+NI
print(f"Graph: {ND}d {NT}t {NP}p {NI}i = {TOTAL} nodes", flush=True)

# ── ECFP4 fingerprints ──
print("Generating ECFP4...", flush=True)
from rdkit import Chem
from rdkit.Chem import AllChem

ECFP_BITS=1024; FP_DIM=128
ecfp={}; missing=[]
for drug in all_drugs:
    s=sc.get(drug)
    if not s: missing.append(drug); continue
    try:
        mol=Chem.MolFromSmiles(s)
        if mol:
            fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=ECFP_BITS)
            arr=np.zeros(ECFP_BITS)
            for b in fp.GetOnBits(): arr[b]=1.0
            ecfp[drug]=arr
        else: missing.append(drug)
    except: missing.append(drug)

print(f"  ECFP4: {len(ecfp)}/{len(all_drugs)}, missing: {len(missing)}", flush=True)

# Random projection
valid=[d for d in all_drugs if d in ecfp]
fp_mat=np.stack([ecfp[d] for d in valid])
rp=GaussianRandomProjection(n_components=FP_DIM,random_state=SEED)
fp_proj=rp.fit_transform(fp_mat)
fp_proj=(fp_proj-fp_proj.mean(0))/(fp_proj.std(0)+1e-8)
drug_fp={d:fp_proj[i].astype(np.float32) for i,d in enumerate(valid)}
mean_fp=fp_proj.mean(0).astype(np.float32)
for d in all_drugs:
    if d not in drug_fp: drug_fp[d]=mean_fp.copy()
print(f"  Projected: {len(drug_fp)} drugs → {FP_DIM}d", flush=True)

# ── Build typed edges ──
def off(t,i):
    return {'drug':i,'target':ND+i,'pathway':ND+NT+i,'disease':ND+NT+NP+i}[t]

typed_edges={}
for et,es in [(0,g['drug_target_edges']),(1,g['target_pathway_edges']),(2,g['pathway_disease_edges'])]:
    srcs,dsts=[],[]
    for s,t in es:
        if et==0:
            if s not in drug_idx or t not in targ_idx: continue
            a,b=off('drug',drug_idx[s]),off('target',targ_idx[t])
            srcs.extend([a,b]); dsts.extend([b,a])
        elif et==1:
            if s not in targ_idx or t not in pw_idx: continue
            a,b=off('target',targ_idx[s]),off('pathway',pw_idx[t])
            srcs.extend([a,b]); dsts.extend([b,a])
        else:
            if s not in pw_idx or t not in dis_idx: continue
            a,b=off('pathway',pw_idx[s]),off('disease',dis_idx[t])
            srcs.extend([a,b]); dsts.extend([b,a])
    if srcs: typed_edges[et]=(torch.LongTensor(srcs),torch.LongTensor(dsts))

# ── Node features ──
all_src=[]; all_dst=[]
for _,(s,d) in typed_edges.items(): all_src.extend(s.tolist()); all_dst.extend(d.tolist())
all_src=torch.LongTensor(all_src); all_dst=torch.LongTensor(all_dst)
deg=torch.bincount(all_src,minlength=TOTAL).float()+torch.bincount(all_dst,minlength=TOTAL).float()
deg_norm=torch.log1p(deg)/torch.log1p(deg).max().clamp(min=1)
hit=torch.zeros(TOTAL); adj=torch.zeros(TOTAL,TOTAL)
adj[all_src,all_dst]=1; adj[all_dst,all_src]=1
for _ in range(100):
    cur=random.randrange(TOTAL)
    for _ in range(10):
        nbrs=(adj[cur]>0).nonzero(as_tuple=True)[0]
        if len(nbrs)==0: break
        cur=nbrs[random.randrange(len(nbrs))].item(); hit[cur]+=1
btw=hit/hit.max().clamp(min=1)
layer_oh=torch.zeros(TOTAL,4)
layer_oh[:ND,0]=1; layer_oh[ND:ND+NT,1]=1
layer_oh[ND+NT:ND+NT+NP,2]=1; layer_oh[ND+NT+NP:,3]=1

x_orig=torch.cat([layer_oh,deg_norm.unsqueeze(1),btw.unsqueeze(1)],dim=1)
IN_ORIG=x_orig.size(1)

ecfp_feats=torch.zeros(TOTAL,FP_DIM)
for i in range(ND):
    if drugs[i] in drug_fp: ecfp_feats[i]=torch.FloatTensor(drug_fp[drugs[i]])
x_ecfp=torch.cat([layer_oh,deg_norm.unsqueeze(1),btw.unsqueeze(1),ecfp_feats],dim=1)
IN_ECFP=x_ecfp.size(1)
print(f"Features: orig={x_orig.shape}, ecfp={x_ecfp.shape}", flush=True)

# ── Labels ──
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive':
        pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in all_drugs for di in diseases]
cold_start=set(all_drugs)-set(drugs)
print(f"Cold-start drugs: {len(cold_start)}", flush=True)

# ── Model ──
class HeteroGNNLayer(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.3):
        super().__init__()
        self.W_msg=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(3)])
        self.W_self=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(4)])
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,te):
        out=torch.zeros(x.size(0),self.W_msg[0].out_features); deg_t=torch.zeros(x.size(0))
        for et,(s,d) in te.items():
            msg=self.W_msg[et](x[s]); out=out.index_add(0,d,msg)
            deg=torch.bincount(d,minlength=x.size(0)).float().clamp(min=1); deg_t+=deg
        deg_t=deg_t.clamp(min=1); out=out/deg_t.unsqueeze(1)
        sl=torch.zeros_like(out)
        sl[:ND]=self.W_self[0](x[:ND]); sl[ND:ND+NT]=self.W_self[1](x[ND:ND+NT])
        sl[ND+NT:ND+NT+NP]=self.W_self[2](x[ND+NT:ND+NT+NP]); sl[ND+NT+NP:]=self.W_self[3](x[ND+NT+NP:])
        return F.relu(self.dropout(out+sl))

class HeteroGNN(nn.Module):
    def __init__(self,in_dim,hidden=128,n_layers=2,dropout=0.4):
        super().__init__()
        self.layers=nn.ModuleList([HeteroGNNLayer(in_dim,hidden,dropout)])
        for _ in range(n_layers-1): self.layers.append(HeteroGNNLayer(hidden,hidden,dropout))
        self.pred=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x,te):
        for l in self.layers: x=l(x,te)
        return x
    def predict(self,h,di,ds):
        return self.pred(torch.cat([h[di],h[ds]],dim=1)).squeeze()

# ── LDO ──
def run_ldo(name,x_ft,in_dim,tagged_drugs,pairs_override=None):
    N_FOLDS=5; HIDDEN=128; N_LAYERS=2; DROPOUT=0.4; LR=0.005; EPOCHS=300; PATIENCE=40; PW=3.0
    nd=len(tagged_drugs); indices=list(range(nd)); random.shuffle(indices); fs=nd//N_FOLDS
    fa,fp=[],[]
    for fold in range(N_FOLDS):
        hs,he=fold*fs,(fold+1)*fs if fold<N_FOLDS-1 else nd; held=set(indices[hs:he])
        ft={}
        for et,(s,d) in typed_edges.items():
            m=torch.ones(len(s),dtype=torch.bool)
            for i in range(len(s)):
                if (s[i]<nd and s[i].item() in held) or (d[i]<nd and d[i].item() in held): m[i]=False
            if m.sum()>0: ft[et]=(s[m],d[m])
        pair_src = pairs_override if pairs_override else all_pairs
        tpairs=[(d,di,y) for d,di,y in pair_src if tagged_drugs.index(d) not in held]
        xpairs=[(d,di,y) for d,di,y in pair_src if tagged_drugs.index(d) in held]
        if not xpairs or not tpairs: continue
        model=HeteroGNN(in_dim,HIDDEN,N_LAYERS,DROPOUT).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        best_loss,pat=float('inf'),0
        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            h=model(x_ft,ft if ft else typed_edges)
            samp=random.sample(tpairs,min(8000,len(tpairs)))
            di=torch.LongTensor([drug_idx.get(d,0) for d,_,_ in samp])
            ds=torch.LongTensor([dis_idx[di] for _,di,_ in samp])
            y=torch.FloatTensor([y for _,_,y in samp])
            pred=model.predict(h,di,ds)
            lp=F.binary_cross_entropy_with_logits(pred[y==1],y[y==1],reduction='sum')
            ln=F.binary_cross_entropy_with_logits(pred[y==0],y[y==0],reduction='sum')
            loss=(lp*PW+ln)/y.numel(); loss.backward(); opt.step()
            if loss.item()<best_loss: best_loss=loss.item(); pat=0
            else: pat+=1
            if pat>=PATIENCE: break
        model.eval()
        with torch.no_grad():
            h=model(x_ft,typed_edges)
            di=torch.LongTensor([drug_idx.get(d,0) for d,_,_ in xpairs])
            ds=torch.LongTensor([dis_idx[di] for _,di,_ in xpairs])
            yt=[y for _,_,y in xpairs]
            scores=model.predict(h,di,ds).cpu().numpy()
        try: auc=roc_auc_score(yt,scores); ap=average_precision_score(yt,scores)
        except: auc,ap=0.5,sum(yt)/len(yt)
        fa.append(auc); fp.append(ap)
        print(f"  {name} Fold {fold+1}: AUC={auc:.4f} AP={ap:.4f}", flush=True)
    return {'mean_auc':float(np.mean(fa)),'std_auc':float(np.std(fa)),
            'mean_ap':float(np.mean(fp)),'std_ap':float(np.std(fp)),
            'fold_aucs':[float(a) for a in fa],'fold_aps':[float(a) for a in fp]}

# Filter pairs to graph drugs only for fair comparison
graph_pairs=[(d,di,y) for d,di,y in all_pairs if d in drugs]

# ── Run ──
print("\nOriginal features (6-dim)...", flush=True)
r_orig=run_ldo("orig",x_orig,IN_ORIG,drugs,graph_pairs)
print("\nECFP-enhanced (134-dim)...", flush=True)
r_ecfp=run_ldo("ecfp",x_ecfp,IN_ECFP,drugs,graph_pairs)

print(f"\n{'='*50}")
print(f"Original:    AUC={r_orig['mean_auc']:.4f}±{r_orig['std_auc']:.4f} AP={r_orig['mean_ap']:.4f}")
print(f"ECFP-enhanced: AUC={r_ecfp['mean_auc']:.4f}±{r_ecfp['std_auc']:.4f} AP={r_ecfp['mean_ap']:.4f}")

result={'original':r_orig,'ecfp_enhanced':r_ecfp,'smiles_found':len(ecfp),'smiles_total':len(all_drugs)}
with open(DATA/'molecular_features_results.json','w') as f: json.dump(result,f,indent=2)
print(f"→ {DATA/'molecular_features_results.json'}", flush=True)
