#!/usr/bin/env python3
"""Evaluate graph-completed model vs original on LDO."""
import json, random, time, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cpu')
DATA=Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

# Load both graphs
with open(DATA/'four_layer_graph_full_v2.json') as f: g_orig=json.load(f)
with open(DATA/'four_layer_graph_expanded_v3.json') as f: g_exp=json.load(f)

with open(DATA/'p016_train_v5_0.json') as f: train=json.load(f)

def build_graph(g):
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

    typed={}
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
        if srcs: typed[et]=(torch.LongTensor(srcs),torch.LongTensor(dsts))

    # Features
    all_src=[]; all_dst=[]
    for _,(s,d) in typed.items(): all_src.extend(s.tolist()); all_dst.extend(d.tolist())
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
    x=torch.cat([layer_oh,deg_norm.unsqueeze(1),btw.unsqueeze(1)],dim=1)

    # Labels
    items=train['splits']['train']; pos_labels=defaultdict(set)
    for it in items:
        if it.get('label')==1 or it.get('conclusion')=='positive':
            pos_labels[it['drug']].add(it['disease'])
    all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

    return typed,x,drugs,diseases,drug_idx,dis_idx,ND,all_pairs

to_orig,x_orig,dr_orig,dis_orig,di_orig,dii_orig,ND_orig,ap_orig = build_graph(g_orig)
to_exp,x_exp,dr_exp,dis_exp,di_exp,dii_exp,ND_exp,ap_exp = build_graph(g_exp)

print(f"Original: {len(dr_orig)}d {ND_orig} PD={len(g_orig['pathway_disease_edges'])} Test pairs={len(ap_orig)}")
print(f"Expanded: {len(dr_exp)}d PD={len(g_exp['pathway_disease_edges'])} Test pairs={len(ap_exp)}")

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
        sl[:ND_orig]=self.W_self[0](x[:ND_orig]); sl[ND_orig:ND_orig+len(g_orig['targets'])]=self.W_self[1](x[ND_orig:ND_orig+len(g_orig['targets'])])
        sl[ND_orig+len(g_orig['targets']):ND_orig+len(g_orig['targets'])+len(g_orig['pathways'])]=self.W_self[2](x[ND_orig+len(g_orig['targets']):ND_orig+len(g_orig['targets'])+len(g_orig['pathways'])])
        sl[ND_orig+len(g_orig['targets'])+len(g_orig['pathways']):]=self.W_self[3](x[ND_orig+len(g_orig['targets'])+len(g_orig['pathways']):])
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
def run_ldo(name,x_ft,typed_edges,tagged_drugs,pairs,drug_idx_map,dis_idx_map,ND_val):
    N_FOLDS=5; HIDDEN=128; N_LAYERS=2; DROPOUT=0.4; LR=0.005; EPOCHS=300; PATIENCE=40; PW=3.0
    nd=len(tagged_drugs); indices=list(range(nd)); random.shuffle(indices); fs=nd//N_FOLDS
    fa,fp=[],[]
    for fold in range(N_FOLDS):
        hs,he=fold*fs,(fold+1)*fs if fold<N_FOLDS-1 else nd; held=set(indices[hs:he])
        ft={}
        for et,(s,d) in typed_edges.items():
            m=torch.ones(len(s),dtype=torch.bool)
            for i in range(len(s)):
                if (s[i]<ND_val and s[i].item() in held) or (d[i]<ND_val and d[i].item() in held): m[i]=False
            if m.sum()>0: ft[et]=(s[m],d[m])
        tpairs=[(d,di,y) for d,di,y in pairs if tagged_drugs.index(d) not in held]
        xpairs=[(d,di,y) for d,di,y in pairs if tagged_drugs.index(d) in held]
        if not xpairs or not tpairs: continue
        model=HeteroGNN(x_ft.size(1),HIDDEN,N_LAYERS,DROPOUT).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        best_loss,pat=float('inf'),0
        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            h=model(x_ft,ft if ft else typed_edges)
            samp=random.sample(tpairs,min(8000,len(tpairs)))
            di=torch.LongTensor([drug_idx_map[d] for d,_,_ in samp])
            ds=torch.LongTensor([dis_idx_map[di] for _,di,_ in samp])
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
            di=torch.LongTensor([drug_idx_map[d] for d,_,_ in xpairs])
            ds=torch.LongTensor([dis_idx_map[di] for _,di,_ in xpairs])
            yt=[y for _,_,y in xpairs]
            scores=model.predict(h,di,ds).cpu().numpy()
        try: auc=roc_auc_score(yt,scores); ap=average_precision_score(yt,scores)
        except: auc,ap=0.5,sum(yt)/len(yt)
        fa.append(auc); fp.append(ap)
        print(f"  {name} Fold {fold+1}: AUC={auc:.4f} AP={ap:.4f}", flush=True)
    return {'mean_auc':float(np.mean(fa)),'std_auc':float(np.std(fa)),
            'mean_ap':float(np.mean(fp)),'std_ap':float(np.std(fp)),
            'fold_aucs':[float(a) for a in fa],'fold_aps':[float(a) for a in fp]}

# ── Run ──
print("\nOriginal graph (200 PD edges)...", flush=True)
r1=run_ldo("orig",x_orig,to_orig,dr_orig,ap_orig,di_orig,dii_orig,ND_orig)
print("\nExpanded graph (571 PD edges)...", flush=True)
r2=run_ldo("exp",x_exp,to_exp,dr_exp,ap_exp,di_exp,dii_exp,ND_orig)  # same ND

print(f"\n{'='*50}")
print(f"Original:  AUC={r1['mean_auc']:.4f}±{r1['std_auc']:.4f} AP={r1['mean_ap']:.4f}")
print(f"Expanded:  AUC={r2['mean_auc']:.4f}±{r2['std_auc']:.4f} AP={r2['mean_ap']:.4f}")

result={'original':r1,'graph_completed':r2,'pd_edges_before':200,'pd_edges_after':571,
        'connected_paths_before':29,'connected_paths_after':54}
with open(DATA/'graph_completion_results.json','w') as f: json.dump(result,f,indent=2)
print(f"→ {DATA/'graph_completion_results.json'}", flush=True)
