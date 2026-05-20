#!/usr/bin/env python3
"""
P0: Mechanism path case studies — extract drug→target→pathway→disease paths
for top-ranked predictions to use as paper case studies.
"""
import json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cpu')

DATA=Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH=DATA/"four_layer_graph_full_v2.json"
TRAIN=DATA/"p016_train_v5_0.json"
OUT=DATA/"mechanism_case_studies.json"

HIDDEN=128; N_LAYERS=2; DROPOUT=0.4; LR=0.005; POS_WEIGHT=3.0

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

# Build typed edges & mappings
dt_map=defaultdict(set)  # drug -> targets
tp_map=defaultdict(set)  # target -> pathways
pd_map=defaultdict(set)  # pathway -> diseases

typed_edges={}
for etype,edges in [(0,g['drug_target_edges']),(1,g['target_pathway_edges']),(2,g['pathway_disease_edges'])]:
    srcs,dsts=[],[]
    for s,t in edges:
        if etype==0:
            if s not in drug_idx or t not in targ_idx: continue
            srcs+=[off('drug',drug_idx[s]),off('target',targ_idx[t])]
            dsts+=[off('target',targ_idx[t]),off('drug',drug_idx[s])]
            dt_map[s].add(t)
        elif etype==1:
            if s not in targ_idx or t not in pw_idx: continue
            srcs+=[off('target',targ_idx[s]),off('pathway',pw_idx[t])]
            dsts+=[off('pathway',pw_idx[t]),off('target',targ_idx[s])]
            tp_map[s].add(t)
        else:
            if s not in pw_idx or t not in dis_idx: continue
            srcs+=[off('pathway',pw_idx[s]),off('disease',dis_idx[t])]
            dsts+=[off('disease',dis_idx[t]),off('pathway',pw_idx[s])]
            pd_map[s].add(t)
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

# AttnHeteroGNN (same as before)
class AttnHeteroGNNLayer(nn.Module):
    def __init__(self,in_dim,out_dim,n_types=3,dropout=0.3):
        super().__init__()
        self.W_msg=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(n_types)])
        self.W_self=nn.ModuleList([nn.Linear(in_dim,out_dim) for _ in range(4)])
        self.q=nn.Parameter(torch.randn(out_dim))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,typed_edges):
        type_msgs=[]
        for etype,(src,dst) in typed_edges.items():
            msg=self.W_msg[etype](x[src])
            agg=torch.zeros(x.size(0),self.W_msg[0].out_features)
            agg=agg.index_add(0,dst,msg)
            deg=torch.bincount(dst,minlength=x.size(0)).float().clamp(min=1)
            type_msgs.append(agg/deg.unsqueeze(1))
        if not type_msgs:
            sl=torch.zeros(x.size(0),self.W_msg[0].out_features)
            sl[:ND]=self.W_self[0](x[:ND]); sl[ND:ND+NT]=self.W_self[1](x[ND:ND+NT])
            sl[ND+NT:ND+NT+NP]=self.W_self[2](x[ND+NT:ND+NT+NP]); sl[ND+NT+NP:]=self.W_self[3](x[ND+NT+NP:])
            return F.relu(self.dropout(sl))
        stacked=torch.stack(type_msgs,dim=1)
        attn_scores=torch.einsum('tnd,d->tn',stacked,self.q)
        attn=F.softmax(attn_scores/np.sqrt(self.q.size(0)),dim=1).unsqueeze(-1)
        out=(stacked*attn).sum(dim=1)
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

# Train full model
print("Training AttnHeteroGNN...")
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive': pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

model=AttnHeteroGNN(IN_DIM,HIDDEN,N_LAYERS,DROPOUT)
opt=torch.optim.Adam(model.parameters(),lr=LR)
best_loss,pat=float('inf'),0
for ep in range(400):
    model.train(); opt.zero_grad()
    h=model(x,typed_edges)
    samp=random.sample(all_pairs,min(8000,len(all_pairs)))
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
    if pat>=30: break

# ─── Score all drug-TMJOA pairs ─────────────────────────────────
model.eval()
with torch.no_grad():
    h=model(x,typed_edges)
    scores=[]
    for d in drugs:
        if 'TMJOA' not in diseases: continue
        s=torch.sigmoid(model.predict(h, torch.LongTensor([drug_idx[d]]), torch.LongTensor([dis_idx['TMJOA']]))).item()
        scores.append((d,s))
scores.sort(key=lambda x:-x[1])

# ─── Extract mechanism paths ────────────────────────────────────
def trace_paths(drug, disease, max_depth=2):
    """Trace all drug→target→pathway→disease paths."""
    paths=[]
    for target in dt_map.get(drug,[]):
        for pathway in tp_map.get(target,[]):
            if disease in pd_map.get(pathway,[]):
                # Direct path
                paths.append({
                    'path':[drug,target,pathway,disease],
                    'target_pathway':f"{target}→{pathway}",
                    'depth':3
                })
            elif max_depth>1:
                # Target contributes to disease-relevant pathway
                # Check if pathway connects to any disease-relevant pathway via shared targets
                for t2 in tp_map.get(target,[]):
                    if disease in pd_map.get(t2,[]):
                        paths.append({
                            'path':[drug,target,t2,disease],
                            'target_pathway':f"{target}→{t2}",
                            'depth':3
                        })
    # Also find targets that are NOT directly connected to disease pathways
    orphan_targets=[t for t in dt_map.get(drug,[]) if not any(
        disease in pd_map.get(p,[]) for p in tp_map.get(t,[])
    )]
    return paths, orphan_targets

# ─── Case studies ───────────────────────────────────────────────
case_drugs=[
    ('BGJ398','FGFR inhibitor — only prediction with 0 PubMed hits, truly novel'),
    ('Chondroitin','known cartilage-protective agent, validated by literature'),
    ('Metformin','AMPK activator, repurposing candidate'),
    ('Quercetin','flavonoid with NF-κB/COX-2 mechanism'),
    ('Dextrose','prolotherapy agent, top-ranked prediction'),
]

target_disease='TMJOA'
case_studies=[]

for drug_name, rationale in case_drugs:
    score=next((s for d,s in scores if d==drug_name),0)
    paths,orphans=trace_paths(drug_name, target_disease)
    
    study={
        'drug':drug_name,
        'rationale':rationale,
        'score':float(score),
        'direct_mechanism_paths':paths,
        'orphan_targets':orphans,
        'graph_connectivity':{
            'n_targets':len(dt_map.get(drug_name,[])),
            'n_pathways':len(set(p for t in dt_map.get(drug_name,[]) for p in tp_map.get(t,[]))),
            'n_disease_paths':len(paths)
        }
    }
    case_studies.append(study)
    
    print(f"\n{'─'*60}")
    print(f"Case: {drug_name} → {target_disease}")
    print(f"  Rationale: {rationale}")
    print(f"  Score: {score:.4f}")
    print(f"  Targets: {sorted(dt_map.get(drug_name,[]))}")
    if paths:
        for p in paths[:5]:
            print(f"  Path: {' → '.join(p['path'])}")
    if orphans:
        print(f"  Orphan targets (no TMJOA path): {orphans}")

# Also extract attention weights
print(f"\n{'─'*60}")
print("Type-Level Attention Weights")
model.eval()
with torch.no_grad():
    # Get attention from first layer
    layer0=model.layers[0]
    type_msgs=[]
    etype_order=[]
    for etype,(src,dst) in sorted(typed_edges.items()):
        msg=layer0.W_msg[etype](x[src])
        agg=torch.zeros(x.size(0),HIDDEN)
        agg=agg.index_add(0,dst,msg)
        deg=torch.bincount(dst,minlength=x.size(0)).float().clamp(min=1)
        type_msgs.append(agg/deg.unsqueeze(1))
        etype_order.append({0:'Drug→Target',1:'Target→Pathway',2:'Pathway→Disease'}[etype])
    stacked=torch.stack(type_msgs,dim=1)
    attn_scores=torch.einsum('tnd,d->tn',stacked,layer0.q)
    attn=F.softmax(attn_scores/np.sqrt(HIDDEN),dim=1)
    avg_attn=attn.mean(dim=0)
    
    attn_weights={
        etype_order[i]:float(avg_attn[i]) for i in range(len(etype_order))
    }
    for k,v in attn_weights.items():
        print(f"  {k}: {v:.4f}")

# Save
result={'target_disease':target_disease,'case_studies':case_studies,'attention_weights':attn_weights}
with open(OUT,'w') as f: json.dump(result,f,indent=2)
print(f"\n→ {OUT}")
