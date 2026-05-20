#!/usr/bin/env python3
"""
Systematic literature validation of Top-K HeteroGNN predictions.
Searches PubMed for each predicted drug-disease pair and counts supporting evidence.
"""

import json, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from collections import defaultdict

DATA=Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH=DATA/"four_layer_graph_full.json"
OUT=DATA/"literature_validation.json"
TRAIN=DATA/"p016_train_v5_0.json"

API_KEY="ee184947af33c882866c84dc66f390964007"
TOP_K=20
DELAY=0.4  # NCBI rate limit: 3/sec

# ─── Load HeteroGNN predictions ────────────────────────────────
# Re-run best model and get scores
import random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
layer_onehot[:ND,0]=1; layer_onehot[ND:ND+NT,1]=1; layer_onehot[ND+NT:ND+NT+NP,2]=1; layer_onehot[ND+NT+NP:,3]=1
x=torch.cat([layer_onehot, deg_norm.unsqueeze(1), btw.unsqueeze(1)], dim=1)
IN_DIM=x.size(1)

# Model class (same as before)
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

# Train full model on all drugs (no LDO split)
print("Training full HeteroGNN for Top-K prediction...")
items=train['splits']['train']; pos_labels=defaultdict(set)
for it in items:
    if it.get('label')==1 or it.get('conclusion')=='positive': pos_labels[it['drug']].add(it['disease'])
all_pairs=[(d,di,1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

model=HeteroGNN(IN_DIM,128,2,0.4)
opt=torch.optim.Adam(model.parameters(),lr=0.005)
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
    loss=(lp*3.0+ln)/y.numel()
    loss.backward(); opt.step()
    if loss.item()<best_loss: best_loss=loss.item(); pat=0
    else: pat+=1
    if pat>=30: break
print(f"  Trained in {ep} epochs, loss={best_loss:.4f}")

# Score all drug-TMJOA pairs
model.eval()
with torch.no_grad():
    h=model(x,typed_edges)
    tmjoa_candidates=[]
    for d in drugs:
        if 'TMJOA' not in diseases: continue
        if d in pos_labels and 'TMJOA' in pos_labels[d]: continue  # skip known positives
        di_idx=drug_idx[d]; ds_idx=dis_idx['TMJOA']
        score=torch.sigmoid(model.predict(h, torch.LongTensor([di_idx]), torch.LongTensor([ds_idx]))).item()
        tmjoa_candidates.append((d, score))

tmjoa_candidates.sort(key=lambda x:-x[1])
top_k=tmjoa_candidates[:TOP_K]

print(f"\nTop-{TOP_K} Predictions for TMJOA:")
for i,(drug,score) in enumerate(top_k):
    print(f"  {i+1}. {drug} (score={score:.4f})")

# ─── PubMed literature validation ──────────────────────────────
def search_pubmed(drug, disease, disease_query=None):
    """Search PubMed for drug + disease combination."""
    if disease_query:
        query=urllib.parse.quote(f'{drug} AND {disease_query}')
    else:
        query=urllib.parse.quote(f'"{drug}" AND "{disease}"')
    
    url=f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=5&api_key={API_KEY}&retmode=json"
    try:
        req=urllib.request.Request(url, headers={"Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data=json.loads(resp.read())
        count=int(data.get('esearchresult',{}).get('count',0))
        ids=data.get('esearchresult',{}).get('idlist',[])
        return count, [f"PMID:{pid}" for pid in ids[:3]]
    except Exception as e:
        return 0, [str(e)]

# Disease aliases for search
DISEASE_QUERIES={
    'TMJOA':'("temporomandibular joint" OR "TMJ" OR "TMD")',
    'TMD':'("temporomandibular disorder" OR "TMD" OR "TMJ")',
    'Osteoarthritis':'("osteoarthritis" OR "OA" OR "degenerative joint")',
    'Rheumatoid arthritis':'("rheumatoid arthritis" OR "RA")',
    'Osteoporosis':'("osteoporosis" OR "bone loss")',
    'Fibromyalgia':'("fibromyalgia" OR "FM" OR "chronic widespread pain")',
    'Chronic pain':'("chronic pain" OR "persistent pain")',
    'Pain':'("pain" OR "nociception" OR "analgesia")',
}

print(f"\n{'='*60}")
print(f"Literature Validation — PubMed Search")
print(f"{'='*60}")

validation_results=[]
for i,(drug,score) in enumerate(top_k):
    dq=DISEASE_QUERIES.get('TMJOA','("temporomandibular joint" OR "TMJ")')
    count,pubs=search_pubmed(drug, "TMJOA", dq)
    evidence='confirmed' if count>0 else 'none'
    if count>=10: evidence='strong'
    elif count>=3: evidence='moderate'
    elif count>=1: evidence='weak'
    
    pub_info=pubs[0] if pubs else ''
    validation_results.append({
        'rank':i+1,'drug':drug,'score':float(score),
        'pubmed_hits':count,'evidence_level':evidence,
        'sample_pmids':pubs
    })
    print(f"  {i+1}. {drug} (score={score:.4f}): {count} hits [{evidence}] {pub_info}")
    time.sleep(DELAY)

# Also search known positive drugs for comparison
print(f"\n--- Known Positive Drug Validation ---")
known_pos_drugs=[d for d in drugs if 'TMJOA' in pos_labels.get(d,set())][:10]
known_results=[]
for drug in known_pos_drugs:
    da=DISEASE_QUERIES.get('TMJOA','("TMJ osteoarthritis")')
    count,pubs=search_pubmed(drug,'TMJOA',da)
    evidence='confirmed' if count>0 else 'none'
    if count>=10: evidence='strong'
    elif count>=3: evidence='moderate'
    known_results.append({'drug':drug,'pubmed_hits':count,'evidence_level':evidence,'sample_pmids':pubs[:2]})
    print(f"  {drug}: {count} hits [{evidence}]")
    time.sleep(DELAY)

result={
    'target_disease':'TMJOA',
    'top_k':TOP_K,
    'model':'HeteroGNN (A+B+C)',
    'predictions':validation_results,
    'known_positives_validation':known_results,
    'summary':{
        'novel_predictions_with_evidence':sum(1 for v in validation_results if v['pubmed_hits']>0),
        'total_novel_predictions':len(validation_results),
        'discovery_rate':f"{sum(1 for v in validation_results if v['pubmed_hits']>0)}/{len(validation_results)}"
    }
}

with open(OUT,'w') as f: json.dump(result,f,indent=2)
print(f"\n  Discovery rate: {result['summary']['discovery_rate']}")
print(f"  → {OUT}")
