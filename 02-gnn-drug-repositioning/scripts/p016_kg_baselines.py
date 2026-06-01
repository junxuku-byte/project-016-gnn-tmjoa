#!/usr/bin/env python3
"""
P0-4: Strong Baselines for Drug Repositioning.
Knowledge Graph Embedding methods: DistMult, ComplEx, RotatE, TransE.
Evaluated under identical inductive LDO protocol.

# ── Paths (portable) ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
"""

import json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

SEED = 42; DEVICE = torch.device('cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA = Path("DATA_DIR")
GRAPH = DATA / "four_layer_graph_full_v2.json"
TRAIN = DATA / "p016_train_v5_0.json"

EMBED_DIM = 128; LR = 0.005; EPOCHS = 400; PATIENCE = 40
N_FOLDS = 5; POS_WEIGHT = 3.0

with open(GRAPH) as f: g = json.load(f)
with open(TRAIN) as f: train = json.load(f)

drugs = sorted(set(e[0] for e in g['drug_target_edges']))
targets = g['targets']
pathways = g['pathways']
diseases = sorted(set(e[1] for e in g['pathway_disease_edges']))

drug_idx = {d:i for i,d in enumerate(drugs)}
targ_idx = {t:i for i,t in enumerate(targets)}
pw_idx = {p:i for i,p in enumerate(pathways)}
dis_idx = {d:i for i,d in enumerate(diseases)}

ND = len(drugs); NT = len(targets); NP = len(pathways); NI = len(diseases)

def off(t, i):
    return {'drug':i, 'target':ND+i, 'pathway':ND+NT+i, 'disease':ND+NT+NP+i}[t]

def build_triples(g):
    """Build (head, rel, tail) triples from 4-layer graph."""
    triples = []
    for d,t in g['drug_target_edges']:
        if d in drug_idx and t in targ_idx:
            triples.append((off('drug',drug_idx[d]), 0, off('target',targ_idx[t])))
            triples.append((off('target',targ_idx[t]), 0, off('drug',drug_idx[d])))
    for t,p in g['target_pathway_edges']:
        if t in targ_idx and p in pw_idx:
            triples.append((off('target',targ_idx[t]), 1, off('pathway',pw_idx[p])))
            triples.append((off('pathway',pw_idx[p]), 1, off('target',targ_idx[t])))
    for p,d in g['pathway_disease_edges']:
        if p in pw_idx and d in dis_idx:
            triples.append((off('pathway',pw_idx[p]), 2, off('disease',dis_idx[d])))
            triples.append((off('disease',dis_idx[d]), 2, off('pathway',pw_idx[p])))
    return triples

triples = build_triples(g)
TOTAL = ND + NT + NP + NI
N_RELS = 3

# Labels
train_items = train['splits']['train']
pos_labels = defaultdict(set)
for it in train_items:
    if it.get('label')==1 or it.get('conclusion')=='positive':
        pos_labels[it['drug']].add(it['disease'])
all_pairs = [(d,di, 1 if di in pos_labels.get(d,set()) else 0) for d in drugs for di in diseases]

# ─── KG Embedding Models ────────────────────────────────────────
class DistMult(nn.Module):
    """Bilinear diagonal: score(h,r,t) = sum(h * r * t)"""
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.E = nn.Embedding(n_ent, dim); nn.init.xavier_uniform_(self.E.weight)
        self.R = nn.Embedding(n_rel, dim); nn.init.xavier_uniform_(self.R.weight)
    def forward(self, h, r, t):
        return (self.E(h) * self.R(r) * self.E(t)).sum(dim=1)

class ComplEx(nn.Module):
    """Complex embeddings: Re(sum(h * r * conj(t)))"""
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.Er = nn.Embedding(n_ent, dim); nn.init.xavier_uniform_(self.Er.weight)
        self.Ei = nn.Embedding(n_ent, dim); nn.init.xavier_uniform_(self.Ei.weight)
        self.Rr = nn.Embedding(n_rel, dim); nn.init.xavier_uniform_(self.Rr.weight)
        self.Ri = nn.Embedding(n_rel, dim); nn.init.xavier_uniform_(self.Ri.weight)
    def forward(self, h, r, t):
        return (self.Er(h)*self.Rr(r)*self.Er(t) + self.Ei(h)*self.Rr(r)*self.Ei(t) +
                self.Er(h)*self.Ri(r)*self.Ei(t) - self.Ei(h)*self.Ri(r)*self.Er(t)).sum(dim=1)

class RotatE(nn.Module):
    """Rotation in complex space: ||h∘r - t||"""
    def __init__(self, n_ent, n_rel, dim, gamma=12.0):
        super().__init__()
        self.Er = nn.Embedding(n_ent, dim); nn.init.uniform_(self.Er.weight, -1, 1)
        self.Ei = nn.Embedding(n_ent, dim); nn.init.uniform_(self.Ei.weight, -1, 1)
        self.Rr = nn.Embedding(n_rel, dim); nn.init.uniform_(self.Rr.weight, -1, 1)
        with torch.no_grad():
            self.Rr.weight.data = self.Rr.weight.data / (self.Rr.weight.data.norm(p=2,dim=1,keepdim=True)+1e-9)
        self.gamma = gamma
    def forward(self, h, r, t):
        hp = self.Er(h) + 1j*self.Ei(h)
        rp = self.Rr(r)  # real-only rotation embedding
        tp = self.Er(t) + 1j*self.Ei(t)
        hp_rot = hp * torch.exp(1j * rp)
        return -torch.norm(hp_rot - tp, p=1, dim=1) + self.gamma  # score, higher=better

class TransE(nn.Module):
    """Translation: score = -||h + r - t||"""
    def __init__(self, n_ent, n_rel, dim, gamma=12.0):
        super().__init__()
        self.E = nn.Embedding(n_ent, dim); nn.init.uniform_(self.E.weight, -1,1)
        self.R = nn.Embedding(n_rel, dim); nn.init.uniform_(self.R.weight, -1,1)
        with torch.no_grad():
            self.E.weight.data = F.normalize(self.E.weight.data, p=2, dim=1)
        self.gamma = gamma
    def forward(self, h, r, t):
        return -torch.norm(self.E(h)+self.R(r)-self.E(t), p=1, dim=1) + self.gamma

# ─── Drug-Disease prediction wrapper ────────────────────────────
def predict_drug_disease(model, model_name, fold_typed_triples):
    """Train KG model on fold triples, evaluate drug-disease prediction."""
    fa, fp = [], []
    n = ND; indices = list(range(n)); random.shuffle(indices)
    fs = n // N_FOLDS

    for fold in range(N_FOLDS):
        hs, he = fold*fs, (fold+1)*fs if fold<N_FOLDS-1 else n
        held = set(indices[hs:he])

        # Filter triples: remove any involving held-out drugs
        train_triples = []
        for h,r,t in triples:
            if h < ND and h in held: continue
            if t < ND and t in held: continue
            train_triples.append((h,r,t))

        if not train_triples:
            fa.append(0.5); fp.append(0.03); continue

        train_h = torch.LongTensor([t[0] for t in train_triples])
        train_r = torch.LongTensor([t[1] for t in train_triples])
        train_t = torch.LongTensor([t[2] for t in train_triples])

        m = model(TOTAL, N_RELS, EMBED_DIM).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR)

        # Negative sampling for KG training
        all_ents = set(range(TOTAL))
        train_set = set((h,r,t) for h,r,t in train_triples)

        best_loss, pat = float('inf'), 0
        for ep in range(EPOCHS):
            m.train(); opt.zero_grad()
            pos_score = m(train_h, train_r, train_t)

            # Sample negatives
            neg_t = torch.randint(0, TOTAL, (len(train_triples),))
            neg_score = m(train_h, train_r, neg_t)

            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            loss = pos_loss + neg_loss

            loss.backward(); opt.step()
            if loss.item() < best_loss: best_loss = loss.item(); pat = 0
            else: pat += 1
            if pat >= PATIENCE: break

        # Evaluate drug-disease prediction for held-out drugs
        m.eval()
        test_pairs = [(d,di,y) for d,di,y in all_pairs if drug_idx[d] in held]
        if not test_pairs: continue

        with torch.no_grad():
            d_idx = torch.LongTensor([drug_idx[d] for d,_,_ in test_pairs])
            di_idx = torch.LongTensor([dis_idx[di] for _,di,_ in test_pairs])
            # Score via drug→drug relation (use relation 0 as proxy) + drug→disease
            ent_emb = m.E.weight if hasattr(m, 'E') else m.Er.weight
            # Use entity embeddings + a shallow MLP for drug-disease scoring
            drug_emb = ent_emb[d_idx]
            dis_emb = ent_emb[di_idx]
            # Simple dot product as link prediction
            scores = (drug_emb * dis_emb).sum(dim=1).cpu().numpy()
            yt = [y for _,_,y in test_pairs]

        try:
            auc = roc_auc_score(yt, scores); ap = average_precision_score(yt, scores)
        except:
            auc, ap = 0.5, sum(yt)/len(yt)

        fa.append(auc); fp.append(ap)
        print(f"  {model_name} Fold {fold+1}: {len(held)} held, AUC={auc:.4f}, AP={ap:.4f}")

    return fa, fp

# ─── MLP-Driven KG evaluator (better for drug-disease prediction) ───
class KGLinkPredictor(nn.Module):
    """Train KG embeddings, then use them with MLP for drug-disease prediction."""
    def __init__(self, n_ent, n_rel, dim, kg_model_class):
        super().__init__()
        self.kg = kg_model_class(n_ent, n_rel, dim)
        self.mlp = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Dropout(0.4), nn.Linear(dim, 1))

    def forward(self, h, r, t):
        return self.kg(h, r, t)

    def predict_dd(self, d_idx, di_idx):
        ed = self.kg.E.weight if hasattr(self.kg, 'E') else self.kg.Er.weight
        return self.mlp(torch.cat([ed[d_idx], ed[di_idx]], dim=1)).squeeze()

def run_kg_baseline(name, kg_class, dim=EMBED_DIM):
    """Run a KG embedding baseline with MLP drug-disease predictor."""
    print(f"\n{'='*60}")
    print(f"{name} Leave-Drug-Out {N_FOLDS}-Fold")
    print(f"{'='*60}")

    n = ND; indices = list(range(n)); random.shuffle(indices)
    fs = n // N_FOLDS
    fa, fp = [], []

    for fold in range(N_FOLDS):
        hs, he = fold*fs, (fold+1)*fs if fold<N_FOLDS-1 else n
        held = set(indices[hs:he])

        train_triples = []
        for h,r,t in triples:
            if h < ND and h in held: continue
            if t < ND and t in held: continue
            train_triples.append((h,r,t))

        if not train_triples:
            fa.append(0.5); fp.append(0.03); continue

        train_h = torch.LongTensor([t[0] for t in train_triples])
        train_r = torch.LongTensor([t[1] for t in train_triples])
        train_t = torch.LongTensor([t[2] for t in train_triples])

        model = KGLinkPredictor(TOTAL, N_RELS, dim, kg_class).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        # Phase 1: KG pre-training
        best_kg_loss, pat = float('inf'), 0
        for ep in range(EPOCHS // 2):
            model.train(); opt.zero_grad()
            pos_score = model(train_h, train_r, train_t)
            neg_t = torch.randint(0, TOTAL, (len(train_triples),))
            neg_score = model(train_h, train_r, neg_t)
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            kg_loss = pos_loss + neg_loss
            kg_loss.backward(); opt.step()
            if kg_loss.item() < best_kg_loss: best_kg_loss = kg_loss.item(); pat = 0
            else: pat += 1
            if pat >= PATIENCE: break

        # Phase 2: Fine-tune MLP for drug-disease prediction
        train_pairs = [(d,di,y) for d,di,y in all_pairs if drug_idx[d] not in held]
        best_lp_loss, pat = float('inf'), 0

        for ep in range(EPOCHS // 2):
            model.train(); opt.zero_grad()
            samp = random.sample(train_pairs, min(8000, len(train_pairs)))
            d_idx = torch.LongTensor([drug_idx[d] for d,_,_ in samp])
            di_idx = torch.LongTensor([dis_idx[di] for _,di,_ in samp])
            y = torch.FloatTensor([y for _,_,y in samp])
            pred = model.predict_dd(d_idx, di_idx)
            lp = F.binary_cross_entropy_with_logits(pred[y==1], y[y==1], reduction='sum')
            ln = F.binary_cross_entropy_with_logits(pred[y==0], y[y==0], reduction='sum')
            lp_loss = (lp * POS_WEIGHT + ln) / y.numel()
            lp_loss.backward(); opt.step()
            if lp_loss.item() < best_lp_loss: best_lp_loss = lp_loss.item(); pat = 0
            else: pat += 1
            if pat >= PATIENCE: break

        test_pairs = [(d,di,y) for d,di,y in all_pairs if drug_idx[d] in held]
        if not test_pairs: continue

        model.eval()
        with torch.no_grad():
            d_idx = torch.LongTensor([drug_idx[d] for d,_,_ in test_pairs])
            di_idx = torch.LongTensor([dis_idx[di] for _,di,_ in test_pairs])
            yt = [y for _,_,y in test_pairs]
            scores = model.predict_dd(d_idx, di_idx).cpu().numpy()

        try:
            auc = roc_auc_score(yt, scores); ap = average_precision_score(yt, scores)
        except:
            auc, ap = 0.5, sum(yt)/len(yt)

        fa.append(auc); fp.append(ap)
        print(f"  Fold {fold+1}: {len(held)} held, {len(test_pairs)} test, AUC={auc:.4f}, AP={ap:.4f}")

    mean_auc = np.mean(fa); std_auc = np.std(fa)
    mean_ap = np.mean(fp); std_ap = np.std(fp)
    print(f"\n  {name} Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  {name} Mean AP:  {mean_ap:.4f} ± {std_ap:.4f}")

    result = {
        'method': name, 'dim': dim,
        'n_folds': N_FOLDS,
        'fold_aucs': [float(a) for a in fa], 'fold_aps': [float(a) for a in fp],
        'mean_auc': float(mean_auc), 'std_auc': float(std_auc),
        'mean_ap': float(mean_ap), 'std_ap': float(std_ap),
    }
    return result

# ─── Run all baselines ──────────────────────────────────────────
results = {}

for name, kg_class in [
    ('DistMult', DistMult),
    ('ComplEx', ComplEx),
    ('RotatE', RotatE),
    ('TransE', TransE),
]:
    results[name] = run_kg_baseline(name, kg_class)

out_path = DATA / "kg_baseline_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("Baseline Summary")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name:12s}: AUC={r['mean_auc']:.4f} ± {r['std_auc']:.4f}, AP={r['mean_ap']:.4f} ± {r['std_ap']:.4f}")
print(f"\n  → {out_path}")
