#!/usr/bin/env python3
"""
Flat drug-disease GNN control experiment (paper Table 2).
Tests whether a flat association network can match the mechanism graph.
Produces: data/flat_graph_and_equivalence.json
"""
import json, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
HIDDEN = 128
N_LAYERS = 2
DROPOUT = 0.4
LR = 0.005
EPOCHS = 400
PATIENCE = 40

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA = SCRIPT_DIR / "data"
GRAPH = DATA / "four_layer_graph_full_v3.json"  # v3 superset (225 drugs), v2 at four_layer_graph_full_v2.json
TRAIN = DATA / "p016_train_v5_1.json"
OUT = DATA / "flat_graph_and_equivalence.json"

# ── Load data ──────────────────────────────────────────────────
with open(GRAPH) as f:
    g = json.load(f)
with open(TRAIN) as f:
    train = json.load(f)

drugs = sorted(set(e[0] for e in g["drug_target_edges"]))
diseases = sorted(set(e[1] for e in g["pathway_disease_edges"]))
drug_idx = {d: i for i, d in enumerate(drugs)}
dis_idx = {d: i for i, d in enumerate(diseases)}
ND = len(drugs)
NI = len(diseases)

# ── Load positive labels ──────────────────────────────────────
pos_labels = defaultdict(set)
for it in train["splits"]["train"]:
    if it.get("label") == 1:
        pos_labels[it["drug"]].add(it["disease"])

# ── Build flat graph (drug→disease only) ─────────────────────
flat_src, flat_dst = [], []
for d, dis_set in pos_labels.items():
    for di in dis_set:
        if d in drug_idx and di in dis_idx:
            flat_src.append(drug_idx[d])
            flat_dst.append(ND + dis_idx[di])
# Bidirectional
flat_src_all = flat_src + flat_dst
flat_dst_all = flat_dst + flat_src
flat_src_t = torch.LongTensor(flat_src_all)
flat_dst_t = torch.LongTensor(flat_dst_all)
flat_N = ND + NI

# ── One-hot node features (drug+ disease) ─────────────────────
x_flat = torch.zeros(flat_N, flat_N)
x_flat[torch.arange(flat_N), torch.arange(flat_N)] = 1.0

# ── Labels (all drug-disease pairs) ───────────────────────────
all_pairs = [
    (d, di, 1 if di in pos_labels.get(d, set()) else 0)
    for d in drugs
    for di in diseases
]

# ── Leave-Drug-Out folds ──────────────────────────────────────
random.shuffle(drugs)
folds = [drugs[i::5] for i in range(5)]

def ldo_mask_flat(heldout_drugs):
    """Remove edges incident to held-out drugs, return train src/dst."""
    he_set = {drug_idx[d] for d in heldout_drugs if d in drug_idx}
    keep = [
        i
        for i in range(len(flat_src))
        if flat_src[i] not in he_set and flat_dst[i] not in he_set
    ]
    s = [flat_src[i] for i in keep] + [flat_dst[i] for i in keep]
    d = [flat_dst[i] for i in keep] + [flat_src[i] for i in keep]
    return torch.LongTensor(s), torch.LongTensor(d)


# ── Homogeneous GNN on flat graph ─────────────────────────────
class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.W_msg = nn.Linear(in_dim, out_dim)
        self.W_self = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src, dst):
        out = torch.zeros(x.size(0), self.W_msg.out_features)
        out = out.index_add(0, dst, self.W_msg(x[src]))
        deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
        return F.relu(self.dropout(out / deg.unsqueeze(1) + self.W_self(x)))


class FlatGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [GNNLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, src, dst):
        h = self.in_proj(x)
        for l in self.layers:
            h = l(h, src, dst)
        return h

    def predict(self, h, di, ds):
        return self.predictor(torch.cat([h[di], h[ds]], dim=1)).squeeze()


# ── Train & evaluate ──────────────────────────────────────────
fold_aucs, fold_aps = [], []
for fi, held in enumerate(folds):
    he_set = {drug_idx[d] for d in held if d in drug_idx}
    tr_src, tr_dst = ldo_mask_flat(held)
    val_mask = [
        i
        for i, (d, di, _) in enumerate(all_pairs)
        if d not in held
    ]
    val_sub = random.sample(val_mask, min(len(val_mask) // 5, len(val_mask)))
    test_mask = [i for i, (d, di, _) in enumerate(all_pairs) if d in held]

    model = FlatGNN(flat_N, HIDDEN, N_LAYERS, DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = 0.0
    best_state = None
    patience = PATIENCE

    for ep in range(EPOCHS):
        model.train()
        h = model(x_flat, tr_src, tr_dst)
        tr_indices = torch.LongTensor(
            [
                i
                for i in range(len(all_pairs))
                if i not in val_mask and i not in test_mask
            ]
        )
        di = torch.LongTensor([drug_idx[all_pairs[i][0]] for i in tr_indices])
        ds = torch.LongTensor([ND + dis_idx[all_pairs[i][1]] for i in tr_indices])
        y_tr = torch.FloatTensor([all_pairs[i][2] for i in tr_indices])
        pred_tr = model.predict(h, di, ds)
        loss = F.binary_cross_entropy_with_logits(pred_tr, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if val_mask:
            model.eval()
            with torch.no_grad():
                h_v = model(x_flat, tr_src, tr_dst)
                di_v = torch.LongTensor(
                    [drug_idx[all_pairs[i][0]] for i in val_mask]
                )
                ds_v = torch.LongTensor(
                    [ND + dis_idx[all_pairs[i][1]] for i in val_mask]
                )
                y_v = torch.FloatTensor([all_pairs[i][2] for i in val_mask])
                pred_v = model.predict(h_v, di_v, ds_v)
                val_auc = roc_auc_score(y_v.numpy(), pred_v.sigmoid().numpy())
            if val_auc > best_val:
                best_val = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = PATIENCE
            else:
                patience -= 1
        if patience <= 0:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        h = model(x_flat, tr_src, tr_dst)
        di_t = torch.LongTensor([drug_idx[all_pairs[i][0]] for i in test_mask])
        ds_t = torch.LongTensor([ND + dis_idx[all_pairs[i][1]] for i in test_mask])
        y_t = [all_pairs[i][2] for i in test_mask]
        pred_t = model.predict(h, di_t, ds_t)
        auc = roc_auc_score(y_t, pred_t.sigmoid().numpy())
        ap = average_precision_score(y_t, pred_t.sigmoid().numpy())
    fold_aucs.append(auc)
    fold_aps.append(ap)
    print(f"  Fold {fi + 1}: AUC = {auc:.4f}, AP = {ap:.4f}")

mean_auc = float(sum(fold_aucs) / 5)
std_auc = float(
    (sum((x - mean_auc) ** 2 for x in fold_aucs) / 5) ** 0.5
)
mean_ap = float(sum(fold_aps) / 5)

result = {
    "flat_graph_gnn": {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_ap": mean_ap,
        "fold_aucs": fold_aucs,
        "n_nodes": flat_N,
    },
    "interpretation": (
        "Flat drug–disease GNN achieves AUC approximately {:.3f} with one-hot identity features. "
        "The 438-node mechanism graph forces prediction through target and pathway intermediates, "
        "preventing identity-based memorization of drug indices."
    ).format(mean_auc),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nFlat GNN: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
print(f"Results written to {OUT}")
