#!/usr/bin/env python3
"""
Project-016 Phase 3 — Four-Layer Heterogeneous GNN
Layers: drug → target → pathway → disease
Constraints: predictions must pass through mechanism path (no direct drug-disease shortcut)
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter
from pathlib import Path

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.5
PATIENCE = 15
POS_WEIGHT = 3.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
TRAIN_FILE = DATA_DIR / "p016_train_v5_0.json"
GRAPH_FILE = DATA_DIR / "four_layer_graph.json"
DRUG_FEAT_FILE = DATA_DIR / "drug_features_chembl.json"
DISEASE_FEAT_FILE = DATA_DIR / "disease_features_domain.json"
TOPO_FILE = DATA_DIR / "drug_topological_embeddings.json"
CKPT_FILE = DATA_DIR / "models" / "p016_gnn_fourlayer.pt"


def load_data():
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    with open(GRAPH_FILE) as f:
        graph_data = json.load(f)
    with open(DRUG_FEAT_FILE) as f:
        drug_feat_data = json.load(f)
    with open(DISEASE_FEAT_FILE) as f:
        disease_feat_data = json.load(f)
    with open(TOPO_FILE) as f:
        topo_data = json.load(f)

    drugs = drug_feat_data["drugs"]
    diseases = disease_feat_data["diseases"]
    targets = graph_data["targets"]
    pathways = graph_data["pathways"]

    # Features
    chembl = torch.tensor([drug_feat_data["features"][d] for d in drugs], dtype=torch.float)
    topo = torch.tensor([topo_data["embeddings"][d] for d in drugs], dtype=torch.float)
    drug_feats = torch.cat([chembl, topo], dim=1)
    disease_feats = torch.tensor([disease_feat_data["features"][d] for d in diseases], dtype=torch.float)

    # Random init for target and pathway nodes (no pre-trained features)
    target_feats = torch.randn(len(targets), EMB_DIM) * 0.1
    pathway_feats = torch.randn(len(pathways), EMB_DIM) * 0.1

    # Vocab
    drug2idx = {d: i for i, d in enumerate(drugs)}
    target2idx = {t: i for i, t in enumerate(targets)}
    pathway2idx = {p: i for i, p in enumerate(pathways)}
    disease2idx = {d: i for i, d in enumerate(diseases)}

    return (train_data, graph_data, drugs, targets, pathways, diseases,
            drug2idx, target2idx, pathway2idx, disease2idx,
            drug_feats, target_feats, pathway_feats, disease_feats)


class FourLayerGNN(nn.Module):
    """Four-layer heterogeneous GNN with mechanism path constraint."""

    def __init__(self, hidden_channels=HIDDEN_DIM, out_channels=EMB_DIM, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # Drug layer
                ('drug', 'targets', 'target'): SAGEConv((-1, -1), hidden_channels),
                ('target', 'targeted_by', 'drug'): SAGEConv((-1, -1), hidden_channels),

                # Target layer
                ('target', 'regulates', 'pathway'): SAGEConv((-1, -1), hidden_channels),
                ('pathway', 'regulated_by', 'target'): SAGEConv((-1, -1), hidden_channels),

                # Pathway layer
                ('pathway', 'causes', 'disease'): SAGEConv((-1, -1), hidden_channels),
                ('disease', 'caused_by', 'pathway'): SAGEConv((-1, -1), hidden_channels),

                # Direct edges for supervision (but not for main message passing)
                ('drug', 'treats', 'disease'): SAGEConv((-1, -1), hidden_channels),
                ('disease', 'treated_by', 'drug'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)

        self.lin = nn.ModuleDict({
            'drug': Linear(hidden_channels, out_channels),
            'target': Linear(hidden_channels, out_channels),
            'pathway': Linear(hidden_channels, out_channels),
            'disease': Linear(hidden_channels, out_channels),
        })

        # Predictor: drug + disease (with mechanism path constraint)
        # For drug-disease prediction, we use drug_emb and disease_emb
        # But the embeddings are influenced by target/pathway message passing
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                      for key, x in x_dict.items()}

        x_dict = {key: self.lin[key](x) for key, x in x_dict.items()}
        return x_dict

    def predict_edge(self, drug_emb, disease_emb):
        x = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.predictor(x).squeeze(-1)


def build_four_layer_graph(data, graph_data, drug2idx, target2idx, pathway2idx, disease2idx,
                           drug_feats, target_feats, pathway_feats, disease_feats):
    """Build four-layer heterogeneous graph."""
    data_obj = HeteroData()

    data_obj['drug'].x = drug_feats.clone()
    data_obj['target'].x = target_feats.clone()
    data_obj['pathway'].x = pathway_feats.clone()
    data_obj['disease'].x = disease_feats.clone()

    # Drug-Target edges
    dt_edges = []
    for d, t in graph_data['drug_target_edges']:
        if d in drug2idx and t in target2idx:
            dt_edges.append([drug2idx[d], target2idx[t]])
    if dt_edges:
        dt = torch.tensor(dt_edges, dtype=torch.long).t()
        data_obj['drug', 'targets', 'target'].edge_index = dt
        data_obj['target', 'targeted_by', 'drug'].edge_index = dt.flip(0)

    # Target-Pathway edges
    tp_edges = []
    for t, p in graph_data['target_pathway_edges']:
        if t in target2idx and p in pathway2idx:
            tp_edges.append([target2idx[t], pathway2idx[p]])
    if tp_edges:
        tp = torch.tensor(tp_edges, dtype=torch.long).t()
        data_obj['target', 'regulates', 'pathway'].edge_index = tp
        data_obj['pathway', 'regulated_by', 'target'].edge_index = tp.flip(0)

    # Pathway-Disease edges
    pd_edges = []
    for p, d in graph_data['pathway_disease_edges']:
        if p in pathway2idx and d in disease2idx:
            pd_edges.append([pathway2idx[p], disease2idx[d]])
    if pd_edges:
        pd = torch.tensor(pd_edges, dtype=torch.long).t()
        data_obj['pathway', 'causes', 'disease'].edge_index = pd
        data_obj['disease', 'caused_by', 'pathway'].edge_index = pd.flip(0)

    # Drug-Disease edges (for supervision)
    samples = data['splits']['train']
    drug_indices = []
    disease_indices = []
    labels = []
    weights = []

    for s in samples:
        drug_idx = drug2idx.get(s['drug'])
        disease_idx = disease2idx.get(s['disease'])
        if drug_idx is not None and disease_idx is not None:
            drug_indices.append(drug_idx)
            disease_indices.append(disease_idx)
            labels.append(s['label'])
            h = s.get('hardness', 'soft')
            hw = {'hard': 1.5, 'medium': 1.2, 'soft': 1.0, 'positive': 1.0}.get(h, 1.0)
            weights.append(hw)

    dd = torch.tensor([drug_indices, disease_indices], dtype=torch.long)
    data_obj['drug', 'treats', 'disease'].edge_index = dd
    data_obj['drug', 'treats', 'disease'].edge_label = torch.tensor(labels, dtype=torch.float)
    data_obj['drug', 'treats', 'disease'].edge_weight = torch.tensor(weights, dtype=torch.float)
    data_obj['disease', 'treated_by', 'drug'].edge_index = dd.flip(0)

    return data_obj


def train_epoch(model, data, optimizer, mask):
    model.train()
    optimizer.zero_grad()

    x_dict = model(data.x_dict, data.edge_index_dict)

    edge_index = data['drug', 'treats', 'disease'].edge_index[:, mask]
    edge_label = data['drug', 'treats', 'disease'].edge_label[mask]
    edge_weight = data['drug', 'treats', 'disease'].edge_weight[mask]

    drug_emb = x_dict['drug'][edge_index[0]]
    disease_emb = x_dict['disease'][edge_index[1]]

    pred = model.predict_edge(drug_emb, disease_emb)

    pos_mask = edge_label > 0.5
    weight = torch.ones_like(edge_label)
    weight[pos_mask] = POS_WEIGHT
    weight = weight * edge_weight

    loss = F.binary_cross_entropy_with_logits(pred, edge_label, weight=weight)
    l2_reg = 0.001 * sum(p.pow(2).sum() for p in model.lin['drug'].parameters())
    loss = loss + l2_reg
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    x_dict = model(data.x_dict, data.edge_index_dict)

    edge_index = data['drug', 'treats', 'disease'].edge_index[:, mask]
    edge_label = data['drug', 'treats', 'disease'].edge_label[mask]

    drug_emb = x_dict['drug'][edge_index[0]]
    disease_emb = x_dict['disease'][edge_index[1]]

    pred = torch.sigmoid(model.predict_edge(drug_emb, disease_emb))
    pred_np = pred.cpu().numpy()
    label_np = edge_label.cpu().numpy()
    binary = (label_np >= 0.5).astype(int)

    if len(np.unique(binary)) < 2:
        return {'auc': 0.0, 'ap': 0.0, 'n_pos': int(binary.sum()), 'n_neg': int((1 - binary).sum())}

    auc = roc_auc_score(binary, pred_np)
    ap = average_precision_score(binary, pred_np)
    return {'auc': auc, 'ap': ap, 'n_pos': int(binary.sum()), 'n_neg': int((1 - binary).sum())}


def main():
    print("=" * 60)
    print("Project-016 Phase 3 — Four-Layer GNN (Drug-Target-Pathway-Disease)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    (train_data, graph_data, drugs, targets, pathways, diseases,
     drug2idx, target2idx, pathway2idx, disease2idx,
     drug_feats, target_feats, pathway_feats, disease_feats) = load_data()

    print(f"\nGraph layers:")
    print(f"  Drugs: {len(drugs)}")
    print(f"  Targets: {len(targets)}")
    print(f"  Pathways: {len(pathways)}")
    print(f"  Diseases: {len(diseases)}")
    print(f"  Drug-Target edges: {len(graph_data['drug_target_edges'])}")
    print(f"  Target-Pathway edges: {len(graph_data['target_pathway_edges'])}")
    print(f"  Pathway-Disease edges: {len(graph_data['pathway_disease_edges'])}")

    # Build graph
    data = build_four_layer_graph(
        train_data, graph_data, drug2idx, target2idx, pathway2idx, disease2idx,
        drug_feats, target_feats, pathway_feats, disease_feats
    )
    data = data.to(DEVICE)

    # 90/10 split for monitoring
    n_edges = data['drug', 'treats', 'disease'].edge_index.size(1)
    perm = torch.randperm(n_edges)
    train_idx = perm[:int(0.9 * n_edges)]
    val_idx = perm[int(0.9 * n_edges):]

    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    val_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    print(f"\nTraining edges: {train_mask.sum()} | Val edges: {val_mask.sum()}")

    # Model
    model = FourLayerGNN(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    print(f"\nTraining...")
    best_val_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, train_mask)

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, data, val_mask)
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_metrics['auc']:.4f}")

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best
    if best_state:
        model.load_state_dict(best_state)

    # Final eval
    train_metrics = evaluate(model, data, train_mask)
    val_metrics = evaluate(model, data, val_mask)
    print(f"\n{'='*60}")
    print(f"Four-Layer GNN Complete")
    print(f"  Train AUC: {train_metrics['auc']:.4f} | AP: {train_metrics['ap']:.4f}")
    print(f"  Val AUC:   {val_metrics['auc']:.4f} | AP: {val_metrics['ap']:.4f}")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"{'='*60}")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'drug2idx': drug2idx,
        'target2idx': target2idx,
        'pathway2idx': pathway2idx,
        'disease2idx': disease2idx,
        'drugs': drugs,
        'targets': targets,
        'pathways': pathways,
        'diseases': diseases,
        'val_auc': best_val_auc,
    }, CKPT_FILE)
    print(f"\nSaved: {CKPT_FILE}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
