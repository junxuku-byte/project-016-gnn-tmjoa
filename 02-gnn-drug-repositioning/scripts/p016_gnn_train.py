#!/usr/bin/env python3
"""
Project-016 GNN Drug Repositioning Training Framework
PyTorch Geometric 2.6.1 + HeteroGraphSAGE

Features:
- Heterogeneous graph: Drug + Disease nodes
- Edge classification: Drug-Disease link prediction
- Weighted BCE loss with sample weights
- Layered evaluation: TMJ-core / cross-disease / repositioning stars
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from collections import defaultdict, Counter
from pathlib import Path

# ========== Config ==========
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
POS_WEIGHT = 2.0  # Over-sample positive class

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== Data Loading ==========

def load_training_set(path='.tmp/p016_train_v4_2.json'):
    """Load v3.9d corrected training set"""
    with open(path) as f:
        data = json.load(f)
    
    all_samples = data['splits']['train'] + data['splits']['val'] + data['splits']['test']
    
    # Build drug/disease vocab
    drugs = sorted(set(s['drug'] for s in all_samples))
    diseases = sorted(set(s['disease'] for s in all_samples))
    
    drug2idx = {d: i for i, d in enumerate(drugs)}
    disease2idx = {d: i for i, d in enumerate(diseases)}
    
    return all_samples, drugs, diseases, drug2idx, disease2idx


def build_hetero_graph(samples, drug2idx, disease2idx):
    """Build heterogeneous graph from training samples"""
    data = HeteroData()
    
    num_drugs = len(drug2idx)
    num_diseases = len(disease2idx)
    
    # Node features: initialize with small random values
    # In production, replace with pre-trained embeddings (ChEMBL, MeSH, etc.)
    data['drug'].x = torch.randn(num_drugs, EMB_DIM) * 0.1
    data['disease'].x = torch.randn(num_diseases, EMB_DIM) * 0.1
    
    # Edge indices and labels
    drug_indices = []
    disease_indices = []
    labels = []
    weights = []
    tmj_flags = []  # For layered evaluation
    
    for s in samples:
        drug_idx = drug2idx[s['drug']]
        disease_idx = disease2idx[s['disease']]
        
        drug_indices.append(drug_idx)
        disease_indices.append(disease_idx)
        labels.append(s['label'])
        weights.append(s.get('weight', 1.0))
        
        # Flag TMJ-core samples
        is_tmj = s.get('tmj_relevance') == 'direct'
        tmj_flags.append(1.0 if is_tmj else 0.0)
    
    # Convert to tensors
    edge_index = torch.tensor([drug_indices, disease_indices], dtype=torch.long)
    edge_label = torch.tensor(labels, dtype=torch.float)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    edge_tmj = torch.tensor(tmj_flags, dtype=torch.float)
    
    data['drug', 'treats', 'disease'].edge_index = edge_index
    data['drug', 'treats', 'disease'].edge_label = edge_label
    data['drug', 'treats', 'disease'].edge_weight = edge_weight
    data['drug', 'treats', 'disease'].edge_tmj = edge_tmj
    
    # Reverse edges for message passing (undirected for GNN)
    data['disease', 'treated_by', 'drug'].edge_index = edge_index.flip(0)
    
    return data


# ========== Model ==========

class DrugRepositionGNN(nn.Module):
    """Heterogeneous GNN for drug-disease link prediction"""
    
    def __init__(self, hidden_channels=HIDDEN_DIM, out_channels=EMB_DIM, num_layers=2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('drug', 'treats', 'disease'): SAGEConv((-1, -1), hidden_channels),
                ('disease', 'treated_by', 'drug'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        
        self.lin = Linear(hidden_channels, out_channels)
        
        # Edge predictor: dot product + MLP
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x_dict, edge_index_dict):
        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Final projection
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict
    
    def predict_edge(self, drug_emb, disease_emb):
        """Predict drug-disease treatment probability"""
        x = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.predictor(x).squeeze(-1)


# ========== Training ==========

def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    # Get train edges
    edge_index = data['drug', 'treats', 'disease'].edge_index[:, train_mask]
    edge_label = data['drug', 'treats', 'disease'].edge_label[train_mask]
    edge_weight = data['drug', 'treats', 'disease'].edge_weight[train_mask]
    
    # Get embeddings for edge endpoints
    drug_emb = x_dict['drug'][edge_index[0]]
    disease_emb = x_dict['disease'][edge_index[1]]
    
    # Predict
    pred = model.predict_edge(drug_emb, disease_emb)
    
    # Weighted BCE loss
    # pos_weight for class imbalance
    pos_mask = edge_label > 0.5
    weight = torch.ones_like(edge_label)
    weight[pos_mask] = POS_WEIGHT
    weight = weight * edge_weight  # Combine with sample weight
    
    loss = F.binary_cross_entropy_with_logits(pred, edge_label, weight=weight)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, split_name=''):
    model.eval()
    
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    edge_index = data['drug', 'treats', 'disease'].edge_index[:, mask]
    edge_label = data['drug', 'treats', 'disease'].edge_label[mask]
    edge_tmj = data['drug', 'treats', 'disease'].edge_tmj[mask]
    
    drug_emb = x_dict['drug'][edge_index[0]]
    disease_emb = x_dict['disease'][edge_index[1]]
    
    pred = torch.sigmoid(model.predict_edge(drug_emb, disease_emb))
    pred_np = pred.cpu().numpy()
    label_np = edge_label.cpu().numpy()
    
    # Overall metrics
    # Convert continuous labels to binary for AUC/AP
    binary_label = (label_np >= 0.5).astype(int)
    
    # Check if we have both classes
    if len(np.unique(binary_label)) < 2:
        print(f"  [{split_name}] Only one class present, skipping AUC")
        return {'auc': 0.0, 'ap': 0.0, 'pred': pred_np, 'label': label_np}
    
    auc = roc_auc_score(binary_label, pred_np)
    ap = average_precision_score(binary_label, pred_np)
    
    # Precision at different thresholds
    # Precision-Recall curve (optional)
    try:
        precision, recall, thresholds = precision_recall_curve(binary_label, pred_np)
    except ValueError:
        precision = recall = thresholds = None
    
    # TMJ-core subset
    tmj_mask = edge_tmj.cpu().numpy() > 0.5
    if tmj_mask.sum() > 10:
        tmj_binary = (label_np[tmj_mask] >= 0.5).astype(int)
        if len(np.unique(tmj_binary)) >= 2:
            tmj_auc = roc_auc_score(tmj_binary, pred_np[tmj_mask])
            tmj_ap = average_precision_score(tmj_binary, pred_np[tmj_mask])
        else:
            tmj_auc = tmj_ap = 0.0
    else:
        tmj_auc = tmj_ap = 0.0
    
    # Repositioning stars evaluation
    # (would need star flag in data)
    
    print(f"  [{split_name}] AUC={auc:.4f} AP={ap:.4f} | TMJ AUC={tmj_auc:.4f} TMJ AP={tmj_ap:.4f}")
    
    return {
        'auc': auc, 'ap': ap,
        'tmj_auc': tmj_auc, 'tmj_ap': tmj_ap,
        'pred': pred_np, 'label': label_np
    }


# ========== Main ==========

def main():
    print("=" * 60)
    print("Project-016 GNN Drug Repositioning Training")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading training set...")
    samples, drugs, diseases, drug2idx, disease2idx = load_training_set()
    print(f"  Drugs: {len(drugs)}, Diseases: {len(diseases)}")
    print(f"  Samples: {len(samples)}")
    
    # Split by original train/val/test
    with open('.tmp/p016_train_v3_9d.json') as f:
        split_data = json.load(f)
    
    train_samples = split_data['splits']['train']
    val_samples = split_data['splits']['val']
    test_samples = split_data['splits']['test']
    
    # Build graph with ALL edges (for message passing)
    # But only train edges have gradients
    print("\n[2] Building heterogeneous graph...")
    data = build_hetero_graph(samples, drug2idx, disease2idx)
    data = data.to(DEVICE)
    
    # Create masks for train/val/test
    all_drug_disease = [(s['drug'], s['disease']) for s in samples]
    
    train_set = set((s['drug'], s['disease']) for s in train_samples)
    val_set = set((s['drug'], s['disease']) for s in val_samples)
    test_set = set((s['drug'], s['disease']) for s in test_samples)
    
    train_mask = torch.tensor([((d, dis) in train_set) for d, dis in all_drug_disease], dtype=torch.bool)
    val_mask = torch.tensor([((d, dis) in val_set) for d, dis in all_drug_disease], dtype=torch.bool)
    test_mask = torch.tensor([((d, dis) in test_set) for d, dis in all_drug_disease], dtype=torch.bool)
    
    print(f"  Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")
    
    # Model
    print("\n[3] Initializing model...")
    model = DrugRepositionGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Dummy forward to initialize LazyModules
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n[4] Training...")
    best_val_auc = 0.0
    best_epoch = 0
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, train_mask)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"\nEpoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f}")
            evaluate(model, data, train_mask, 'Train')
            val_metrics = evaluate(model, data, val_mask, 'Val  ')
            evaluate(model, data, test_mask, 'Test ')
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'drug2idx': drug2idx,
                    'disease2idx': disease2idx,
                }, '.tmp/p016_gnn_best.pt')
                print(f"  -> New best! Saved checkpoint")
    
    print(f"\n{'='*60}")
    print(f"Training complete. Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"Checkpoint: .tmp/p016_gnn_best.pt")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
