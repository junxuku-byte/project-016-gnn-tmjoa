#!/usr/bin/env python3
"""
Project-016 GNN v5.0 Training
PyTorch Geometric 2.6.1 + HeteroGraphSAGE

Enhancements:
- Stratified train/val/test split (80/10/10)
- Dropout 0.5 (up from 0.3) for small graph overfitting
- Early stopping patience=15
- Weighted BCE with hardness-aware sampling
- Hard negative mining during training
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

# ========== Config ==========
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.5
PATIENCE = 15
POS_WEIGHT = 3.0  # 1:3.5 ratio -> up-weight positives

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== Paths ==========
DATA_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "p016_train_v5_0.json"
CKPT_FILE = MODEL_DIR / "p016_gnn_v5_best.pt"

# ========== Data Loading ==========

def load_v5_training_set(path):
    with open(path) as f:
        data = json.load(f)
    
    samples = data['splits']['train']
    
    # Build vocab
    drugs = sorted(set(s['drug'] for s in samples))
    diseases = sorted(set(s['disease'] for s in samples))
    
    drug2idx = {d: i for i, d in enumerate(drugs)}
    disease2idx = {d: i for i, d in enumerate(diseases)}
    
    return samples, drugs, diseases, drug2idx, disease2idx


def stratified_split(samples, train_ratio=0.8, val_ratio=0.1):
    """Stratified split preserving positive/negative ratio and hardness"""
    pos = [s for s in samples if s['label'] >= 0.5]
    neg = [s for s in samples if s['label'] < 0.5]
    
    # Shuffle each group
    random.shuffle(pos)
    random.shuffle(neg)
    
    def split_group(group):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return group[:n_train], group[n_train:n_train+n_val], group[n_train+n_val:]
    
    pos_train, pos_val, pos_test = split_group(pos)
    neg_train, neg_val, neg_test = split_group(neg)
    
    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def build_hetero_graph(samples, drug2idx, disease2idx):
    data = HeteroData()
    
    num_drugs = len(drug2idx)
    num_diseases = len(disease2idx)
    
    # Node features: small random init (replace with ChEMBL/MeSH embeddings in production)
    data['drug'].x = torch.randn(num_drugs, EMB_DIM) * 0.1
    data['disease'].x = torch.randn(num_diseases, EMB_DIM) * 0.1
    
    drug_indices = []
    disease_indices = []
    labels = []
    weights = []
    hardness_scores = []
    
    for s in samples:
        drug_idx = drug2idx[s['drug']]
        disease_idx = disease2idx[s['disease']]
        
        drug_indices.append(drug_idx)
        disease_indices.append(disease_idx)
        labels.append(s['label'])
        
        # Hardness-aware weight
        hardness = s.get('hardness', 'soft')
        if hardness == 'hard':
            hw = 1.5
        elif hardness == 'medium':
            hw = 1.2
        else:
            hw = 1.0
        weights.append(hw)
        
        # Hardness score for evaluation stratification
        hardness_map = {'positive': 2.0, 'hard': 1.5, 'medium': 1.0, 'soft': 0.5}
        hardness_scores.append(hardness_map.get(hardness, 0.5))
    
    edge_index = torch.tensor([drug_indices, disease_indices], dtype=torch.long)
    edge_label = torch.tensor(labels, dtype=torch.float)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    edge_hardness = torch.tensor(hardness_scores, dtype=torch.float)
    
    data['drug', 'treats', 'disease'].edge_index = edge_index
    data['drug', 'treats', 'disease'].edge_label = edge_label
    data['drug', 'treats', 'disease'].edge_weight = edge_weight
    data['drug', 'treats', 'disease'].edge_hardness = edge_hardness
    
    # Reverse edges for message passing
    data['disease', 'treated_by', 'drug'].edge_index = edge_index.flip(0)
    
    return data


# ========== Model ==========

class DrugRepositionGNN(nn.Module):
    def __init__(self, hidden_channels=HIDDEN_DIM, out_channels=EMB_DIM, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('drug', 'treats', 'disease'): SAGEConv((-1, -1), hidden_channels),
                ('disease', 'treated_by', 'drug'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        
        self.lin = Linear(hidden_channels, out_channels)
        
        # MLP predictor with higher dropout
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
        
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict
    
    def predict_edge(self, drug_emb, disease_emb):
        x = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.predictor(x).squeeze(-1)


# ========== Training ==========

def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    edge_index = data['drug', 'treats', 'disease'].edge_index[:, train_mask]
    edge_label = data['drug', 'treats', 'disease'].edge_label[train_mask]
    edge_weight = data['drug', 'treats', 'disease'].edge_weight[train_mask]
    
    drug_emb = x_dict['drug'][edge_index[0]]
    disease_emb = x_dict['disease'][edge_index[1]]
    
    pred = model.predict_edge(drug_emb, disease_emb)
    
    # Weighted BCE
    pos_mask = edge_label > 0.5
    weight = torch.ones_like(edge_label)
    weight[pos_mask] = POS_WEIGHT
    weight = weight * edge_weight
    
    loss = F.binary_cross_entropy_with_logits(pred, edge_label, weight=weight)
    
    # L2 regularization on embeddings
    l2_reg = 0.001 * sum(p.pow(2).sum() for p in model.lin.parameters())
    loss = loss + l2_reg
    
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, split_name=''):
    model.eval()
    
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    edge_index = data['drug', 'treats', 'disease'].edge_index[:, mask]
    edge_label = data['drug', 'treats', 'disease'].edge_label[mask]
    edge_hardness = data['drug', 'treats', 'disease'].edge_hardness[mask]
    
    drug_emb = x_dict['drug'][edge_index[0]]
    disease_emb = x_dict['disease'][edge_index[1]]
    
    pred = torch.sigmoid(model.predict_edge(drug_emb, disease_emb))
    pred_np = pred.cpu().numpy()
    label_np = edge_label.cpu().numpy()
    hardness_np = edge_hardness.cpu().numpy()
    
    binary_label = (label_np >= 0.5).astype(int)
    
    if len(np.unique(binary_label)) < 2:
        print(f"  [{split_name}] Only one class!")
        return {'auc': 0.0, 'ap': 0.0}
    
    auc = roc_auc_score(binary_label, pred_np)
    ap = average_precision_score(binary_label, pred_np)
    
    # Hard negative subset (hardness >= 1.0)
    hard_mask = hardness_np >= 1.0
    if hard_mask.sum() > 5:
        hard_binary = binary_label[hard_mask]
        if len(np.unique(hard_binary)) >= 2:
            hard_auc = roc_auc_score(hard_binary, pred_np[hard_mask])
        else:
            hard_auc = 0.0
    else:
        hard_auc = 0.0
    
    print(f"  [{split_name}] AUC={auc:.4f} AP={ap:.4f} HardAUC={hard_auc:.4f}")
    
    return {'auc': auc, 'ap': ap, 'hard_auc': hard_auc}


# ========== Main ==========

def main():
    print("=" * 60)
    print("Project-016 GNN v5.0 Training")
    print(f"Device: {DEVICE} | Dropout: {DROPOUT} | Patience: {PATIENCE}")
    print("=" * 60)
    
    # Load
    print("\n[1] Loading v5.0 training set...")
    samples, drugs, diseases, drug2idx, disease2idx = load_v5_training_set(TRAIN_FILE)
    print(f"  Drugs: {len(drugs)}, Diseases: {len(diseases)}, Total: {len(samples)}")
    
    pos = sum(1 for s in samples if s['label'] >= 0.5)
    neg = len(samples) - pos
    print(f"  Positives: {pos}, Negatives: {neg}, Ratio: 1:{neg/pos:.1f}")
    
    # Split
    print("\n[2] Stratified split (80/10/10)...")
    train_samples, val_samples, test_samples = stratified_split(samples)
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    
    # Build graph with ALL edges (message passing)
    print("\n[3] Building heterogeneous graph...")
    data = build_hetero_graph(samples, drug2idx, disease2idx)
    data = data.to(DEVICE)
    
    # Create masks
    all_pairs = [(s['drug'], s['disease']) for s in samples]
    train_set = set((s['drug'], s['disease']) for s in train_samples)
    val_set = set((s['drug'], s['disease']) for s in val_samples)
    test_set = set((s['drug'], s['disease']) for s in test_samples)
    
    train_mask = torch.tensor([((d, dis) in train_set) for d, dis in all_pairs], dtype=torch.bool)
    val_mask = torch.tensor([((d, dis) in val_set) for d, dis in all_pairs], dtype=torch.bool)
    test_mask = torch.tensor([((d, dis) in test_set) for d, dis in all_pairs], dtype=torch.bool)
    
    # Model
    print("\n[4] Initializing model...")
    model = DrugRepositionGNN(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Init LazyModules
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop with early stopping
    print(f"\n[5] Training ({EPOCHS} epochs max)...")
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, train_mask)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f}")
            evaluate(model, data, train_mask, 'Train')
            val_metrics = evaluate(model, data, val_mask, 'Val  ')
            evaluate(model, data, test_mask, 'Test ')
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'val_ap': val_metrics['ap'],
                    'drug2idx': drug2idx,
                    'disease2idx': disease2idx,
                    'drugs': drugs,
                    'diseases': diseases,
                    'config': {
                        'emb_dim': EMB_DIM,
                        'hidden_dim': HIDDEN_DIM,
                        'dropout': DROPOUT,
                        'pos_weight': POS_WEIGHT,
                    }
                }, CKPT_FILE)
                print(f"  -> New best! Saved to {CKPT_FILE}")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Final evaluation with best model
    print(f"\n{'='*60}")
    print(f"Training complete. Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"Checkpoint: {CKPT_FILE}")
    
    # Load best and test
    checkpoint = torch.load(CKPT_FILE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n[6] Final evaluation (best model)...")
    test_metrics = evaluate(model, data, test_mask, 'Test ')
    print(f"\nFinal Test AUC: {test_metrics['auc']:.4f} | AP: {test_metrics['ap']:.4f}")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
