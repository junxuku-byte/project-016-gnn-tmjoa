#!/usr/bin/env python3
"""
Project-016 GNN v5.0 Inference
Load trained model, score all candidate drug-disease pairs, output Top-K.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from pathlib import Path
from collections import defaultdict

# ========== Config ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/models/p016_gnn_v5_best.pt")
TRAIN_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/p016_train_v5_0.json")
OUT_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/p016_top_predictions_v5.json")

# ========== Model (must match training) ==========

class DrugRepositionGNN(nn.Module):
    def __init__(self, hidden_channels=128, out_channels=64, num_layers=2, dropout=0.5):
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


def main():
    print("=" * 60)
    print("Project-016 GNN v5.0 Inference")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\n[1] Loading checkpoint: {CKPT_FILE}")
    checkpoint = torch.load(CKPT_FILE, weights_only=False, map_location=DEVICE)
    
    drug2idx = checkpoint['drug2idx']
    disease2idx = checkpoint['disease2idx']
    drugs = checkpoint['drugs']
    diseases = checkpoint['diseases']
    config = checkpoint.get('config', {})
    
    print(f"  Drugs: {len(drugs)}, Diseases: {len(diseases)}")
    print(f"  Best Val AUC: {checkpoint.get('val_auc', 0):.4f} (Epoch {checkpoint.get('epoch', 0)})")
    
    # Load training set to identify existing edges
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    
    existing_pairs = set()
    for s in train_data['splits']['train']:
        existing_pairs.add((s['drug'], s['disease']))
    
    # Build graph with ALL edges (for message passing)
    print("\n[2] Building graph...")
    data = HeteroData()
    emb_dim = config.get('emb_dim', 64)
    data['drug'].x = torch.randn(len(drugs), emb_dim) * 0.1
    data['disease'].x = torch.randn(len(diseases), emb_dim) * 0.1
    
    # All known edges for message passing
    drug_indices = [drug2idx[s['drug']] for s in train_data['splits']['train']]
    disease_indices = [disease2idx[s['disease']] for s in train_data['splits']['train']]
    edge_index = torch.tensor([drug_indices, disease_indices], dtype=torch.long)
    data['drug', 'treats', 'disease'].edge_index = edge_index
    data['disease', 'treated_by', 'drug'].edge_index = edge_index.flip(0)
    data = data.to(DEVICE)
    
    # Initialize model
    model = DrugRepositionGNN(
        hidden_channels=config.get('hidden_dim', 128),
        out_channels=emb_dim,
        num_layers=2,
        dropout=config.get('dropout', 0.5)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Warm-up (init lazy modules)
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)
    
    # Generate ALL candidate pairs
    print("\n[3] Scoring candidate pairs...")
    all_pairs = []
    for d in drugs:
        for dis in diseases:
            if (d, dis) not in existing_pairs:
                all_pairs.append((d, dis))
    
    print(f"  Candidate pairs: {len(all_pairs)}")
    
    # Batch scoring
    batch_size = 1024
    all_scores = []
    
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)
        
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i+batch_size]
            drug_idx = torch.tensor([drug2idx[d] for d, _ in batch], dtype=torch.long, device=DEVICE)
            disease_idx = torch.tensor([disease2idx[dis] for _, dis in batch], dtype=torch.long, device=DEVICE)
            
            drug_emb = x_dict['drug'][drug_idx]
            disease_emb = x_dict['disease'][disease_idx]
            scores = torch.sigmoid(model.predict_edge(drug_emb, disease_emb))
            all_scores.extend(scores.cpu().tolist())
    
    # Sort by score
    scored_pairs = list(zip(all_pairs, all_scores))
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Output Top-K by disease
    top_k = 20
    print(f"\n[4] Top-{top_k} Overall Predictions:")
    for i, ((d, dis), score) in enumerate(scored_pairs[:top_k]):
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")
    
    # Top by disease (TMJOA focus)
    print(f"\n[5] Top-10 for TMJOA:")
    tmjoa_pairs = [p for p in scored_pairs if p[0][1] == 'TMJOA']
    for i, ((d, dis), score) in enumerate(tmjoa_pairs[:10]):
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")
    
    # Top for TMJ
    print(f"\n[6] Top-10 for TMJ:")
    tmj_pairs = [p for p in scored_pairs if p[0][1] == 'TMJ']
    for i, ((d, dis), score) in enumerate(tmj_pairs[:10]):
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")
    
    # Save results
    output = {
        "metadata": {
            "version": "v5_0",
            "checkpoint": str(CKPT_FILE),
            "total_candidates": len(all_pairs),
            "scored_at": "2026-05-19T20:30:00+08:00",
        },
        "top_20_overall": [
            {"rank": i+1, "drug": d, "disease": dis, "score": round(score, 4)}
            for i, ((d, dis), score) in enumerate(scored_pairs[:top_k])
        ],
        "top_10_tmjoa": [
            {"rank": i+1, "drug": d, "disease": dis, "score": round(score, 4)}
            for i, ((d, dis), score) in enumerate(tmjoa_pairs[:10])
        ],
        "top_10_tmj": [
            {"rank": i+1, "drug": d, "disease": dis, "score": round(score, 4)}
            for i, ((d, dis), score) in enumerate(tmj_pairs[:10])
        ],
        "all_predictions": [
            {"drug": d, "disease": dis, "score": round(score, 4)}
            for (d, dis), score in scored_pairs
        ],
    }
    
    with open(OUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {OUT_FILE}")
    print(f"   Total candidates scored: {len(all_pairs)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
