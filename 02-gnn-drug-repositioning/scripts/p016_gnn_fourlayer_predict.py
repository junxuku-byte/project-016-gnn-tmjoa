#!/usr/bin/env python3
"""
Phase 3 Four-Layer GNN Inference + Mechanism Path Extraction
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from pathlib import Path

CKPT_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/models/p016_gnn_fourlayer.pt")
TRAIN_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/p016_train_v5_0.json")
GRAPH_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/four_layer_graph.json")
OUT_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/p016_top_predictions_fourlayer.json")

DEVICE = torch.device('cpu')


class FourLayerGNN(nn.Module):
    def __init__(self, hidden_channels=128, out_channels=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('drug', 'targets', 'target'): SAGEConv((-1, -1), hidden_channels),
                ('target', 'targeted_by', 'drug'): SAGEConv((-1, -1), hidden_channels),
                ('target', 'regulates', 'pathway'): SAGEConv((-1, -1), hidden_channels),
                ('pathway', 'regulated_by', 'target'): SAGEConv((-1, -1), hidden_channels),
                ('pathway', 'causes', 'disease'): SAGEConv((-1, -1), hidden_channels),
                ('disease', 'caused_by', 'pathway'): SAGEConv((-1, -1), hidden_channels),
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
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout),
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


def main():
    print("=" * 60)
    print("Phase 3 Inference — Four-Layer GNN + Mechanism Path")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
    drug2idx = checkpoint['drug2idx']
    target2idx = checkpoint['target2idx']
    pathway2idx = checkpoint['pathway2idx']
    disease2idx = checkpoint['disease2idx']
    drugs = checkpoint['drugs']
    targets = checkpoint['targets']
    pathways = checkpoint['pathways']
    diseases = checkpoint['diseases']

    # Load graph
    with open(GRAPH_FILE) as f:
        graph_data = json.load(f)

    # Build mechanism adjacency for path tracing
    drug_to_targets = {d: [] for d in drugs}
    target_to_pathways = {t: [] for t in targets}
    pathway_to_diseases = {p: [] for p in pathways}

    for d, t in graph_data['drug_target_edges']:
        if d in drug_to_targets:
            drug_to_targets[d].append(t)
    for t, p in graph_data['target_pathway_edges']:
        if t in target_to_pathways:
            target_to_pathways[t].append(p)
    for p, d in graph_data['pathway_disease_edges']:
        if p in pathway_to_diseases:
            pathway_to_diseases[p].append(d)

    # Build graph
    data = HeteroData()
    data['drug'].x = torch.randn(len(drugs), 73) * 0.1
    data['target'].x = torch.randn(len(targets), 64) * 0.1
    data['pathway'].x = torch.randn(len(pathways), 64) * 0.1
    data['disease'].x = torch.randn(len(diseases), 6) * 0.1

    def add_edges(edge_list, src2idx, dst2idx, src_type, edge_type, dst_type):
        edges = []
        for s, d in edge_list:
            if s in src2idx and d in dst2idx:
                edges.append([src2idx[s], dst2idx[d]])
        if edges:
            e = torch.tensor(edges, dtype=torch.long).t()
            data[src_type, edge_type, dst_type].edge_index = e
            data[dst_type, edge_type + '_by', src_type].edge_index = e.flip(0)

    add_edges(graph_data['drug_target_edges'], drug2idx, target2idx, 'drug', 'targets', 'target')
    add_edges(graph_data['target_pathway_edges'], target2idx, pathway2idx, 'target', 'regulates', 'pathway')
    add_edges(graph_data['pathway_disease_edges'], pathway2idx, disease2idx, 'pathway', 'causes', 'disease')

    # Load train edges
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    existing_pairs = set((s['drug'], s['disease']) for s in train_data['splits']['train'])

    drug_indices = [drug2idx[s['drug']] for s in train_data['splits']['train']]
    disease_indices = [disease2idx[s['disease']] for s in train_data['splits']['train']]
    dd = torch.tensor([drug_indices, disease_indices], dtype=torch.long)
    data['drug', 'treats', 'disease'].edge_index = dd
    data['disease', 'treated_by', 'drug'].edge_index = dd.flip(0)

    data = data.to(DEVICE)

    # Load model
    model = FourLayerGNN(dropout=0.5).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)
        x_dict = model(data.x_dict, data.edge_index_dict)

    # Score candidates
    candidates = []
    for d in drugs:
        for dis in diseases:
            if (d, dis) not in existing_pairs:
                candidates.append((d, dis))

    print(f"\nScoring {len(candidates)} candidates...")

    batch_size = 1024
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            drug_idx = torch.tensor([drug2idx[d] for d, _ in batch], dtype=torch.long, device=DEVICE)
            disease_idx = torch.tensor([disease2idx[dis] for _, dis in batch], dtype=torch.long, device=DEVICE)

            drug_emb = x_dict['drug'][drug_idx]
            disease_emb = x_dict['disease'][disease_idx]
            scores = torch.sigmoid(model.predict_edge(drug_emb, disease_emb))
            all_scores.extend(scores.cpu().tolist())

    scored = list(zip(candidates, all_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Find mechanism paths for top predictions
    def find_mechanism_paths(drug, disease):
        """Find drug → target → pathway → disease paths."""
        paths = []
        for t in drug_to_targets.get(drug, []):
            for p in target_to_pathways.get(t, []):
                if disease in pathway_to_diseases.get(p, []):
                    paths.append(f"{drug} → {t} → {p} → {disease}")
        return paths[:3]  # Top 3 paths

    # Output
    print(f"\nTop-20 Overall (with mechanism paths):")
    results = []
    for i, ((d, dis), score) in enumerate(scored[:20]):
        paths = find_mechanism_paths(d, dis)
        path_str = " | ".join(paths) if paths else "No direct mechanism path found"
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")
        if paths:
            for p in paths:
                print(f"       └─ {p}")
        results.append({
            'rank': i+1, 'drug': d, 'disease': dis, 'score': round(score, 4),
            'mechanism_paths': paths
        })

    # TMJOA focus
    print(f"\nTop-10 for TMJOA:")
    tmjoa_results = []
    tmjoa = [p for p in scored if p[0][1] == 'TMJOA']
    for i, ((d, dis), score) in enumerate(tmjoa[:10]):
        paths = find_mechanism_paths(d, dis)
        print(f"  {i+1:2d}. {d} → {dis} | score={score:.4f}")
        if paths:
            for p in paths:
                print(f"       └─ {p}")
        tmjoa_results.append({
            'rank': i+1, 'drug': d, 'disease': dis, 'score': round(score, 4),
            'mechanism_paths': paths
        })

    # Save
    output = {
        'metadata': {
            'version': 'phase3_fourlayer',
            'checkpoint': str(CKPT_FILE),
            'total_candidates': len(candidates),
            'val_auc': checkpoint.get('val_auc', 0),
        },
        'top_20': results,
        'top_10_tmjoa': tmjoa_results,
    }

    with open(OUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: {OUT_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()
