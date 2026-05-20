"""
Pure PyTorch GNN for Link Prediction on LabKG.
No PyTorch Geometric dependency - uses only torch + numpy + networkx.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from pathlib import Path
import csv
import json


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer: H' = D^-0.5 A D^-0.5 H W"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, adj_norm):
        # adj_norm: normalized adjacency matrix (sparse or dense)
        support = self.linear(x)
        output = torch.sparse.mm(adj_norm, support) if adj_norm.is_sparse else torch.mm(adj_norm, support)
        return output


class GraphSAGELayer(nn.Module):
    """GraphSAGE layer: mean aggregation + linear transform"""
    def __init__(self, in_dim, out_dim, aggregator='mean'):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)  # concat self + neighbor
        self.aggregator = aggregator
    
    def forward(self, x, adj):
        # x: (N, in_dim)
        # adj: dense adjacency matrix (N, N)
        if adj.is_sparse:
            adj = adj.to_dense()
        
        # Neighbor aggregation
        neighbor_sum = torch.mm(adj, x)  # (N, in_dim)
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
        
        if self.aggregator == 'mean':
            neighbor_agg = neighbor_sum / degree
        elif self.aggregator == 'sum':
            neighbor_agg = neighbor_sum
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        # Concatenate self + neighbor
        combined = torch.cat([x, neighbor_agg], dim=1)  # (N, in_dim * 2)
        output = self.linear(combined)
        return F.normalize(output, p=2, dim=1) if self.aggregator == 'mean' else output


class GNNEncoder(nn.Module):
    """Multi-layer GNN encoder"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3, gnn_type='sage'):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.layers.append(GCNLayer(dims[i], dims[i+1]))
            elif gnn_type == 'sage':
                self.layers.append(GraphSAGELayer(dims[i], dims[i+1]))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class InnerProductDecoder(nn.Module):
    """Inner product decoder for link prediction: score = z_u^T z_v"""
    def forward(self, z, edge_index):
        # z: (N, embed_dim)
        # edge_index: (2, num_edges) or list of (u, v) tuples
        if isinstance(edge_index, list):
            src = torch.LongTensor([e[0] for e in edge_index])
            dst = torch.LongTensor([e[1] for e in edge_index])
        else:
            src, dst = edge_index[0], edge_index[1]
        
        scores = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(scores)


class MLPDecoder(nn.Module):
    """MLP decoder: more expressive than inner product"""
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z, edge_index):
        if isinstance(edge_index, list):
            src = torch.LongTensor([e[0] for e in edge_index])
            dst = torch.LongTensor([e[1] for e in edge_index])
        else:
            src, dst = edge_index[0], edge_index[1]
        
        combined = torch.cat([z[src], z[dst]], dim=1)
        return torch.sigmoid(self.mlp(combined).squeeze())


class LinkPredictionGNN(nn.Module):
    """End-to-end GNN for link prediction"""
    def __init__(self, input_dim, hidden_dim=128, embed_dim=64, num_layers=2, 
                 decoder_type='mlp', gnn_type='sage', dropout=0.3):
        super().__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim, embed_dim, num_layers, dropout, gnn_type)
        
        if decoder_type == 'inner_product':
            self.decoder = InnerProductDecoder()
        elif decoder_type == 'mlp':
            self.decoder = MLPDecoder(embed_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown decoder: {decoder_type}")
    
    def forward(self, x, adj, pos_edges, neg_edges):
        z = self.encoder(x, adj)  # Node embeddings
        
        pos_scores = self.decoder(z, pos_edges)
        neg_scores = self.decoder(z, neg_edges)
        
        return pos_scores, neg_scores, z
    
    def predict(self, x, adj, edge_list):
        """Inference: score all candidate edges"""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x, adj)
            scores = self.decoder(z, edge_list)
        return scores, z


def build_normalized_adjacency(edge_list, num_nodes):
    """Build normalized adjacency matrix A_norm = D^-0.5 A D^-0.5"""
    # Create sparse adjacency matrix
    rows, cols = [], []
    for src, dst in edge_list:
        rows.append(src)
        cols.append(dst)
        # Undirected: add reverse
        rows.append(dst)
        cols.append(src)
    
    data = [1.0] * len(rows)
    indices = torch.LongTensor([rows, cols])
    values = torch.FloatTensor(data)
    
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    
    # Compute degree matrix D
    adj_dense = adj.to_dense()
    degree = adj_dense.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj_dense), D_inv_sqrt)
    
    return adj_norm


def build_dense_adjacency(edge_list, num_nodes):
    """Build dense adjacency matrix (for GraphSAGE)"""
    adj = torch.zeros(num_nodes, num_nodes)
    for src, dst in edge_list:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # Undirected
    # Add self-loops
    adj += torch.eye(num_nodes)
    return adj


def negative_sampling(edge_list, num_nodes, num_neg_samples, exclude_edges=None):
    """Sample negative edges (non-existing edges)"""
    existing_edges = set((min(e), max(e)) for e in edge_list)
    if exclude_edges:
        existing_edges.update((min(e), max(e)) for e in exclude_edges)
    
    neg_edges = []
    max_attempts = num_neg_samples * 10
    attempts = 0
    
    while len(neg_edges) < num_neg_samples and attempts < max_attempts:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v:
            attempts += 1
            continue
        edge = (min(u, v), max(u, v))
        if edge not in existing_edges:
            neg_edges.append((u, v))
            existing_edges.add(edge)
        attempts += 1
    
    return neg_edges


def train_link_prediction(model, x, adj, pos_edges, num_neg_samples, num_epochs=200, lr=0.01, weight_decay=5e-4):
    """Train link prediction model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Split edges into train/val/test
    np.random.shuffle(pos_edges)
    n = len(pos_edges)
    train_edges = pos_edges[:int(0.8 * n)]
    val_edges = pos_edges[int(0.8 * n):int(0.9 * n)]
    test_edges = pos_edges[int(0.9 * n):]
    
    best_val_auc = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        
        # Sample negative edges for training
        neg_edges = negative_sampling(train_edges, x.size(0), len(train_edges))
        
        optimizer.zero_grad()
        pos_scores, neg_scores, z = model(x, adj, train_edges, neg_edges)
        
        # Binary cross-entropy loss
        pos_loss = -torch.log(pos_scores + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                neg_val = negative_sampling(val_edges, x.size(0), len(val_edges))
                pos_val_scores, neg_val_scores, _ = model(x, adj, val_edges, neg_val)
                
                # Compute AUC
                from sklearn.metrics import roc_auc_score
                y_true = [1] * len(pos_val_scores) + [0] * len(neg_val_scores)
                y_scores = torch.cat([pos_val_scores, neg_val_scores]).cpu().numpy()
                val_auc = roc_auc_score(y_true, y_scores)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), '/tmp/best_gnn_model.pt')
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Load best model and test
    model.load_state_dict(torch.load('/tmp/best_gnn_model.pt'))
    model.eval()
    with torch.no_grad():
        neg_test = negative_sampling(test_edges, x.size(0), len(test_edges))
        pos_test_scores, neg_test_scores, _ = model(x, adj, test_edges, neg_test)
        
        y_true = [1] * len(pos_test_scores) + [0] * len(neg_test_scores)
        y_scores = torch.cat([pos_test_scores, neg_test_scores]).cpu().numpy()
        test_auc = roc_auc_score(y_true, y_scores)
        print(f"\nTest AUC: {test_auc:.4f}")
    
    return model, test_auc


def load_labkg_data(data_dir):
    """Load exported LabKG data"""
    data_dir = Path(data_dir)
    
    # Node features
    node_features = np.load(data_dir / "node_features.npy")
    
    # Edge list
    edges = []
    with open(data_dir / "edge_list.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((int(row["source_idx"]), int(row["target_idx"])))
    
    # Metadata
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return node_features, edges, metadata


def prepare_drug_disease_prediction(node_features, edges, metadata, edge_type_filter=None):
    """Prepare data for drug-disease link prediction"""
    num_nodes = node_features.shape[0]
    
    # Get drug and disease node indices
    drug_nodes = [metadata["node_to_idx"][n] for n in metadata.get("drug_nodes", []) if n in metadata["node_to_idx"]]
    disease_nodes = [metadata["node_to_idx"][n] for n in metadata.get("disease_nodes", []) if n in metadata["node_to_idx"]]
    
    print(f"Drugs: {len(drug_nodes)}, Diseases: {len(disease_nodes)}")
    
    # Find existing drug-disease edges (positive samples)
    pos_edges = []
    for u, v in edges:
        if (u in drug_nodes and v in disease_nodes) or (v in drug_nodes and u in disease_nodes):
            # Normalize direction: drug -> disease
            if u in disease_nodes:
                pos_edges.append((v, u))
            else:
                pos_edges.append((u, v))
    
    print(f"Positive drug-disease edges: {len(pos_edges)}")
    
    if len(pos_edges) == 0:
        print("WARNING: No drug-disease edges found! Need to add more drug-disease relationships to LabKG.")
        return None, None, None, None, None
    
    # Create all possible drug-disease pairs (for inference)
    all_pairs = []
    for d in drug_nodes:
        for dis in disease_nodes:
            all_pairs.append((d, dis))
    
    # Remove existing edges from candidate list
    existing = set(pos_edges)
    candidate_pairs = [p for p in all_pairs if p not in existing]
    
    # Convert to tensors
    x = torch.FloatTensor(node_features)
    adj = build_dense_adjacency(edges, num_nodes)
    
    return x, adj, pos_edges, candidate_pairs, (drug_nodes, disease_nodes)


if __name__ == "__main__":
    import sys
    
    data_dir = Path.home() / "morph-lab" / "projects" / "project-016-chip-tmjoa" / "02-gnn-drug-repositioning" / "data"
    
    print("Loading LabKG data...")
    node_features, edges, metadata = load_labkg_data(data_dir)
    
    print(f"Nodes: {node_features.shape[0]}, Features: {node_features.shape[1]}")
    print(f"Edges: {len(edges)}")
    
    result = prepare_drug_disease_prediction(node_features, edges, metadata)
    
    if result[0] is None:
        print("\nDrug-disease edges too sparse. Need to enrich LabKG with more drug-disease relationships.")
        print("Suggested actions:")
        print("1. Add drug indications from DrugBank/ChEMBL")
        print("2. Add TMJOA-specific treatments from literature")
        print("3. Add OA disease-modifying drug targets")
        sys.exit(0)
    
    x, adj, pos_edges, candidate_pairs, node_groups = result
    
    # Initialize model
    model = LinkPredictionGNN(
        input_dim=x.size(1),
        hidden_dim=128,
        embed_dim=64,
        num_layers=2,
        decoder_type='mlp',
        gnn_type='sage',
        dropout=0.3
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    print("\nTraining...")
    model, test_auc = train_link_prediction(model, x, adj, pos_edges, num_neg_samples=len(pos_edges), num_epochs=200)
    
    # Inference: rank candidate drug-disease pairs
    print("\nRanking candidate drug-disease pairs...")
    scores, embeddings = model.predict(x, adj, candidate_pairs)
    
    # Get top predictions
    top_k = 20
    sorted_indices = torch.argsort(scores, descending=True)[:top_k]
    
    print(f"\nTop {top_k} drug-disease predictions:")
    idx_to_node = {v: k for k, v in metadata["node_to_idx"].items()}
    
    for i, idx in enumerate(sorted_indices):
        u, v = candidate_pairs[idx]
        drug_name = idx_to_node.get(u, f"node_{u}")
        disease_name = idx_to_node.get(v, f"node_{v}")
        score = scores[idx].item()
        print(f"  {i+1}. {drug_name} -> {disease_name} | score={score:.4f}")
