#!/usr/bin/env python3
"""
Generate topological embeddings for all drugs using graph structure.
Falls back to random-walk co-occurrence + SVD if node2vec library unavailable.

Outputs: drug_topological_embeddings.json (171 x 64)
"""

import json
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

TRAIN_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/p016_train_v5_0.json")
OUT_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/drug_topological_embeddings.json")

DIM = 64
WALK_LENGTH = 30
NUM_WALKS = 200
WINDOW = 5


def build_drug_graph(samples):
    """Build drug co-occurrence graph: two drugs share an edge if they appear for same disease."""
    # Group drugs by disease
    disease_to_drugs = defaultdict(set)
    for s in samples:
        disease_to_drugs[s["disease"]].add(s["drug"])
    
    drugs = sorted(set(s["drug"] for s in samples))
    drug2idx = {d: i for i, d in enumerate(drugs)}
    N = len(drugs)
    
    # Build adjacency: drugs co-occurring for same disease
    adj = np.zeros((N, N))
    for dis, drug_set in disease_to_drugs.items():
        drug_list = list(drug_set)
        for i in range(len(drug_list)):
            for j in range(i+1, len(drug_list)):
                u = drug2idx[drug_list[i]]
                v = drug2idx[drug_list[j]]
                adj[u, v] += 1
                adj[v, u] += 1
    
    # Add self-loops
    adj += np.eye(N)
    
    # Normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_norm = adj / row_sums
    
    return drugs, drug2idx, adj_norm


def random_walk_embeddings(adj_norm, drug2idx, drugs, dim=DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALKS, window=WINDOW):
    """Random walk on normalized adjacency + co-occurrence matrix + SVD."""
    N = len(drugs)
    
    print(f"Generating random walks: {num_walks} walks per node, length={walk_length}...")
    
    # Build transition matrix (row-normalized)
    # adj_norm is already row-normalized
    
    # Co-occurrence matrix
    cooccur = np.zeros((N, N))
    
    for node_idx in range(N):
        if node_idx % 20 == 0:
            print(f"  Walks for node {node_idx}/{N}...")
        
        for _ in range(num_walks):
            current = node_idx
            path = [current]
            
            for _ in range(walk_length - 1):
                # Sample next node from transition probabilities
                probs = adj_norm[current]
                if probs.sum() == 0:
                    break
                next_node = np.random.choice(N, p=probs)
                path.append(next_node)
                current = next_node
            
            # Update co-occurrence for pairs within window
            for i in range(len(path)):
                for j in range(max(0, i - window), min(len(path), i + window + 1)):
                    if i != j:
                        cooccur[path[i], path[j]] += 1.0 / abs(i - j)
    
    # PMI-like transformation
    cooccur_sum = cooccur.sum()
    if cooccur_sum > 0:
        P_ij = cooccur / cooccur_sum
        P_i = P_ij.sum(axis=1, keepdims=True)
        P_j = P_ij.sum(axis=0, keepdims=True)
        
        # Avoid division by zero
        P_i[P_i == 0] = 1e-10
        P_j[P_j == 0] = 1e-10
        
        pmi = np.log(P_ij / (P_i @ P_j) + 1e-10)
        pmi[pmi < 0] = 0  # Positive PMI
    else:
        pmi = cooccur
    
    # SVD for embedding
    print("Running SVD...")
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=dim, random_state=42)
    embeddings = svd.fit_transform(pmi)
    
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    return embeddings


def main():
    print("=" * 60)
    print("Drug Topological Embedding Generation")
    print("=" * 60)
    
    with open(TRAIN_FILE) as f:
        data = json.load(f)
    samples = data["splits"]["train"]
    
    print(f"\nLoaded {len(samples)} training samples")
    
    # Build drug graph
    drugs, drug2idx, adj_norm = build_drug_graph(samples)
    print(f"Drugs: {len(drugs)}, Graph density: {adj_norm.sum() / (len(drugs) * len(drugs)):.4f}")
    
    # Generate embeddings
    embeddings = random_walk_embeddings(adj_norm, drug2idx, drugs)
    
    # Save
    embeddings_dict = {drug: embeddings[i].tolist() for i, drug in enumerate(drugs)}
    
    output = {
        "drugs": drugs,
        "embeddings": embeddings_dict,
        "dim": DIM,
        "method": "random_walk_ppmi_svd",
        "walk_length": WALK_LENGTH,
        "num_walks": NUM_WALKS,
        "window": WINDOW,
    }
    
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {OUT_FILE} ({len(drugs)} x {DIM})")
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.4f}")


if __name__ == "__main__":
    main()
