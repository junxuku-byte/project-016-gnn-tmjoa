"""
Node2Vec baseline for drug repositioning on LabKG.
No labels needed - unsupervised node embeddings + similarity-based ranking.
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path


def load_graph(data_dir):
    """Load LabKG as NetworkX graph"""
    data_dir = Path(data_dir)
    
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Build graph from edge list
    G = nx.Graph()
    
    # Add all nodes
    for node_id, idx in metadata["node_to_idx"].items():
        G.add_node(idx, name=node_id)
    
    # Add edges
    import csv
    with open(data_dir / "edge_list.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["source_idx"])
            v = int(row["target_idx"])
            G.add_edge(u, v, edge_type=row.get("edge_type", ""))
    
    return G, metadata


def node2vec_embeddings(G, dimensions=64, walk_length=30, num_walks=200, p=1.0, q=1.0, window=10):
    """
    Node2Vec embedding using random walk + skip-gram.
    If node2vec library not available, use NetworkX's own approach.
    """
    try:
        from node2vec import Node2Vec
        print("Using node2vec library...")
        n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, 
                       num_walks=num_walks, p=p, q=q, workers=4)
        model = n2v.fit(window=window, min_count=1, batch_words=4)
        
        # Extract embeddings
        embeddings = {}
        for node in G.nodes():
            if str(node) in model.wv:
                embeddings[node] = model.wv[str(node)]
            else:
                embeddings[node] = np.random.randn(dimensions)
        
        return embeddings, model
        
    except ImportError:
        print("node2vec library not available. Using random walk + SVD fallback...")
        return _random_walk_svd(G, dimensions, walk_length, num_walks)


def _random_walk_svd(G, dimensions, walk_length, num_walks):
    """Fallback: random walk co-occurrence matrix + SVD"""
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    print(f"  Building co-occurrence matrix for {N} nodes...")
    
    # Build co-occurrence matrix
    cooccur = np.zeros((N, N))
    
    total_walks = num_walks * N
    for walk_i in range(num_walks):
        if walk_i % 10 == 0:
            print(f"  Walk batch {walk_i}/{num_walks}...")
        for start_node in nodes:
            walk = [start_node]
            current = start_node
            
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                walk.append(current)
            
            # Count co-occurrences within window
            for i, node_i in enumerate(walk):
                for j in range(max(0, i-3), min(len(walk), i+4)):
                    if i != j:
                        idx_i = node_to_idx[node_i]
                        idx_j = node_to_idx[walk[j]]
                        cooccur[idx_i, idx_j] += 1
    
    print(f"  Running SVD (dim={dimensions})...")
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=dimensions, random_state=42)
    embeddings_matrix = svd.fit_transform(cooccur)
    
    print(f"  Explained variance: {sum(svd.explained_variance_ratio_):.2%}")
    
    embeddings = {}
    for node in nodes:
        embeddings[node] = embeddings_matrix[node_to_idx[node]]
    
    return embeddings, None


def rank_drug_disease_pairs(embeddings, metadata, top_k=20):
    """Rank all drug-disease pairs by embedding similarity"""
    drug_nodes = [metadata["node_to_idx"][n] for n in metadata.get("drug_nodes", []) 
                  if n in metadata["node_to_idx"]]
    disease_nodes = [metadata["node_to_idx"][n] for n in metadata.get("disease_nodes", []) 
                     if n in metadata["node_to_idx"]]
    
    # Get embedding matrix
    all_nodes = sorted(embeddings.keys())
    embed_matrix = np.array([embeddings[n] for n in all_nodes])
    
    # Compute similarity
    sim_matrix = cosine_similarity(embed_matrix)
    
    # Create index mapping
    idx_in_sim = {n: i for i, n in enumerate(all_nodes)}
    
    # Rank all drug-disease pairs
    predictions = []
    
    for drug_idx in drug_nodes:
        for disease_idx in disease_nodes:
            if drug_idx in idx_in_sim and disease_idx in idx_in_sim:
                sim = sim_matrix[idx_in_sim[drug_idx], idx_in_sim[disease_idx]]
                predictions.append((drug_idx, disease_idx, sim))
    
    # Sort by similarity
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    # Get names
    idx_to_node = {v: k for k, v in metadata["node_to_idx"].items()}
    
    results = []
    for drug_idx, disease_idx, score in predictions[:top_k]:
        drug_name = idx_to_node.get(drug_idx, f"node_{drug_idx}")
        disease_name = idx_to_node.get(disease_idx, f"node_{disease_idx}")
        
        # Clean names
        drug_name = drug_name.replace("concept:", "").replace("_", " ")
        disease_name = disease_name.replace("concept:", "").replace("_", " ")
        
        results.append({
            "drug": drug_name,
            "disease": disease_name,
            "score": float(score),
            "drug_idx": drug_idx,
            "disease_idx": disease_idx,
        })
    
    return results


def analyze_drug_neighbors(G, embeddings, metadata, drug_idx, top_n=10):
    """Analyze what nodes are closest to a given drug in embedding space"""
    all_nodes = sorted(embeddings.keys())
    embed_matrix = np.array([embeddings[n] for n in all_nodes])
    sim_matrix = cosine_similarity(embed_matrix)
    idx_in_sim = {n: i for i, n in enumerate(all_nodes)}
    
    if drug_idx not in idx_in_sim:
        return []
    
    drug_sim = sim_matrix[idx_in_sim[drug_idx]]
    top_indices = np.argsort(drug_sim)[::-1][1:top_n+1]  # Exclude self
    
    idx_to_node = {v: k for k, v in metadata["node_to_idx"].items()}
    
    neighbors = []
    for idx in top_indices:
        node = all_nodes[idx]
        node_name = idx_to_node.get(node, f"node_{node}")
        node_name = node_name.replace("concept:", "").replace("_", " ")
        node_type = G.nodes[node].get("name", "")[:20]
        
        neighbors.append({
            "node": node_name,
            "similarity": float(drug_sim[idx]),
            "degree": G.degree(node),
        })
    
    return neighbors


if __name__ == "__main__":
    import sys
    
    data_dir = Path.home() / "morph-lab" / "projects" / "project-016-chip-tmjoa" / "02-gnn-drug-repositioning" / "data"
    
    print("Loading LabKG...")
    G, metadata = load_graph(data_dir)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Node2Vec embeddings
    print("\nComputing Node2Vec embeddings...")
    print("  (This may take 1-2 minutes for 10K+ nodes)")
    embeddings, model = node2vec_embeddings(G, dimensions=64, walk_length=15, 
                                           num_walks=50, p=1.0, q=0.5)
    
    # Rank drug-disease pairs
    print("\n" + "=" * 60)
    print("TOP DRUG-DISEASE REPOSITIONING CANDIDATES")
    print("=" * 60)
    
    results = rank_drug_disease_pairs(embeddings, metadata, top_k=20)
    
    for i, r in enumerate(results):
        print(f"{i+1:2d}. {r['drug']:25s} -> {r['disease']:30s} | sim={r['score']:.4f}")
    
    # Save results
    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "node2vec_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'node2vec_predictions.json'}")
    
    # Analyze specific drugs of interest
    drugs_of_interest = ["denosumab", "burosumab", "metformin", "rapamycin", "curcumin"]
    
    print("\n" + "=" * 60)
    print("DRUG NEIGHBORHOOD ANALYSIS")
    print("=" * 60)
    
    for drug_name in drugs_of_interest:
        drug_id = f"concept:{drug_name}"
        if drug_id in metadata["node_to_idx"]:
            drug_idx = metadata["node_to_idx"][drug_id]
            neighbors = analyze_drug_neighbors(G, embeddings, metadata, drug_idx, top_n=5)
            
            print(f"\n{drug_name.upper()}:")
            for n in neighbors:
                print(f"  - {n['node']:30s} (sim={n['similarity']:.3f}, degree={n['degree']})")
        else:
            print(f"\n{drug_name}: Not found in LabKG")
    
    print("\nNOTE: These are unsupervised similarity scores from Node2Vec.")
    print("For supervised link prediction, need >50 drug-disease training edges.")
    print("Current LabKG has only 4 drug-disease edges.")
    print("\nRecommended next steps:")
    print("1. Enrich LabKG with drug-disease relationships from DrugBank/CTD")
    print("2. Add TMJOA-specific drug indications")
    print("3. Add OA disease-modifying drug targets")
    print("4. Re-export and retrain with enriched graph")
