#!/usr/bin/env python3
"""
Fast network-based drug repositioning for TMJOA.
No training needed - uses graph topology features:
1. Common Neighbors
2. Adamic-Adar index
3. Shortest path length
4. Personalized PageRank proximity
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import json
from pathlib import Path


def load_graph(data_dir):
    """Load LabKG as NetworkX graph"""
    data_dir = Path(data_dir)
    
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    G = nx.Graph()
    
    # Add nodes with attributes
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


def compute_drug_disease_scores(G, metadata):
    """Compute topology-based scores for all drug-disease pairs"""
    
    # Get drug and disease nodes (expanded criteria)
    drug_indices = set()
    disease_indices = set()
    
    for node_name in metadata.get("drug_nodes", []):
        if node_name in metadata["node_to_idx"]:
            drug_indices.add(metadata["node_to_idx"][node_name])
    
    for node_name in metadata.get("disease_nodes", []):
        if node_name in metadata["node_to_idx"]:
            disease_indices.add(metadata["node_to_idx"][node_name])
    
    # Also include TMJ/OA-related nodes regardless of category
    for node_name, idx in metadata["node_to_idx"].items():
        lower_name = node_name.lower()
        if any(kw in lower_name for kw in ['tmjoa', 'osteoarthritis', 'tmjd', 'tmj_arthralgia', 'disc_displacement']):
            disease_indices.add(idx)
        # Also include pathology/symptom nodes as potential disease targets
        if "pathology" in lower_name or "symptom" in lower_name:
            disease_indices.add(idx)
    
    # Find existing edges (positive samples)
    existing_edges = set()
    for u, v in G.edges():
        if (u in drug_indices and v in disease_indices) or (v in drug_indices and u in disease_indices):
            existing_edges.add((min(u, v), max(u, v)))
    
    print(f"Drugs: {len(drug_indices)}, Disease-like nodes: {len(disease_indices)}")
    print(f"Existing drug-disease edges: {len(existing_edges)}")
    
    # Precompute node metrics
    print("\nPrecomputing node metrics...")
    degree = dict(G.degree())
    
    # Precompute common neighbors for efficiency
    print("Precomputing common neighbors...")
    common_neighbors_cache = {}
    
    # Compute scores for all pairs
    results = []
    total_pairs = len(drug_indices) * len(disease_indices)
    processed = 0
    
    print(f"\nScoring {total_pairs} drug-disease pairs...")
    
    for drug_idx in drug_indices:
        drug_neighbors = set(G.neighbors(drug_idx))
        
        for disease_idx in disease_indices:
            if drug_idx == disease_idx:
                continue
            
            pair = (min(drug_idx, disease_idx), max(drug_idx, disease_idx))
            is_existing = pair in existing_edges
            
            # 1. Common Neighbors
            disease_neighbors = set(G.neighbors(disease_idx))
            cn = len(drug_neighbors & disease_neighbors)
            
            # 2. Jaccard similarity
            union = len(drug_neighbors | disease_neighbors)
            jaccard = cn / union if union > 0 else 0
            
            # 3. Adamic-Adar index
            aa = 0
            for common in drug_neighbors & disease_neighbors:
                deg = degree.get(common, 1)
                if deg > 1:
                    aa += 1 / np.log(deg)
            
            # 4. Preferential Attachment
            pa = degree.get(drug_idx, 1) * degree.get(disease_idx, 1)
            
            # 5. Shortest path length (if connected)
            try:
                sp = nx.shortest_path_length(G, drug_idx, disease_idx)
            except nx.NetworkXNoPath:
                sp = float('inf')
            
            # 6. Resource Allocation index
            ra = 0
            for common in drug_neighbors & disease_neighbors:
                deg = degree.get(common, 1)
                ra += 1 / deg
            
            # Composite score (weighted combination)
            # Higher common neighbors, AA, RA, PA = more likely
            # Lower shortest path = more likely
            sp_score = 1 / sp if sp < float('inf') and sp > 0 else 0
            
            composite = (
                0.3 * np.log1p(cn) +
                0.2 * aa +
                0.2 * ra +
                0.1 * np.log1p(pa) +
                0.2 * sp_score
            )
            
            results.append({
                "drug_idx": drug_idx,
                "disease_idx": disease_idx,
                "drug_name": metadata["node_to_idx"].get(str(drug_idx), f"node_{drug_idx}"),
                "disease_name": metadata["node_to_idx"].get(str(disease_idx), f"node_{disease_idx}"),
                "common_neighbors": cn,
                "jaccard": jaccard,
                "adamic_adar": aa,
                "preferential_attachment": pa,
                "shortest_path": sp if sp < float('inf') else -1,
                "resource_allocation": ra,
                "composite_score": composite,
                "existing": is_existing,
            })
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed}/{total_pairs} pairs...")
    
    return results


def rank_for_disease(results, disease_keyword, top_k=20):
    """Rank drugs for a specific disease"""
    filtered = [r for r in results if disease_keyword.lower() in r["disease_name"].lower()]
    filtered.sort(key=lambda x: x["composite_score"], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"TOP {top_k} DRUG CANDIDATES FOR '{disease_keyword.upper()}'")
    print(f"{'='*70}")
    
    idx_to_node = {}
    # We need reverse mapping
    
    for i, r in enumerate(filtered[:top_k]):
        drug_name = r["drug_name"].replace("concept:", "").replace("_", " ")
        disease_name = r["disease_name"].replace("concept:", "").replace("_", " ")
        status = "✓ KNOWN" if r["existing"] else "NEW"
        
        print(f"{i+1:2d}. {drug_name:28s} | score={r['composite_score']:6.3f} | "
              f"CN={r['common_neighbors']:2d} | SP={r['shortest_path']:2d} | {status}")
        print(f"    AA={r['adamic_adar']:.3f} | RA={r['resource_allocation']:.3f} | "
              f"Jaccard={r['jaccard']:.3f}")
    
    return filtered[:top_k]


def rank_all_novel(results, top_k=30):
    """Rank top novel (non-existing) drug-disease pairs across all diseases"""
    novel = [r for r in results if not r["existing"]]
    novel.sort(key=lambda x: x["composite_score"], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"TOP {top_k} NOVEL DRUG-DISEASE REPOSITIONING CANDIDATES")
    print(f"{'='*70}")
    
    for i, r in enumerate(novel[:top_k]):
        drug_name = r["drug_name"].replace("concept:", "").replace("_", " ")
        disease_name = r["disease_name"].replace("concept:", "").replace("_", " ")
        
        print(f"{i+1:2d}. {drug_name:25s} -> {disease_name:25s}")
        print(f"    Score={r['composite_score']:.3f} | CN={r['common_neighbors']} | "
              f"SP={r['shortest_path']:.0f} | AA={r['adamic_adar']:.2f}")
    
    return novel[:top_k]


if __name__ == "__main__":
    data_dir = Path.home() / "morph-lab" / "projects" / "project-016-chip-tmjoa" / "02-gnn-drug-repositioning" / "data"
    
    print("Loading LabKG...")
    G, metadata = load_graph(data_dir)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Build reverse mapping for name lookup
    idx_to_name = {v: k for k, v in metadata["node_to_idx"].items()}
    
    # Compute all scores
    print("\nComputing topology-based drug-disease scores...")
    all_results = compute_drug_disease_scores(G, metadata)
    
    # Fix names in results
    for r in all_results:
        r["drug_name"] = idx_to_name.get(r["drug_idx"], f"node_{r['drug_idx']}")
        r["disease_name"] = idx_to_name.get(r["disease_idx"], f"node_{r['disease_idx']}")
    
    # Save results
    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "topology_predictions.json", "w") as f:
        # Convert to serializable format
        serializable = []
        for r in all_results:
            s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
            serializable.append(s)
        json.dump(serializable, f, indent=2)
    
    print(f"\nSaved {len(all_results)} predictions to {output_dir / 'topology_predictions.json'}")
    
    # Rank for TMJOA
    tmjoa_results = rank_for_disease(all_results, "tmj", top_k=20)
    
    # Rank for general OA
    oa_results = rank_for_disease(all_results, "osteoarthritis", top_k=20)
    
    # Top novel candidates overall
    novel_results = rank_all_novel(all_results, top_k=20)
    
    print("\n" + "="*70)
    print("NOTE: These are topology-based heuristic scores.")
    print("Higher = more graph proximity between drug and disease.")
    print("Validation needed: literature search + mechanism check.")
    print("="*70)
