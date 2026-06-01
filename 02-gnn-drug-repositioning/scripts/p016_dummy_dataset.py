#!/usr/bin/env python3
"""
Generate a minimal dummy dataset for testing the Project-016 pipeline.

Produces a synthetic 4-layer mechanism graph and a reduced training set
(5 drugs, 7 targets, 9 pathways, 3 diseases) with random labels.

Usage: python3 scripts/p016_dummy_dataset.py
Output: data/dummy_four_layer_graph.json, data/dummy_train_set.json
"""
import json, random
from pathlib import Path

SEED = 42
random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA = SCRIPT_DIR / "data"

# ── Synthetic graph ────────────────────────────────────────────
drugs = [f"Drug_{i}" for i in range(5)]
targets = [f"Target_{i}" for i in range(7)]
pathways = [f"Pathway_{i}" for i in range(9)]
diseases = ["TMJOA", "Osteoarthritis", "Rheumatoid Arthritis"]

# Generate random edges
drug_target_edges = [
    [d, t] for d in random.sample(drugs, 4) for t in random.sample(targets, 3)
]
target_pathway_edges = [
    [t, p] for t in random.sample(targets, 5) for p in random.sample(pathways, 4)
]
pathway_disease_edges = [
    [p, di] for p in random.sample(pathways, 4) for di in diseases
]

graph = {
    "metadata": {
        "version": "dummy_v1",
        "description": "Synthetic dummy graph for pipeline testing",
    },
    "targets": targets,
    "pathways": pathways,
    "drug_target_edges": drug_target_edges,
    "target_pathway_edges": target_pathway_edges,
    "pathway_disease_edges": pathway_disease_edges,
    "num_drugs": len(drugs),
    "num_targets": len(targets),
    "num_pathways": len(pathways),
    "num_diseases": len(diseases),
}

with open(DATA / "dummy_four_layer_graph.json", "w") as f:
    json.dump(graph, f, indent=2)
print(f"  ✓ Dummy graph: {len(drugs)}d × {len(targets)}t × {len(pathways)}p × {len(diseases)}di")

# ── Synthetic training set ─────────────────────────────────────
train_entries = []
for drug in drugs:
    for disease in diseases:
        label = random.choices([0, 1], weights=[0.8, 0.2])[0]
        train_entries.append({
            "drug": drug,
            "disease": disease,
            "label": float(label),
            "source": f"dummy|seed={SEED}|drug={drug}|disease={disease}",
        })

train_set = {
    "metadata": {
        "version": "dummy_v1",
        "total": len(train_entries),
        "pos": sum(1 for e in train_entries if e["label"] == 1),
        "neg": sum(1 for e in train_entries if e["label"] == 0),
        "source": "synthetic_dummy",
    },
    "splits": {"train": train_entries},
}

with open(DATA / "dummy_train_set.json", "w") as f:
    json.dump(train_set, f, indent=2)
print(f"  ✓ Dummy training set: {len(train_entries)} entries "
      f"(pos={train_set['metadata']['pos']}, neg={train_set['metadata']['neg']})")
