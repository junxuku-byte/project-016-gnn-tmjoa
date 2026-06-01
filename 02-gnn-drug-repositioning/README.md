# Project-016: Mechanism-Constrained Graph Learning for Inductive Drug Repositioning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code, data, and reproduction pipeline for the manuscript:

> **Mechanism-Constrained Graph Learning for Inductive Drug Repositioning**
>
> *Li H, et al. (2026). Submitted to Bioinformatics.*

## Overview

We reformulate computational drug repositioning as a mechanism-constrained hierarchical prediction problem (drug → target → pathway → disease) and demonstrate that graph schema design — not GNN architectural complexity — is the primary determinant of inductive generalization performance.

### Key Findings

- **Architecture independence**: All five GNN architectures (homogeneous through attention) achieve indistinguishable inductive AUC ≈ 0.85
- **Graph topology dominates**: Randomizing target–pathway edges causes the largest performance drop (−8 AUC points)
- **Node2Vec leakage**: Transductive evaluation overestimates performance by 21 AUC points
- **Multi-source validation**: 18/20 top TMJOA candidates confirmed across PubMed, OpenAlex, and Scopus
- **Novel prediction**: BGJ398 (infigratinib, FGFR inhibitor) — a genuinely literature-novel TMJOA repositioning candidate

### Graph Versions

- **v2** (paper v9): 132 drugs, 134 targets, 157 pathways — used for all paper experiments
- **v3** (extended): 225 drugs, 176 targets, 157 pathways — activates 93 cold-start drugs with pharmacology-based target mapping (SSRIs, NSAIDs, antihistamines, opioids, immunotherapy, etc.)

Run `python3 scripts/p016_activate_cold_start.py` to generate v3 from v2. All training scripts use v3 by default (v3 is a superset of v2).

## Quick Start

```bash
# Clone and run
git clone https://github.com/junxuku-byte/project-016-gnn-tmjoa.git
cd project-016-gnn-tmjoa
pip install -r requirements.txt
bash reproduce.sh
```

### Quick Test with Dummy Dataset

```bash
python3 scripts/p016_dummy_dataset.py  # Generate synthetic test data
```

### Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- NumPy, scikit-learn, scipy
- RDKit (optional, for ECFP4 molecular feature experiment)

Install with conda/pip:

```bash
pip install torch numpy scikit-learn scipy rdkit-pypi
```

## Repository Structure

```
.
├── reproduce.sh                  # One-click reproduction script
├── README.md                     # This file
├── data/
│   ├── four_layer_graph_full_v2.json   # 4-layer mechanism graph
│   ├── p016_train_v5_1.json            # Training set (1075 entries, source-annotated)
│   ├── paper_full_draft_v9.md          # Latest manuscript (Bioinformatics)
│   ├── supplementary_s1_search_strategy.json
│   ├── supplementary_s2_target_pathway.json
│   ├── supplementary_s3_edge_provenance.json
│   ├── supplementary_s4_full_validation.json
│   └── ...                             # Additional result files
├── scripts/
│   ├── expand_graph.py                 # Graph construction
│   ├── build_v5_training_set.py        # Training set construction
│   ├── p016_gnn_fourlayer_ldo.py       # Architecture ablation (True Homog. GNN)
│   ├── p016_hgnn_ldo.py                # HeteroGNN
│   ├── p016_hgnn_abc_ldo.py            # HeteroGNN + features + class weight
│   ├── p016_attn_hgnn_ldo.py           # AttnHeteroGNN
│   ├── p016_true_homogeneous.py        # Standalone True Homogeneous GNN
│   ├── p016_rwr_baseline.py            # Random Walk with Restart
│   ├── p016_node2vec_baseline.py       # Node2Vec (transductive)
│   ├── p016_node2vec_inductive.py      # Node2Vec (inductive)
│   ├── p016_flat_gnn.py                # Flat drug–disease GNN control
│   ├── p016_kg_baselines.py            # DistMult, ComplEx, RotatE, TransE
│   ├── p016_graph_ablation.py          # Graph ablation + Feature ablation
│   ├── p016_ecfp_ldo.py                # ECFP4 molecular features
│   ├── p016_permutation_test.py        # Permutation test
│   ├── p016_clean_validation.py        # Literature validation (PubMed)
│   ├── p016_multisource_validation.py  # Multi-source validation
│   ├── p016_mechanism_cases.py         # Mechanism path case studies
│   ├── p016_supplementary.py           # Supplementary tables S1-S4
│   └── p016_compile_figures.py         # Figure data compilation
└── models/
    ├── gnn_pure_pytorch.py             # Core GNN implementation
    ├── node2vec_baseline.py            # Node2Vec implementation
    └── topology_baseline.py            # Topology scoring baseline
```

## Results Summary

| Experiment | Key Metric | Value |
|------------|-----------|-------|
| Architecture ablation | AUC (5 models) | 0.849–0.851 |
| TP edge randomization | ΔAUC | −0.083 |
| Flat graph GNN | AUC | ~0.95 (identity leak) |
| Node2Vec trans→ind | AUC drop | 21 points |
| KG baselines (best) | ComplEx AUC | 0.83 |
| ECFP4 enhancement | ΔAUC | +0.025 |
| Literature validation | 3-DB consensus | 18/20 |

## Citation

If you use this code or data, please cite:

```bibtex
@article{li2026mechanism,
  title={Mechanism-Constrained Graph Learning for Inductive Drug Repositioning},
  author={Li, Haosen and others},
  journal={Bioinformatics},
  year={2026},
  note={Submitted}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Haosen Li (chongchong@tjh.tjmu.edu.cn)  
Department of Stomatology, Tongji Hospital, Tongji Medical College  
Huazhong University of Science and Technology
