#!/bin/bash
# ================================================================
# Project-016: Mechanism-Constrained Graph Learning for Inductive
#               Drug Repositioning — Full Reproduction Pipeline
#
# Target: Bioinformatics (Oxford)
# Paper:  "Mechanism-Constrained Graph Learning for Inductive
#          Drug Repositioning" (v9)
#
# Usage:  bash reproduce.sh
# Requirements: Python 3.8+, PyTorch, numpy, scikit-learn, scipy,
#               rdkit (optional, for ECFP4 experiment)
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "================================================================"
echo "  Project-016: Full Reproduction Pipeline"
echo "  Paper: Mechanism-Constrained Graph Learning (v9)"
echo "================================================================"

# ── Requirements ───────────────────────────────────────────────
echo ""
echo "[1/8] Checking Python dependencies..."
python3 -c "
import torch; import numpy; import sklearn;
print('  ✓ PyTorch {}, NumPy {}, scikit-learn {}'.format(
    torch.__version__, numpy.__version__, sklearn.__version__))
"
# rdkit is optional — only needed for ECFP4 experiment
python3 -c "import rdkit; print('  ✓ RDKit {} (ECFP4 experiment)'.format(rdkit.__version__))" 2>/dev/null || echo "  ⚠️  RDKit not found — ECFP4 experiment will use cached SMILES"

# ── Graph Construction ─────────────────────────────────────────
echo ""
echo "[2/8] Building 4-layer mechanism graph..."
python3 scripts/expand_graph.py
echo "  ✓ data/four_layer_graph_full_v2.json"

# ── Training Set ───────────────────────────────────────────────
echo ""
echo "[3/8] Building training set v5.1..."
python3 scripts/build_v5_training_set.py
python3 scripts/p016_fix_source.py
echo "  ✓ data/p016_train_v5_1.json (1075 entries, source-annotated)"

# ── Architecture Ablation ──────────────────────────────────────
echo ""
echo "[4/8] Architecture ablation (5 models × 5-fold LDO)..."
python3 scripts/p016_gnn_fourlayer_ldo.py      # True Homogeneous GNN
python3 scripts/p016_hgnn_ldo.py                # HeteroGNN
python3 scripts/p016_hgnn_abc_ldo.py            # HeteroGNN + features + class weight
python3 scripts/p016_attn_hgnn_ldo.py           # AttnHeteroGNN
python3 scripts/p016_true_homogeneous.py        # Standalone True Homogeneous baseline
echo "  ✓ Architecture ablation → Table 1"
echo "  ✓ Equivalence analysis → data/flat_graph_and_equivalence.json"

# ── Baselines ──────────────────────────────────────────────────
echo ""
echo "[5/8] Classical baselines..."
python3 scripts/p016_rwr_baseline.py             # Random Walk with Restart
python3 scripts/p016_node2vec_baseline.py        # Node2Vec (transductive)
python3 scripts/p016_node2vec_inductive.py       # Node2Vec (inductive)
echo "  ✓ RWR + Node2Vec (trans/ind) → Table 2"

echo ""
echo "[6/8] Knowledge graph baselines..."
python3 scripts/p016_kg_baselines.py             # DistMult, ComplEx, RotatE, TransE
echo "  ✓ KG baselines → Table 3"

# ── Flat Graph Control ─────────────────────────────────────────
echo ""
echo "[7/8] Flat graph GNN control..."
python3 scripts/p016_flat_gnn.py                 # Flat drug–disease GNN (identity leak control)
echo "  ✓ Flat graph GNN → Table 2"

# ── Ablation Studies ───────────────────────────────────────────
echo ""
echo "[8/8] Ablation experiments..."
python3 scripts/p016_graph_ablation.py            # Graph ablation (5 variants) + Feature ablation (5 variants)
echo "  ✓ Graph ablation → Table 4"
echo "  ✓ Feature ablation → Table 5"

# ── ECFP4 Molecular Features ──────────────────────────────────
echo ""
echo "[9/9] Molecular features..."
python3 scripts/p016_ecfp_ldo.py                  # ECFP4 fingerprint enhancement
echo "  ✓ ECFP4 molecular features → Table 6"

# ── Permutation Test ───────────────────────────────────────────
echo ""
echo "[10/10] Statistical validation..."
python3 scripts/p016_permutation_test.py           # 1000-shuffle permutation test
echo "  ✓ Permutation test (p < 0.001, d = 3.3)"

# ── Literature Validation ──────────────────────────────────────
echo ""
echo "[11/11] Multi-source literature validation..."
python3 scripts/p016_clean_validation.py           # PubMed (excl. graph PMIDs)
python3 scripts/p016_multisource_validation.py     # OpenAlex + Scopus
echo "  ✓ Multi-source validation → Table 7"

# ── Mechanism Path Extraction ──────────────────────────────────
echo ""
echo "[12/12] Mechanism path case studies..."
python3 scripts/p016_mechanism_cases.py            # BGJ398, Chondroitin, Metformin
echo "  ✓ Mechanism path case studies"

# ── Compile Figures ───────────────────────────────────────────
echo ""
echo "[13/13] Compiling figure data..."
python3 scripts/p016_compile_figures.py            # paper_figures_data.json
echo "  ✓ data/paper_figures_data.json"

# ── Supplementary Tables ──────────────────────────────────────
echo ""
echo "[14/14] Generating supplementary tables..."
python3 scripts/p016_supplementary.py              # S1-S4
echo "  ✓ Supplementary Tables S1–S4"

# ── Results Summary ────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  REPRODUCTION COMPLETE"
echo "================================================================"
echo ""
echo "  Output files:"
echo "    Architecture ablation:   data/ldo_results.json"
echo "    Flat graph + equivalence: data/flat_graph_and_equivalence.json"
echo "    KG baselines:            data/kg_baseline_results.json"
echo "    Graph ablation:          data/graph_ablation_results.json"
echo "    ECFP features:           data/molecular_features_results.json"
echo "    Permutation test:        data/permutation_test_results.json"
echo "    Multi-source validation: data/multisource_validation.json"
echo "    Mechanism cases:         data/mechanism_case_studies.json"
echo "    Training set v5.1:       data/p016_train_v5_1.json"
echo "    Figure data:             data/paper_figures_data.json"
echo "    Supplementary tables:    data/supplementary_s*.json"
echo "    Graph file:              data/four_layer_graph_full_v2.json"
echo ""
echo "  Key results (expected ranges):"
echo "    Architecture ablation:   AUC ≈ 0.85 (all 5 models within ±0.002)"
echo "    Node2Vec transductive:   AUC ≈ 0.90 (21-point leak vs inductive)"
echo "    Flat graph GNN:          AUC ≈ 0.95 (identity memorization)"
echo "    TP edge randomization:   ΔAUC ≈ −0.08 (largest ablation effect)"
echo "    ECFP4 enhancement:       +2.5 AUC points"
echo "    Literature validation:   18/20 confirmed (≥2 databases)"
echo "================================================================"
