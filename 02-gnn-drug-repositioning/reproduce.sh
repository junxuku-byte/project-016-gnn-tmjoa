#!/bin/bash
# ================================================================
# Project-016: Mechanism-Constrained GNN Drug Repositioning
# One-click reproduction script
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Project-016 Reproduction Pipeline"
echo "=================================="

# ── Requirements ──
echo "[1/6] Checking requirements..."
python3 -c "import torch; import numpy; import sklearn; print('  ✓ Python deps OK')"

# ── Graph Construction ──
echo "[2/6] Building 4-layer mechanism graph..."
python3 scripts/expand_graph.py
echo "  ✓ four_layer_graph_full_v2.json"

# ── Training Set ──
echo "[3/6] Building training set v5..."
python3 scripts/build_v5_training_set.py
python3 scripts/p016_fix_source.py
echo "  ✓ p016_train_v5_1.json"

# ── Model Evaluation ──
echo "[4/6] Running architecture ablation (5 models × 5-fold LDO)..."
python3 scripts/p016_gnn_fourlayer_ldo.py      # True Homogeneous GNN
python3 scripts/p016_hgnn_ldo.py                # HeteroGNN
python3 scripts/p016_hgnn_abc_ldo.py            # HeteroGNN + features + class weight
python3 scripts/p016_attn_hgnn_ldo.py           # AttnHeteroGNN
echo "  ✓ Architecture ablation complete"

echo "[5/6] Running baselines + ablation..."
python3 scripts/p016_kg_baselines.py            # DistMult, ComplEx, RotatE, TransE
python3 scripts/p016_graph_ablation.py           # Graph ablation (5 variants)
python3 scripts/p016_ecfp_ldo.py                 # ECFP4 molecular features
echo "  ✓ Baselines + ablation complete"

echo "[6/6] Running literature validation..."
python3 scripts/p016_clean_validation.py         # PubMed (excl. graph PMIDs)
python3 scripts/p016_multisource_validation.py   # OpenAlex + Scopus
echo "  ✓ Multi-source validation complete"

# ── Results ──
echo ""
echo "=================================="
echo "Reproduction complete."
echo ""
echo "Key results:"
echo "  Architecture ablation: data/ldo_results.json"
echo "  KG baselines:          data/kg_baseline_results.json"
echo "  Graph ablation:        data/graph_ablation_results.json"
echo "  ECFP features:         data/molecular_features_results.json"
echo "  Multi-source validation: data/multisource_validation.json"
echo "  Training set v5.1:     data/p016_train_v5_1.json"
echo "  Supplementary tables:  data/supplementary_s*.json"
echo ""
echo "Graph files:"
echo "  four_layer_graph_full_v2.json"
echo "=================================="
