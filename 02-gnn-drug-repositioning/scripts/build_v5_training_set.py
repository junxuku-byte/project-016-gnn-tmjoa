#!/usr/bin/env python3
"""
Build Project-016 v5.0 training set from v3.5 positives + v3.8 negative pool.

Rules:
1. Exclude repositioning stars from negatives
2. Degree-match drug/disease distribution against positives
3. Stratified sampling by strategy (mechanism_exclusion / temporal / random)
4. Target ratio 1:3.5 (balanced between 1:3 and 1:5)

v3.8 strategy name corrections:
  title_cooccurrence_zero  → cooccurrence_exclusion (all 8 are stars, unusable)
  unrelated_drug_class   → mechanism_exclusion
  failed_oa_trial        → temporal_anchor
"""

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/from_workspace_tmp")
OUT  = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

POS_FILE  = BASE / "p016_train_v3_5.json"
POOL_FILE = BASE / "p016_negative_pool_v3_8.json"
OUT_FILE  = OUT  / "p016_train_v5_0.json"

# ── Repositioning stars (never as negative) ──────────────────────
REPOSITIONING_STARS = {
    "Metformin", "Rapamycin", "Sirolimus", "Everolimus",
    "Resveratrol", "Curcumin", "Quercetin", "Fisetin",
    "Ginger", "Omega-3", "SAMe", "EGCG", "Denosumab",
    "Bisphosphonate", "Zoledronic acid", "Alendronate",
    "Risedronate", "Ibandronate",
}

# ── Load data ────────────────────────────────────────────────────
with open(POS_FILE) as f:
    pos_data = json.load(f)

with open(POOL_FILE) as f:
    pool_data = json.load(f)

positives = [x for x in pos_data["splits"]["train"] if x.get("label", 0) >= 0.5]
all_negs  = pool_data["candidates"]

# ── 1. Filter pool ───────────────────────────────────────────────
filtered = []
star_count = Counter()
for cand in all_negs:
    drug = cand.get("drug", "")
    is_star = any(star.lower() in drug.lower() for star in REPOSITIONING_STARS)
    if is_star:
        star_count[cand.get("strategy", "unknown")] += 1
        continue
    filtered.append(cand)

print(f"Pool: {len(all_negs)} total → {len(filtered)} after star-filter")
print(f"Stars removed by strategy: {dict(star_count)}")

# ── 2. Map v3.8 strategies to v1 strategy tiers ──────────────────
STRATEGY_MAP = {
    "title_cooccurrence_zero": "cooccurrence_exclusion",
    "bone_harmful": "mechanism_exclusion",
    "unrelated_drug_class": "mechanism_exclusion",
    "arthralgia_side_effect": "mechanism_exclusion",
    "cartilage_inhibitory": "mechanism_exclusion",
    "failed_oa_trial": "temporal_anchor",
    "random_baseline": "random",
}

for cand in filtered:
    old = cand.get("strategy", "")
    cand["strategy_v1"] = STRATEGY_MAP.get(old, old)

# Show post-filter distribution
post_filter_dist = Counter(c["strategy_v1"] for c in filtered)
print(f"Post-filter distribution: {dict(post_filter_dist)}")

# ── 3. Target counts (1:3.5 ratio) ──────────────────────────────
n_pos = len(positives)
target_neg = int(n_pos * 3.5)
print(f"Positives: {n_pos} | Target negatives: {target_neg}")

# Allocation: cooccurrence_exclusion is 0 after star-filter, redistribute
# Available: mechanism_exclusion (~700), temporal_anchor (~42), random (~329)
# Target: ~836 negatives
# Allocation adjusted for actual availability:
ALLOCATION = {
    "mechanism_exclusion": 0.65,
    "temporal_anchor": 0.15,
    "random": 0.20,
}

targets = {k: int(target_neg * v) for k, v in ALLOCATION.items()}
# Adjust rounding
while sum(targets.values()) < target_neg:
    targets["mechanism_exclusion"] += 1
print(f"Allocation: {targets}")

# ── 4. Degree matching helpers ─────────────────────────────────────
drug_deg_pos   = Counter(x["drug"] for x in positives)
disease_deg_pos = Counter(x["disease"] for x in positives)

def stratified_sample(candidates, target_n, strategy_name, drug_counter, disease_counter):
    """Sample with degree matching."""
    pool = [c for c in candidates if c.get("strategy_v1") == strategy_name]
    if len(pool) <= target_n:
        return pool

    scored = []
    for c in pool:
        d_score = drug_counter.get(c.get("drug", ""), 0)
        ds_score = disease_counter.get(c.get("disease", ""), 0)
        score = d_score + ds_score
        scored.append((score, random.random(), c))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [c for _, _, c in scored[:target_n]]
    return selected

# ── 5. Sample negatives ──────────────────────────────────────────
selected_negs = []
selected_drugs = Counter()
selected_diseases = Counter()

for strategy, target_n in targets.items():
    batch = stratified_sample(filtered, target_n, strategy, drug_deg_pos, disease_deg_pos)
    for c in batch:
        selected_drugs[c.get("drug", "")] += 1
        selected_diseases[c.get("disease", "")] += 1
    selected_negs.extend(batch)
    print(f"  {strategy}: sampled {len(batch)}")

# ── 6. Build unified training samples ────────────────────────────
train_samples = []

# Positives (label=1.0)
for p in positives:
    train_samples.append({
        "pmid": p.get("pmid", ""),
        "title": p.get("title", ""),
        "drug": p["drug"],
        "disease": p["disease"],
        "year": p.get("year", 2020),
        "journal_if": p.get("journal_if", 2.0),
        "design": p.get("confirmed_design", p.get("design", "Unknown")),
        "conclusion": p.get("conclusion_direction", "positive"),
        "final_score": p.get("final_score", 0.5),
        "label": 1.0,
        "is_synthetic": False,
        "hardness": "positive",
        "strategy": "verified_positive",
        "is_repositioning_star": p.get("is_repositioning_star", False),
        "tmj_relevance": p.get("tmj_relevance", "direct"),
    })

# Negatives (label=0.0)
for n in selected_negs:
    hardness = n.get("hardness", "soft")
    # failed_oa_trial and cartilage_inhibitory = harder negatives
    if n.get("strategy") in ("failed_oa_trial", "cartilage_inhibitory"):
        hardness = "hard"
    elif n.get("strategy") == "bone_harmful":
        hardness = "hard"
    elif n.get("strategy") == "arthralgia_side_effect":
        hardness = "medium"

    train_samples.append({
        "pmid": n.get("pmid", "synthetic_v5_0"),
        "title": n.get("title", f"Negative: {n['drug']} for {n['disease']}"),
        "drug": n["drug"],
        "disease": n["disease"],
        "year": n.get("year", 2024),
        "journal_if": n.get("journal_if", 2.0),
        "design": n.get("design", "synthetic_negative"),
        "conclusion": "negative",
        "final_score": -0.5,
        "label": 0.0,
        "is_synthetic": True,
        "hardness": hardness,
        "strategy": n.get("strategy_v1", n.get("strategy", "unknown")),
        "is_repositioning_star": False,
        "tmj_relevance": n.get("tmj_relevance", "N/A"),
    })

# Shuffle
random.shuffle(train_samples)

# ── 7. Build output ──────────────────────────────────────────────
n_pos_out = sum(1 for x in train_samples if x["label"] >= 0.5)
n_neg_out = sum(1 for x in train_samples if x["label"] < 0.5)

output = {
    "metadata": {
        "version": "v5_0_scopus_rebuilt",
        "created_at": "2026-05-19T19:20:00+08:00",
        "total": len(train_samples),
        "pos": n_pos_out,
        "neg": n_neg_out,
        "ratio": f"1:{n_neg_out/n_pos_out:.1f}",
        "source": "v3.5_positives + v3.8_negative_pool",
        "repositioning_stars_excluded": list(REPOSITIONING_STARS),
        "allocation": ALLOCATION,
        "changes": [
            "REBUILD: fresh negative sampling from v3.8 pool",
            "FILTER: exclude repositioning stars",
            "MATCH: degree-matched drug/disease distribution",
            "STRATIFY: 65% mechanism / 15% temporal / 20% random (cooccurrence all stars)",
        ],
    },
    "statistics": {
        "drug_distribution_pos": dict(Counter(x["drug"] for x in train_samples if x["label"] >= 0.5).most_common(20)),
        "drug_distribution_neg": dict(Counter(x["drug"] for x in train_samples if x["label"] < 0.5).most_common(20)),
        "disease_distribution_pos": dict(Counter(x["disease"] for x in train_samples if x["label"] >= 0.5).most_common(10)),
        "disease_distribution_neg": dict(Counter(x["disease"] for x in train_samples if x["label"] < 0.5).most_common(10)),
        "hardness_distribution": dict(Counter(x["hardness"] for x in train_samples)),
        "strategy_distribution": dict(Counter(x["strategy"] for x in train_samples)),
    },
    "splits": {
        "train": train_samples,
    },
}

with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ v5.0 written to: {OUT_FILE}")
print(f"   Total: {len(train_samples)} | Pos: {n_pos_out} | Neg: {n_neg_out} | Ratio: 1:{n_neg_out/n_pos_out:.1f}")
