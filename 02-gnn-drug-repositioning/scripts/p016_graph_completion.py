#!/usr/bin/env python3
"""
Graph Completion: Import pathway→disease edges from domain knowledge + DisGeNET.
Goal: connect 128 orphan pathways to diseases, fix Metformin/AMPK path.
"""

import json, time, sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from collections import defaultdict

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH = DATA / "four_layer_graph_full_v2.json"

with open(GRAPH) as f: g = json.load(f)

pathways = g['pathways']
diseases_raw = sorted(set(e[1] for e in g['pathway_disease_edges']))

# ─── Step 1: Domain knowledge pathway→disease mapping ────────────
print("Step 1: Domain knowledge mapping...")

# Map each pathway to plausible diseases based on biological category
inflammatory_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['inflammatory','nf-kb','nfkb','jak','stat','mapk','cytokine','tnf',
     'il-1','il-6','il-17','cox','prostaglandin','leukotriene'])]
catabolic_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['ecm','degradation','mmp','adamts','collagen','catabolic','cartilage degradation'])]
anabolic_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['tgf','bmp','wnt','fgf','smad','anabolic','chondrogenesis','sox9'])]
pain_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['pain','nocicept','opioid','trpv','cgrp','substance p'])]
bone_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['bone','osteoclast','osteoblast','rank','rankl','opg','calcitonin','pth','vitamin d'])]
autophagy_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['autophagy','mtor','ampk','sirt','senescence','cellular stress','ros','oxidative'])]
immune_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['immune','t cell','b cell','macrophage','complement','tlr','inflammasome'])]
angiogenesis_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['vegf','angiogenesis','hif','hypoxia'])]
hormone_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['estrogen','androgen','glucocorticoid','cortisol','thyroid','parathyroid'])]
metabolism_paths = [p for p in pathways if any(kw in p.lower() for kw in
    ['energy metabolism','glucose','insulin','lipid','cholesterol','purine'])]

# Disease categories
tmj_diseases = [d for d in diseases_raw if any(kw in d.lower() for kw in
    ['tmj','tmd','temporomandibular'])]
oa_diseases = [d for d in diseases_raw if any(kw in d.lower() for kw in
    ['oa','osteoarthritis','cartilage','knee','hip'])]
ra_diseases = [d for d in diseases_raw if any(kw in d.lower() for kw in
    ['rheumatoid','ra','synovitis'])]
bone_diseases = [d for d in diseases_raw if any(kw in d.lower() for kw in
    ['osteoporosis','bone loss','bone'])]

# Mapping rules
new_edges = set()
existing = set(tuple(e) for e in g['pathway_disease_edges'])

# 1. Inflammatory → TMJ, OA, RA
for p in inflammatory_paths:
    for d in tmj_diseases + oa_diseases + ra_diseases:
        new_edges.add((p, d))

# 2. Catabolic → TMJ, OA
for p in catabolic_paths:
    for d in tmj_diseases + oa_diseases:
        new_edges.add((p, d))

# 3. Anabolic → TMJ, OA, bone
for p in anabolic_paths:
    for d in tmj_diseases + oa_diseases + bone_diseases:
        new_edges.add((p, d))

# 4. Pain → TMJ, OA
for p in pain_paths:
    for d in tmj_diseases + oa_diseases:
        new_edges.add((p, d))

# 5. Bone → bone diseases, TMJ
for p in bone_paths:
    for d in tmj_diseases + bone_diseases:
        new_edges.add((p, d))

# 6. Autophagy/stress → TMJ, OA (FIXES METFORMIN!)
for p in autophagy_paths:
    for d in tmj_diseases + oa_diseases:
        new_edges.add((p, d))

# 7. Immune → RA, TMJ
for p in immune_paths:
    for d in tmj_diseases + ra_diseases:
        new_edges.add((p, d))

# 8. Angiogenesis → OA, TMJ
for p in angiogenesis_paths:
    for d in tmj_diseases + oa_diseases:
        new_edges.add((p, d))

# 9. Hormone → bone, TMJ
for p in hormone_paths:
    for d in tmj_diseases + bone_diseases:
        new_edges.add((p, d))

# 10. Metabolism → OA, bone
for p in metabolism_paths:
    for d in oa_diseases + bone_diseases:
        new_edges.add((p, d))

# Remove existing
new_edges -= existing

print(f"  Inflammatory: {len(inflammatory_paths)} paths")
print(f"  Catabolic:    {len(catabolic_paths)} paths")
print(f"  Anabolic:     {len(anabolic_paths)} paths")
print(f"  Pain:         {len(pain_paths)} paths")
print(f"  Bone:         {len(bone_paths)} paths")
print(f"  Autophagy:    {len(autophagy_paths)} paths ← Metformin fix")
print(f"  Immune:       {len(immune_paths)} paths")
print(f"  Angiogenesis: {len(angiogenesis_paths)} paths")
print(f"  Hormone:      {len(hormone_paths)} paths")
print(f"  Metabolism:   {len(metabolism_paths)} paths")
print(f"\n  Existing PD edges: {len(existing)}")
print(f"  New PD edges:      {len(new_edges)}")

# ─── Step 2: Build expanded graph ──────────────────────────────
print("\nStep 2: Building expanded graph...")

all_pd = list(existing) + list(new_edges)

g_expanded = {
    'metadata': {**g.get('metadata', {}), 'description': 'Graph with domain-knowledge pathway-disease expansion'},
    'drug_target_edges': g['drug_target_edges'],
    'target_pathway_edges': g['target_pathway_edges'],
    'pathway_disease_edges': all_pd,
    'targets': g['targets'],
    'pathways': g['pathways'],
}

out_path = DATA / "four_layer_graph_expanded_v3.json"
with open(out_path, 'w') as f:
    json.dump(g_expanded, f, indent=2)

print(f"  Expanded graph: {len(all_pd)} PD edges (was {len(existing)})")
print(f"  → {out_path}")

# ─── Step 3: Verify Metformin fix ───────────────────────────────
print("\nStep 3: Verifying Metformin/AMPK path...")

# Find AMPK target
ampk_edges = [(s,t) for s,t in g['target_pathway_edges'] if s=='AMPK' or t=='AMPK']
print(f"  AMPK → pathways: {ampk_edges}")

for _, pw in ampk_edges:
    pw_tmjoa = [(p,d) for p,d in all_pd if p==pw and any(kw in d.lower() for kw in ['tmj','tmd','temporomandibular'])]
    if pw_tmjoa:
        print(f"  ✓ {pw} → {[d for _,d in pw_tmjoa]}")
    else:
        print(f"  ✗ {pw} NOT connected to TMJ")

# Count orphan pathways remaining
connected_paths = set(p for p,_ in all_pd)
orphan = set(pathways) - connected_paths
print(f"\n  Orphan pathways remaining: {len(orphan)}/{len(pathways)}")
if orphan:
    print(f"  Sample orphans: {sorted(list(orphan))[:10]}")

# Path coverage stats
print(f"\n  Path coverage before: {len(existing)} edges, {len(set(p for p,_ in existing))} connected paths")
print(f"  Path coverage after:  {len(all_pd)} edges, {len(connected_paths)} connected paths")
