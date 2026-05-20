#!/usr/bin/env python3
"""Generate Supplementary Tables S1-S4 for editorial submission."""
import json
from pathlib import Path
from collections import defaultdict

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

# ── S1: PubMed Search Strategy ──
print("=== S1: Literature Search Strategy ===")
with open(DATA/'from_workspace_tmp/p016_evidence_v3_5.json') as f: ev = json.load(f)

s1 = {
    "databases": ["PubMed", "Scopus", "OpenAlex"],
    "search_date": "2026-05",
    "topics": [
        "TMJOA treatment (drugs, injections, arthrocentesis)",
        "TMD pharmacological management",
        "Osteoarthritis drug repositioning",
        "TMJ condylar cartilage mechanisms",
        "Musculoskeletal pain pharmacology"
    ],
    "total_retrieved": "1,392 (Scopus-enhanced)",
    "after_dedup": "400 unique PMIDs",
    "inclusion": [
        "Human/animal studies on TMJOA/TMD/OA pharmacological treatment",
        "Mechanism studies linking drug targets to TMJOA pathways",
        "RCTs, meta-analyses, cohort studies"
    ],
    "exclusion": [
        "Pure surgical technique papers",
        "Non-English without English abstract",
        "Non-musculoskeletal disease focused"
    ],
    "screening_method": "Abstract-level review by two independent reviewers",
    "evidence_levels": {
        "strong_positive": "RCT/Meta with positive conclusion for TMJOA/TMD",
        "weak_positive": "Animal/in-vitro with positive signal",
        "mechanism_only": "Mechanism evidence without efficacy data",
        "negative": "Study concluded no benefit"
    }
}
with open(DATA/'supplementary_s1_search_strategy.json', 'w') as f:
    json.dump(s1, f, indent=2)
print("  → supplementary_s1_search_strategy.json")

# ── S2: Target-Pathway Mapping ──
print("\n=== S2: Target-Pathway Assignment Table ===")
with open(DATA/'four_layer_graph_full_v2.json') as f: g = json.load(f)

tp_by_target = defaultdict(list)
for t, p in g['target_pathway_edges']:
    tp_by_target[t].append(p)

categories = {
    'inflammatory': ['nf-kb','jak','stat','mapk','cytokine','tnf','il-1','il-6','il-17','cox','prostaglandin','leukotriene'],
    'catabolic': ['ecm','degradation','mmp','adamts','collagen','catabolic'],
    'anabolic': ['tgf','bmp','wnt','fgf','smad','chondrogenesis','sox9'],
    'pain': ['pain','nocicept','opioid','trpv','cgrp','substance p'],
    'bone': ['bone','osteoclast','osteoblast','rank','rankl','opg'],
    'autophagy': ['autophagy','mtor','ampk','sirt','senescence'],
}

s2 = []
for target, pathways in sorted(tp_by_target.items()):
    cat = 'other'
    for cname, keywords in categories.items():
        if any(kw in ' '.join(pathways).lower() for kw in keywords):
            cat = cname
            break
    s2.append({
        'target': target,
        'pathways': pathways,
        'n_pathways': len(pathways),
        'source': 'KEGG R112 / Reactome v89 / domain knowledge',
        'category': cat
    })

with open(DATA/'supplementary_s2_target_pathway.json', 'w') as f:
    json.dump(s2, f, indent=2)
print(f"  {len(s2)} targets → supplementary_s2_target_pathway.json")

# ── S3: Graph Edge Provenance ──
print("\n=== S3: Edge Provenance Table ===")
with open(DATA/'mechanism_quadruples_raw.json') as f: mq = json.load(f)

s3 = {
    'graph_version': 'four_layer_graph_full_v2',
    'total_nodes': f"{len(g['drug_target_edges'])} drugs + {len(g['targets'])} targets + {len(g['pathways'])} pathways + {len(set(e[1] for e in g['pathway_disease_edges']))} diseases",
    'edge_types': {
        'drug_target': {
            'count': len(g['drug_target_edges']),
            'sources': ['LabKG literature cards (420 mechanism cards from 753 PubMed articles)', 'ChEMBL v34 mechanism-of-action database (confidence ≥ 4)']
        },
        'target_pathway': {
            'count': len(g['target_pathway_edges']),
            'sources': ['KEGG Release 112', 'Reactome v89', 'Domain-curated signaling modules (4 categories)']
        },
        'pathway_disease': {
            'count': len(g['pathway_disease_edges']),
            'sources': ['LabKG evidence pool (753 PubMed articles)', 'Each pathway connected to diseases cited as mechanistically relevant in ≥1 curated article']
        }
    },
    'mechanism_cards': {
        'total_labkg_cards': 753,
        'mechanism_relevant_cards': 420,
        'unique_targets': mq['metadata'].get('unique_targets', 0),
        'unique_pathways': mq['metadata'].get('unique_pathways', 0)
    }
}

with open(DATA/'supplementary_s3_edge_provenance.json', 'w') as f:
    json.dump(s3, f, indent=2)
print("  → supplementary_s3_edge_provenance.json")

# ── S4: Full Multi-Source Validation (top 20) ──
print("\n=== S4: Full Multi-Source Validation ===")
with open(DATA/'multisource_validation.json') as f: ms = json.load(f)

s4 = []
for i, (pm, oa, sc) in enumerate(zip(
    ms['pubmed_excl_graph']['results'],
    ms['openalex'],
    ms['scopus'] if ms['scopus'] else [{}]*20
)):
    s4.append({
        'rank': i+1,
        'drug': pm['drug'],
        'pubmed_excl_graph': pm['clean'],
        'openalex': oa['count'],
        'scopus': sc.get('count','--'),
        'confirmed_by': sum([
            pm['clean'] > 0,
            oa['count'] > 0,
            sc.get('count',0) > 0
        ])
    })

with open(DATA/'supplementary_s4_full_validation.json', 'w') as f:
    json.dump(s4, f, indent=2)
print("  → supplementary_s4_full_validation.json")

print(f"\n✓ All 4 supplementary tables generated.")
