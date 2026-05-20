#!/usr/bin/env python3
"""
Phase 3b: Expand target-pathway-disease coverage.
Strategy: Merge domain knowledge + literature extraction.
"""

import json
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
ORIG = OUT_DIR / "four_layer_graph.json"
RAW = OUT_DIR / "mechanism_quadruples_raw.json"
OUT = OUT_DIR / "four_layer_graph_expanded.json"

with open(ORIG) as f:
    g = json.load(f)

with open(RAW) as f:
    raw = json.load(f)

# Derive node lists from edges
drugs = set()
for e in g['drug_target_edges']:
    drugs.add(e[0])

targets = set(g['targets'])
pathways = set(g['pathways'])

diseases = set()
for e in g['pathway_disease_edges']:
    diseases.add(e[1])

dt_edges = set(tuple(e) for e in g['drug_target_edges'])
tp_edges = set(tuple(e) for e in g['target_pathway_edges'])
pd_edges = set(tuple(e) for e in g['pathway_disease_edges'])

# === Domain knowledge: drug → targets (high-confidence only) ===
DRUG_KNOWN_TARGETS = {
    "Metformin": ["AMPK", "mTOR", "SIRT1"],
    "Rapamycin": ["mTOR", "mTORC1", "FKBP12"],
    "Resveratrol": ["SIRT1", "AMPK", "NF-κB"],
    "Curcumin": ["NF-κB", "COX-2", "STAT3"],
    "Quercetin": ["NF-κB", "COX-2", "Nrf2"],
    "EGCG": ["EGFR", "PI3K", "MMP"],
    "Vitamin D": ["VDR", "RANKL", "OPG"],
    "Glucosamine": ["COX-2", "NF-κB", "MMP"],
    "Chondroitin": ["MMP", "TGF-β", "NF-κB"],
    "Hyaluronic acid": ["CD44", "TLR2", "TLR4"],
    "Collagen": ["COL2A1", "MMP", "integrin"],
    "NSAIDs": ["COX-1", "COX-2", "NF-κB"],
    "Diclofenac": ["COX-1", "COX-2"],
    "Celecoxib": ["COX-2", "NF-κB"],
    "Dexamethasone": ["GR", "NF-κB", "AP-1"],
    "Duloxetine": ["SERT", "NET", "5-HT"],
    "Pregabalin": ["CaV2.2", "α2δ-1"],
    "Gabapentin": ["CaV2.2", "α2δ-1"],
    "Alendronate": ["FPPS", "RANKL", "osteoclast"],
    "Denosumab": ["RANKL", "osteoclast"],
    "Atorvastatin": ["HMGCR", "LDLR"],
    "Statin": ["HMGCR", "LDLR"],
    "Omega-3": ["COX", "LOX"],
    "SAMe": ["MAT", "COMT", "DNMT"],
    "Spironolactone": ["MR", "aldosterone"],
    "Botulinum toxin": ["SNAP-25", "SNARE", "ACh"],
    "Tanezumab": ["NGF", "TrkA"],
    "Aspirin": ["COX-1", "COX-2", "NF-κB"],
    "Paracetamol": ["COX", "TRPV1"],
    "Lidocaine": ["NaV1.7", "KCNQ"],
    "Amitriptyline": ["SERT", "NET", "5-HT"],
    "Carbamazepine": ["NaV", "GABA"],
    "Insulin": ["INSR", "IRS", "PI3K"],
    "Methotrexate": ["DHFR", "TS"],
    "Cyclosporine": ["Cyclophilin", "calcineurin"],
    "Tacrolimus": ["FKBP12", "calcineurin"],
    "Infliximab": ["TNF-α"],
    "Etanercept": ["TNF-α"],
    "Tocilizumab": ["IL-6R"],
    "Anakinra": ["IL-1R"],
    "Tofacitinib": ["JAK1", "TYK2"],
}

# === Target → Pathway (domain knowledge) ===
TARGET_KNOWN_PATHWAY = {
    "AMPK": ["AMPK signaling", "energy metabolism"],
    "mTOR": ["mTOR signaling", "cell growth", "autophagy"],
    "mTORC1": ["mTOR signaling", "cell growth"],
    "SIRT1": ["SIRT signaling", "longevity"],
    "NF-κB": ["NF-κB signaling", "inflammatory response"],
    "COX-2": ["inflammatory response", "prostaglandin synthesis"],
    "COX-1": ["prostaglandin synthesis"],
    "STAT3": ["JAK-STAT", "immune response"],
    "Nrf2": ["antioxidant response", "cytoprotection"],
    "EGFR": ["EGF signaling", "cell growth"],
    "PI3K": ["PI3K-AKT", "cell survival"],
    "MMP": ["ECM degradation", "OA cartilage degradation"],
    "VDR": ["vitamin D signaling", "bone metabolism"],
    "RANKL": ["bone metabolism", "osteoclast differentiation"],
    "OPG": ["bone metabolism", "osteoclast inhibition"],
    "TGF-β": ["TGF-β/Smad", "cell growth"],
    "CD44": ["cell adhesion", "hyaluronan receptor"],
    "TLR2": ["innate immunity", "inflammatory response"],
    "TLR4": ["innate immunity", "inflammatory response"],
    "COL2A1": ["cartilage matrix"],
    "integrin": ["cell adhesion"],
    "GR": ["glucocorticoid signaling", "anti-inflammatory"],
    "AP-1": ["MAPK signaling", "inflammatory response"],
    "SERT": ["serotonin reuptake", "serotonergic signaling"],
    "NET": ["norepinephrine reuptake", "noradrenergic signaling"],
    "5-HT": ["serotonergic signaling", "mood"],
    "CaV2.2": ["N-type calcium channel", "pain"],
    "α2δ-1": ["calcium channel auxiliary", "analgesic"],
    "FPPS": ["bone metabolism", "farnesyl pyrophosphate synthase"],
    "osteoclast": ["bone resorption", "osteoclast differentiation"],
    "HMGCR": ["cholesterol synthesis"],
    "LDLR": ["cholesterol uptake"],
    "MAT": ["methionine cycle"],
    "COMT": ["catecholamine metabolism"],
    "MR": ["aldosterone signaling", "mineralocorticoid"],
    "SNAP-25": ["neurotransmitter release", "synaptic vesicle"],
    "SNARE": ["neurotransmitter release", "synaptic vesicle"],
    "ACh": ["cholinergic signaling", "muscle contraction"],
    "NGF": ["neurotrophin signaling", "pain", "neuron survival"],
    "TrkA": ["neurotrophin signaling", "pain"],
    "NaV1.7": ["pain signaling", "nociception"],
    "KCNQ": ["K+ channel", "neuronal excitability"],
    "TRPV1": ["pain signaling", "nociception"],
    "NaV": ["sodium channel", "neuronal excitability"],
    "GABA": ["GABAergic signaling", "inhibitory neurotransmission"],
    "INSR": ["insulin signaling", "glucose uptake"],
    "IRS": ["insulin signaling", "PI3K-AKT"],
    "DHFR": ["folate metabolism", "DNA synthesis"],
    "TS": ["folate metabolism", "DNA synthesis"],
    "Cyclophilin": ["immunosuppressive", "calcineurin inhibition"],
    "calcineurin": ["immunosuppressive", "T cell activation"],
    "FKBP12": ["immunosuppressive", "mTOR inhibition"],
    "TNF-α": ["inflammatory cytokine pathway", "NF-κB signaling"],
    "IL-6R": ["inflammatory cytokine pathway", "JAK-STAT"],
    "IL-1R": ["inflammatory cytokine pathway", "NF-κB signaling"],
    "JAK1": ["JAK-STAT", "immune response"],
    "TYK2": ["JAK-STAT", "interferon signaling"],
}

# === Disease → Pathway (from cards + domain knowledge) ===
DISEASE_PATHWAY = {
    "Temporomandibular joint osteoarthritis": [
        "OA cartilage degradation", "inflammatory cytokine pathway", "bone metabolism",
        "cartilage matrix", "ECM degradation", "pain signaling", "nociception",
        "TGF-β/Smad", "Wnt/β-catenin", "NF-κB signaling", "MAPK signaling"
    ],
    "TMD": [
        "inflammatory cytokine pathway", "pain signaling", "nociception",
        "NF-κB signaling", "cartilage matrix"
    ],
    "Rheumatoid arthritis": [
        "inflammatory cytokine pathway", "NF-κB signaling", "JAK-STAT",
        "bone metabolism", "ECM degradation", "autoimmune", "immune response"
    ],
    "Osteoarthritis": [
        "OA cartilage degradation", "ECM degradation", "inflammatory cytokine pathway",
        "bone metabolism", "cartilage matrix", "pain signaling"
    ],
    "Osteoporosis": [
        "bone metabolism", "osteoclast differentiation", "osteoblast differentiation",
        "RANK/RANKL/OPG", "Wnt/β-catenin", "calcium signaling"
    ],
    "Fibromyalgia": [
        "pain signaling", "nociception", "serotonergic signaling", "noradrenergic signaling",
        "central sensitization"
    ],
    "Chronic pain": [
        "pain signaling", "nociception", "inflammatory cytokine pathway",
        "central sensitization", "neuroplasticity"
    ],
    "Pain": [
        "pain signaling", "nociception", "inflammatory cytokine pathway"
    ],
}

# Add known drug-target edges
for drug, targs in DRUG_KNOWN_TARGETS.items():
    if drug in drugs:
        for t in targs:
            targets.add(t)
            dt_edges.add((drug, t))

# Add known target-pathway edges
for target, pws in TARGET_KNOWN_PATHWAY.items():
    if target in targets:
        for p in pws:
            pathways.add(p)
            tp_edges.add((target, p))

# Add disease-pathway edges
for disease, pws in DISEASE_PATHWAY.items():
    if disease in diseases:
        for p in pws:
            pathways.add(p)
            pd_edges.add((p, disease))

# Add pathway-disease from raw cards
for quad in raw.get('quadruples', []):
    if quad.get('pathway') and quad.get('disease'):
        pathways.add(quad['pathway'])
        diseases.add(quad['disease'])
        pd_edges.add((quad['pathway'], quad['disease']))

# Final graph
result = {
    'metadata': {
        'n_drugs': len(drugs),
        'n_targets': len(targets),
        'n_pathways': len(pathways),
        'n_diseases': len(diseases)
    },
    'targets': sorted(targets),
    'pathways': sorted(pathways),
    'drug_target_edges': [list(e) for e in dt_edges],
    'target_pathway_edges': [list(e) for e in tp_edges],
    'pathway_disease_edges': [list(e) for e in pd_edges],
}

with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Expanded graph saved to {OUT}")
print(f"  Drugs: {len(drugs)}")
print(f"  Targets: {len(targets)}")
print(f"  Pathways: {len(pathways)}")
print(f"  Diseases: {len(diseases)}")
print(f"  DT edges: {len(dt_edges)}")
print(f"  TP edges: {len(tp_edges)}")
print(f"  PD edges: {len(pd_edges)}")
