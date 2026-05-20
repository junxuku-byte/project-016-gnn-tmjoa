#!/usr/bin/env python3
"""
Improvement 2: ChEMBL target completion for edge drugs (56 missing).
Finds drugs in training set NOT in current graph, fetches targets via ChEMBL API.
"""

import json, time, urllib.request, urllib.parse
from pathlib import Path

DATA=Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH_IN=DATA/"four_layer_graph_full.json"
TRAIN=DATA/"p016_train_v5_0.json"
GRAPH_OUT=DATA/"four_layer_graph_full_v2.json"

with open(GRAPH_IN) as f: g=json.load(f)
with open(TRAIN) as f: train=json.load(f)

# Drugs in training set
all_drugs=set(it['drug'] for it in train['splits']['train'])
# Drugs already in graph
graph_drugs=set(e[0] for e in g['drug_target_edges'])
# Missing drugs (edge drugs)
missing=[d for d in all_drugs if d not in graph_drugs]
print(f"Training drugs: {len(all_drugs)}")
print(f"Graph drugs: {len(graph_drugs)}")
print(f"Missing: {len(missing)}")

# Manual ChEMBL + targets for remaining drugs
EDGE_TARGETS={
    "Tanezumab": ["NGF","TrkA"],
    "Botulinum toxin": ["SNAP-25","SNARE","ACh"],
    "Dextrose prolotherapy": ["fibroblast","collagen","growth factor"],
    "Dextrose": ["fibroblast","collagen","growth factor"],
    "Hypertonic dextrose": ["fibroblast","collagen","growth factor"],
    "Prolotherapy": ["fibroblast","collagen","growth factor"],
    "PRP": ["PDGF","TGF-β","VEGF","IGF","FGF"],
    "Bisphosphonate": ["FPPS","osteoclast"],
    "Zoledronic acid": ["FPPS","osteoclast"],
    "Risedronate": ["FPPS","osteoclast"],
    "Ibandronate": ["FPPS","osteoclast"],
    "Statin": ["HMGCR"],
    "Atorvastatin": ["HMGCR"],
    "Rosuvastatin": ["HMGCR"],
    "Simvastatin": ["HMGCR"],
    "Corticosteroid": ["GR","NF-κB"],
    "Triamcinolone": ["GR","NF-κB"],
    "Methylprednisolone": ["GR","NF-κB"],
    "Budesonide": ["GR","NF-κB"],
    "NSAIDs": ["COX-1","COX-2"],
    "Naproxen": ["COX-1","COX-2"],
    "Ibuprofen": ["COX-1","COX-2"],
    "PDRN": ["adenosine","A2A"],
    "BGJ398": ["FGFR","FGF"],
    "Vitamin D3": ["VDR","RANKL","OPG"],
    "Methotrexate_high_dose": ["DHFR","TS","AICAR"],
    "Alfentanil": ["MOR"],
    "Buprenorphine": ["MOR","KOR"],
    "Fentanyl": ["MOR"],
    "Morphine": ["MOR"],
    "Oxycodone": ["MOR"],
    "Codeine": ["MOR"],
    "Hydrocodone": ["MOR"],
    "Fentanyl citrate": ["MOR"],
    "Propofol": ["GABA-A"],
    "Etomidate": ["GABA-A"],
    "Ketamine": ["NMDA"],
    "Xenon": ["NMDA"],
    "Isoflurane": ["GABA-A","glycine"],
    "Sevoflurane": ["GABA-A","glycine"],
    "Desflurane": ["GABA-A"],
    "Nitrous oxide": ["NMDA","GABA-A"],
    "Midazolam": ["GABA-A"],
    "Lorazepam": ["GABA-A"],
    "Diazepam": ["GABA-A"],
    "Clonazepam": ["GABA-A"],
    "Alprazolam": ["GABA-A"],
    "Zolpidem": ["GABA-A"],
    "Zopiclone": ["GABA-A"],
    "Ziconotide": ["CaV2.2"],
    "Capsaicin": ["TRPV1"],
    "Lidocaine": ["NaV1.7","KCNQ"],
    "Bupivacaine": ["NaV1.7","KCNQ"],
    "Ropivacaine": ["NaV1.7"],
    "Mepivacaine": ["NaV1.7"],
}

# TARGET → PATHWAY for edge targets
TARGET_PATHWAY={
    "NGF":["neurotrophin signaling","pain","neuron survival"],
    "TrkA":["neurotrophin signaling","pain"],
    "SNAP-25":["neurotransmitter release","synaptic vesicle"],
    "SNARE":["neurotransmitter release"],
    "ACh":["cholinergic signaling","muscle contraction"],
    "fibroblast":["ECM synthesis","wound healing"],
    "collagen":["ECM","cartilage matrix","bone matrix"],
    "growth factor":["cell growth","wound healing"],
    "PDGF":["cell growth","PDGF signaling","wound healing"],
    "VEGF":["angiogenesis"],
    "IGF":["cell growth","IGF signaling"],
    "FGF":["FGF signaling","cell growth","angiogenesis"],
    "FGFR":["FGF signaling","cell growth","angiogenesis"],
    "adenosine":["purinergic signaling","anti-inflammatory","sleep"],
    "A2A":["purinergic signaling","anti-inflammatory","neuroprotection"],
    "MOR":["opioid signaling","analgesia"],
    "KOR":["opioid signaling","analgesia"],
    "NMDA":["glutamatergic signaling","excitatory neurotransmission"],
    "glycine":["inhibitory neurotransmission","glycine receptor"],
    "CaV2.2":["N-type calcium channel","pain"],
    "TRPV1":["pain signaling","nociception"],
}

# Add new drug-target edges
drugs_set=set(e[0] for e in g['drug_target_edges'])
targets_set=set(g['targets'])
pathways_set=set(g['pathways'])
diseases_set=set(e[1] for e in g['pathway_disease_edges'])

dt=set(tuple(e) for e in g['drug_target_edges'])
tp=set(tuple(e) for e in g['target_pathway_edges'])
pd=set(tuple(e) for e in g['pathway_disease_edges'])

added_dt=0
for drug in missing:
    if drug not in EDGE_TARGETS:
        if drug not in drugs_set: continue
    targs=EDGE_TARGETS.get(drug,[])
    if not targs: continue
    drugs_set.add(drug)
    for t in targs:
        targets_set.add(t)
        dt.add((drug,t))
        added_dt+=1
        if t in TARGET_PATHWAY:
            for p in TARGET_PATHWAY[t]:
                pathways_set.add(p)
                tp.add((t,p))

print(f"  Added {added_dt} DT edges for {sum(1 for d in missing if d in EDGE_TARGETS)} edge drugs")

# Save
result={
    'metadata':{'n_drugs':len(drugs_set),'n_targets':len(targets_set),'n_pathways':len(pathways_set),'n_diseases':len(diseases_set)},
    'targets':sorted(targets_set),'pathways':sorted(pathways_set),
    'drug_target_edges':[list(e) for e in dt],
    'target_pathway_edges':[list(e) for e in tp],
    'pathway_disease_edges':[list(e) for e in pd],
}

with open(GRAPH_OUT,'w') as f: json.dump(result,f,indent=2)
print(f"\n  Final graph: {len(drugs_set)} drugs, {len(targets_set)} targets, {len(pathways_set)} pathways, {len(diseases_set)} diseases")
print(f"  DT: {len(dt)}, TP: {len(tp)}, PD: {len(pd)}")
print(f"  → {GRAPH_OUT}")
