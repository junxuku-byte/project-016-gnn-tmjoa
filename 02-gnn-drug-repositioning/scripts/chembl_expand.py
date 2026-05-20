#!/usr/bin/env python3
"""
Phase 3d: ChEMBL bulk drug-target mapping to expand four-layer graph coverage.
1. Map 171 drugs to ChEMBL compound IDs via API
2. Fetch known protein targets from ChEMBL
3. Map targets to pathways via domain knowledge
4. Build expanded graph with more drug-target edges
"""

import json, time, urllib.request, urllib.parse, urllib.error
from collections import defaultdict
from pathlib import Path

BASE = "https://www.ebi.ac.uk/chembl/api/data"
DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
GRAPH_IN = DATA / "four_layer_graph_expanded.json"
GRAPH_OUT = DATA / "four_layer_graph_chembl.json"
TRAIN = DATA / "p016_train_v5_0.json"
DRUG_MAP_OUT = DATA / "chembl_drug_targets.json"

# Load current graph & training data
with open(GRAPH_IN) as f:
    g = json.load(f)
with open(TRAIN) as f:
    train = json.load(f)

# Get all 171 drugs from training data
all_drugs = set()
for item in train['splits']['train']:
    all_drugs.add(item['drug'])
all_drugs = sorted(all_drugs)

print(f"Total drugs in training set: {len(all_drugs)}")
print(f"Drugs with current mechanism data: {len(set(e[0] for e in g['drug_target_edges']))}")

# ─── Step 1: Map drug names to ChEMBL IDs ───────────────────────
def chembl_request(endpoint, params=None, retries=3):
    url = f"{BASE}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ChEMBL error {endpoint}: {e}")
                return None

drug_chembl = {}  # drug_name -> chembl_id

# Bulk search by drug name
print("\n--- Step 1: Drug name → ChEMBL ID ---")
# Common name overrides (manual mapping for known drugs)
MANUAL_CHEMBL = {
    "Metformin": "CHEMBL1431",
    "Rapamycin": "CHEMBL413",
    "Sirolimus": "CHEMBL413",
    "Resveratrol": "CHEMBL165",
    "Curcumin": "CHEMBL116",
    "Quercetin": "CHEMBL50",
    "Fisetin": "CHEMBL29424",
    "EGCG": "CHEMBL297453",
    "Vitamin D": "CHEMBL527",
    "Vitamin D3": "CHEMBL527",
    "Glucosamine": "CHEMBL493827",
    "Chondroitin": "CHEMBL1201449",
    "Hyaluronic acid": "CHEMBL1235252",
    "Celecoxib": "CHEMBL118",
    "Diclofenac": "CHEMBL139",
    "Ibuprofen": "CHEMBL521",
    "Naproxen": "CHEMBL154",
    "Aspirin": "CHEMBL25",
    "Paracetamol": "CHEMBL112",
    "Dexamethasone": "CHEMBL384467",
    "Prednisone": "CHEMBL635",
    "Methotrexate": "CHEMBL34259",
    "Duloxetine": "CHEMBL1175",
    "Pregabalin": "CHEMBL1059",
    "Gabapentin": "CHEMBL940",
    "Tramadol": "CHEMBL1066",
    "Amitriptyline": "CHEMBL629",
    "Fluoxetine": "CHEMBL41",
    "Carbamazepine": "CHEMBL108",
    "Lidocaine": "CHEMBL79",
    "Riluzole": "CHEMBL744",
    "Spironolactone": "CHEMBL453",
    "Atorvastatin": "CHEMBL1487",
    "Simvastatin": "CHEMBL1064",
    "Omega-3": "CHEMBL1237052",
    "SAMe": "CHEMBL24924",
    "Tanezumab": None,  # monoclonal antibody, no small molecule target
    "Botulinum toxin": None,  # protein toxin
    "Dextrose prolotherapy": None,
    "PRP": None,
    "Collagen": None,
    "Tocilizumab": None,
    "Infliximab": None,
    "Etanercept": None,
    "Adalimumab": None,
    "Anakinra": None,
    "Denosumab": None,
    "Insulin": "CHEMBL1201552",
    "Cyclosporine": "CHEMBL160",
    "Tacrolimus": "CHEMBL269732",
}

for drug in all_drugs:
    if drug in MANUAL_CHEMBL:
        if MANUAL_CHEMBL[drug]:
            drug_chembl[drug] = MANUAL_CHEMBL[drug]
        continue
    # API search
    result = chembl_request("molecule", {"pref_name__iexact": drug, "format": "json", "limit": 1})
    if result and result.get("molecules"):
        drug_chembl[drug] = result["molecules"][0]["molecule_chembl_id"]
        time.sleep(0.2)
    else:
        # Try fuzzy search
        result = chembl_request("molecule", {"pref_name__icontains": drug, "format": "json", "limit": 1})
        if result and result.get("molecules"):
            mid = result["molecules"][0]["molecule_chembl_id"]
            name = result["molecules"][0].get("pref_name", "?")
            print(f"  {drug} → {mid} ({name})")
            drug_chembl[drug] = mid
            time.sleep(0.2)

print(f"  Mapped: {len(drug_chembl)}/{len(all_drugs)} drugs")

# ─── Step 2: Fetch targets for each drug ────────────────────────
print("\n--- Step 2: ChEMBL ID → Target proteins ---")

drug_target_proteins = {}  # drug → [protein_names]

# Quick target lookup for common drugs (literature-backed, saves API calls)
KNOWN_TARGETS = {
    "CHEMBL1431": ["AMPK", "Complex I", "GPD2", "SIRT1"],  # Metformin
    "CHEMBL413": ["mTOR", "FKBP12", "mTORC1"],  # Rapamycin/Sirolimus
    "CHEMBL165": ["SIRT1", "AMPK", "NF-κB", "COX-2"],  # Resveratrol
    "CHEMBL116": ["NF-κB", "COX-2", "STAT3", "TNF-α", "Nrf2"],  # Curcumin
    "CHEMBL50": ["NF-κB", "COX-2", "Nrf2", "AMPK", "Akt"],  # Quercetin
    "CHEMBL29424": ["SIRT1", "AMPK", "NF-κB", "COX-2"],  # Fisetin
    "CHEMBL297453": ["EGFR", "PI3K", "MMP", "COX-2"],  # EGCG
    "CHEMBL527": ["VDR", "RANKL", "OPG"],  # Vitamin D
    "CHEMBL493827": ["COX-2", "NF-κB", "MMP"],  # Glucosamine
    "CHEMBL1201449": ["MMP", "TGF-β", "NF-κB", "ADAMTS"],  # Chondroitin
    "CHEMBL1235252": ["CD44", "TLR2", "TLR4"],  # Hyaluronic acid
    "CHEMBL118": ["COX-2"],  # Celecoxib
    "CHEMBL139": ["COX-1", "COX-2"],  # Diclofenac
    "CHEMBL521": ["COX-1", "COX-2"],  # Ibuprofen
    "CHEMBL154": ["COX-1", "COX-2"],  # Naproxen
    "CHEMBL25": ["COX-1", "COX-2", "TXA2", "NF-κB"],  # Aspirin
    "CHEMBL112": ["COX", "TRPV1"],  # Paracetamol
    "CHEMBL384467": ["GR", "NF-κB", "AP-1"],  # Dexamethasone
    "CHEMBL635": ["GR", "NF-κB"],  # Prednisone
    "CHEMBL34259": ["DHFR", "TS", "AICAR"],  # Methotrexate
    "CHEMBL1175": ["SERT", "NET"],  # Duloxetine
    "CHEMBL1059": ["CaV2.2", "α2δ-1"],  # Pregabalin
    "CHEMBL940": ["CaV2.2", "α2δ-1"],  # Gabapentin
    "CHEMBL1066": ["MOR", "SERT", "NET"],  # Tramadol
    "CHEMBL629": ["SERT", "NET", "5-HT"],  # Amitriptyline
    "CHEMBL41": ["SERT", "5-HT"],  # Fluoxetine
    "CHEMBL108": ["NaV", "GABA"],  # Carbamazepine
    "CHEMBL79": ["NaV1.7", "KCNQ"],  # Lidocaine
    "CHEMBL744": ["NaV1.7", "KCNQ", "GABA"],  # Riluzole
    "CHEMBL453": ["MR", "aldosterone"],  # Spironolactone
    "CHEMBL1487": ["HMGCR"],  # Atorvastatin
    "CHEMBL1064": ["HMGCR"],  # Simvastatin
    "CHEMBL1237052": ["COX", "LOX"],  # Omega-3
    "CHEMBL24924": ["MAT", "COMT", "DNMT"],  # SAMe
    "CHEMBL1201552": ["INSR", "IRS", "PI3K"],  # Insulin
    "CHEMBL160": ["Cyclophilin", "calcineurin"],  # Cyclosporine
    "CHEMBL269732": ["FKBP12", "calcineurin"],  # Tacrolimus
}

for drug, chembl_id in drug_chembl.items():
    if chembl_id in KNOWN_TARGETS:
        drug_target_proteins[drug] = KNOWN_TARGETS[chembl_id]
        continue

    # API fetch
    result = chembl_request("mechanism", {
        "molecule_chembl_id": chembl_id,
        "format": "json", "limit": 50
    })
    targets = set()
    if result and result.get("mechanisms"):
        for mech in result["mechanisms"]:
            tname = mech.get("target_chembl_id") or mech.get("target_pref_name", "")
            if tname:
                targets.add(tname)
    if targets:
        drug_target_proteins[drug] = list(targets)
    time.sleep(0.3)

print(f"  Drugs with targets: {len(drug_target_proteins)}")

# ─── Step 3: Map ChEMBL targets to our pathway targets ────────
print("\n--- Step 3: Target → Pathway mapping ---")

# Load existing targets for pathway lookup
TARGET_PATHWAY = {
    "AMPK": ["AMPK signaling", "energy metabolism"],
    "mTOR": ["mTOR signaling", "cell growth", "autophagy"],
    "mTORC1": ["mTOR signaling", "cell growth"],
    "FKBP12": ["mTOR inhibition"],
    "SIRT1": ["SIRT signaling", "longevity"],
    "NF-κB": ["NF-κB signaling", "inflammatory response"],
    "COX-2": ["inflammatory response", "prostaglandin synthesis"],
    "COX-1": ["prostaglandin synthesis"],
    "COX": ["prostaglandin synthesis"],
    "STAT3": ["JAK-STAT", "immune response"],
    "Nrf2": ["antioxidant response", "cytoprotection"],
    "EGFR": ["EGF signaling", "cell growth"],
    "PI3K": ["PI3K-AKT", "cell survival"],
    "MMP": ["ECM degradation", "OA cartilage degradation"],
    "ADAMTS": ["ECM degradation", "OA cartilage degradation"],
    "VDR": ["vitamin D signaling", "bone metabolism"],
    "RANKL": ["bone metabolism", "osteoclast differentiation"],
    "OPG": ["bone metabolism", "osteoclast inhibition"],
    "TGF-β": ["TGF-β/Smad", "cell growth"],
    "CD44": ["cell adhesion", "hyaluronan receptor"],
    "TLR2": ["innate immunity", "inflammatory response"],
    "TLR4": ["innate immunity", "inflammatory response"],
    "GR": ["glucocorticoid signaling", "anti-inflammatory"],
    "AP-1": ["MAPK signaling", "inflammatory response"],
    "SERT": ["serotonin reuptake", "serotonergic signaling"],
    "NET": ["norepinephrine reuptake", "noradrenergic signaling"],
    "5-HT": ["serotonergic signaling", "mood"],
    "CaV2.2": ["N-type calcium channel", "pain"],
    "α2δ-1": ["calcium channel auxiliary", "analgesic"],
    "HMGCR": ["cholesterol synthesis"],
    "MOR": ["opioid signaling", "analgesia"],
    "MAT": ["methionine cycle"],
    "COMT": ["catecholamine metabolism"],
    "DNMT": ["DNA methylation"],
    "MR": ["aldosterone signaling", "mineralocorticoid"],
    "NaV1.7": ["pain signaling", "nociception"],
    "NaV": ["sodium channel", "neuronal excitability"],
    "KCNQ": ["K+ channel", "neuronal excitability"],
    "GABA": ["GABAergic signaling", "inhibitory neurotransmission"],
    "TRPV1": ["pain signaling", "nociception"],
    "DHFR": ["folate metabolism", "DNA synthesis"],
    "TS": ["folate metabolism", "DNA synthesis"],
    "AICAR": ["AMPK signaling"],
    "Cyclophilin": ["immunosuppressive", "T cell activation"],
    "calcineurin": ["immunosuppressive", "T cell activation"],
    "TNF-α": ["inflammatory cytokine pathway", "NF-κB signaling"],
    "TXA2": ["prostaglandin synthesis", "platelet aggregation"],
    "INSR": ["insulin signaling", "glucose uptake"],
    "IRS": ["insulin signaling", "PI3K-AKT"],
    "Akt": ["PI3K-AKT", "cell survival"],
    "LOX": ["leukotriene synthesis", "inflammatory response"],
    "aldosterone": ["aldosterone signaling"],
    "Complex I": ["oxidative phosphorylation", "mitochondrial respiration"],
    "GPD2": ["glycerol phosphate shuttle"],
}

# Build final edges
drugs_set = set(e[0] for e in g['drug_target_edges'])
targets_set = set(g['targets'])
pathways_set = set(g['pathways'])
diseases_set = set(e[1] for e in g['pathway_disease_edges'])

dt_edges = set(tuple(e) for e in g['drug_target_edges'])
tp_edges = set(tuple(e) for e in g['target_pathway_edges'])
pd_edges = set(tuple(e) for e in g['pathway_disease_edges'])

added_drugs = 0
added_dt = 0
for drug, targets in drug_target_proteins.items():
    if drug not in drugs_set:
        drugs_set.add(drug)
        added_drugs += 1
    for t in targets:
        targets_set.add(t)
        dt_edges.add((drug, t))
        added_dt += 1
        # Add target-pathway edges
        if t in TARGET_PATHWAY:
            for p in TARGET_PATHWAY[t]:
                pathways_set.add(p)
                tp_edges.add((t, p))

print(f"  New drugs added: {added_drugs}")
print(f"  New DT edges: {added_dt}")
print(f"  Total targets: {len(targets_set)}")
print(f"  Total pathways: {len(pathways_set)}")

# ─── Step 4: Build & save expanded graph ───────────────────────
result = {
    'metadata': {
        'n_drugs': len(drugs_set),
        'n_targets': len(targets_set),
        'n_pathways': len(pathways_set),
        'n_diseases': len(diseases_set)
    },
    'targets': sorted(targets_set),
    'pathways': sorted(pathways_set),
    'drug_target_edges': [list(e) for e in dt_edges],
    'target_pathway_edges': [list(e) for e in tp_edges],
    'pathway_disease_edges': [list(e) for e in pd_edges],
}

with open(GRAPH_OUT, 'w') as f:
    json.dump(result, f, indent=2)

# Save drug-target mapping
with open(DRUG_MAP_OUT, 'w') as f:
    json.dump(drug_target_proteins, f, indent=2)

print(f"\n--- Final Graph Statistics ---")
print(f"  Drugs: {len(drugs_set)}")
print(f"  Targets: {len(targets_set)}")
print(f"  Pathways: {len(pathways_set)}")
print(f"  Diseases: {len(diseases_set)}")
print(f"  DT edges: {len(dt_edges)}")
print(f"  TP edges: {len(tp_edges)}")
print(f"  PD edges: {len(pd_edges)}")
print(f"\n  Output: {GRAPH_OUT}")
