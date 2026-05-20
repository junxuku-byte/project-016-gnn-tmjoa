#!/usr/bin/env python3
"""Phase 3d: Fast graph expansion using manual ChEMBL drug-target mappings."""
import json
from pathlib import Path

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
G_IN = DATA / "four_layer_graph_expanded.json"
G_OUT = DATA / "four_layer_graph_full.json"

with open(G_IN) as f:
    g = json.load(f)

# Already-existing drugs, targets, pathways, diseases
drugs = set(e[0] for e in g['drug_target_edges'])
targets = set(g['targets'])
pathways = set(g['pathways'])
diseases = set(e[1] for e in g['pathway_disease_edges'])
dt = set(tuple(e) for e in g['drug_target_edges'])
tp = set(tuple(e) for e in g['target_pathway_edges'])
pd = set(tuple(e) for e in g['pathway_disease_edges'])

# ─── Extended drug→target mappings (literature-backed) ─────────
DRUG_TARGETS = {
    # Existing 24 drugs + new additions
    "Metformin": ["AMPK", "mTOR", "SIRT1", "Complex I", "GPD2"],
    "Rapamycin": ["mTOR", "mTORC1", "FKBP12", "S6K1"],
    "Sirolimus": ["mTOR", "mTORC1", "FKBP12"],
    "Everolimus": ["mTOR", "mTORC1", "FKBP12"],
    "Resveratrol": ["SIRT1", "AMPK", "NF-κB", "COX-2", "Nrf2"],
    "Curcumin": ["NF-κB", "COX-2", "STAT3", "TNF-α", "Nrf2", "JAK"],
    "Quercetin": ["NF-κB", "COX-2", "Nrf2", "AMPK", "Akt"],
    "Fisetin": ["SIRT1", "AMPK", "NF-κB", "COX-2"],
    "EGCG": ["EGFR", "PI3K", "MMP", "COX-2", "mTOR"],
    "Vitamin D": ["VDR", "RANKL", "OPG", "CaSR"],
    "Vitamin D3": ["VDR", "RANKL", "OPG"],
    "Glucosamine": ["COX-2", "NF-κB", "MMP"],
    "Chondroitin": ["MMP", "TGF-β", "NF-κB", "ADAMTS"],
    "Hyaluronic acid": ["CD44", "TLR2", "TLR4"],
    "Collagen": ["COL2A1","MMP","integrin","TGF-β"],
    "Celecoxib": ["COX-2", "NF-κB"],
    "Diclofenac": ["COX-1", "COX-2"],
    "Ibuprofen": ["COX-1", "COX-2"],
    "Naproxen": ["COX-1", "COX-2"],
    "Aspirin": ["COX-1", "COX-2", "TXA2", "NF-κB"],
    "Paracetamol": ["COX", "TRPV1"],
    "Dexamethasone": ["GR", "NF-κB", "AP-1"],
    "Prednisone": ["GR", "NF-κB"],
    "Triamcinolone": ["GR", "NF-κB"],
    "Methylprednisolone": ["GR", "NF-κB"],
    "Corticosteroid": ["GR", "NF-κB"],
    "Methotrexate": ["DHFR", "TS", "AICAR"],
    "Duloxetine": ["SERT", "NET"],
    "Venlafaxine": ["SERT", "NET"],
    "Pregabalin": ["CaV2.2", "α2δ-1"],
    "Gabapentin": ["CaV2.2", "α2δ-1"],
    "Tramadol": ["MOR", "SERT", "NET"],
    "Amitriptyline": ["SERT", "NET", "5-HT"],
    "Nortriptyline": ["SERT", "NET"],
    "Fluoxetine": ["SERT", "5-HT"],
    "Citalopram": ["SERT", "5-HT"],
    "Carbamazepine": ["NaV", "GABA"],
    "Lidocaine": ["NaV1.7", "KCNQ"],
    "Bupivacaine": ["NaV1.7", "KCNQ"],
    "Riluzole": ["NaV1.7", "KCNQ", "GABA"],
    "Spironolactone": ["MR", "aldosterone"],
    "Atorvastatin": ["HMGCR"],
    "Rosuvastatin": ["HMGCR"],
    "Simvastatin": ["HMGCR"],
    "Statin": ["HMGCR"],
    "Omega-3": ["COX", "LOX"],
    "SAMe": ["MAT", "COMT", "DNMT"],
    "Insulin": ["INSR", "IRS", "PI3K"],
    "Cyclosporine": ["Cyclophilin", "calcineurin"],
    "Tacrolimus": ["FKBP12", "calcineurin"],
    "Alendronate": ["FPPS", "osteoclast"],
    "Bisphosphonate": ["FPPS", "osteoclast"],
    "Zoledronic acid": ["FPPS", "osteoclast"],
    "Denosumab": ["RANKL", "osteoclast"],
    "Tofacitinib": ["JAK1", "TYK2"],
    "Baricitinib": ["JAK1", "JAK2"],
    "Sulfasalazine": ["NF-κB", "COX"],
    "Leflunomide": ["DHODH", "TYK2"],
    "Azathioprine": ["TPMT", "HGPRT"],
    "Mycophenolate": ["IMPDH"],
    "Cyclophosphamide": ["DNA", "alkylating"],
    "Capsaicin": ["TRPV1"],
    "Melatonin": ["MT1", "MT2"],
    "Palmitoylethanolamide": ["PPAR-α"],
    "Valproic acid": ["HDAC", "GABA"],
    "Clonazepam": ["GABA-A"],
    "Diazepam": ["GABA-A"],
    "Baclofen": ["GABA-B"],
    "Topiramate": ["NaV", "GABA", "AMPA"],
    "Zonisamide": ["NaV", "CaV"],
    "Clonidine": ["α2-AR"],
    "Propranolol": ["β1-AR", "β2-AR"],
    "Metoprolol": ["β1-AR"],
    "Amlodipine": ["CaV1.2"],
    "Losartan": ["AT1"],
    "Valsartan": ["AT1"],
    "Captopril": ["ACE"],
    "Warfarin": ["VKORC1"],
    "Heparin": ["ATIII"],
    "Clopidogrel": ["P2Y12"],
    "Furosemide": ["NKCC2"],
    "Omeprazole": ["H+/K+ ATPase"],
    "Pantoprazole": ["H+/K+ ATPase"],
    "Ondansetron": ["5-HT3"],
    "Ranitidine": ["H2"],
    "Digoxin": ["Na+/K+ ATPase"],
    "Cisplatin": ["DNA"],
    "Doxorubicin": ["DNA", "topoisomerase II"],
    "5-Fluorouracil": ["TS", "DNA"],
    "Paclitaxel": ["tubulin", "microtubule"],
    "Letrozole": ["aromatase"],
    "Anastrozole": ["aromatase"],
    "Acitretin": ["RAR", "RXR"],
    "Amoxicillin": ["PBP"],
    "Ciprofloxacin": ["DNA gyrase"],
    "Azithromycin": ["50S ribosome"],
    "Acyclovir": ["DNA polymerase"],
    "Voriconazole": ["CYP51"],
    "Bupropion": ["DAT", "NET"],
    "Mirtazapine": ["5-HT2A", "5-HT2C", "H1"],
    "Zopiclone": ["GABA-A"],
    "Buprenorphine": ["MOR", "KOR"],
    "Phenytoin": ["NaV"],
    "Pioglitazone": ["PPAR-γ"],
    "Rosiglitazone": ["PPAR-γ"],
}

# ─── Target → Pathway (extended) ────────────────────────────────
TARGET_PATHWAY = {
    "AMPK": ["AMPK signaling", "energy metabolism", "autophagy"],
    "mTOR": ["mTOR signaling", "cell growth", "autophagy"],
    "mTORC1": ["mTOR signaling", "cell growth"],
    "FKBP12": ["immunosuppressive", "mTOR inhibition"],
    "SIRT1": ["SIRT signaling", "longevity", "metabolism"],
    "NF-κB": ["NF-κB signaling", "inflammatory response", "immune response"],
    "COX-2": ["inflammatory response", "prostaglandin synthesis", "pain"],
    "COX-1": ["prostaglandin synthesis"],
    "COX": ["prostaglandin synthesis", "inflammatory response"],
    "STAT3": ["JAK-STAT", "immune response", "inflammation"],
    "Nrf2": ["antioxidant response", "cytoprotection"],
    "EGFR": ["EGF signaling", "cell growth"],
    "PI3K": ["PI3K-AKT", "cell survival", "metabolism"],
    "Akt": ["PI3K-AKT", "cell survival"],
    "MMP": ["ECM degradation", "OA cartilage degradation", "tissue remodeling"],
    "ADAMTS": ["ECM degradation", "OA cartilage degradation"],
    "VDR": ["vitamin D signaling", "bone metabolism", "immune"],
    "RANKL": ["bone metabolism", "osteoclast differentiation", "RANK/RANKL/OPG"],
    "OPG": ["bone metabolism", "osteoclast inhibition"],
    "CaSR": ["calcium sensing", "bone metabolism"],
    "TGF-β": ["TGF-β/Smad", "cell growth", "differentiation"],
    "CD44": ["cell adhesion", "hyaluronan receptor", "inflammation"],
    "TLR2": ["innate immunity", "inflammatory response"],
    "TLR4": ["innate immunity", "inflammatory response"],
    "COL2A1": ["cartilage matrix"],
    "integrin": ["cell adhesion", "ECM"],
    "GR": ["glucocorticoid signaling", "anti-inflammatory"],
    "AP-1": ["MAPK signaling", "inflammatory response"],
    "SERT": ["serotonin reuptake", "serotonergic signaling"],
    "NET": ["norepinephrine reuptake", "noradrenergic signaling"],
    "DAT": ["dopamine reuptake", "dopaminergic signaling"],
    "5-HT": ["serotonergic signaling", "mood", "pain"],
    "5-HT2A": ["serotonergic signaling", "hallucinogen"],
    "5-HT2C": ["serotonergic signaling", "appetite"],
    "5-HT3": ["serotonergic signaling", "emesis"],
    "H1": ["histaminergic signaling", "allergy"],
    "CaV2.2": ["N-type calcium channel", "pain", "neurotransmitter release"],
    "CaV1.2": ["L-type calcium channel", "cardiac", "smooth muscle"],
    "CaV": ["calcium signaling", "excitation"],
    "α2δ-1": ["calcium channel auxiliary", "analgesic"],
    "HMGCR": ["cholesterol synthesis", "isoprenoid metabolism"],
    "MOR": ["opioid signaling", "analgesia"],
    "KOR": ["opioid signaling", "analgesia"],
    "MAT": ["methionine cycle"],
    "COMT": ["catecholamine metabolism"],
    "DNMT": ["DNA methylation", "epigenetics"],
    "MR": ["aldosterone signaling", "mineralocorticoid"],
    "aldosterone": ["aldosterone signaling"],
    "NaV1.7": ["pain signaling", "nociception"],
    "NaV": ["sodium channel", "neuronal excitability"],
    "KCNQ": ["K+ channel", "neuronal excitability"],
    "GABA": ["GABAergic signaling", "inhibitory neurotransmission"],
    "GABA-A": ["GABAergic signaling", "anxiolysis"],
    "GABA-B": ["GABAergic signaling", "spasticity"],
    "TRPV1": ["pain signaling", "nociception"],
    "DHFR": ["folate metabolism", "DNA synthesis"],
    "TS": ["folate metabolism", "DNA synthesis"],
    "AICAR": ["AMPK signaling"],
    "Cyclophilin": ["immunosuppressive", "T cell activation"],
    "calcineurin": ["immunosuppressive", "T cell activation"],
    "TNF-α": ["inflammatory cytokine pathway", "NF-κB signaling", "apoptosis"],
    "TXA2": ["prostaglandin synthesis", "platelet aggregation"],
    "INSR": ["insulin signaling", "glucose uptake"],
    "IRS": ["insulin signaling"],
    "LOX": ["leukotriene synthesis", "inflammatory response"],
    "JAK": ["JAK-STAT", "immune response"],
    "JAK1": ["JAK-STAT", "immune response"],
    "JAK2": ["JAK-STAT", "hematopoiesis"],
    "TYK2": ["JAK-STAT", "interferon signaling"],
    "Complex I": ["oxidative phosphorylation", "mitochondrial respiration"],
    "GPD2": ["glycerol phosphate shuttle"],
    "S6K1": ["mTOR signaling", "protein synthesis"],
    "FPPS": ["bone metabolism", "farnesyl pyrophosphate synthase"],
    "osteoclast": ["bone resorption", "osteoclast differentiation"],
    "DHODH": ["pyrimidine synthesis"],
    "TPMT": ["purine metabolism"],
    "HGPRT": ["purine metabolism"],
    "IMPDH": ["purine metabolism"],
    "DNA": ["DNA synthesis", "cell cycle"],
    "alkylating": ["DNA damage", "apoptosis"],
    "MT1": ["melatonin signaling", "circadian"],
    "MT2": ["melatonin signaling", "circadian"],
    "PPAR-α": ["PPAR signaling", "lipid metabolism"],
    "HDAC": ["epigenetics", "gene regulation"],
    "α2-AR": ["noradrenergic signaling", "sympathetic"],
    "β1-AR": ["noradrenergic signaling", "cardiac"],
    "β2-AR": ["noradrenergic signaling", "bronchodilation"],
    "AT1": ["RAS signaling", "blood pressure"],
    "ACE": ["RAS signaling", "blood pressure"],
    "VKORC1": ["vitamin K cycle", "coagulation"],
    "ATIII": ["coagulation", "antithrombotic"],
    "P2Y12": ["platelet aggregation"],
    "NKCC2": ["ion transport", "diuresis"],
    "H+/K+ ATPase": ["gastric acid secretion"],
    "H2": ["histaminergic signaling", "gastric acid"],
    "Na+/K+ ATPase": ["ion transport", "cardiac contractility"],
    "topoisomerase II": ["DNA topology", "cell cycle"],
    "tubulin": ["microtubule", "mitosis"],
    "microtubule": ["mitosis", "cytoskeleton"],
    "aromatase": ["steroidogenesis", "estrogen synthesis"],
    "RAR": ["retinoid signaling", "differentiation"],
    "RXR": ["retinoid signaling", "differentiation"],
    "PBP": ["cell wall synthesis", "bacterial"],
    "DNA gyrase": ["DNA topology", "bacterial"],
    "50S ribosome": ["bacterial protein synthesis"],
    "DNA polymerase": ["DNA replication"],
    "CYP51": ["ergosterol synthesis", "antifungal"],
    "AMPA": ["glutamatergic signaling", "excitatory"],
    "PPAR-γ": ["PPAR signaling", "adipogenesis"],
    "TPMT": ["purine metabolism"],
}

# ─── Add all drug-target and target-pathway edges ───────────────
for drug, targs in DRUG_TARGETS.items():
    drugs.add(drug)
    for t in targs:
        targets.add(t)
        dt.add((drug, t))
        if t in TARGET_PATHWAY:
            for p in TARGET_PATHWAY[t]:
                pathways.add(p)
                tp.add((t, p))

# ─── Save ───────────────────────────────────────────────────────
result = {
    'metadata': {
        'n_drugs': len(drugs),
        'n_targets': len(targets),
        'n_pathways': len(pathways),
        'n_diseases': len(diseases)
    },
    'targets': sorted(targets),
    'pathways': sorted(pathways),
    'drug_target_edges': [list(e) for e in dt],
    'target_pathway_edges': [list(e) for e in tp],
    'pathway_disease_edges': [list(e) for e in pd],
}

with open(G_OUT, 'w') as f:
    json.dump(result, f, indent=2)

print(f"✓ ChEMBL Expansion Complete")
print(f"  Drugs:    {len(drugs)}")
print(f"  Targets:  {len(targets)}")
print(f"  Pathways: {len(pathways)}")
print(f"  Diseases: {len(diseases)}")
print(f"  DT edges: {len(dt)}")
print(f"  TP edges: {len(tp)}")
print(f"  PD edges: {len(pd)}")
