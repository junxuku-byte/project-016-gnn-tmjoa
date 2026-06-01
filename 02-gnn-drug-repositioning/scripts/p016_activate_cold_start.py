#!/usr/bin/env python3
"""
Activate 93 cold-start drugs by adding drug-target edges with known pharmacology.

For each cold-start drug, maps to its established protein targets based on
DrugBank/ChEMBL mechanism-of-action data, then adds target→pathway edges
to existing signaling pathways for mechanism-grounded GNN prediction.

Also adds user-specified clinical drugs:
  - Celecoxib, Pregabalin, Salmon calcitonin
  - Anti-anxiety/antidepressants (Escitalopram, Sertraline already in cold-start list)

Output: four_layer_graph_full_v3.json (v2 + cold-start drug-target edges)
"""
import json, copy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA = SCRIPT_DIR / "data"

# ── Load existing graph ───────────────────────────────────────
with open(DATA / "four_layer_graph_full_v2.json") as f:
    g = json.load(f)

existing_targets = set(g['targets'])
existing_edges = {(e[0], e[1]) for e in g['drug_target_edges']}
existing_tp = {(e[0], e[1]) for e in g['target_pathway_edges']}

# ── Pharmacology-based target mapping ──────────────────────────
# Format: drug → [(target_protein, [list_of_pathways])]
# Pathways must exist in the graph or be added

DRUG_TARGET_MAP = {
    # === NSAIDs (COX inhibitors) ===
    "Flurbiprofen": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain"]),
                     ("COX-1", ["prostaglandin synthesis"])],
    "Meloxicam": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain"]),
                  ("COX-1", ["prostaglandin synthesis"])],
    "Piroxicam": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain"]),
                  ("COX-1", ["prostaglandin synthesis"])],
    "Ketoprofen": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain"]),
                   ("COX-1", ["prostaglandin synthesis"])],
    "Tenoxicam": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain"]),
                  ("COX-1", ["prostaglandin synthesis"])],
    "Lornoxicam": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain"]),
                   ("COX-1", ["prostaglandin synthesis"])],
    "Indomethacin_high_dose": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain"]),
                                ("COX-1", ["prostaglandin synthesis"])],

    # === SSRIs (SERT) ===
    "Escitalopram": [("SERT", ["serotonin reuptake", "serotonergic signaling", "mood"])],
    "Sertraline": [("SERT", ["serotonin reuptake", "serotonergic signaling", "mood"]),
                   ("5-HT", ["serotonergic signaling"])],
    "Fluoxetine": [("SERT", ["serotonin reuptake", "serotonergic signaling", "mood"]),
                   ("5-HT", ["serotonergic signaling"])],

    # === SNRIs & NDRIs ===
    "Atomoxetine": [("NET", ["norepinephrine reuptake", "noradrenergic signaling"])],
    # NET = norepinephrine transporter

    # === Antipsychotics (D2, 5-HT2) ===
    "Olanzapine": [("5-HT2A", ["serotonergic signaling"]),
                   ("D2", ["dopaminergic signaling"])],
    "Quetiapine": [("5-HT2A", ["serotonergic signaling"]),
                   ("D2", ["dopaminergic signaling"]),
                   ("H1", ["histaminergic signaling", "allergy"])],
    "Risperidone": [("5-HT2A", ["serotonergic signaling"]),
                    ("D2", ["dopaminergic signaling"])],
    "Haloperidol": [("D2", ["dopaminergic signaling"])],

    # === Antihistamines ===
    "Cetirizine": [("H1", ["histaminergic signaling", "allergy", "inflammation"])],
    "Loratadine": [("H1", ["histaminergic signaling", "allergy", "inflammation"])],
    "Fexofenadine": [("H1", ["histaminergic signaling", "allergy", "inflammation"])],
    "Diphenhydramine": [("H1", ["histaminergic signaling", "allergy"])],

    # === Proton Pump Inhibitors ===
    "Esomeprazole": [("H+/K+ ATPase", ["gastric acid secretion"])],
    "Lansoprazole": [("H+/K+ ATPase", ["gastric acid secretion"])],
    "Omeprazole": [("H+/K+ ATPase", ["gastric acid secretion"])],

    # === H2 Blockers ===
    "Famotidine": [("H2", ["gastric acid"])],

    # === Leukotriene ===
    "Montelukast": [("CysLT1", ["leukotriene synthesis", "inflammation", "allergy"])],

    # === DPP-4 Inhibitors ===
    "Sitagliptin": [("DPP4", ["insulin signaling", "glucose uptake", "inflammation"])],

    # === Sulfonylureas ===
    "Glipizide": [("KATP", ["insulin signaling", "glucose uptake"])],
    "Glyburide": [("KATP", ["insulin signaling", "glucose uptake"])],

    # === Opioids ===
    "Methadone": [("μ-opioid", ["analgesia", "opioid signaling", "pain signaling"]),
                  ("NMDA", ["glutamatergic signaling", "pain"])],
    "Sufentanil": [("μ-opioid", ["analgesia", "opioid signaling", "pain signaling"])],
    "Fentanyl": [("μ-opioid", ["analgesia", "opioid signaling", "pain signaling"])],
    "Buprenorphine": [("μ-opioid", ["analgesia", "opioid signaling", "pain signaling"]),
                      ("κ-opioid", ["analgesia"])],
    "Tapentadol": [("μ-opioid", ["analgesia", "opioid signaling", "pain signaling"]),
                   ("NET", ["norepinephrine reuptake", "noradrenergic signaling"])],

    # === Opioid Antagonists ===
    "Naloxone": [("μ-opioid", ["opioid signaling"])],
    "Naltrexone": [("μ-opioid", ["opioid signaling"]),
                   ("κ-opioid", ["opioid signaling"])],

    # === Calcium Channel ===
    "Pregabalin": [("α2δ", ["calcium channel auxiliary", "pain signaling",
                             "nociception", "neuronal excitability"])],
    # user-requested clinical drug

    # === COX-2 Selective ===
    "Celecoxib": [("COX-2", ["prostaglandin synthesis", "inflammation", "pain",
                              "anti-inflammatory"])],
    # user-requested clinical drug

    # === Calcitonin ===
    "Salmon_calcitonin": [("CALCR", ["bone metabolism", "osteoclast inhibition",
                                      "bone resorption", "calcium sensing"])],
    # user-requested clinical drug

    # === Muscle Relaxants ===
    "Cyclobenzaprine": [("5-HT2A", ["serotonergic signaling", "spasticity"]),
                        ("H1", ["histaminergic signaling"])],
    "Carisoprodol": [("GABA-A", ["GABAergic signaling", "inhibitory neurotransmission"])],
    "Methocarbamol": [("VGSC", ["sodium channel", "muscle contraction"])],
    "Orphenadrine": [("H1", ["histaminergic signaling"]),
                     ("NMDA", ["glutamatergic signaling"])],
    "Tizanidine": [("α2-AR", ["noradrenergic signaling", "sympathetic", "spasticity"])],

    # === Antiepileptics ===
    "Levetiracetam": [("SV2A", ["synaptic vesicle", "neurotransmitter release"])],
    "Sodium valproate": [("HDAC", ["epigenetics", "gene regulation"]),
                         ("VGSC", ["sodium channel", "neuronal excitability"]),
                         ("GABA", ["GABAergic signaling"])],
    "Phenobarbital": [("GABA-A", ["GABAergic signaling", "inhibitory neurotransmission"])],

    # === Stimulants ===
    "Methylphenidate": [("DAT", ["dopamine reuptake", "dopaminergic signaling"]),
                        ("NET", ["norepinephrine reuptake", "noradrenergic signaling"])],
    "Modafinil": [("DAT", ["dopamine reuptake", "dopaminergic signaling"])],
    "Caffeine": [("A1", ["purine metabolism"]),
                 ("A2A", ["purine metabolism", "dopaminergic signaling"]),
                 ("PDE", ["purine metabolism"])],

    # === Smoking Cessation ===
    "Nicotine": [("nAChR", ["cholinergic signaling", "dopaminergic signaling"])],
    "Varenicline": [("nAChR", ["cholinergic signaling", "dopaminergic signaling"])],

    # === PDE5 ===
    "Sildenafil": [("PDE5", ["vasodilation"])],
    "Tadalafil": [("PDE5", ["vasodilation"])],

    # === Alpha Blockers ===
    "Tamsulosin": [("α1A-AR", ["noradrenergic signaling", "sympathetic"])],
    "Guanfacine": [("α2A-AR", ["noradrenergic signaling", "sympathetic", "blood pressure"])],

    # === Beta Blockers ===
    "Atenolol": [("β1-AR", ["cardiac", "blood pressure"])],
    "Timolol": [("β-AR", ["cardiac", "blood pressure"])],

    # === ACE Inhibitors ===
    "Enalapril": [("ACE", ["blood pressure"])],

    # === Antibiotics ===
    "Doxycycline": [("30S ribosome", ["bacterial protein synthesis"]),
                    ("MMP", ["ECM degradation", "OA cartilage degradation", "cartilage matrix"])],
    "Clarithromycin": [("50S ribosome", ["bacterial protein synthesis"])],
    "Levofloxacin": [("DNA gyrase", ["DNA topology", "bacterial"]),
                     ("Topo IV", ["DNA topology"])],
    "Moxifloxacin": [("DNA gyrase", ["DNA topology", "bacterial"]),
                     ("Topo IV", ["DNA topology"])],
    "Metronidazole": [("DNA", ["DNA damage", "bacterial"])],

    # === Antivirals ===
    "Oseltamivir": [("Neuraminidase", ["immune response"])],
    "Valacyclovir": [("DNA polymerase", ["DNA synthesis"])],
    "Remdesivir": [("RNA polymerase", ["immune response"])],

    # === Antifungals ===
    "Fluconazole": [("CYP51", ["ergosterol synthesis", "antifungal"])],
    "Itraconazole": [("CYP51", ["ergosterol synthesis", "antifungal"]),
                     ("Hedgehog", ["cell growth", "differentiation"])],

    # === Retinoids ===
    "Isotretinoin": [("RAR", ["retinoid signaling", "cell growth", "differentiation"]),
                     ("RXR", ["retinoid signaling"])],
    "Tretinoin": [("RAR", ["retinoid signaling", "cell growth", "differentiation"])],
    "Adapalene": [("RAR-β", ["retinoid signaling"]),
                  ("RAR-γ", ["retinoid signaling", "differentiation"])],

    # === Bronchodilators ===
    "Salbutamol": [("β2-AR", ["bronchodilation"])],
    "Ipratropium": [("M3", ["cholinergic signaling", "bronchodilation"])],
    "Theophylline": [("PDE4", ["bronchodilation", "anti-inflammatory", "purine metabolism"]),
                     ("A1", ["purine metabolism"])],

    # === Others ===
    "Diacerein": [("IL-1", ["inflammatory cytokine pathway", "inflammation",
                             "OA cartilage degradation"]),
                  ("COX-2", ["prostaglandin synthesis"])],
    "Montelukast": [("CysLT1", ["leukotriene synthesis", "inflammation", "allergy"])],
    "Edaravone": [("ROS", ["antioxidant response", "oxidative stress"])],
    "Fasinumab": [("NGF", ["neurotrophin signaling", "pain signaling", "nociception"])],
    "Fulranumab": [("NGF", ["neurotrophin signaling", "pain signaling", "nociception"])],
    "Tanezumab": [("NGF", ["neurotrophin signaling", "pain signaling", "nociception"])],
    "GSK2831781": [("LAG3", ["T cell activation", "immune response", "autoimmune"])],
    "M40403": [("SOD", ["antioxidant response"])],

    # === Immunotherapy ===
    "Pembrolizumab": [("PD-1", ["T cell activation", "immune response", "immunosuppressive"])],
    "Nivolumab": [("PD-1", ["T cell activation", "immune response", "immunosuppressive"])],
    "Ipilimumab": [("CTLA-4", ["T cell activation", "immune response", "immunosuppressive"])],

    # === Hormone-related ===
    "Exemestane": [("Aromatase", ["estrogen synthesis", "steroidogenesis"])],
    "Goserelin": [("GnRH", ["steroidogenesis"])],
    "Leuprolide": [("GnRH", ["steroidogenesis"])],
    "Finasteride": [("5α-reductase", ["steroidogenesis"])],
    "Teriparatide": [("PTH1R", ["bone metabolism", "osteoblast differentiation",
                                 "Wnt/β-catenin"])],
    "Corticosteroid_systemic_longterm": [("GR", ["glucocorticoid signaling",
                                                  "immunosuppressive", "anti-inflammatory"])],

    # === GI ===
    "Bisacodyl": [("CFTR", ["ion transport"])],
    "Domperidone": [("D2", ["dopaminergic signaling", "emesis"])],
    "Metoclopramide": [("D2", ["dopaminergic signaling", "emesis"]),
                       ("5-HT3", ["serotonergic signaling", "emesis"]),
                       ("5-HT4", ["serotonergic signaling"])],
    "Loperamide": [("μ-opioid", ["opioid signaling"])],
    "Senna": [("CFTR", ["ion transport"])],

    # === Other CNS ===
    "Lithium": [("GSK3β", ["Wnt/β-catenin", "cell survival", "neurotrophin signaling"]),
                ("IMPase", ["phosphoinositide signaling"])],
    "Esketamine": [("NMDA", ["glutamatergic signaling", "pain", "neurotrophin signaling"])],
    "Ketamine_anesthesia": [("NMDA", ["glutamatergic signaling", "pain", "analgesia"])],

    # === Ophthalmics ===
    "Latanoprost": [("FP", ["prostaglandin synthesis"])],
    "Brimonidine": [("α2-AR", ["noradrenergic signaling", "sympathetic"])],

    # === Mixed ===
    "Aluminum hydroxide": [("TLR", ["innate immunity"])],
    "Naltrexone": [("μ-opioid", ["opioid signaling"]),
                   ("κ-opioid", ["opioid signaling"]),
                   ("TLR4", ["innate immunity", "inflammation"])],
}

# ── Build new targets and edges ────────────────────────────────
new_targets = set()
new_dt_edges = []
new_tp_edges = []
drugs_activated = set()
total_new_dt = 0
total_new_tp = 0

for drug, target_pathways in DRUG_TARGET_MAP.items():
    drugs_activated.add(drug)
    for target, pathways in target_pathways:
        new_targets.add(target)
        # Drug→Target edge
        if (drug, target) not in existing_edges:
            new_dt_edges.append([drug, target])
            total_new_dt += 1
        # Target→Pathway edges
        for pw in pathways:
            if (target, pw) not in existing_tp:
                new_tp_edges.append([target, pw])
                total_new_tp += 1

# ── Update graph ───────────────────────────────────────────────
g2 = copy.deepcopy(g)
g2['metadata']['version'] = 'four_layer_graph_full_v3'
g2['metadata']['description'] = (
    f"v2 + {total_new_dt} drug-target edges and {total_new_tp} target-pathway edges "
    f"for {len(drugs_activated)} cold-start drugs"
)
g2['metadata']['cold_start_drugs_activated'] = len(drugs_activated)

# Add new targets
all_targets = set(g['targets']) | new_targets
g2['targets'] = sorted(all_targets)

# Add drug-target edges
dt_set = {(e[0], e[1]) for e in g['drug_target_edges']}
for dt in new_dt_edges:
    dt_set.add((dt[0], dt[1]))
g2['drug_target_edges'] = [[d, t] for d, t in dt_set]

# Add target-pathway edges
tp_set = {(e[0], e[1]) for e in g['target_pathway_edges']}
for tp in new_tp_edges:
    tp_set.add((tp[0], tp[1]))
g2['target_pathway_edges'] = [[t, p] for t, p in tp_set]

# ── Stats ──────────────────────────────────────────────────────
n_drugs_before = len(set(e[0] for e in g['drug_target_edges']))
n_drugs_after = len(set(e[0] for e in g2['drug_target_edges']))
n_targets_before = len(g['targets'])
n_targets_after = len(g2['targets'])
n_dt_before = len(g['drug_target_edges'])
n_dt_after = len(g2['drug_target_edges'])
n_tp_before = len(g['target_pathway_edges'])
n_tp_after = len(g2['target_pathway_edges'])

# ── Save ───────────────────────────────────────────────────────
out_path = DATA / "four_layer_graph_full_v3.json"
with open(out_path, "w") as f:
    json.dump(g2, f, indent=2)

print("=" * 60)
print("  Cold-Start Drug Activation — Complete")
print("=" * 60)
print(f"\n  Drugs with target edges: {n_drugs_before} → {n_drugs_after}")
print(f"  Targets: {n_targets_before} → {n_targets_after}")
print(f"  Drug→Target edges: {n_dt_before} → {n_dt_after} (+{n_dt_after-n_dt_before})")
print(f"  Target→Pathway edges: {n_tp_before} → {n_tp_after} (+{n_tp_after-n_tp_before})")
print(f"\n  New targets added:")
for t in sorted(new_targets):
    n_paths = sum(1 for e in new_tp_edges if e[0] == t)
    print(f"    {t:<25s} → {n_paths} pathways")
print(f"\n  Output: {out_path}")
print(f"  Size: {out_path.stat().st_size:,} bytes")
