#!/usr/bin/env python3
"""
Build 'Common Bone Metabolism Abnormalities → TMJOA' parallel network in LabKG.
Captures routinely ignored bone metabolism issues that coexist with TMJOA.

NOTE (2026-05-18): CHIP-related edges removed upon project pivot to GNN drug repositioning.
Preserved as commented-out code block below for potential future reference.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store

PROJECT_ID = "project-016"


def nid(name: str) -> str:
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '').replace(',', '')[:80]}"


def import_bone_metabolism_network():
    store = get_graph_store()
    print("="*70)
    print("Building Common Bone Metabolism → TMJOA Parallel Network")
    print("="*70)

    # ============================================================
    # 1. CONCEPT NODES: Screening targets + diseases + biomarkers
    # ============================================================
    concepts = {
        # Screening panel
        "vitamin_d": {"category": "biomarker", "name": "Vitamin D"},
        "25_oh_d": {"category": "biomarker", "name": "25(OH)D"},
        "vitamin_d_deficiency": {"category": "disease", "name": "Vitamin D deficiency"},
        "vdr": {"category": "protein", "name": "Vitamin D receptor"},
        "calcium": {"category": "biomarker", "name": "Serum calcium"},
        "phosphorus": {"category": "biomarker", "name": "Serum phosphorus"},
        "alp": {"category": "biomarker", "name": "Alkaline phosphatase"},
        "pth": {"category": "biomarker", "name": "Parathyroid hormone"},
        "tsh": {"category": "biomarker", "name": "TSH"},
        "ft4": {"category": "biomarker", "name": "Free T4"},
        "egfr": {"category": "biomarker", "name": "eGFR"},
        "creatinine": {"category": "biomarker", "name": "Creatinine"},
        "hba1c": {"category": "biomarker", "name": "HbA1c"},
        "fgf23": {"category": "biomarker", "name": "FGF23"},
        "bmp2": {"category": "protein", "name": "BMP2"},
        "bmp7": {"category": "protein", "name": "BMP7"},
        "serum_iron": {"category": "biomarker", "name": "Serum iron"},
        "ferritin": {"category": "biomarker", "name": "Ferritin"},
        "transferrin_saturation": {"category": "biomarker", "name": "Transferrin saturation"},
        "hfe_gene": {"category": "gene", "name": "HFE gene"},
        "homocysteine": {"category": "biomarker", "name": "Homocysteine"},
        "mthfr_gene": {"category": "gene", "name": "MTHFR gene"},
        "magnesium": {"category": "biomarker", "name": "Magnesium"},
        "p1np": {"category": "biomarker", "name": "P1NP"},
        "ctx": {"category": "biomarker", "name": "CTX"},
        "dxa": {"category": "method", "name": "DXA bone densitometry"},
        "bone_density": {"category": "outcome", "name": "Bone mineral density"},
        "bone_quality": {"category": "outcome", "name": "Bone quality"},

        # Diseases / conditions
        "ckd": {"category": "disease", "name": "Chronic kidney disease"},
        "ckd_mbd": {"category": "disease", "name": "CKD-mineral bone disorder"},
        "hypothyroidism": {"category": "disease", "name": "Hypothyroidism"},
        "hyperthyroidism": {"category": "disease", "name": "Hyperthyroidism"},
        "subclinical_hypothyroidism": {"category": "disease", "name": "Subclinical hypothyroidism"},
        "hyperparathyroidism": {"category": "disease", "name": "Hyperparathyroidism"},
        "primary_hyperparathyroidism": {"category": "disease", "name": "Primary hyperparathyroidism"},
        "hypophosphatemia": {"category": "disease", "name": "Hypophosphatemia"},
        "tumor_induced_osteomalacia": {"category": "disease", "name": "Tumor-induced osteomalacia"},
        "xlh": {"category": "disease", "name": "X-linked hypophosphatemia"},
        "hemochromatosis": {"category": "disease", "name": "Hemochromatosis"},
        "iron_overload": {"category": "disease", "name": "Iron overload"},
        "hyperhomocysteinemia": {"category": "disease", "name": "Hyperhomocysteinemia"},
        "glucocorticoid_induced_osteoporosis": {"category": "disease", "name": "Glucocorticoid-induced osteoporosis"},
        "diabetic_bone_disease": {"category": "disease", "name": "Diabetic bone disease"},
        "osteomalacia": {"category": "disease", "name": "Osteomalacia"},
        "osteoporosis": {"category": "disease", "name": "Osteoporosis"},
        "type_2_diabetes": {"category": "disease", "name": "Type 2 diabetes"},
        "ages": {"category": "biomarker", "name": "Advanced glycation end products"},
        "bone_marrow_fat": {"category": "anatomy", "name": "Bone marrow fat"},

        # Drugs / interventions
        "vitamin_d_supplement": {"category": "drug", "name": "Vitamin D supplement"},
        "calcium_supplement": {"category": "drug", "name": "Calcium supplement"},
        "active_vitamin_d": {"category": "drug", "name": "Active vitamin D analog"},
        "phosphate_binder": {"category": "drug", "name": "Phosphate binder"},
        "levothyroxine": {"category": "drug", "name": "Levothyroxine"},
        "methimazole": {"category": "drug", "name": "Methimazole"},
        "parathyroidectomy": {"category": "surgery", "name": "Parathyroidectomy"},
        "chelation_therapy": {"category": "drug", "name": "Iron chelation therapy"},
        "phlebotomy": {"category": "procedure", "name": "Therapeutic phlebotomy"},
        "folate": {"category": "drug", "name": "Folate supplement"},
        "vitamin_b12": {"category": "drug", "name": "Vitamin B12 supplement"},
        "magnesium_supplement": {"category": "drug", "name": "Magnesium supplement"},
        "ppi": {"category": "drug", "name": "Proton pump inhibitor"},
        "aromatase_inhibitor": {"category": "drug", "name": "Aromatase inhibitor"},
        "antiepileptic": {"category": "drug", "name": "Antiepileptic drug"},
        "thiazolidinedione": {"category": "drug", "name": "Thiazolidinedione"},
        "methotrexate": {"category": "drug", "name": "Methotrexate"},
        "bisphosphonate": {"category": "drug", "name": "Bisphosphonate"},

        # Mechanisms
        "collagen_crosslinking": {"category": "process", "name": "Collagen crosslinking"},
        "bone_mineralization": {"category": "process", "name": "Bone mineralization"},
        "bone_resorption": {"category": "process", "name": "Bone resorption"},
        "bone_formation": {"category": "process", "name": "Bone formation"},
        "oxidative_stress": {"category": "process", "name": "Oxidative stress"},
        "mitochondrial_dysfunction": {"category": "process", "name": "Mitochondrial dysfunction"},
        "estrogen_deficiency": {"category": "process", "name": "Estrogen deficiency"},
        "gut_barrier": {"category": "anatomy", "name": "Gut barrier"},
        "tight_junction": {"category": "protein", "name": "Tight junction proteins"},
        "lysyloxidase": {"category": "protein", "name": "Lysyl oxidase"},
        "osteoblast": {"category": "cell_type", "name": "Osteoblast"},
        "osteocyte": {"category": "cell_type", "name": "Osteocyte"},
        "hypertrophic_chondrocyte": {"category": "cell_type", "name": "Hypertrophic chondrocyte"},
        "subchondral_bone": {"category": "anatomy", "name": "Subchondral bone"},
        "condyle": {"category": "anatomy", "name": "Mandibular condyle"},
        "articular_cartilage": {"category": "anatomy", "name": "Articular cartilage"},
        "sclerosis": {"category": "pathology", "name": "Subchondral bone sclerosis"},
        "cyst": {"category": "pathology", "name": "Bone cyst"},
    }

    added_nodes = 0
    for key, data in concepts.items():
        node_id = nid(key)
        if node_id not in store.graph.nodes:
            store.graph.add_node(
                node_id,
                node_type="concept",
                name=data["name"],
                category=data["category"],
                project_ids=[PROJECT_ID],
                source="project-016_bone_metabolism",
            )
            added_nodes += 1
    print(f"[1] Added {added_nodes} bone metabolism concept nodes")

    # ============================================================
    # 2. EVIDENCE EDGES: Bone metabolism → TMJOA mechanisms
    # ============================================================
    edges = [
        # Vitamin D axis
        ("vitamin_d_deficiency", "vitamin_d", "causes", "Low 25(OH)D <20 ng/mL"),
        ("vitamin_d", "vdr", "activates", "Vit D binds VDR in chondrocytes/osteocytes"),
        ("vdr", "bone_mineralization", "promotes", "VDR signaling promotes mineralization"),
        ("vdr", "bone_formation", "promotes", "VDR promotes osteoblast differentiation"),
        ("vitamin_d_deficiency", "bone_mineralization", "impairs", "Deficiency impairs mineralization"),
        ("vitamin_d_deficiency", "bone_density", "reduces", "BMD decreases with deficiency"),
        ("vitamin_d_deficiency", "subchondral_bone", "weakens", "Weakens subchondral bone plate"),
        ("vitamin_d_deficiency", "condyle", "degrades", "Condylar bone quality degrades"),
        ("vitamin_d_deficiency", "tmjoa", "predisposes", "Epidemiological link: deficiency associated with OA severity"),
        ("vitamin_d_supplement", "vitamin_d", "restores", "Supplementation restores 25(OH)D"),
        ("vitamin_d_supplement", "bone_density", "increases", "Increases BMD when deficient"),

        # Calcium / PTH axis
        ("calcium", "bone_mineralization", "required_for", "Essential for hydroxyapatite formation"),
        ("phosphorus", "bone_mineralization", "required_for", "Essential for hydroxyapatite formation"),
        ("pth", "bone_resorption", "promotes", "PTH promotes osteoclast activity"),
        ("pth", "bone_formation", "dual_role", "Intermittent PTH promotes formation, continuous promotes resorption"),
        ("primary_hyperparathyroidism", "pth", "elevates", "PTH autonomous secretion"),
        ("primary_hyperparathyroidism", "calcium", "elevates", "Hypercalcemia"),
        ("primary_hyperparathyroidism", "bone_resorption", "increases", "Osteitis fibrosa cystica"),
        ("primary_hyperparathyroidism", "condyle", "resorbs", "Mandibular condyle bone resorption"),
        ("hypophosphatemia", "phosphorus", "decreases", "Low serum phosphorus"),
        ("hypophosphatemia", "bone_mineralization", "impairs", "Impaired mineralization → osteomalacia"),
        ("xlh", "fgf23", "elevates", "FGF23 mutation → phosphaturia"),
        ("fgf23", "phosphorus", "reduces", "FGF23 promotes renal phosphate wasting"),
        ("fgf23", "bone_mineralization", "impairs", "FGF23 excess impairs mineralization"),

        # Thyroid axis
        ("subclinical_hypothyroidism", "tsh", "elevates", "TSH >4.5 mIU/L with normal FT4"),
        ("hypothyroidism", "bone_formation", "delays", "Delayed bone remodeling"),
        ("hypothyroidism", "bone_density", "may_increase", "Paradoxically may increase BMD due to low turnover"),
        ("hyperthyroidism", "bone_resorption", "accelerates", "High turnover osteoporosis"),
        ("hyperthyroidism", "bone_density", "decreases", "BMD loss"),
        ("hyperthyroidism", "tmjoa", "may_accelerate", "Accelerated condylar bone loss"),
        ("levothyroxine", "tsh", "normalizes", "Restores euthyroid state"),

        # CKD-MBD axis
        ("ckd", "egfr", "reduces", "eGFR <60 mL/min/1.73m2"),
        ("ckd", "phosphorus", "retains", "Phosphate retention"),
        ("ckd", "pth", "elevates", "Secondary hyperparathyroidism"),
        ("ckd", "fgf23", "elevates", "FGF23 early compensatory rise"),
        ("ckd_mbd", "bone_mineralization", "impairs", "Mixed bone disease: high and low turnover"),
        ("ckd_mbd", "vascular_calcification", "promotes", "Soft tissue calcification"),
        ("ckd_mbd", "condyle", "abnormal", "Abnormal condylar bone remodeling"),
        ("phosphate_binder", "phosphorus", "binds", "Reduces intestinal phosphate absorption"),
        ("active_vitamin_d", "pth", "suppresses", "Suppresses secondary hyperparathyroidism"),

        # Glucocorticoid-induced osteoporosis
        ("glucocorticoid_induced_osteoporosis", "osteoblast", "apoptosis", "Glucocorticoids induce osteoblast apoptosis"),
        ("glucocorticoid_induced_osteoporosis", "osteoclast", "prolongs_lifespan", "Prolongs osteoclast lifespan"),
        ("glucocorticoid_induced_osteoporosis", "bone_formation", "suppresses", "Suppresses bone formation"),
        ("glucocorticoid_induced_osteoporosis", "bone_density", "reduces", "Rapid BMD loss"),
        ("glucocorticoid_induced_osteoporosis", "condyle", "weakens", "Weakens condylar bone"),
        ("glucocorticoid_induced_osteoporosis", "tmjoa", "predisposes", "Predisposes to TMJOA progression"),

        # Iron overload
        ("hemochromatosis", "serum_iron", "elevates", "Iron overload"),
        ("hemochromatosis", "ferritin", "elevates", "Elevated ferritin"),
        ("iron_overload", "osteoblast", "dysfunction", "Iron deposition in osteoblasts"),
        ("iron_overload", "bone_density", "decreases", "Associated with osteoporosis"),
        ("iron_overload", "oxidative_stress", "increases", "ROS via Fenton reaction"),
        ("chelation_therapy", "serum_iron", "reduces", "Reduces iron burden"),
        ("phlebotomy", "ferritin", "reduces", "Reduces iron stores"),

        # Homocysteine / MTHFR
        ("hyperhomocysteinemia", "homocysteine", "elevates", "Hcy >15 μmol/L"),
        ("homocysteine", "collagen_crosslinking", "impairs", "Impairs collagen crosslinking"),
        ("homocysteine", "bone_quality", "reduces", "Increases bone fragility"),
        ("mthfr_gene", "homocysteine", "elevates", "C677T polymorphism raises Hcy"),
        ("folate", "homocysteine", "reduces", "Folate remethylates homocysteine"),
        ("vitamin_b12", "homocysteine", "reduces", "B12 cofactor for remethylation"),

        # Magnesium
        ("magnesium", "vitamin_d", "activates", "Mg required for 25-OH → 1,25-OH conversion"),
        ("magnesium_deficiency", "pth", "resistance", "PTH resistance"),
        ("magnesium_deficiency", "bone_mineralization", "impairs", "Impairs mineralization"),
        ("magnesium_supplement", "magnesium", "restores", "Restores Mg levels"),

        # Diabetes bone disease
        ("type_2_diabetes", "hba1c", "elevates", "HbA1c >6.5%"),
        ("type_2_diabetes", "ages", "accumulates", "AGEs accumulate in collagen"),
        ("ages", "collagen_crosslinking", "abnormal", "Non-enzymatic crosslinking → brittle collagen"),
        ("ages", "bone_quality", "reduces", "Reduces bone toughness despite normal BMD"),
        ("ages", "articular_cartilage", "stiffens", "Cartilage stiffness increases"),
        ("diabetic_bone_disease", "bone_density", "may_preserve", "BMD may appear normal"),
        ("diabetic_bone_disease", "bone_quality", "reduces", "But bone quality is poor"),
        ("diabetic_bone_disease", "tmjoa", "predisposes", "Predisposes to TMJOA"),
        ("thiazolidinedione", "bone_marrow_fat", "increases", "Increases marrow adiposity"),
        ("thiazolidinedione", "bone_formation", "suppresses", "Suppresses osteoblastogenesis"),

        # Drug-induced bone loss
        ("ppi", "calcium", "reduces_absorption", "Reduces intestinal calcium absorption"),
        ("ppi", "bone_density", "may_reduce", "Epidemiological association"),
        ("aromatase_inhibitor", "estrogen", "reduces", "Blocks peripheral estrogen synthesis"),
        ("aromatase_inhibitor", "bone_resorption", "increases", "Increases bone resorption"),
        ("antiepileptic", "vitamin_d", "accelerates_catabolism", "Accelerates vit D catabolism"),
        ("antiepileptic", "bone_density", "reduces", "Associated with low BMD"),
        ("methotrexate", "folate", "antagonizes", "Folate antagonism"),
        ("methotrexate", "bone_formation", "suppresses", "Suppresses bone formation"),

        # Common convergence on TMJOA
        ("osteoporosis", "subchondral_bone", "weakens", "Weakened subchondral bone plate"),
        ("osteoporosis", "condyle", "fragile", "Fragile condylar bone"),
        ("osteoporosis", "tmjoa", "predisposes", "Low BMD associated with OA severity"),
        ("osteomalacia", "bone_mineralization", "fails", "Failed mineralization"),
        ("osteomalacia", "condyle", "softens", "Softened condylar bone"),
        ("osteomalacia", "tmjoa", "predisposes", "Pseudofractures in condyle possible"),
        ("bone_marrow_fat", "bone_formation", "suppresses", "Adipocytes suppress osteoblasts"),
        ("bone_marrow_fat", "subchondral_bone", "weakens", "Fatty marrow weakens bone"),

        # Subchondral bone → cartilage crosstalk
        ("subchondral_bone", "articular_cartilage", "supports", "Subchondral bone supports cartilage"),
        ("sclerosis", "articular_cartilage", "damages", "Sclerosis alters mechanical loading → cartilage damage"),
        ("cyst", "articular_cartilage", "damages", "Cyst formation alters biomechanics"),
        ("condyle", "articular_cartilage", "covers", "Condyle covered by articular cartilage"),
        ("condyle", "subchondral_bone", "contains", "Condyle contains subchondral bone"),

        # Bone turnover markers
        ("p1np", "bone_formation", "reflects", "Marker of bone formation"),
        ("ctx", "bone_resorption", "reflects", "Marker of bone resorption"),
        ("p1np", "ctx", "ratio", "P1NP/CTX ratio reflects bone balance"),
    ]

    # NOTE: CHIP-related edges removed 2026-05-18 upon project pivot to GNN drug repositioning.
    # Preserved below as comments for potential future reference:
    # CHIP interaction (confounding / synergy):
    #   ("vitamin_d_deficiency", "chip", "synergizes", "Vit D deficiency + CHIP = worse bone outcomes")
    #   ("estrogen_deficiency", "vitamin_d_deficiency", "synergizes", "Menopause + low Vit D common")
    #   ("estrogen_deficiency", "bone_marrow_fat", "increases", "Estrogen loss increases marrow fat")
    #   ("bone_marrow_fat", "chip", "promotes", "FBM promotes DNMT3A-CHIP advantage (Zioni 2023)")
    #   ("type_2_diabetes", "chip", "may_coexist", "Diabetes and CHIP share aging/inflammation risk")
    #   ("ckd", "chip", "may_coexist", "CKD patients have accelerated aging → higher CHIP prevalence")
    #   ("hyperhomocysteinemia", "chip", "may_coexist", "MTHFR variant may associate with clonal hematopoiesis")
    #   ("oxidative_stress", "chip", "promotes", "ROS promotes HSPC mutations")
    #   ("iron_overload", "oxidative_stress", "increases", "Iron overload increases ROS")
    #   ("iron_overload", "chip", "may_promote", "ROS may promote clonal expansion")
    ]

    added_edges = 0
    for s_name, t_name, rel, ev in edges:
        s_id = nid(s_name)
        t_id = nid(t_name)
        for node_id in [s_id, t_id]:
            if node_id not in store.graph.nodes:
                store.graph.add_node(
                    node_id,
                    node_type="concept",
                    name=node_id.replace("concept:", "").replace("_", " "),
                    category="mechanism",
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
        if not store.graph.has_edge(s_id, t_id):
            store.graph.add_edge(
                s_id, t_id,
                edge_type=rel,
                project_ids=[PROJECT_ID],
                evidence=ev,
            )
            added_edges += 1
    print(f"[2] Added {added_edges} bone metabolism edges")

    # ============================================================
    # 3. Ensure TMJOA connectivity for key screening targets
    # ============================================================
    bridges = [
        ("vitamin_d_deficiency", "tmjoa"),
        ("osteoporosis", "tmjoa"),
        ("osteomalacia", "tmjoa"),
        ("glucocorticoid_induced_osteoporosis", "tmjoa"),
        ("hyperthyroidism", "tmjoa"),
        ("hypothyroidism", "tmjoa"),
        ("ckd_mbd", "tmjoa"),
        ("diabetic_bone_disease", "tmjoa"),
        ("iron_overload", "tmjoa"),
        ("hyperhomocysteinemia", "tmjoa"),
        ("bone_marrow_fat", "tmjoa"),
        ("sclerosis", "tmjoa"),
        ("cyst", "tmjoa"),
    ]

    added_bridges = 0
    for s_name, t_name in bridges:
        s_id = nid(s_name)
        t_id = nid(t_name)
        if s_id in store.graph.nodes and t_id in store.graph.nodes:
            if not store.graph.has_edge(s_id, t_id):
                store.graph.add_edge(
                    s_id, t_id,
                    edge_type="predisposes_to",
                    project_ids=[PROJECT_ID],
                    evidence="Clinical association: bone metabolism abnormality increases TMJOA risk",
                )
                added_bridges += 1
    print(f"[3] Added {added_bridges} TMJOA bridge edges")

    # Save
    store.save()

    # Summary
    project_nodes = [n for n, d in store.graph.nodes(data=True)
                     if PROJECT_ID in d.get("project_ids", [])]
    papers = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "paper"]
    print(f"\n{'='*70}")
    print("✅ Bone Metabolism Network complete!")
    print(f"{'='*70}")
    print(f"LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016: {len(project_nodes)} nodes ({len(papers)} papers)")
    print(f"Bone metabolism concepts: {len(concepts)}")

    return True


if __name__ == "__main__":
    success = import_bone_metabolism_network()
    sys.exit(0 if success else 1)
