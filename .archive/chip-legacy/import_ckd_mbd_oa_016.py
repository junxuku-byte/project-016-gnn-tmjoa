#!/usr/bin/env python3
"""
Update LabKG with CKD-MBD → OA connection knowledge.
Based on deep literature learning: FGF23-Wnt axis bridges CKD-MBD and OA.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store

PROJECT_ID = "project-016"


def nid(name: str) -> str:
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '').replace(',', '')[:80]}"


def import_ckd_mbd_oa():
    store = get_graph_store()
    print("="*70)
    print("Importing CKD-MBD → OA connection knowledge")
    print("="*70)

    # ============================================================
    # 1. NEW CONCEPTS from deep literature learning
    # ============================================================
    new_concepts = {
        # Core CKD-MBD molecules
        "fgf23_receptor": {"category": "protein", "name": "FGFR1 (FGF23 receptor)"},
        "klotho": {"category": "protein", "name": "Klotho (FGF23 co-receptor)"},
        "sost": {"category": "protein", "name": "Sclerostin (SOST)"},
        "dkk1": {"category": "protein", "name": "Dickkopf-1"},
        "wnt3a": {"category": "protein", "name": "Wnt3a"},
        "wnt5a": {"category": "protein", "name": "Wnt5a"},
        "lrp5": {"category": "protein", "name": "LRP5"},
        "lrp6": {"category": "protein", "name": "LRP6"},
        "beta_catenin": {"category": "protein", "name": "β-catenin"},
        "sox9": {"category": "protein", "name": "SOX9 (chondrocyte master regulator)"},
        "col10a1": {"category": "protein", "name": "Col10a1 (hypertrophic marker)"},
        "x_linked_hypophosphatemia": {"category": "disease", "name": "X-linked hypophosphatemia (XLH)"},
        "dmp1": {"category": "gene", "name": "DMP1 (dentin matrix protein 1)"},
        "hmwtg": {"category": "model", "name": "HMWTg mouse model"},
        "bgj398": {"category": "drug", "name": "BGJ398 (FGFR inhibitor)"},
        "fgf23_neutralizing_antibody": {"category": "drug", "name": "FGF23 neutralizing antibody"},
        "romosozumab": {"category": "drug", "name": "Romosozumab (anti-sclerostin)"},
        "cinacalcet": {"category": "drug", "name": "Cinacalcet (calcimimetic)"},
        "sevelamer": {"category": "drug", "name": "Sevelamer (phosphate binder)"},
        "lanthanum_carbonate": {"category": "drug", "name": "Lanthanum carbonate"},
        "calcitriol": {"category": "drug", "name": "Calcitriol (active Vit D)"},
        "paricalcitol": {"category": "drug", "name": "Paricalcitol (VDRA)"},
        "high_turnover_bone_disease": {"category": "disease", "name": "High-turnover renal osteodystrophy"},
        "low_turnover_bone_disease": {"category": "disease", "name": "Low-turnover renal osteodystrophy"},
        "adynamic_bone_disease": {"category": "disease", "name": "Adynamic bone disease"},
        "vascular_calcification": {"category": "pathology", "name": "Vascular calcification"},
        "valvular_calcification": {"category": "pathology", "name": "Valvular calcification"},
        "renal_osteodystrophy": {"category": "disease", "name": "Renal osteodystrophy"},
        "kdigo_guideline": {"category": "guideline", "name": "KDIGO CKD-MBD guideline"},
        "egfr": {"category": "biomarker", "name": "eGFR"},
        "ckd_epi": {"category": "method", "name": "CKD-EPI equation"},
        "stage_3a_ckd": {"category": "disease", "name": "CKD Stage 3a (eGFR 45-59)"},
        "stage_3b_ckd": {"category": "disease", "name": "CKD Stage 3b (eGFR 30-44)"},
        "stage_4_ckd": {"category": "disease", "name": "CKD Stage 4 (eGFR 15-29)"},
        "stage_5_ckd": {"category": "disease", "name": "CKD Stage 5 (ESRD)"},
        "dialysis": {"category": "treatment", "name": "Dialysis"},
        "kidney_transplant": {"category": "treatment", "name": "Kidney transplant"},
        "fracture_risk": {"category": "outcome", "name": "Fracture risk"},
        "hip_fracture": {"category": "outcome", "name": "Hip fracture"},
        "extraskeletal_calcification": {"category": "pathology", "name": "Extraskeletal calcification"},
        "coronary_calcification": {"category": "pathology", "name": "Coronary artery calcification"},
        "abdominal_aortic_calcification": {"category": "pathology", "name": "Abdominal aortic calcification"},
        "pulse_wave_velocity": {"category": "biomarker", "name": "Pulse wave velocity"},
        "arterial_stiffness": {"category": "pathology", "name": "Arterial stiffness"},
        "fetuin_a": {"category": "protein", "name": "Fetuin-A (calcification inhibitor)"},
        "matrix_gla_protein": {"category": "protein", "name": "Matrix Gla protein (MGP)"},
        "oxidative_ldl": {"category": "biomarker", "name": "Oxidized LDL"},
        "hyperphosphatemia": {"category": "disease", "name": "Hyperphosphatemia"},
        "hypocalcemia": {"category": "disease", "name": "Hypocalcemia"},
        "secondary_hyperparathyroidism": {"category": "disease", "name": "Secondary hyperparathyroidism"},
        "tertiary_hyperparathyroidism": {"category": "disease", "name": "Tertiary hyperparathyroidism"},
        "parathyroidectomy": {"category": "surgery", "name": "Parathyroidectomy"},
        "calciphylaxis": {"category": "disease", "name": "Calciphylaxis"},
        "nsaid_nephrotoxicity": {"category": "pathology", "name": "NSAID nephrotoxicity"},
        "contrast_induced_nephropathy": {"category": "pathology", "name": "Contrast-induced nephropathy"},
        "hyperuricemia": {"category": "disease", "name": "Hyperuricemia"},
        "urate_crystal": {"category": "pathology", "name": "Urate crystal deposition"},
        "gout": {"category": "disease", "name": "Gout"},
        "gouty_arthritis": {"category": "disease", "name": "Gouty arthritis"},
    }

    added_nodes = 0
    for key, data in new_concepts.items():
        node_id = nid(key)
        if node_id not in store.graph.nodes:
            store.graph.add_node(
                node_id,
                node_type="concept",
                name=data["name"],
                category=data["category"],
                project_ids=[PROJECT_ID],
                source="project-016_ckd_mbd_oa",
            )
            added_nodes += 1
    print(f"[1] Added {added_nodes} CKD-MBD/OA concept nodes")

    # ============================================================
    # 2. CORE MECHANISTIC EDGES: FGF23-Wnt-OA axis
    # ============================================================
    edges = [
        # FGF23 signaling details
        ("fgf23", "fgf23_receptor", "binds", "FGF23 binds FGFR1 in cartilage (Klotho-independent)"),
        ("fgf23_receptor", "klotho", "klotho_independent", "In cartilage, FGF23 signals without Klotho"),
        ("fgf23_receptor", "beta_catenin", "activates", "FGFR1 activates Wnt/β-catenin"),
        ("fgf23", "sost", "downregulates", "FGF23 reduces Sost expression"),
        ("fgf23", "dkk1", "downregulates", "FGF23 reduces Dkk1 expression"),
        ("sost", "lrp5", "inhibits", "Sost inhibits LRP5"),
        ("sost", "lrp6", "inhibits", "Sost inhibits LRP6"),
        ("dkk1", "lrp5", "inhibits", "Dkk1 inhibits LRP5/6"),
        ("dkk1", "lrp6", "inhibits", "Dkk1 inhibits LRP5/6"),
        ("lrp5", "beta_catenin", "promotes", "LRP5 promotes β-catenin stabilization"),
        ("lrp6", "beta_catenin", "promotes", "LRP6 promotes β-catenin stabilization"),
        ("beta_catenin", "mmp13", "upregulates", "β-catenin induces MMP13 transcription"),
        ("beta_catenin", "adamts5", "upregulates", "β-catenin induces ADAMTS5 transcription"),
        ("beta_catenin", "col10a1", "upregulates", "β-catenin induces Col10a1 (hypertrophy)"),
        ("beta_catenin", "sox9", "downregulates", "β-catenin suppresses SOX9"),
        ("sox9", "cartilage", "maintains", "SOX9 maintains chondrocyte phenotype"),
        ("sox9", "col2a1", "promotes", "SOX9 promotes Col2a1 (anabolic)"),
        ("il_1_beta", "fgf23", "induces", "IL-1β dose-dependently induces FGF23 in chondrocytes"),
        ("t_nf", "fgf23", "induces", "TNF-α induces FGF23"),
        ("il_6", "fgf23", "induces", "IL-6 induces FGF23"),
        ("il_1_beta", "sox9", "suppresses", "IL-1β suppresses SOX9"),

        # CKD-MBD core axis
        ("ckd", "egfr", "reduces", "CKD reduces eGFR"),
        ("stage_3a_ckd", "fgf23", "elevates", "FGF23 rises early in CKD 3a"),
        ("stage_3b_ckd", "fgf23", "elevates", "FGF23 rises further"),
        ("stage_4_ckd", "fgf23", "elevates", "FGF23 markedly elevated"),
        ("stage_5_ckd", "fgf23", "markedly_elevates", "FGF23 extremely high"),
        ("ckd", "phosphorus", "retains", "Phosphate retention in CKD"),
        ("phosphorus", "fgf23", "induces", "Hyperphosphatemia induces FGF23"),
        ("fgf23", "phosphorus", "reduces", "FGF23 promotes phosphaturia"),
        ("fgf23", "calcitriol", "reduces", "FGF23 suppresses 1α-hydroxylase → 1,25(OH)2D↓"),
        ("calcitriol", "calcium", "increases", "Active Vit D increases Ca absorption"),
        ("calcitriol", "pth", "suppresses", "Active Vit D suppresses PTH"),
        ("vitamin_d_deficiency", "calcitriol", "reduces", "Vit D deficiency reduces calcitriol synthesis"),
        ("calcium", "pth", "suppresses", "High Ca suppresses PTH"),
        ("phosphorus", "pth", "induces", "High P induces PTH"),
        ("pth", "bone_resorption", "promotes", "PTH promotes osteoclast activity"),
        ("pth", "bone_formation", "dual_role", "Intermittent PTH promotes formation"),
        ("pth", "vitamin_d", "required_for", "PTH requires Vit D for bone effects"),
        ("secondary_hyperparathyroidism", "high_turnover_bone_disease", "causes", "SHPT causes fibrotic bone disease"),
        ("pth", "secondary_hyperparathyroidism", "elevates", "Elevated PTH defines SHPT"),
        ("cinacalcet", "pth", "suppresses", "Calcimimetic suppresses PTH secretion"),
        ("sevelamer", "phosphorus", "binds", "Sevelamer binds intestinal phosphate"),
        ("lanthanum_carbonate", "phosphorus", "binds", "Lanthanum binds intestinal phosphate"),
        ("calcitriol", "active_vitamin_d", "is", "Calcitriol is active Vit D"),
        ("paricalcitol", "vdr", "activates", "Paricalcitol activates VDR with less hypercalcemia"),

        # CKD-MBD bone types
        ("high_turnover_bone_disease", "pth", "driven_by", "Driven by elevated PTH"),
        ("low_turnover_bone_disease", "pth", "suppressed", "PTH oversuppression or aluminum toxicity"),
        ("low_turnover_bone_disease", "bone_mineralization", "impairs", "Impaired mineralization → osteomalacia"),
        ("adynamic_bone_disease", "low_turnover_bone_disease", "is_a", "ABD is low-turnover variant"),
        ("adynamic_bone_disease", "bone_formation", "severely_suppresses", "Severely suppressed bone formation"),

        # Vascular calcification
        ("ckd", "vascular_calcification", "promotes", "CKD promotes vascular calcification"),
        ("hyperphosphatemia", "vascular_calcification", "promotes", "High P promotes VSMC osteogenic transdifferentiation"),
        ("calcium", "vascular_calcification", "promotes", "High Ca promotes calcification"),
        ("fetuin_a", "vascular_calcification", "inhibits", "Fetuin-A inhibits calcium crystal formation"),
        ("matrix_gla_protein", "vascular_calcification", "inhibits", "MGP is potent calcification inhibitor (Vit K-dependent)"),
        ("vitamin_k", "matrix_gla_protein", "activates", "Vit K carboxylates MGP"),
        ("vascular_calcification", "arterial_stiffness", "increases", "Calcification increases PWV"),
        ("arterial_stiffness", "pulse_wave_velocity", "measured_by", "PWV measures arterial stiffness"),
        ("vascular_calcification", "coronary_calcification", "includes", "Coronary calcification is subset"),
        ("vascular_calcification", "abdominal_aortic_calcification", "includes", "AAC is subset"),
        ("vascular_calcification", "valvular_calcification", "includes", "Valvular calcification"),

        # CKD-MBD clinical outcomes
        ("ckd_mbd", "fracture_risk", "increases", "CKD-MBD increases fracture risk 3-5x"),
        ("ckd_mbd", "hip_fracture", "increases", "Hip fracture risk markedly increased"),
        ("ckd_mbd", "extraskeletal_calcification", "promotes", "Extra-skeletal calcification"),
        ("ckd_mbd", "cardiovascular_disease", "promotes", "CV mortality is leading cause of death in CKD"),
        ("fracture_risk", "condyle", "includes", "Condyle fracture risk in severe CKD"),

        # Treatment: Denosumab in CKD
        ("denosumab", "pth", "may_increase", "Denosumab may transiently increase PTH"),
        ("denosumab", "calcium", "may_reduce", "Denosumab may reduce serum Ca"),
        ("denosumab", "ckd", "used_in", "Used in CKD patients with osteoporosis"),
        ("bisphosphonate", "ckd", "caution_in", "Caution in eGFR<30"),
        ("bisphosphonate", "low_turnover_bone_disease", "may_worsen", "May worsen ABD in CKD"),
        ("romosozumab", "sost", "blocks", "Anti-sclerostin antibody"),
        ("romosozumab", "bone_formation", "increases", "Increases bone formation"),
        ("romosozumab", "vascular_calcification", "may_increase", "Concerns about CV safety in CKD"),

        # FGF23 in XLH and OA
        ("x_linked_hypophosphatemia", "fgf23", "markedly_elevates", "PHEX mutation → FGF23↑↑"),
        ("x_linked_hypophosphatemia", "oa", "causes", "XLH causes spontaneous degenerative joint disease"),
        ("x_linked_hypophosphatemia", "osteomalacia", "causes", "Rickets/osteomalacia in children"),
        ("dmp1", "fgf23", "regulates", "DMP1 regulates FGF23"),
        ("dmp1", "oa", "model_of", "Dmp1 KO is OA model"),
        ("fgf23_neutralizing_antibody", "fgf23", "blocks", "Blocks FGF23 signaling"),
        ("fgf23_neutralizing_antibody", "xlh", "treats", "Approved for XLH (burosumab)"),
        ("fgf23_neutralizing_antibody", "oa", "rescues", "Rescues OA in mouse models"),
        ("bgj398", "fgf23_receptor", "inhibits", "FGFR tyrosine kinase inhibitor"),
        ("bgj398", "oa", "rescues", "Rescues OA changes in subchondral bone and cartilage"),

        # KDIGO
        ("kdigo_guideline", "ckd_mbd", "defines", "Defines CKD-MBD diagnosis and management"),
        ("kdigo_guideline", "renal_osteodystrophy", "distinguishes", "Distinguishes renal osteodystrophy (bone biopsy) from CKD-MBD"),

        # CKD-OA comorbidity
        ("ckd", "oa", "coexists", "CKD and OA commonly coexist"),
        ("ckd", "osteoarthritis", "increases_risk", "CKD increases OA risk"),
        ("oa", "nsaid", "treats_with", "OA treated with NSAIDs"),
        ("nsaid", "nsaid_nephrotoxicity", "causes", "NSAIDs cause nephrotoxicity"),
        ("nsaid_nephrotoxicity", "ckd", "may_induce", "May induce/accelerate CKD"),
        ("contrast_induced_nephropathy", "ckd", "risk_factor", "CKD is major risk factor"),

        # Uric acid / gout
        ("ckd", "hyperuricemia", "causes", "CKD reduces uric acid excretion"),
        ("hyperuricemia", "urate_crystal", "forms", "Supersaturation → crystal formation"),
        ("urate_crystal", "gout", "causes", "Crystal deposition causes gout"),
        ("urate_crystal", "gouty_arthritis", "causes", "Acute inflammatory arthritis"),
        ("gout", "oa", "may_overlap", "Chronic gout may overlap with OA"),
        ("hyperuricemia", "oa", "associated_with", "Associated with OA severity"),

        # Sclerostin paradox
        ("sost", "oa", "decreased_in", "Sost decreased in OA cartilage → Wnt disinhibition"),
        ("sost", "ckd", "elevated_in", "Sost elevated in CKD serum → protective?"),
        ("sost", "bone_formation", "inhibits", "Sost inhibits osteoblast Wnt signaling"),

        # Denosumab in CKD-OA
        ("denosumab", "fgf23", "indirectly_reduces", "By improving bone health, may reduce FGF23 drive"),
        ("denosumab", "ckd_mbd", "used_in", "Approved for CKD osteoporosis"),

        # Estrogen-CKD-OA axis
        ("estrogen_deficiency", "ckd", "may_accelerate", "Estrogen loss may accelerate CKD progression"),
        ("estrogen_deficiency", "fgf23", "may_increase", "Estrogen regulates FGF23"),

        # Inflammation convergence
        ("inflammation", "fgf23", "induces", "Systemic inflammation induces FGF23"),
        ("inflammation", "ckd", "promotes", "Inflammation promotes CKD progression"),
        ("inflammation", "oa", "promotes", "Inflammation promotes OA progression"),
        ("chip", "fgf23", "may_induce", "CHIP inflammation may induce FGF23"),
        ("chip", "ckd", "may_coexist", "CHIP and CKD share aging/inflammation risk"),
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
    print(f"[2] Added {added_edges} CKD-MBD/OA mechanistic edges")

    # ============================================================
    # 3. Bridge to TMJOA
    # ============================================================
    bridges = [
        ("ckd", "tmjoa", "increases_risk_of", "CKD increases TMJOA risk via bone metabolism"),
        ("ckd_mbd", "tmjoa", "predisposes_to", "CKD-MBD predisposes to TMJOA"),
        ("fgf23", "tmjoa", "may_accelerate", "High FGF23 may accelerate TMJ cartilage degradation"),
        ("renal_osteodystrophy", "tmjoa", "predisposes_to", "Renal osteodystrophy weakens condylar bone"),
        ("vascular_calcification", "tmjoa", "may_associate", "Shared aging mechanisms"),
        ("x_linked_hypophosphatemia", "tmjoa", "may_cause", "XLH causes degenerative joint disease including TMJ"),
        ("nsaid_nephrotoxicity", "tmjoa", "irony", "TMJOA treated with NSAIDs → may worsen CKD"),
    ]

    added_bridges = 0
    for s_name, t_name, rel, ev in bridges:
        s_id = nid(s_name)
        t_id = nid(t_name)
        if s_id in store.graph.nodes and t_id in store.graph.nodes:
            if not store.graph.has_edge(s_id, t_id):
                store.graph.add_edge(
                    s_id, t_id,
                    edge_type=rel,
                    project_ids=[PROJECT_ID],
                    evidence=ev,
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
    print("✅ CKD-MBD → OA knowledge import complete!")
    print(f"{'='*70}")
    print(f"LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016: {len(project_nodes)} nodes ({len(papers)} papers)")

    return True


if __name__ == "__main__":
    success = import_ckd_mbd_oa()
    sys.exit(0 if success else 1)
