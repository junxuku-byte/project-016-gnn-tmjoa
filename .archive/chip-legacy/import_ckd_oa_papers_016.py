#!/usr/bin/env python3
"""
Import FGF23-Wnt-OA (PMID 29718273) + CKD-OA interactions (PMID 37562858)
into LabKG for Project-016.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store

PROJECT_ID = "project-016"


def nid(name: str) -> str:
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '').replace(',', '')[:80]}"


def import_ckd_oa_papers():
    store = get_graph_store()
    print("="*70)
    print("Importing FGF23-Wnt-OA + CKD-OA interaction papers")
    print("="*70)

    # ============================================================
    # 1. PAPER NODES
    # ============================================================
    papers = {
        "paper:29718273": {
            "title": "FGF23 regulates Wnt/β-catenin signaling-mediated osteoarthritis in mice overexpressing high molecular weight FGF2",
            "authors": "Burt PM, Xiao L, Hurley MM",
            "year": 2018,
            "journal": "Endocrinology",
            "doi": "10.1210/en.2018-00184",
            "pmid": "29718273",
            "key_finding": "FGF23 is a catabolic regulator of Wnt/β-catenin-mediated OA cartilage destruction. FGF23 neutralizing antibody partially rescues OA phenotype in vivo. FGF23 signals through FGFR1 independent of Klotho in cartilage.",
            "category": "mechanism",
            "citation_count": 0,
        },
        "paper:37562858": {
            "title": "Disease-modifying interactions between chronic kidney disease and osteoarthritis: a new comorbid mouse model",
            "authors": "Julovi SM, Dao A, Trinh K, et al.",
            "year": 2023,
            "journal": "RMD Open",
            "doi": "10.1136/rmdopen-2023-003109",
            "pmid": "37562858",
            "key_finding": "World-first CKD-OA comorbid model (5/6Nx+DMM). Bidirectional disease-modifying effects: CKD reduces OA severity via subchondral bone resorption (pseudo-protection), but DMM alone induces renal fibrosis. Synovial MMP-13 increased in CKD.",
            "category": "preclinical_model",
            "citation_count": 0,
        },
    }

    added_papers = 0
    for pid, data in papers.items():
        if pid not in store.graph.nodes:
            store.graph.add_node(
                pid,
                node_type="paper",
                title=data["title"],
                authors=data["authors"],
                year=data["year"],
                journal=data["journal"],
                doi=data["doi"],
                pmid=data["pmid"],
                key_finding=data["key_finding"],
                category=data["category"],
                citation_count=data["citation_count"],
                project_ids=[PROJECT_ID],
                source="project-016_ckd_oa_deep_read",
            )
            added_papers += 1
    print(f"[1] Added {added_papers} paper nodes")

    # ============================================================
    # 2. NEW CONCEPTS from both papers
    # ============================================================
    new_concepts = {
        # From PMID 29718273
        "iwr_1": {"category": "drug", "name": "IWR-1 (Wnt inhibitor, Axin2 stabilizer)"},
        "fgf23_neutralizing_antibody": {"category": "drug", "name": "FGF23 neutralizing antibody"},
        "burosumab": {"category": "drug", "name": "Burosumab (anti-FGF23 mAb, KRN23)"},
        "hmwtg": {"category": "model", "name": "HMWTg mouse (high molecular weight FGF2 overexpression)"},
        "lmwtg": {"category": "model", "name": "LMWTg mouse (low molecular weight FGF2 overexpression)"},
        "wisp1": {"category": "protein", "name": "WISP1 (Wnt-induced signaling protein 1)"},
        "wnt7b": {"category": "protein", "name": "Wnt7b (canonical Wnt ligand)"},
        "wnt16": {"category": "protein", "name": "Wnt16"},
        "axin2": {"category": "protein", "name": "Axin2"},
        "lef1": {"category": "protein", "name": "LEF1 (lymphoid enhancer-binding factor 1)"},
        "p_gsk3b": {"category": "protein", "name": "Phospho-GSK3β (inactive form)"},
        "nuclear_beta_catenin": {"category": "protein", "name": "Nuclear β-catenin (active)"},
        "col10a1": {"category": "protein", "name": "Col10a1 (hypertrophic marker)"},
        "fgfr1c": {"category": "protein", "name": "FGFR1c isoform"},
        "fgfr3": {"category": "protein", "name": "FGFR3"},
        "fgfr3c": {"category": "protein", "name": "FGFR3c isoform"},
        "fgf18": {"category": "protein", "name": "FGF18 (anabolic in cartilage)"},
        "nkx3_2": {"category": "protein", "name": "Nkx3.2 (chondrocyte maturation regulator)"},
        "sostdc1": {"category": "protein", "name": "Sostdc1"},
        "ihh": {"category": "protein", "name": "Indian hedgehog (Ihh)"},
        "col1a1": {"category": "protein", "name": "Col1a1"},
        "mmp9": {"category": "protein", "name": "MMP9"},
        "mmp19": {"category": "protein", "name": "MMP19"},
        "alcian_blue": {"category": "method", "name": "Alcian blue staining (GAG)"},
        "alkaline_phosphatase_staining": {"category": "method", "name": "Alkaline phosphatase staining (mineralization)"},
        "safranin_o": {"category": "method", "name": "Safranin-O staining"},
        "dmoad": {"category": "concept", "name": "Disease-modifying osteoarthritis drug (DMOAD)"},

        # From PMID 37562858
        "5_6_nephrectomy": {"category": "model", "name": "5/6 nephrectomy (CKD model)"},
        "dmm": {"category": "model", "name": "DMM (destabilization of medial meniscus, OA model)"},
        "trap": {"category": "method", "name": "TRAP staining (tartrate-resistant acid phosphatase, osteoclast marker)"},
        "renal_fibrosis": {"category": "pathology", "name": "Renal fibrosis"},
        "tubulointerstitial_lesion": {"category": "pathology", "name": "Tubulointerstitial lesion (TIL)"},
        "proteinuria": {"category": "biomarker", "name": "Proteinuria"},
        "blood_urea_nitrogen": {"category": "biomarker", "name": "Blood urea nitrogen (BUN)"},
        "serum_creatinine": {"category": "biomarker", "name": "Serum creatinine"},
        "urine_creatinine": {"category": "biomarker", "name": "Urine creatinine"},
        "bone_volume_fraction": {"category": "biomarker", "name": "BV/TV (bone volume fraction)"},
        "trabecular_thickness": {"category": "biomarker", "name": "Tb.Th (trabecular thickness)"},
        "trabecular_number": {"category": "biomarker", "name": "Tb.N (trabecular number)"},
        "trabecular_separation": {"category": "biomarker", "name": "Tb.Sp (trabecular separation)"},
        "bone_mineral_density": {"category": "biomarker", "name": "BMD (bone mineral density)"},
        "bone_surface_volume": {"category": "biomarker", "name": "BS/BV (bone surface/bone volume)"},
        "chondrocyte_hypertrophy": {"category": "pathology", "name": "Chondrocyte hypertrophy"},
        "proteoglycan_loss": {"category": "pathology", "name": "Proteoglycan loss"},
        "structural_damage": {"category": "pathology", "name": "Structural cartilage damage"},
        "marginal_osteophyte": {"category": "pathology", "name": "Marginal osteophyte"},
        "synovitis": {"category": "pathology", "name": "Synovitis"},
        "osteoclast_activity": {"category": "pathology", "name": "Osteoclast activity"},
        "subchondral_bone_sclerosis": {"category": "pathology", "name": "Subchondral bone sclerosis"},
        "bone_marrow_loss": {"category": "pathology", "name": "Bone marrow loss"},
        "vascular_invasion": {"category": "pathology", "name": "Vascular invasion"},
        "synovial_pannus": {"category": "pathology", "name": "Synovial pannus"},
        "cortical_bone_erosion": {"category": "pathology", "name": "Cortical bone erosion"},
        "synoviocyte_hyperplasia": {"category": "pathology", "name": "Synoviocyte hyperplasia"},
        "intrasynovial_exudate": {"category": "pathology", "name": "Intrasynovial exudate"},
        "loading_stress": {"category": "concept", "name": "Mechanical loading stress on cartilage"},
        "bidirectional_interaction": {"category": "concept", "name": "Bidirectional disease interaction"},
        "comorbid_model": {"category": "concept", "name": "Comorbid disease model"},
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
                source="project-016_ckd_oa_papers",
            )
            added_nodes += 1
    print(f"[2] Added {added_nodes} concept nodes")

    # ============================================================
    # 3. MECHANISTIC EDGES from PMID 29718273
    # ============================================================
    edges_29718273 = [
        # Paper→findings
        ("paper:29718273", "fgf23", "reports", "FGF23 is catabolic regulator of Wnt-mediated OA"),
        ("paper:29718273", "fgf23_receptor", "reports", "FGFR1 mediates FGF23 effects in cartilage"),
        ("paper:29718273", "klotho", "reports", "Klotho-independent FGF23 signaling in cartilage"),
        ("paper:29718273", "wnt3a", "reports", "Wnt pathway components altered in HMWTg"),
        ("paper:29718273", "beta_catenin", "reports", "Nuclear β-catenin only in HMWTg cartilage"),
        ("paper:29718273", "mmp13", "reports", "MMP13 increased in HMWTg, rescued by FGF23Ab"),
        ("paper:29718273", "mmp9", "reports", "MMP9 increased in HMWTg, rescued by FGF23Ab"),
        ("paper:29718273", "adamts5", "reports", "ADAMTS5 increased in HMWTg chondrocytes"),
        ("paper:29718273", "col10a1", "reports", "ColX marker of hypertrophy, reduced by FGF23Ab"),
        ("paper:29718273", "il_1_beta", "reports", "IL-1β increased in HMWTg chondrocytes"),
        ("paper:29718273", "sox9", "reports", "SOX9 decreased in LMWTg (protected) chondrocytes"),
        ("paper:29718273", "sost", "reports", "Sost absent in HMWTg subchondral bone"),
        ("paper:29718273", "dkk1", "reports", "Dkk1 decreased in HMWTg joints"),
        ("paper:29718273", "lrp5", "reports", "LRP5 strongly expressed in HMWTg cartilage"),
        ("paper:29718273", "lrp6", "reports", "p-LRP6 decreased in HMWTg cartilage"),
        ("paper:29718273", "wnt7b", "reports", "Wnt7b increased in HMWTg (exclusively canonical)"),
        ("paper:29718273", "wnt5a", "reports", "Wnt5a increased in HMWTg"),
        ("paper:29718273", "dmoad", "reports", "FGF23Ab is potential DMOAD (first in vivo evidence)"),
        ("paper:29718273", "fgf23_neutralizing_antibody", "reports", "Partial rescue of OA phenotype in vivo"),
        ("paper:29718273", "x_linked_hypophosphatemia", "reports", "HMWTg phenocopies XLH"),

        # Mechanism edges (refined from paper)
        ("fgf23", "fgf23_receptor", "binds", "FGF23 binds FGFR1 in cartilage"),
        ("fgf23_receptor", "klotho", "klotho_independent", "No Klotho required in cartilage"),
        ("fgf23_receptor", "beta_catenin", "activates_via", "FGFR1 activates canonical Wnt/β-catenin"),
        ("fgf23", "sost", "downregulates", "Sost expression reduced in HMWTg"),
        ("fgf23", "dkk1", "downregulates", "Dkk1 expression reduced"),
        ("fgf23", "lrp5", "upregulates", "LRP5 increased (promotes cartilage destruction)"),
        ("fgf23", "lrp6", "downregulates", "Phospho-LRP6 decreased"),
        ("fgf23", "wnt7b", "upregulates", "Wnt7b increased in chondrocytes"),
        ("fgf23", "wnt5a", "upregulates", "Wnt5a increased"),
        ("fgf23", "wisp1", "upregulates", "WISP1 increased"),
        ("fgf23", "adamts5", "upregulates", "ADAMTS5 increased"),
        ("fgf23", "mmp13", "upregulates", "MMP13 increased (in vivo)"),
        ("fgf23", "mmp9", "upregulates", "MMP9 increased (in vivo)"),
        ("fgf23", "il_1_beta", "upregulates", "IL-1β increased"),
        ("fgf23", "col10a1", "upregulates", "ColX hypertrophic marker increased"),
        ("fgf23", "p_gsk3b", "increases", "Phospho-GSK3β increased (inactive)"),
        ("fgf23", "lef1", "upregulates", "LEF1 increased"),
        ("fgf23", "axin2", "upregulates", "Axin2 increased"),
        ("fgf23", "nuclear_beta_catenin", "promotes", "Nuclear β-catenin present"),
        ("beta_catenin", "fgf23", "upregulates", "β-catenin activates FGF23 promoter (positive feedback)"),
        ("wnt7b", "beta_catenin", "activates", "Wnt7b exclusively canonical → β-catenin"),
        ("wnt5a", "lrp5", "activates_canonical", "Wnt5a activates canonical pathway via LRP5"),
        ("lrp5", "beta_catenin", "promotes", "LRP5 essential for Wnt-induced cartilage destruction"),
        ("sost", "lrp5", "inhibits", "Sost inhibits LRP5"),
        ("dkk1", "lrp5", "inhibits", "Dkk1 inhibits LRP5/6"),
        ("p_gsk3b", "beta_catenin", "stabilizes", "Inactive GSK3β prevents β-catenin degradation"),
        ("beta_catenin", "adamts5", "upregulates", "β-catenin transcriptionally upregulates ADAMTS5"),
        ("beta_catenin", "mmp13", "upregulates", "β-catenin upregulates MMP13"),
        ("beta_catenin", "col10a1", "upregulates", "β-catenin upregulates ColX"),
        ("il_1_beta", "fgf23", "induces", "IL-1β dose-dependently induces FGF23 (from prior studies)"),

        # Drug effects
        ("iwr_1", "axin2", "stabilizes", "IWR-1 stabilizes Axin2 → β-catenin degradation"),
        ("iwr_1", "beta_catenin", "destabilizes", "IWR-1 leads to β-catenin degradation"),
        ("iwr_1", "fgf23", "reduces", "IWR-1 reduces FGF23 expression in HMWTg"),
        ("iwr_1", "adamts5", "reduces", "IWR-1 rescues ADAMTS5 in HMWTg"),
        ("iwr_1", "wnt7b", "reduces", "IWR-1 reduces Wnt7b"),
        ("fgf23_neutralizing_antibody", "fgf23", "blocks", "Neutralizes FGF23 signaling"),
        ("fgf23_neutralizing_antibody", "subchondral_bone", "rescues", "Restores subchondral bone microarchitecture"),
        ("fgf23_neutralizing_antibody", "cartilage", "rescues", "Restores cartilage thickness"),
        ("fgf23_neutralizing_antibody", "mmp13", "reduces", "Eliminates MMP13 protein expression"),
        ("fgf23_neutralizing_antibody", "mmp9", "reduces", "Eliminates MMP9 protein expression"),
        ("fgf23_neutralizing_antibody", "col10a1", "reduces", "Reduces ColX expression"),
        ("burosumab", "fgf23", "blocks", "Burosumab is FGF23 neutralizing antibody (approved for XLH)"),
        ("burosumab", "xlh", "treats", "Approved for X-linked hypophosphatemia"),
        ("burosumab", "oa", "potential", "Potential DMOAD based on 29718273"),

        # Model comparisons
        ("hmwtg", "fgf23", "overexpresses", "HMWTg overexpresses FGF23 in bone and serum"),
        ("hmwtg", "oa", "develops", "HMWTg develops spontaneous OA at 2 months"),
        ("lmwtg", "fgf23", "decreases", "LMWTg has decreased FGF23"),
        ("lmwtg", "oa", "protected_from", "LMWTg protected from OA"),
        ("x_linked_hypophosphatemia", "fgf23", "markedly_elevates", "PHEX mutation → FGF23↑↑"),
        ("x_linked_hypophosphatemia", "oa", "causes", "XLH causes degenerative joint disease"),
    ]

    # ============================================================
    # 4. MECHANISTIC EDGES from PMID 37562858
    # ============================================================
    edges_37562858 = [
        # Paper→findings
        ("paper:37562858", "5_6_nephrectomy", "reports", "5/6Nx successfully models CKD in B6 mice"),
        ("paper:37562858", "dmm", "reports", "DMM induces OA pathology"),
        ("paper:37562858", "renal_fibrosis", "reports", "DMM alone induces renal fibrosis (unexpected)"),
        ("paper:37562858", "proteinuria", "reports", "DMM alone causes proteinuria"),
        ("paper:37562858", "trap", "reports", "CKD increases subchondral osteoclast activity"),
        ("paper:37562858", "mmp13", "reports", "Synovial MMP-13 increased in CKD; cartilage MMP-13 unchanged"),
        ("paper:37562858", "bidirectional_interaction", "reports", "OA and CKD have bidirectional disease-modifying effects"),
        ("paper:37562858", "subchondral_bone", "reports", "CKD alters subchondral bone microarchitecture"),
        ("paper:37562858", "bone_mineral_density", "reports", "BMD significantly reduced in CKD"),

        # CKD model effects
        ("5_6_nephrectomy", "ckd", "models", "5/6 nephrectomy is standard CKD model"),
        ("5_6_nephrectomy", "egfr", "reduces", "Removes 70-80% kidney mass"),
        ("5_6_nephrectomy", "renal_fibrosis", "induces", "Induces tubulointerstitial fibrosis"),
        ("5_6_nephrectomy", "blood_urea_nitrogen", "elevates", "Serum urea elevated"),
        ("5_6_nephrectomy", "serum_creatinine", "elevates", "Serum creatinine elevated"),
        ("5_6_nephrectomy", "proteinuria", "causes", "Proteinuria in CKD model"),
        ("5_6_nephrectomy", "alkaline_phosphatase", "elevates", "Serum ALP elevated"),
        ("5_6_nephrectomy", "phosphorus", "elevates", "Serum phosphate elevated"),
        ("5_6_nephrectomy", "calcium", "elevates", "Serum calcium elevated"),
        ("5_6_nephrectomy", "bone_volume_fraction", "reduces", "BV/TV decreased"),
        ("5_6_nephrectomy", "trabecular_thickness", "reduces", "Tb.Th decreased"),
        ("5_6_nephrectomy", "trabecular_number", "reduces", "Tb.N decreased"),
        ("5_6_nephrectomy", "bone_mineral_density", "reduces", "BMD decreased"),
        ("5_6_nephrectomy", "osteoclast_activity", "increases", "TRAP+ osteoclasts increased"),

        # CKD alone effects on joint
        ("ckd", "cartilage", "no_direct_damage", "CKD alone does not cause cartilage pathology"),
        ("ckd", "synovitis", "no_change", "No synovitis in CKD alone"),
        ("ckd", "osteophyte", "reduces", "CKD reduces osteophyte formation"),
        ("ckd", "subchondral_bone_sclerosis", "reduces", "CKD reduces subchondral sclerosis (bone resorption dominates)"),
        ("ckd", "synovial_mmp13", "increases", "Synovial MMP-13 increased"),
        ("ckd", "cartilage_mmp13", "no_change", "Cartilage MMP-13 unchanged"),

        # DMM alone effects
        ("dmm", "oa", "induces", "DMM induces knee OA pathology"),
        ("dmm", "cartilage", "damages", "Cartilage proteoglycan loss, structural damage"),
        ("dmm", "subchondral_bone_sclerosis", "increases", "Subchondral bone sclerosis increased"),
        ("dmm", "marginal_osteophyte", "increases", "Marginal osteophyte formation"),
        ("dmm", "synovitis", "increases", "Synovitis increased"),
        ("dmm", "mmp13", "increases", "Cartilage and synovial MMP-13 increased"),

        # Bidirectional: DMM → Kidney
        ("dmm", "renal_fibrosis", "induces", "DMM alone induces renal fibrosis (without NSAIDs)"),
        ("dmm", "proteinuria", "induces", "DMM causes proteinuria"),
        ("dmm", "blood_urea_nitrogen", "increases", "DMM increases serum urea when combined with 5/6Nx"),
        ("oa", "ckd", "may_induce", "OA may directly damage kidney"),

        # CKD+OA interaction
        ("ckd", "oa", "reduces_severity", "CKD reduces DMM-induced OA severity"),
        ("ckd", "cartilage", "pseudo_protects", "Reduced cartilage damage (load-mediated, not biological)"),
        ("ckd", "marginal_osteophyte", "reduces", "Reduced osteophyte development"),
        ("ckd", "bone_volume_fraction", "reduces", "Reduced BV/TV persists in comorbid model"),
        ("ckd", "bone_mineral_density", "reduces", "BMD lower than DMM alone"),
        ("ckd", "osteoclast_activity", "inhibited_by_oa", "DMM inhibits 5/6Nx-induced osteoclasts (bone formation phase)"),
        ("oa", "phosphorus", "reduces", "DMM reduces CKD-induced serum phosphate"),
        ("oa", "calcium", "reduces", "DMM reduces CKD-induced serum calcium"),
        ("loading_stress", "cartilage", "protects", "Reduced loading stress from bone resorption protects cartilage"),

        # Bridge to TMJOA
        ("ckd", "tmjoa", "increases_risk", "CKD increases TMJOA risk via bone metabolism + FGF23"),
        ("ckd_mbd", "tmjoa", "predisposes", "CKD-MBD predisposes to TMJOA"),
        ("dmm", "tmjoa", "analogous", "DMM is knee OA model; TMJOA has similar biomechanical triggers"),
        ("fgf23", "tmjoa", "may_accelerate", "High FGF23 may accelerate TMJ cartilage degradation"),
        ("5_6_nephrectomy", "tmjoa", "models_risk", "5/6Nx model demonstrates CKD→OA risk mechanisms"),
    ]

    all_edges = edges_29718273 + edges_37562858

    added_edges = 0
    for s_name, t_name, rel, ev in all_edges:
        s_id = nid(s_name) if not s_name.startswith("paper:") else s_name
        t_id = nid(t_name) if not t_name.startswith("paper:") else t_name

        # Ensure nodes exist
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
    print(f"[3] Added {added_edges} mechanistic edges")

    # ============================================================
    # 5. Cross-paper connections
    # ============================================================
    cross_edges = [
        ("paper:29718273", "paper:37562858", "complements", "FGF23-Wnt mechanism (29718273) + CKD-OA comorbid model (37562858) together explain CKD-MBD→OA pathway"),
        ("fgf23", "5_6_nephrectomy", "elevated_in", "5/6Nx model should have elevated FGF23 (not measured in 37562858)"),
        ("fgf23_neutralizing_antibody", "ckd", "treats", "Could treat both CKD-MBD and OA simultaneously"),
    ]

    for s_name, t_name, rel, ev in cross_edges:
        s_id = nid(s_name) if not s_name.startswith("paper:") else s_name
        t_id = nid(t_name) if not t_name.startswith("paper:") else t_name
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

    # Save
    store.save()

    # Summary
    project_nodes = [n for n, d in store.graph.nodes(data=True)
                     if PROJECT_ID in d.get("project_ids", [])]
    papers_nodes = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "paper"]
    print(f"\n{'='*70}")
    print("✅ FGF23-Wnt-OA + CKD-OA papers import complete!")
    print(f"{'='*70}")
    print(f"LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016: {len(project_nodes)} nodes ({len(papers_nodes)} papers)")

    return True


if __name__ == "__main__":
    success = import_ckd_oa_papers()
    sys.exit(0 if success else 1)
