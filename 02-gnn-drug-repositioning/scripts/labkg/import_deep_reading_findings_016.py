#!/usr/bin/env python3
"""
Import deep reading findings from 2 fulltext papers + 8 abstract-only papers into LabKG.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store

PROJECT_ID = "project-016"


def nid(name: str) -> str:
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '').replace(',', '')[:80]}"


def import_reading_findings():
    store = get_graph_store()
    print("=" * 70)
    print("Importing Deep Reading Findings into LabKG")
    print("=" * 70)

    # ============================================================
    # 1. NEW CONCEPT NODES (from readings)
    # ============================================================
    new_concepts = {
        # From Denosumab paper
        "trinetx": {"name": "TriNetX Network", "category": "database"},
        "propensity_score_matching": {"name": "Propensity score matching (PSM)", "category": "method"},
        "real_world_evidence": {"name": "Real-world evidence (RWE)", "category": "concept"},
        "electronic_health_record": {"name": "Electronic health record (EHR)", "category": "database"},
        "active_comparator": {"name": "Active comparator design", "category": "method"},
        "new_user_design": {"name": "New-user design", "category": "method"},
        "cumulative_incidence": {"name": "Cumulative incidence probability", "category": "outcome"},
        "subgroup_analysis": {"name": "Subgroup analysis", "category": "method"},
        "sensitivity_analysis": {"name": "Sensitivity analysis", "category": "method"},
        
        # From Riluzole paper
        "scrna_seq": {"name": "Single-cell RNA sequencing (scRNA-seq)", "category": "method"},
        "drug_repositioning": {"name": "Drug repositioning/repurposing", "category": "concept"},
        "network_proximity": {"name": "Network proximity-based drug repositioning", "category": "method"},
        "mendelian_randomization": {"name": "Mendelian randomization (MR)", "category": "method"},
        "zebrafish_model": {"name": "Zebrafish OA model", "category": "model"},
        "molecular_docking": {"name": "Molecular docking", "category": "method"},
        "autodock": {"name": "AutoDock", "category": "software"},
        "tcc": {"name": "Triclocarban (TCC)", "category": "chemical"},
        "slc7a11": {"name": "SLC7A11 (xCT)", "category": "protein"},
        "ctss": {"name": "CTSS (cathepsin S)", "category": "protein"},
        "nos1": {"name": "NOS1 (neuronal nitric oxide synthase)", "category": "protein"},
        "pi4ka": {"name": "PI4KA (phosphatidylinositol 4-kinase alpha)", "category": "protein"},
        "plcg2": {"name": "PLCG2 (phospholipase C gamma 2)", "category": "protein"},
        "qpct": {"name": "QPCT (glutaminyl-peptide cyclotransferase)", "category": "protein"},
        "cxcl12": {"name": "CXCL12 (SDF-1)", "category": "cytokine"},
        "cxcr4": {"name": "CXCR4", "category": "protein"},
        "retn": {"name": "RETN (resistin)", "category": "protein"},
        "cap1": {"name": "CAP1 (adenylyl cyclase-associated protein 1)", "category": "protein"},
        "mapk_erk": {"name": "MAPK/ERK pathway", "category": "pathway"},
        "binding_energy": {"name": "Binding energy (kcal/mol)", "category": "biomarker"},
        
        # From other abstracts
        "oarsi": {"name": "OARSI (Osteoarthritis Research Society International)", "category": "guideline"},
        "target_trial_emulation": {"name": "Target trial emulation", "category": "method"},
        "nested_case_control": {"name": "Nested case-control study", "category": "method"},
        "claims_data": {"name": "Health insurance claims data", "category": "database"},
        "finngen": {"name": "FinnGen database", "category": "database"},
        "ieu_opengwas": {"name": "IEU OpenGWAS", "category": "database"},
        "inverse_variance_weighting": {"name": "Inverse variance weighting (IVW)", "category": "method"},
        "chitosan_hyaluronate": {"name": "Chitosan-hyaluronate gel", "category": "material"},
        "fmoc_ddt": {"name": "Fmoc-DDT", "category": "chemical"},
        "fos": {"name": "Fos (formaldehyde oligomeric surfactant)", "category": "chemical"},
        "cell_free_fat": {"name": "Cell-free fat extract", "category": "biological"},
        "il_38": {"name": "IL-38", "category": "cytokine"},
        "recombinant_il_38": {"name": "Recombinant IL-38", "category": "drug"},
        "sgk1": {"name": "SGK1 (serum/glucocorticoid-regulated kinase 1)", "category": "protein"},
        "foxo1": {"name": "FoxO1", "category": "protein"},
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
                source="project-016_deep_reading",
            )
            added_nodes += 1
    print(f"[1] Added {added_nodes} new concept nodes")

    # ============================================================
    # 2. PAPER NODES (with fulltext reading)
    # ============================================================
    papers = {
        "paper:41520765": {
            "title": "Association between denosumab use and risk of osteoarthritis among adults with osteoporosis in a real-world cohort",
            "authors": "Zhu Z, Huang JY, Wang W, et al.",
            "year": 2026,
            "journal": "Osteoarthritis and Cartilage",
            "pmid": "41520765",
            "doi": "10.1016/j.joca.2026.01.001",
            "key_finding": "Denosumab vs bisphosphonate: HR 0.87 (0.84-0.91) for knee OA, HR 0.96 (0.94-0.98) for any OA, n=59,157 PSM pairs, 10-year follow-up. Asian subgroup HR 0.73 (0.63-0.84). First real-world evidence for denosumab's protective effect on knee OA.",
            "category": "epidemiology",
        },
        "paper:41993107": {
            "title": "Single-Cell Transcriptomics Reveals Riluzole as an Osteoarthritis Candidate Drug via OB-NE Signaling Modulation and CTSS/NOS1 Inhibition",
            "authors": "Liu K, Li JL, Chen Y, et al.",
            "year": 2026,
            "journal": "Drug Design, Development and Therapy",
            "pmid": "41993107",
            "doi": "10.2147/DDDT.S544803",
            "key_finding": "Riluzole identified via scRNA-seq + network proximity drug repositioning. MR: SLC7A11 expression inverse to OA (OR=0.84, P=0.027). Zebrafish OA model: riluzole reduces joint degeneration. Molecular docking: CTSS (-7 kcal/mol) and NOS1 (-6.9 kcal/mol). Downregulates MAPK/ERK.",
            "category": "drug_repositioning",
        },
        "paper:42130404": {
            "title": "Single Intra-Articular Anakinra (IL-1Ra) Versus Betamethasone in Rabbit Post-Traumatic Knee Osteoarthritis",
            "authors": "Tepedelenlioğlu HE, et al.",
            "year": 2026,
            "journal": "Cell Biochemistry and Function",
            "pmid": "42130404",
            "key_finding": "Single IA anakinra comparable to betamethasone in rabbit PTOA. IL-8 suppression. Chondrocyte viability preserved.",
            "category": "preclinical",
        },
        "paper:42048677": {
            "title": "Analgesic, anti-inflammatory and joint protective effects of ACD137, a selective negative allosteric modulator of TrkA",
            "authors": "Forsell P, et al.",
            "year": 2026,
            "journal": "Scandinavian Journal of Pain",
            "pmid": "42048677",
            "doi": "10.1515/sjpain-2026-0007",
            "key_finding": "ACD137: selective TrkA NAM. Analgesic + anti-inflammatory + joint protective. Selective over p75NTR (safety advantage vs tanezumab).",
            "category": "preclinical",
        },
        "paper:42097689": {
            "title": "Oral bisphosphonates and risk of incident osteoarthritis in individuals with osteoporosis: a target trial emulation",
            "authors": "Hatano M, et al.",
            "year": 2026,
            "journal": "RMD Open",
            "pmid": "42097689",
            "key_finding": "Oral bisphosphonates may reduce OA risk in osteoporosis (Japanese claims data, n=10,844, target trial emulation).",
            "category": "epidemiology",
        },
        "paper:42043713": {
            "title": "Metformin for knee osteoarthritis in overweight and obese adults: a systematic review and meta-analysis",
            "authors": "Amerneni KC, et al.",
            "year": 2026,
            "journal": "Inflammopharmacology",
            "pmid": "42043713",
            "doi": "10.1007/s10787-026-02218-1",
            "key_finding": "Metformin shows efficacy, safety, and anti-inflammatory potential in knee OA (meta-analysis).",
            "category": "meta_analysis",
        },
        "paper:41991265": {
            "title": "Metformin use and the risk of incident osteoarthritis among individuals with diabetes: a register-based nested case-control study",
            "authors": "Dell'Isola A, et al.",
            "year": 2026,
            "journal": "BMJ Open",
            "pmid": "41991265",
            "key_finding": "Metformin reduces incident OA risk in diabetics (Swedish register, n>1.4M), dose-dependent protective effect.",
            "category": "epidemiology",
        },
        "paper:40707728": {
            "title": "The use of statins can reduce the risk of osteoarthritis and osteoporosis",
            "authors": "Zeng X, et al.",
            "year": 2025,
            "journal": "Inflammopharmacology",
            "pmid": "40707728",
            "doi": "10.1007/s10787-025-01864-1",
            "key_finding": "Mendelian randomization: statins causally reduce OA and osteoporosis risk (FinnGen + UKB). IVW primary method.",
            "category": "genetics",
        },
        "paper:42001864": {
            "title": "Recombinant IL-38 Alleviates Temporomandibular Joint Synovial Inflammation",
            "authors": "Not specified",
            "year": 2026,
            "pmid": "42001864",
            "key_finding": "Recombinant IL-38 reduces TMJ synovial inflammation.",
            "category": "preclinical",
        },
        "paper:41716349": {
            "title": "Fmoc-DDT@Fos hydrogel mitigates temporomandibular joint osteoarthritis",
            "authors": "Not specified",
            "year": 2026,
            "pmid": "41716349",
            "key_finding": "Hydrogel drug delivery system (Fmoc-DDT@Fos) for TMJOA mitigation.",
            "category": "drug_delivery",
        },
        "paper:41470005": {
            "title": "SGK1 triggers cartilage degradation in TMJOA via FoxO1/autophagy",
            "authors": "Not specified",
            "year": 2025,
            "pmid": "41470005",
            "key_finding": "SGK1→FoxO1/autophagy axis drives TMJOA cartilage degradation. Therapeutic target.",
            "category": "mechanism",
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
                pmid=data["pmid"],
                doi=data.get("doi", ""),
                key_finding=data["key_finding"],
                category=data["category"],
                project_ids=[PROJECT_ID],
                source="project-016_deep_reading",
            )
            added_papers += 1
    print(f"[2] Added {added_papers} paper nodes")

    # ============================================================
    # 3. EVIDENCE EDGES (paper -> findings)
    # ============================================================
    evidence_edges = [
        # Denosumab paper evidence
        ("paper:41520765", "denosumab", "reports", "Denosumab reduces knee OA risk vs bisphosphonate (HR 0.87)"),
        ("paper:41520765", "bisphosphonate", "compares_to", "Active comparator"),
        ("paper:41520765", "oa", "reports", "Any OA HR 0.96 (0.94-0.98)"),
        ("paper:41520765", "knee_oa", "reports", "Knee OA HR 0.87 (0.84-0.91), primary finding"),
        ("paper:41520765", "hip_oa", "reports", "Hip OA HR 0.98 (0.94-1.03), not significant"),
        ("paper:41520765", "real_world_evidence", "uses", "TriNetX EHR database"),
        ("paper:41520765", "propensity_score_matching", "uses", "1:1 greedy nearest neighbor, caliper 0.1"),
        ("paper:41520765", "trinetx", "uses", "US Collaborative Network, 66 healthcare organizations"),
        ("paper:41520765", "cumulative_incidence", "reports", "10-year incidence probability"),
        ("paper:41520765", "subgroup_analysis", "reports", "Asian HR 0.73, female HR 0.87, ≥65y HR 0.88"),
        ("paper:41520765", "osteoporosis", "studies", "All participants had osteoporosis diagnosis"),
        ("paper:41520765", "denosumab", "reduces_risk_of", "Knee OA: 19.2% vs 21.4% at 10 years"),
        
        # Riluzole paper evidence
        ("paper:41993107", "riluzole", "identifies", "Candidate OA drug via scRNA-seq + network proximity"),
        ("paper:41993107", "scrna_seq", "uses", "Human femoral head: 3 control + 3 OA"),
        ("paper:41993107", "drug_repositioning", "uses", "Network proximity-based approach"),
        ("paper:41993107", "mendelian_randomization", "uses", "SLC7A11 expression → OA risk (OR=0.84, P=0.027)"),
        ("paper:41993107", "slc7a11", "reports", "Inverse association with OA risk (MR validated)"),
        ("paper:41993107", "zebrafish_model", "uses", "TCC-induced OA model, 21-day treatment"),
        ("paper:41993107", "molecular_docking", "uses", "AutoDock, CTSS -7 kcal/mol, NOS1 -6.9 kcal/mol"),
        ("paper:41993107", "ctss", "inhibits", "Riluzole binds CTSS strongly (-7 kcal/mol)"),
        ("paper:41993107", "nos1", "inhibits", "Riluzole binds NOS1 strongly (-6.9 kcal/mol)"),
        ("paper:41993107", "mapk_erk", "downregulates", "Riluzole suppresses MAPK/ERK pathway"),
        ("paper:41993107", "cxcl12", "reports", "OB-NE signaling: CXCL12-CXCR4 pathway enhanced in OA"),
        ("paper:41993107", "retn", "reports", "OB-NE signaling: RETN-CAP1 pathway enhanced in OA"),
        ("paper:41993107", "osteoimmunology", "concept", "OB-immune cell interactions in OA"),
        ("paper:41993107", "riluzole", "reduces", "Joint degeneration in zebrafish OA model"),
        
        # Other papers
        ("paper:42130404", "anakinra", "reports", "Single IA comparable to betamethasone in rabbit PTOA"),
        ("paper:42130404", "il_8", "suppresses", "IL-8 immunopositivity reduced"),
        ("paper:42048677", "acd137", "reports", "TrkA NAM: analgesic + anti-inflammatory + joint protective"),
        ("paper:42048677", "trka", "selectively_inhibits", "Negative allosteric modulator, selective over p75NTR"),
        ("paper:42097689", "alendronate", "may_reduce_risk", "Oral bisphosphonates and OA risk (Japanese claims)"),
        ("paper:42043713", "metformin", "meta_analysis", "Efficacy + safety + anti-inflammatory in knee OA"),
        ("paper:41991265", "metformin", "reduces_risk", "Incident OA in diabetics, dose-dependent"),
        ("paper:40707728", "statin", "causally_reduces", "MR: FinnGen + UKB, IVW method"),
        ("paper:40707728", "mendelian_randomization", "uses", "Causal inference for statin → OA"),
        ("paper:42001864", "il_38", "alleviates", "TMJ synovial inflammation"),
        ("paper:41716349", "fmoc_ddt", "mitigates", "TMJOA via hydrogel delivery"),
        ("paper:41470005", "sgk1", "triggers", "Cartilage degradation via FoxO1/autophagy"),
    ]

    added_evidence = 0
    for s_name, t_name, rel, ev in evidence_edges:
        s_id = s_name if s_name.startswith("paper:") else nid(s_name)
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
            added_evidence += 1
    print(f"[3] Added {added_evidence} evidence edges")

    # ============================================================
    # 4. CROSS-PAPER SYNTHESIS EDGES
    # ============================================================
    synthesis_edges = [
        # Denosumab + Riluzole: both OA protective via different mechanisms
        ("denosumab", "riluzole", "complementary_mechanism", "Denosumab: bone-mediated; Riluzole: OB-NE/CTSS/NOS1"),
        ("denosumab", "ctss", "may_affect", "RANKL inhibition may indirectly affect osteoclast-derived CTSS"),
        ("denosumab", "mapk_erk", "may_affect", "Subchondral bone remodeling affects cartilage signaling"),
        
        # Metformin + Denosumab: both protective, different populations
        ("metformin", "denosumab", "complementary_population", "Metformin: diabetics; Denosumab: osteoporosis"),
        ("metformin", "riluzole", "both_repositioned", "Both identified via non-traditional approaches for OA"),
        
        # Statin + Denosumab: MR vs RWE
        ("statin", "denosumab", "different_evidence_level", "Statin: MR (genetic); Denosumab: RWE (clinical)"),
        ("statin", "metformin", "both_epidemiological", "Both have population-level protective evidence"),
        
        # Anakinra + IL-38: both anti-inflammatory cytokine approaches
        ("anakinra", "il_38", "same_class", "Both IL-1 family modulators for TMJ/OA"),
        ("anakinra", "il_1_beta", "blocks", "IL-1 receptor antagonist"),
        ("il_38", "il_1_beta", "anti_inflammatory", "Anti-inflammatory cytokine"),
        
        # ACD137 + Tanezumab: TrkA/NGF axis
        ("acd137", "tanezumab", "same_axis", "Both target NGF/TrkA, different mechanisms (NAM vs antibody)"),
        ("acd137", "ngf", "indirectly_blocks", "TrkA NAM blocks NGF signal transduction"),
        ("acd137", "p75ntr", "spares", "Safety advantage: selective TrkA sparing p75NTR"),
        
        # FSTL1 (from Hu 2025) + Denosumab (from Zhu 2026)
        ("fstl1", "denosumab", "epidemiological_support", "Hu 2025 mechanism + Zhu 2026 population evidence"),
        ("fstl1", "rankl", "upstream", "FSTL1 upregulates RANKL (Hu 2025)"),
        
        # SGK1 + TMJOA
        ("sgk1", "foxo1", "inhibits", "SGK1 suppresses FoxO1 → autophagy↓ → cartilage degradation"),
        ("sgk1", "autophagy", "suppresses", "Via FoxO1 inhibition"),
        ("sgk1", "tmjoa", "therapeutic_target", "PMID 41470005"),
        
        # Hydrogel delivery
        ("fmoc_ddt", "tmjoa", "delivery_system", "Hydrogel for sustained intra-articular release"),
        ("chitosan_hyaluronate", "tmjoa", "delivery_system", "Arthroscope-guided injection"),
    ]

    added_synthesis = 0
    for s_name, t_name, rel, ev in synthesis_edges:
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
            added_synthesis += 1
    print(f"[4] Added {added_synthesis} synthesis edges")

    # Save
    store.save()

    # Summary
    project_nodes = [n for n, d in store.graph.nodes(data=True)
                     if PROJECT_ID in d.get("project_ids", [])]
    papers_nodes = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "paper"]
    drugs = [n for n in project_nodes if store.graph.nodes[n].get("category") == "drug"]
    diseases = [n for n in project_nodes if store.graph.nodes[n].get("category") == "disease"]
    
    print(f"\n{'='*70}")
    print("✅ Deep reading findings import complete!")
    print(f"{'='*70}")
    print(f"LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016: {len(project_nodes)} nodes ({len(papers_nodes)} papers)")
    print(f"Drugs: {len(drugs)}, Diseases: {len(diseases)}")
    
    return True


if __name__ == "__main__":
    success = import_reading_findings()
    sys.exit(0 if success else 1)
