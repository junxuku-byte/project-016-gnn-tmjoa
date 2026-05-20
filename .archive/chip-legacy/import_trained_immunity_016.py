#!/usr/bin/env python3
"""
Import Chen J Adv Res 2025 Trained Immunity Modulators into LabKG.
Key contribution: bridges trained_immunity and CHIP concepts.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store

PROJECT_ID = "project-016"

# Paper node
PAPER = {
    "node_id": "paper:40975126",
    "pmid": "40975126",
    "doi": "10.1016/j.jare.2025.09.029",
    "title": "Trained immunity modulators: A new frontier in immunoregulation and disease intervention",
    "first_author": "Chen J",
    "year": 2025,
    "journal": "J Adv Res",
    "key_finding": "Trained immunity is mediated by metabolic reprogramming (glycolysis, TCA cycle) + epigenetic reprogramming (H3K4me3, H3K27ac, H3K18la). Inducers: BCG, beta-glucan, LPS, OMVs. Suppressors: rapamycin (mTORi), metformin (AMPK), IL-37/IL-38, MyD88 inhibitors, hydroxychloroquine, GSK2033 (LXR antagonist). CHIP is conceptually equivalent to aberrant trained immunity in HSPCs.",
    "priority": "P0",
    "concepts": [
        "trained_immunity", "innate_immune_memory", "metabolic_reprogramming", "epigenetic_reprogramming",
        "h3k4me3", "h3k27ac", "h3k18la", "histone_methylation", "histone_acetylation", "histone_lactylation",
        "glycolysis", "tca_cycle", "mtor", "hif1a", "akt",
        "bcg", "beta_glucan", "lps", "chitin", "omv", "mtp10_hdl",
        "rapamycin", "metformin", "il_37", "il_38", "myd88_inhibitor", "hydroxychloroquine", "gsk2033",
        "kdm5", "hat", "hdac",
        "hsPC", "monocyte", "macrophage", "neutrophil", "nk_cell", "dendritic_cell",
        "prr", "tlr", "clr", "nod_like_receptor",
        "inflammation", "autoimmune", "atherosclerosis", "cancer", "infection",
    ],
}

# Core mechanistic edges
MECHANISTIC_EDGES = [
    # Metabolic reprogramming
    ("prr", "akt", "activates", "PAMP/DAMP recognition triggers signaling"),
    ("akt", "mtor", "activates", "Core signaling axis"),
    ("mtor", "hif1a", "stabilizes", "mTOR activates HIF1α"),
    ("hif1a", "glycolysis", "upregulates", "Warburg effect"),
    ("hif1a", "tca_cycle", "reprograms", "TCA cycle interruption/rewiring"),
    # Epigenetic reprogramming
    ("glycolysis", "acetyl_coa", "produces", "Metabolic intermediate"),
    ("tca_cycle", "fumarate", "accumulates", "Inhibits KDM5"),
    ("fumarate", "kdm5", "inhibits", "Fumarate blocks histone demethylase"),
    ("kdm5", "h3k4me3", "removes", "KDM5 removes methyl groups"),
    ("acetyl_coa", "hat", "activates", "Acetyl-CoA activates HATs"),
    ("hat", "h3k27ac", "adds", "Histone acetylation"),
    ("hat", "h3k18ac", "adds", "Histone acetylation"),
    ("h3k4me3", "inflammation", "activates", "Opens pro-inflammatory promoters"),
    ("h3k27ac", "inflammation", "enhances", "Enhancer activation"),
    ("lactate", "ldha", "produces", "LDHA-dependent lactate production"),
    ("ldha", "h3k18la", "catalyzes", "Histone lactylation"),
    ("h3k18la", "trained_immunity", "maintains", "BCG-induced memory marker"),
    # Trained immunity suppressors
    ("rapamycin", "mtor", "inhibits", "Cheng et al. suppresses beta-glucan training"),
    ("metformin", "ampk", "activates", "AMPK activation"),
    ("ampk", "mtor", "inhibits", "AMPK suppresses mTOR"),
    ("il_37", "ampk", "activates", "Anti-inflammatory cytokine"),
    ("il_37", "il_1_beta", "blocks", "Blocks ASC oligomerization"),
    ("il_38", "akt", "disrupts", "Disrupts AKT/mTOR/S6K"),
    ("il_38", "h3k4me3", "downregulates", "Reverses beta-glucan training"),
    ("myd88_inhibitor", "tlr", "blocks", "TJ-M2010-5 blocks TLR signaling"),
    ("myd88_inhibitor", "nf_kappa_b", "prevents", "Prevents NF-κB nuclear translocation"),
    ("myd88_inhibitor", "h3k4me3", "reduces", "Reduces H3K4me3 at IL-6/TNFα loci"),
    ("hydroxychloroquine", "mtor", "inhibits", "Lysosomal alkalization suppresses mTOR"),
    ("hydroxychloroquine", "h3k27ac", "prevents", "Prevents histone acetylation"),
    ("hydroxychloroquine", "h3k4me3", "prevents", "Prevents histone methylation"),
    ("gsk2033", "lxr", "antagonizes", "LXR antagonist"),
    ("gsk2033", "glycolysis", "inhibits", "Inhibits aerobic glycolysis"),
    ("gsk2033", "h3k27ac", "reduces", "Reduces at IL-6/TNFα promoters"),
    # CHIP-trained immunity bridge
    ("dnmt3a", "trained_immunity", "drives", "DNMT3A mutation causes aberrant trained immunity"),
    ("tet2", "trained_immunity", "drives", "TET2 mutation causes aberrant trained immunity"),
    ("chip", "trained_immunity", "is_a", "CHIP is conceptually equivalent to aberrant trained immunity"),
    ("hsPC", "trained_immunity", "acquires", "HSPCs acquire memory-like phenotypes"),
    ("bone_marrow", "trained_immunity", "site_of", "Bone marrow is primary site of training"),
    ("monocyte", "trained_immunity", "exhibits", "Circulating monocytes show trained phenotypes"),
    ("macrophage", "trained_immunity", "exhibits", "Tissue macrophages show trained phenotypes"),
    # Disease connections
    ("trained_immunity", "infection", "protects_against", "Enhanced host defense"),
    ("trained_immunity", "cancer", "dual_role", "Anti-tumor vs pro-tumor"),
    ("trained_immunity", "autoimmune", "exacerbates", "RA, atherosclerosis, transplant rejection"),
    ("trained_immunity", "atherosclerosis", "promotes", "Sustained inflammation in plaques"),
    ("trained_immunity", "alzheimer", "accelerates", "Microglia reprogramming"),
    # Specific inducers
    ("bcg", "trained_immunity", "induces", "TLR2/4 + NOD2 → H3K4me3 + glycolysis"),
    ("beta_glucan", "trained_immunity", "induces", "Dectin-1 → AKT/mTOR/HIF1α"),
    ("lps", "trained_immunity", "induces", "Low dose: p38/MAPK → H3K4me3"),
    ("lps", "tolerance", "induces", "High dose: TLR4 downregulation"),
]


def normalize_id(name: str) -> str:
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_')[:80]}"


def import_trained_immunity():
    print("="*70)
    print("Importing Chen 2025 Trained Immunity into LabKG")
    print("="*70)
    
    store = get_graph_store()
    project_node_id = f"project:{PROJECT_ID}"
    
    # 1. Add paper node
    paper_id = PAPER["node_id"]
    if paper_id not in store.graph.nodes:
        store.graph.add_node(
            paper_id,
            node_type="paper",
            pmid=PAPER["pmid"],
            doi=PAPER["doi"],
            title=PAPER["title"],
            first_author=PAPER["first_author"],
            year=PAPER["year"],
            journal=PAPER["journal"],
            key_finding=PAPER["key_finding"],
            project_ids=[PROJECT_ID],
            priority=PAPER["priority"],
            concepts=PAPER["concepts"],
            source="project-016_trained_immunity",
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )
        store.graph.add_edge(project_node_id, paper_id, edge_type="contains", project_ids=[PROJECT_ID])
        print(f"[1] Added paper: {paper_id}")
    else:
        print(f"[1] Paper already exists: {paper_id}")
    
    # 2. Add concept stubs and paper→concept edges
    print(f"[2] Adding {len(PAPER['concepts'])} concepts...")
    added_concepts = 0
    for concept_name in PAPER["concepts"]:
        concept_id = normalize_id(concept_name)
        if concept_id not in store.graph.nodes:
            store.graph.add_node(
                concept_id,
                node_type="concept",
                name=concept_name.replace("_", " "),
                category="mechanism",
                project_ids=[PROJECT_ID],
                source="project-016_stub",
            )
            added_concepts += 1
        
        if not store.graph.has_edge(paper_id, concept_id):
            store.graph.add_edge(
                paper_id, concept_id,
                edge_type="mentions",
                project_ids=[PROJECT_ID],
            )
    print(f"    Added {added_concepts} new concepts")
    
    # 3. Add mechanistic edges
    print(f"[3] Adding {len(MECHANISTIC_EDGES)} mechanistic edges...")
    added_edges = 0
    for source_name, target_name, relation, evidence in MECHANISTIC_EDGES:
        source_id = normalize_id(source_name)
        target_id = normalize_id(target_name)
        
        for node_id in [source_id, target_id]:
            if node_id not in store.graph.nodes:
                store.graph.add_node(
                    node_id,
                    node_type="concept",
                    name=node_id.replace("concept:", "").replace("_", " "),
                    category="mechanism",
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
        
        if not store.graph.has_edge(source_id, target_id):
            store.graph.add_edge(
                source_id, target_id,
                edge_type=relation,
                project_ids=[PROJECT_ID],
                evidence=evidence,
            )
            added_edges += 1
    print(f"    Added {added_edges} new edges")
    
    # 4. Add trained_immunity→CHIP bridge edges (if CHIP nodes exist)
    print("[4] Bridging trained_immunity ↔ CHIP network...")
    bridge_edges = [
        ("concept:chip", "concept:trained_immunity", "is_a", "CHIP is aberrant trained immunity"),
        ("concept:dnmt3a", "concept:trained_immunity", "causes", "DNMT3A mutation drives aberrant training"),
        ("concept:tet2", "concept:trained_immunity", "causes", "TET2 mutation drives aberrant training"),
        ("concept:rapamycin", "concept:chip", "inhibits", "Wang 2024: reverses CHIP inflammation"),
        ("concept:metformin", "concept:chip", "may_inhibit", "Hypothesis: AMPK→mTOR→CHIP suppression"),
        ("concept:il_37", "concept:chip", "may_inhibit", "Hypothesis: IL-37 blocks mTOR in CHIP"),
        ("concept:il_38", "concept:chip", "may_inhibit", "Hypothesis: IL-38 reverses H3K4me3 in CHIP"),
        ("concept:myd88_inhibitor", "concept:chip", "may_inhibit", "Hypothesis: blocks TLR→NF-κB in CHIP"),
        ("concept:hydroxychloroquine", "concept:chip", "may_inhibit", "Hypothesis: lysosomal→mTOR→CHIP"),
    ]
    
    added_bridges = 0
    for source, target, relation, evidence in bridge_edges:
        if source in store.graph.nodes and target in store.graph.nodes:
            if not store.graph.has_edge(source, target):
                store.graph.add_edge(
                    source, target,
                    edge_type=relation,
                    project_ids=[PROJECT_ID],
                    evidence=evidence,
                )
                added_bridges += 1
    # Add bridge: trained_immunity -> inflammation (to connect to TMJOA network)
    if not store.graph.has_edge('concept:trained_immunity', 'concept:inflammation'):
        store.graph.add_edge(
            'concept:trained_immunity', 'concept:inflammation',
            edge_type='drives',
            project_ids=[PROJECT_ID],
            evidence='Aberrant trained immunity causes sustained inflammation',
        )
        added_bridges += 1
    
    # Add bridge: h3k4me3 -> inflammation
    if not store.graph.has_edge('concept:h3k4me3', 'concept:inflammation'):
        store.graph.add_edge(
            'concept:h3k4me3', 'concept:inflammation',
            edge_type='activates',
            project_ids=[PROJECT_ID],
            evidence='H3K4me3 opens pro-inflammatory gene promoters',
        )
        added_bridges += 1
    
    # Save
    store.save()
    
    # Summary
    project_nodes = [n for n, d in store.graph.nodes(data=True) 
                     if PROJECT_ID in d.get("project_ids", [])]
    papers = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "paper"]
    
    print(f"\n{'='*70}")
    print("✅ Trained Immunity import complete!")
    print(f"{'='*70}")
    print(f"LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016: {len(project_nodes)} nodes ({len(papers)} papers)")
    
    return True


if __name__ == "__main__":
    success = import_trained_immunity()
    sys.exit(0 if success else 1)
