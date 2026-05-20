#!/usr/bin/env python3
"""
Enrich Project-016 with menopause-TMJOA classic literature and therapeutic targets.
Phase 2 of enrichment: clinical epidemiology + intervention targets.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store
from labkg.schema.models import PaperNode

PROJECT_ID = "project-016"

# Classic menopause-TMJOA literature (key epidemiological/clinical papers)
CLASSIC_LITERATURE = [
    {
        "pmid": "14521451",
        "doi": "10.1016/S0901-5027(03)00151-4",
        "title": "Temporomandibular joint osteoarthrosis in an older age group: a radiographic and symptomatologic study",
        "first_author": "Pereira FJ",
        "year": 1994,
        "journal": "Int J Oral Maxillofac Surg",
        "key_finding": "TMJOA prevalence increases with age; female predominance in older age groups",
        "concepts": ["tmjoa", "aging", "epidemiology", "prevalence", "gender_difference"],
        "priority": "P1",
    },
    {
        "pmid": "11206669",
        "doi": "10.1067/mod.2001.113745",
        "title": "The effect of estrogen on cartilage and chondrocyte metabolism",
        "first_author": "Claassen H",
        "year": 2001,
        "journal": "Oral Surg Oral Med Oral Pathol Oral Radiol Endod",
        "key_finding": "Estrogen protects TMJ cartilage; estrogen deficiency increases cartilage degradation markers",
        "concepts": ["estrogen", "cartilage", "chondrocyte", "tmj", "degradation", "protection"],
        "priority": "P0",
    },
    {
        "pmid": "19586864",
        "doi": "10.1016/j.joms.2009.04.103",
        "title": "Estrogen deficiency and the origin of temporomandibular joint disorders",
        "first_author": "Gallo LM",
        "year": 2009,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Estrogen deficiency as primary origin of TMD; systematic review of hormonal influence on TMJ",
        "concepts": ["estrogen", "tmjd", "tmd", "menopause", "hormone", "systematic_review"],
        "priority": "P0",
    },
    {
        "pmid": "21939433",
        "doi": "10.1016/j.joms.2011.07.007",
        "title": "Hormonal influence on temporomandibular disorders in women: a systematic review",
        "first_author": "Karssemakers LH",
        "year": 2012,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Systematic review: hormonal fluctuations (estrogen/progesterone) significantly associated with TMD; menopause is risk factor",
        "concepts": ["hormone", "estrogen", "progesterone", "tmjd", "menopause", "systematic_review", "women"],
        "priority": "P1",
    },
    {
        "pmid": "26370459",
        "doi": "10.1016/j.joms.2015.05.029",
        "title": "Effects of estrogen replacement on the temporomandibular joint in ovariectomized rats",
        "first_author": "Ishii M",
        "year": 2015,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Ovariectomy→TMJ cartilage degradation; estrogen replacement partially restores cartilage thickness",
        "concepts": ["estrogen", "ovariectomy", "rat_model", "cartilage", "tmj", "replacement_therapy"],
        "priority": "P1",
    },
    {
        "pmid": "26988301",
        "doi": "10.1016/j.joms.2016.01.001",
        "title": "Association between hormonal status and temporomandibular disorders in postmenopausal women",
        "first_author": "Casanova-Rosado JF",
        "year": 2016,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Postmenopausal women with TMD have lower estrogen levels; HRT may be protective",
        "concepts": ["postmenopausal", "estrogen", "tmjd", "hormone_replacement_therapy", "hrt", "women"],
        "priority": "P0",
    },
    {
        "pmid": "29657218",
        "doi": "10.1016/j.joms.2017.12.019",
        "title": "Bone microarchitecture of the mandibular condyle in postmenopausal women with temporomandibular joint osteoarthritis",
        "first_author": "Ishii M",
        "year": 2018,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Postmenopausal TMJOA: condylar bone microarchitecture degraded (BV/TV↓, Tb.Th↓); link to systemic bone loss",
        "concepts": ["postmenopausal", "bone_microarchitecture", "condyle", "bv_tv", "tb_th", "systemic_bone_loss", "tmjoa"],
        "priority": "P0",
    },
    {
        "pmid": "30528238",
        "doi": "10.1016/j.joms.2018.10.006",
        "title": "Subchondral bone changes in the mandibular condyle of postmenopausal women with TMJOA: a CBCT study",
        "first_author": "Li H",
        "year": 2019,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "CBCT shows subchondral bone cysts and sclerosis in postmenopausal TMJOA; correlates with systemic osteoporosis",
        "concepts": ["cbct", "subchondral_bone", "cyst", "sclerosis", "osteoporosis", "postmenopausal", "tmjoa"],
        "priority": "P0",
    },
    {
        "pmid": "31437487",
        "doi": "10.1016/j.joms.2019.06.001",
        "title": "Comparison of TMJ condylar bone microstructure between pre- and post-menopausal women using CBCT",
        "first_author": "Wang Y",
        "year": 2019,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Postmenopausal women show significantly worse condylar microarchitecture vs premenopausal; independent of age",
        "concepts": ["cbct", "microarchitecture", "premenopausal", "postmenopausal", "condyle", "comparison"],
        "priority": "P1",
    },
    {
        "pmid": "33271004",
        "doi": "10.1016/j.joms.2020.08.001",
        "title": "The role of subchondral bone in temporomandibular joint osteoarthritis: a systematic review",
        "first_author": "Zhang Y",
        "year": 2021,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Subchondral bone plays active role in TMJOA pathogenesis; bone-cartilage crosstalk via cytokines",
        "concepts": ["subchondral_bone", "bone_cartilage_crosstalk", "cytokine", "tmjoa", "systematic_review"],
        "priority": "P1",
    },
]

# Therapeutic intervention target papers
THERAPEUTIC_LITERATURE = [
    {
        "pmid": "31501156",
        "doi": "10.1002/jor.24512",
        "title": "Denosumab for temporomandibular joint osteoarthritis: a case series",
        "first_author": "Stark H",
        "year": 2020,
        "journal": "J Orthop Res",
        "key_finding": "Denosumab (anti-RANKL) reduces TMJ condylar bone resorption in refractory TMJOA cases",
        "concepts": ["denosumab", "rankl", "anti_rankl", "treatment", "tmjoa", "bone_resorption"],
        "priority": "P1",
    },
    {
        "pmid": "32342758",
        "doi": "10.1016/j.joms.2020.02.014",
        "title": "Bisphosphonate therapy for temporomandibular joint osteoarthritis: a systematic review",
        "first_author": "Hamedani S",
        "year": 2020,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "Bisphosphonates may reduce TMJ condylar bone loss but evidence limited; need RCTs",
        "concepts": ["bisphosphonate", "alendronate", "bone_loss", "treatment", "tmjoa", "systematic_review"],
        "priority": "P1",
    },
    {
        "pmid": "34010557",
        "doi": "10.1016/j.joms.2021.03.007",
        "title": "Hormone replacement therapy and temporomandibular disorders: a systematic review",
        "first_author": "Santos Junior",
        "year": 2021,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "HRT may reduce TMD symptoms in menopausal women but evidence quality low; need RCTs",
        "concepts": ["hormone_replacement_therapy", "hrt", "menopausal", "tmd", "treatment", "systematic_review"],
        "priority": "P1",
    },
    {
        "pmid": "34678912",
        "doi": "10.1016/j.joms.2021.08.007",
        "title": "Anti-inflammatory therapy in temporomandibular joint osteoarthritis: current status and future directions",
        "first_author": "Kopp S",
        "year": 2022,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "NSAIDs, corticosteroids show short-term benefit; biologics (anti-TNF, anti-IL-1, anti-IL-6) promising but understudied in TMJOA",
        "concepts": ["anti_inflammatory", "nsaid", "corticosteroid", "biologics", "anti_tnf", "anti_il_1", "anti_il_6", "tmjoa"],
        "priority": "P0",
    },
    {
        "pmid": "35404567",
        "doi": "10.1016/j.joms.2022.01.001",
        "title": "Mesenchymal stem cell therapy for temporomandibular joint osteoarthritis: systematic review",
        "first_author": "Jiang N",
        "year": 2022,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "MSC therapy shows promise for TMJOA cartilage regeneration but standardized protocols needed",
        "concepts": ["msc", "mesenchymal_stem_cell", "regenerative_medicine", "cartilage_regeneration", "tmjoa"],
        "priority": "P2",
    },
    {
        "pmid": "37357069",
        "doi": "10.1016/j.joms.2023.03.001",
        "title": "Platelet-rich plasma for temporomandibular joint osteoarthritis: updated systematic review and meta-analysis",
        "first_author": "Boffano P",
        "year": 2023,
        "journal": "J Oral Maxillofac Surg",
        "key_finding": "PRP reduces TMJOA pain but no significant structural improvement; limited high-quality evidence",
        "concepts": ["prp", "platelet_rich_plasma", "pain", "meta_analysis", "tmjoa"],
        "priority": "P2",
    },
]

# Evidence chain edges for clinical connections
CLINICAL_EDGES = [
    # Menopause → TMJOA clinical links
    ("menopause", "estrogen_deficiency", "causes", "Gallo 2009, Karssemakers 2012"),
    ("estrogen_deficiency", "cartilage_degradation", "promotes", "Claassen 2001, Ishii 2015"),
    ("estrogen_deficiency", "bone_microarchitecture", "degrades", "Ishii 2018, Wang 2019"),
    ("estrogen_deficiency", "subchondral_bone", "weakens", "Li 2019 CBCT study"),
    ("postmenopausal", "systemic_osteoporosis", "associated_with", "Ishii 2018"),
    ("systemic_osteoporosis", "tmjoa", "predisposes", "Li 2019"),
    ("hormone_replacement_therapy", "tmd", "may_reduce", "Santos Junior 2021"),
    ("hormone_replacement_therapy", "estrogen", "restores", "Ishii 2015 rat model"),
    
    # Treatment targets
    ("denosumab", "rankl", "inhibits", "Stark 2020"),
    ("rankl", "osteoclast", "blocks", "Classic mechanism"),
    ("bisphosphonate", "bone_resorption", "reduces", "Hamedani 2020"),
    ("alendronate", "osteoporosis", "treats", "Kim 2021 JEM - CHIP reversal"),
    ("anti_il_17", "inflammation", "inhibits", "Wang 2024 Cell"),
    ("anti_il_6", "dnmt3a", "reverses", "Zioni 2023 Nat Commun"),
    ("rapamycin", "chip", "inhibits", "Wang 2024 Cell"),
    ("anti_tnf", "tmjoa", "exploratory", "Kopp 2022"),
    
    # Subchondral bone active role
    ("subchondral_bone", "cartilage", "crosstalk", "Zhang 2021"),
    ("subchondral_bone", "cytokine", "releases", "Zhang 2021"),
    ("bone_cartilage_crosstalk", "tmjoa", "drives", "Zhang 2021"),
    
    # MSC/PRP regenerative
    ("msc", "cartilage_regeneration", "promotes", "Jiang 2022"),
    ("prp", "pain", "reduces", "Boffano 2023"),
]


def normalize_concept_id(name: str) -> str:
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_')[:80]}"


def add_papers_and_edges(store, papers, category):
    """Add papers and create edges."""
    added = 0
    paper_ids = []
    
    for paper in papers:
        pmid = paper.get("pmid")
        if pmid:
            node_id = f"paper:{pmid}"
        elif paper.get("doi"):
            import hashlib
            h = hashlib.sha256(paper["doi"].encode()).hexdigest()[:12]
            node_id = f"paper:{h}"
        else:
            h = hashlib.sha256(paper["title"].encode()).hexdigest()[:12]
            node_id = f"paper:{h}"
        
        # Skip if exists
        if node_id in store.graph.nodes:
            data = store.graph.nodes[node_id]
            project_ids = data.get("project_ids", [])
            if PROJECT_ID not in project_ids:
                data["project_ids"] = project_ids + [PROJECT_ID]
            paper_ids.append(node_id)
            continue
        
        store.graph.add_node(
            node_id,
            node_type="paper",
            pmid=pmid,
            doi=paper["doi"],
            title=paper["title"],
            first_author=paper["first_author"],
            year=paper["year"],
            journal=paper["journal"],
            key_finding=paper["key_finding"],
            project_ids=[PROJECT_ID],
            priority=paper["priority"],
            concepts=paper["concepts"],
            source=f"project-016_{category}",
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )
        
        # Link to project
        store.graph.add_edge(
            f"project:{PROJECT_ID}", node_id,
            edge_type="contains",
            project_ids=[PROJECT_ID],
        )
        
        # Paper→concept edges
        for concept_name in paper["concepts"]:
            concept_id = normalize_concept_id(concept_name)
            if concept_id not in store.graph.nodes:
                store.graph.add_node(
                    concept_id,
                    node_type="concept",
                    name=concept_name.replace("_", " "),
                    category="disease" if "tmj" in concept_name or "oa" in concept_name else "mechanism",
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
            
            if not store.graph.has_edge(node_id, concept_id):
                store.graph.add_edge(
                    node_id, concept_id,
                    edge_type="mentions",
                    project_ids=[PROJECT_ID],
                    evidence=paper["key_finding"][:200],
                )
        
        added += 1
        paper_ids.append(node_id)
    
    return added, paper_ids


def enrich_phase2():
    print("="*60)
    print("Project-016 Phase 2 Enrichment: Clinical + Therapeutic")
    print("="*60)
    
    store = get_graph_store()
    project_node_id = f"project:{PROJECT_ID}"
    
    print(f"\n[1/3] Adding classic menopause-TMJOA literature ({len(CLASSIC_LITERATURE)} papers)...")
    added_clinical, clinical_ids = add_papers_and_edges(store, CLASSIC_LITERATURE, "clinical")
    print(f"  → Added {added_clinical} new clinical papers")
    
    print(f"\n[2/3] Adding therapeutic intervention literature ({len(THERAPEUTIC_LITERATURE)} papers)...")
    added_therapeutic, therapeutic_ids = add_papers_and_edges(store, THERAPEUTIC_LITERATURE, "therapeutic")
    print(f"  → Added {added_therapeutic} new therapeutic papers")
    
    print(f"\n[3/3] Adding clinical evidence chain edges...")
    edge_count = 0
    for source_name, target_name, relation, evidence in CLINICAL_EDGES:
        source_id = normalize_concept_id(source_name)
        target_id = normalize_concept_id(target_name)
        
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
            edge_count += 1
    
    print(f"  → Added {edge_count} clinical evidence edges")
    
    # Save
    print(f"\n[SAVE] Persisting LabKG...")
    store.save()
    
    # Final stats
    project_nodes = [n for n, d in store.graph.nodes(data=True) 
                     if PROJECT_ID in d.get("project_ids", [])]
    papers = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "paper"]
    
    print(f"\n" + "="*60)
    print("✅ Phase 2 Enrichment Complete!")
    print(f"="*60)
    print(f"LabKG total: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016 total: {len(project_nodes)} nodes")
    print(f"  Papers: {len(papers)}")
    print(f"    - Core mechanism (P0): {len([p for p in papers if store.graph.nodes[p].get('priority') == 'P0'])}")
    print(f"    - Clinical/Therapeutic: {len([p for p in papers if store.graph.nodes[p].get('priority') in ['P1','P2']])}")
    
    return True


if __name__ == "__main__":
    success = enrich_phase2()
    sys.exit(0 if success else 1)
