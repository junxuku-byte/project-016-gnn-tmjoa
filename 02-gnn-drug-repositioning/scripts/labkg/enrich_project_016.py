#!/usr/bin/env python3
"""
Enrich Project-016 LabKG with core literature papers and evidence chain edges.
1. Fix project_ids on existing concept nodes
2. Add 11 core papers as PaperNodes
3. Create paper→concept edges
4. Create key evidence chain edges
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store
from labkg.schema.models import PaperNode

PROJECT_ID = "project-016"

# Core literature — evidence chain foundation
CORE_PAPERS = [
    {
        "pmid": "38838669",
        "doi": "10.1016/j.cell.2024.05.003",
        "title": "Clonal Hematopoiesis Driven by Mutated DNMT3A Promotes Inflammatory Bone Loss",
        "first_author": "Wang H",
        "year": 2024,
        "journal": "Cell",
        "key_finding": "DNMT3A-CHIP causes inflammatory bone loss via IL-17-dependent pathway; reversible with rapamycin",
        "concepts": ["dnmt3a", "chip", "il_17", "inflammatory_bone_loss", "periodontitis", "osteoclast", "rapamycin"],
        "priority": "P0",
    },
    {
        "pmid": "PMC12972371",
        "doi": "10.1007/s11357-025-01843-y",
        "title": "Clonal hematopoiesis promotes osteoarthritis risk through inflammatory pathways",
        "first_author": "Li P",
        "year": 2025,
        "journal": "GeroScience",
        "key_finding": "UK Biobank 45,380: CHIP VAF>10% → OA HR 1.46; 7 mediator proteins identified",
        "concepts": ["chip", "osteoarthritis", "uk_biobank", "inflammation", "mediator_proteins"],
        "priority": "P0",
    },
    {
        "pmid": "34290333",
        "doi": "10.1084/jem.20211872",
        "title": "Dnmt3a-mutated clonal hematopoiesis promotes osteoporosis",
        "first_author": "Kim K",
        "year": 2021,
        "journal": "J Exp Med",
        "key_finding": "Dnmt3a-/- macrophages secrete IL-20 → Irf3-NF-κB → osteoclastogenesis↑; reversible with alendronate",
        "concepts": ["dnmt3a", "chip", "il_20", "osteoporosis", "macrophage", "nf_kappa_b", "osteoclast", "alendronate"],
        "priority": "P0",
    },
    {
        "pmid": "36813799",
        "doi": "10.1038/s41467-023-36906-1",
        "title": "DNMT3A clonal hematopoiesis is selectively favored in fatty bone marrow",
        "first_author": "Zioni M",
        "year": 2023,
        "journal": "Nat Commun",
        "key_finding": "FBM provides selective advantage to DNMT3A-mutant HSPCs via IL-6/IFN-γ; anti-IL-6 reversible",
        "concepts": ["dnmt3a", "chip", "fatty_bone_marrow", "fbm", "menopause", "estrogen", "il_6", "ifn_gamma", "hspc"],
        "priority": "P0",
    },
    {
        "pmid": "28428328",
        "doi": "10.1126/science.aan4673",
        "title": "Clonal hematopoiesis in human aging and disease",
        "first_author": "Jaiswal S",
        "year": 2019,
        "journal": "Science",
        "key_finding": "Foundational review: DNMT3A/TET2/ASXL1 CHIP prevalence, age-related expansion, disease associations",
        "concepts": ["chip", "dnmt3a", "tet2", "asxl1", "clonal_hematopoiesis", "aging", "inflammaging"],
        "priority": "P1",
    },
    {
        "pmid": "28636844",
        "doi": "10.1056/NEJMoa1701719",
        "title": "Age-related clonal hematopoiesis associated with adverse outcomes",
        "first_author": "Jaiswal S",
        "year": 2017,
        "journal": "N Engl J Med",
        "key_finding": "Landmark CHIP→CVD study: VAF≥2% associated with 2× all-cause mortality and coronary heart disease",
        "concepts": ["chip", "cvd", "cardiovascular_disease", "mortality", "vaf", "clonal_hematopoiesis"],
        "priority": "P1",
    },
    {
        "pmid": None,
        "doi": "10.1146/annurev-pathmechdis-111523-023442",
        "title": "Clonal Hematopoiesis and Inflammation",
        "first_author": "Koh Y",
        "year": 2026,
        "journal": "Annu Rev Pathol",
        "key_finding": "Comprehensive CHIP review: DNMT3A→IL-17/IL-20→osteoclastogenesis; inflammation as unifying mechanism",
        "concepts": ["chip", "dnmt3a", "il_17", "il_20", "inflammation", "osteoclast", "bone_loss", "review"],
        "priority": "P1",
    },
    {
        "pmid": "39602602",
        "doi": "10.1126/sciadv.adt9846",
        "title": "Clonal hematopoiesis and risk of rheumatoid arthritis",
        "first_author": "Hiitola T",
        "year": 2025,
        "journal": "Sci Adv",
        "key_finding": "CHIP→RA: FINRISK OR 2.06, FinnGen OR 1.49; DNMT3A→seropositive RA, TET2→seronegative RA",
        "concepts": ["chip", "dnmt3a", "tet2", "rheumatoid_arthritis", "ra", "autoimmune", "inflammation"],
        "priority": "P1",
    },
    {
        "pmid": "38874099",
        "doi": "10.1056/NEJMoa2403664",
        "title": "Semaglutide in Patients with Knee Osteoarthritis",
        "first_author": "Bliddal H",
        "year": 2024,
        "journal": "N Engl J Med",
        "key_finding": "STEP 9: Semaglutide 2.4mg → WOMAC pain -41.7 vs -27.5; NLRP3/IL-1β inhibition proposed mechanism",
        "concepts": ["semaglutide", "glp_1", "knee_osteoarthritis", "obesity", "inflammation", "nlrp3", "il_1_beta"],
        "priority": "P2",
    },
    {
        "pmid": None,
        "doi": "10.3389/fbioe.2025.1596143",
        "title": "Craniofacial Bone Marrow: Anti-Aging Properties and Regenerative Potential",
        "first_author": "Koh Y",
        "year": 2025,
        "journal": "Front Bioeng Biotechnol",
        "key_finding": "Craniofacial marrow retains more hematopoietic precursors than long bone; FCSCs form bone with hematopoietic niche",
        "concepts": ["craniofacial_bone", "bone_marrow", "hematopoietic_niche", "fcsc", "fibrocartilage_stem_cells", "anti_aging", "mandible"],
        "priority": "P2",
    },
    {
        "pmid": "27667684",
        "doi": "10.1038/ncomms13073",
        "title": "Fibrocartilage Stem Cells Generate the Hematopoietic Microenvironment of the Temporomandibular Joint",
        "first_author": "Embree MC",
        "year": 2016,
        "journal": "Nat Commun",
        "key_finding": "FCSCs from TMJ condyle superficial zone generate bone with hematopoietic microenvironment; Wnt/sclerostin regulated",
        "concepts": ["fcsc", "fibrocartilage_stem_cells", "tmj", "temporomandibular_joint", "hematopoietic_microenvironment", "condyle", "wnt", "sclerostin"],
        "priority": "P2",
    },
]

# Key evidence chain edges (concept → concept with evidence citation)
EVIDENCE_EDGES = [
    # CHIP → disease mechanisms
    ("chip", "inflammation", "drives", "Jaiswal 2017, Jaiswal 2019"),
    ("chip", "bone_loss", "causes", "Wang 2024 Cell"),
    ("chip", "osteoporosis", "causes", "Kim 2021 JEM"),
    ("chip", "osteoarthritis", "promotes", "Li 2025 GeroScience"),
    ("chip", "rheumatoid_arthritis", "predisposes", "Hiitola 2025 Sci Adv"),
    
    # DNMT3A specific
    ("dnmt3a", "il_17", "upregulates", "Wang 2024 Cell"),
    ("dnmt3a", "il_20", "upregulates", "Kim 2021 JEM"),
    ("dnmt3a", "osteoclast", "promotes", "Wang 2024, Kim 2021"),
    ("dnmt3a", "macrophage", "polarizes", "Kim 2021 JEM"),
    
    # Menopause pathway
    ("menopause", "fbm", "increases", "Zioni 2023 Nat Commun"),
    ("menopause", "estrogen", "decreases", "Classic endocrinology"),
    ("fbm", "dnmt3a", "selectively_favors", "Zioni 2023 Nat Commun"),
    ("estrogen", "tmjoa", "protects", "Classic theory"),
    
    # Inflammatory cascade
    ("il_17", "osteoclast", "promotes", "Wang 2024 Cell"),
    ("il_20", "osteoclast", "promotes", "Kim 2021 JEM"),
    ("inflammation", "bone_loss", "causes", "Multiple sources"),
    ("inflammation", "osteoarthritis", "drives", "Li 2025 GeroScience"),
    
    # TMJ specific
    ("fcsc", "hematopoietic_microenvironment", "generates", "Embree 2016 Nat Commun"),
    ("tmj", "condyle", "contains", "Embree 2016 Nat Commun"),
    ("mandible", "hematopoietic_precursors", "retains", "Koh 2025 Front Bioeng"),
    
    # Treatment links
    ("rapamycin", "chip", "inhibits", "Wang 2024 Cell"),
    ("alendronate", "osteoporosis", "reverses", "Kim 2021 JEM"),
    ("il_6", "dnmt3a", "promotes", "Zioni 2023 Nat Commun"),
    
    # GLP-1 speculative
    ("glp_1", "inflammation", "modulates", "Bliddal 2024 NEJM (indirect)"),
    ("semaglutide", "knee_osteoarthritis", "improves", "Bliddal 2024 NEJM"),
]


def normalize_concept_id(name: str) -> str:
    """Normalize concept name to LabKG ID format."""
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_')[:80]}"


def enrich_project_016():
    print(f"[1/5] Connecting to LabKG...")
    store = get_graph_store()
    
    # Check current state
    project_node_id = f"project:{PROJECT_ID}"
    existing_project_nodes = [n for n, d in store.graph.nodes(data=True) 
                              if PROJECT_ID in d.get("project_ids", [])]
    print(f"  → LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"  → Current project-016 tagged: {len(existing_project_nodes)} nodes")
    
    print(f"\n[2/5] Tagging existing concept nodes with project-016...")
    # Find nodes from kg.json import that should have project-016 tag
    kg_path = Path.home() / "morph-lab" / "projects" / "project-016-chip-tmjoa" / "00-文献" / "kg_output" / "kg.json"
    if kg_path.exists():
        with open(kg_path, "r") as f:
            kg_data = json.load(f)
        
        tagged = 0
        for node in kg_data.get("nodes", []):
            raw_id = node.get("id", "").strip()
            if not raw_id:
                continue
            norm_id = normalize_concept_id(raw_id)
            if norm_id in store.graph.nodes:
                data = store.graph.nodes[norm_id]
                project_ids = data.get("project_ids", [])
                if PROJECT_ID not in project_ids:
                    data["project_ids"] = project_ids + [PROJECT_ID]
                    tagged += 1
        print(f"  → Tagged {tagged} existing nodes with project-016")
    else:
        print(f"  → kg.json not found, skipping")
    
    print(f"\n[3/5] Adding {len(CORE_PAPERS)} core papers...")
    added_papers = 0
    paper_node_ids = []
    
    for paper in CORE_PAPERS:
        # Determine node ID
        if paper["pmid"] and paper["pmid"] != "PMC12972371":
            node_id = f"paper:{paper['pmid']}"
        elif paper["doi"]:
            # Use DOI hash for PMC-only papers
            import hashlib
            h = hashlib.sha256(paper["doi"].encode()).hexdigest()[:12]
            node_id = f"paper:{h}"
        else:
            h = hashlib.sha256(paper["title"].encode()).hexdigest()[:12]
            node_id = f"paper:{h}"
        
        # Skip if already exists
        if node_id in store.graph.nodes:
            # Tag with project-016 if not already
            data = store.graph.nodes[node_id]
            project_ids = data.get("project_ids", [])
            if PROJECT_ID not in project_ids:
                data["project_ids"] = project_ids + [PROJECT_ID]
            paper_node_ids.append(node_id)
            continue
        
        # Create paper node
        store.graph.add_node(
            node_id,
            node_type="paper",
            pmid=paper.get("pmid"),
            doi=paper["doi"],
            title=paper["title"],
            first_author=paper["first_author"],
            year=paper["year"],
            journal=paper["journal"],
            key_finding=paper["key_finding"],
            project_ids=[PROJECT_ID],
            priority=paper["priority"],
            concepts=paper["concepts"],
            source="project-016_core",
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )
        added_papers += 1
        paper_node_ids.append(node_id)
        
        # Link paper to project
        store.graph.add_edge(
            project_node_id, node_id,
            edge_type="contains",
            project_ids=[PROJECT_ID],
        )
    
    print(f"  → Added {added_papers} new papers, linked {len(paper_node_ids)} total")
    
    print(f"\n[4/5] Creating paper→concept edges...")
    paper_concept_edges = 0
    
    for i, paper in enumerate(CORE_PAPERS):
        paper_id = paper_node_ids[i]
        for concept_name in paper["concepts"]:
            concept_id = normalize_concept_id(concept_name)
            
            # Ensure concept exists (create stub if needed)
            if concept_id not in store.graph.nodes:
                store.graph.add_node(
                    concept_id,
                    node_type="concept",
                    name=concept_name.replace("_", " "),
                    category="biomarker",  # default
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
                print(f"    [STUB] Created concept: {concept_id}")
            
            # Add paper→concept edge if not exists
            if not store.graph.has_edge(paper_id, concept_id):
                store.graph.add_edge(
                    paper_id, concept_id,
                    edge_type="mentions",
                    project_ids=[PROJECT_ID],
                    evidence=paper["key_finding"][:200],
                )
                paper_concept_edges += 1
    
    print(f"  → Created {paper_concept_edges} paper→concept edges")
    
    print(f"\n[5/5] Creating evidence chain edges...")
    evidence_edges = 0
    
    for source_name, target_name, relation, evidence in EVIDENCE_EDGES:
        source_id = normalize_concept_id(source_name)
        target_id = normalize_concept_id(target_name)
        
        # Ensure both nodes exist
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
                print(f"    [STUB] Created: {node_id}")
        
        # Add edge
        if not store.graph.has_edge(source_id, target_id):
            store.graph.add_edge(
                source_id, target_id,
                edge_type=relation,
                project_ids=[PROJECT_ID],
                evidence=evidence,
            )
            evidence_edges += 1
        else:
            # Update evidence if edge exists
            existing = store.graph.edges[source_id, target_id]
            existing_evidence = existing.get("evidence") or ""
            if evidence not in existing_evidence:
                existing["evidence"] = f"{existing_evidence}; {evidence}" if existing_evidence else evidence
    
    print(f"  → Created {evidence_edges} evidence chain edges")
    
    # Save
    print(f"\n[SAVE] Persisting LabKG...")
    store.save()
    
    # Summary
    final_project_nodes = [n for n, d in store.graph.nodes(data=True) 
                           if PROJECT_ID in d.get("project_ids", [])]
    print(f"\n✅ Project-016 enrichment complete!")
    print(f"   LabKG now: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"   Project-016 total: {len(final_project_nodes)} nodes")
    print(f"   Papers: {len([n for n in final_project_nodes if store.graph.nodes[n].get('node_type') == 'paper'])}")
    print(f"   Concepts: {len([n for n in final_project_nodes if store.graph.nodes[n].get('node_type') == 'concept'])}")
    
    return True


if __name__ == "__main__":
    success = enrich_project_016()
    sys.exit(0 if success else 1)
