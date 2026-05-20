#!/usr/bin/env python3
"""
Import Project-016 CHIP-TMJOA KG into LabKG main graph.
Converts kg.json nodes/edges into LabKG-compatible format.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store
from labkg.schema.models import ConceptNode, ProjectNode

KG_PATH = Path.home() / "morph-lab" / "projects" / "project-016-chip-tmjoa" / "00-文献" / "kg_output" / "kg.json"
PROJECT_ID = "project-016"


def map_type_to_category(node_type: str) -> str:
    """Map raw kg.json type to LabKG concept category."""
    type_lower = node_type.lower()
    if any(x in type_lower for x in ["cell", "tissue", "anatom", "structure", "bone"]):
        return "anatomy"
    elif any(x in type_lower for x in ["process", "pathway", "signal"]):
        return "mechanism"
    elif any(x in type_lower for x in ["disease", "pathology", "syndrome", "disorder", "risk", "arthritis", "cancer"]):
        return "disease"
    elif any(x in type_lower for x in ["protein", "gene", "molecule", "reagent", "enzyme", "hormone"]):
        return "biomarker"
    elif any(x in type_lower for x in ["method", "technique", "assay", "analysis"]):
        return "method"
    else:
        return "mechanism"  # default


def import_project_016():
    print(f"[1/4] Loading Project-016 kg.json from {KG_PATH}")
    if not KG_PATH.exists():
        print(f"[ERROR] {KG_PATH} not found!")
        return False
    
    with open(KG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])
    print(f"  → Raw nodes: {len(raw_nodes)}, edges: {len(raw_edges)}")
    
    print(f"[2/4] Connecting to LabKG store...")
    store = get_graph_store()
    print(f"  → Current LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    
    print(f"[3/4] Importing nodes...")
    added_nodes = 0
    skipped = 0
    node_id_map = {}  # map raw id → normalized id
    
    for node in raw_nodes:
        raw_id = node.get("id", "").strip()
        if not raw_id:
            skipped += 1
            continue
        
        # Normalize ID
        norm_id = f"concept:{raw_id.lower().replace(' ', '_').replace('-', '_')[:80]}"
        node_id_map[raw_id] = norm_id
        
        if norm_id in store.graph.nodes:
            # Node exists — just tag with project-016
            existing = store.graph.nodes[norm_id]
            project_ids = existing.get("project_ids", [])
            if PROJECT_ID not in project_ids:
                existing["project_ids"] = project_ids + [PROJECT_ID]
            skipped += 1
            continue
        
        # Create new concept node
        category = map_type_to_category(node.get("type", ""))
        concept = ConceptNode(
            name=raw_id,
            category=category,
            definition=f"Extracted from Project-016 literature (type: {node.get('type', 'unknown')})",
            synonyms=[raw_id] if len(raw_id) < 30 else [],
            paper_count=0,
            project_count=1,
        )
        
        store.graph.add_node(
            norm_id,
            node_type="concept",
            name=concept.name,
            category=concept.category,
            definition=concept.definition,
            synonyms=concept.synonyms,
            paper_count=concept.paper_count,
            project_count=concept.project_count,
            first_appearance=concept.first_appearance.isoformat(),
            project_ids=[PROJECT_ID],
            source="project-016_kg",
        )
        added_nodes += 1
    
    # Add Project node
    project_node_id = f"project:{PROJECT_ID}"
    if project_node_id not in store.graph.nodes:
        store.graph.add_node(
            project_node_id,
            node_type="project",
            project_id=PROJECT_ID,
            name="CHIP与更年期女性TMJ骨关节炎",
            status="active",
            key_concepts=list(node_id_map.values())[:20],
            paper_count=11,
        )
        added_nodes += 1
        print(f"  → Added project node: {project_node_id}")
    
    print(f"  → Added: {added_nodes}, Skipped (exists): {skipped}")
    
    print(f"[4/4] Importing edges...")
    added_edges = 0
    for edge in raw_edges:
        source_raw = edge.get("from") or edge.get("source", "")
        target_raw = edge.get("to") or edge.get("target", "")
        
        source_id = node_id_map.get(source_raw)
        target_id = node_id_map.get(target_raw)
        
        if not source_id or not target_id:
            continue
        if not store.graph.has_node(source_id) or not store.graph.has_node(target_id):
            continue
        
        edge_type = edge.get("edge_type") or edge.get("relation", "related_to")
        
        # Check if edge already exists
        if store.graph.has_edge(source_id, target_id):
            # Update existing edge with project-016 tag if not present
            existing = store.graph.edges[source_id, target_id]
            projects = existing.get("project_ids", [])
            if PROJECT_ID not in projects:
                existing["project_ids"] = projects + [PROJECT_ID]
            continue
        
        store.graph.add_edge(
            source_id, target_id,
            edge_type=edge_type,
            project_ids=[PROJECT_ID],
            weight=edge.get("weight", 1.0),
            evidence=edge.get("evidence", "Project-016 literature extraction"),
        )
        added_edges += 1
    
    # Add project→concept edges
    project_edge_count = 0
    for norm_id in node_id_map.values():
        if store.graph.has_node(norm_id):
            store.graph.add_edge(
                project_node_id, norm_id,
                edge_type="contains",
                project_ids=[PROJECT_ID],
            )
            project_edge_count += 1
    
    print(f"  → Added {added_edges} concept edges, {project_edge_count} project→concept edges")
    
    # Save
    print(f"[SAVE] Persisting LabKG...")
    store.save()
    
    print(f"\n✅ Import complete!")
    print(f"   LabKG now: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"   Project-016 contribution: {added_nodes} new nodes, {added_edges + project_edge_count} new edges")
    
    return True


if __name__ == "__main__":
    success = import_project_016()
    sys.exit(0 if success else 1)
