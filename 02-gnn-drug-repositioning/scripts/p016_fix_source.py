#!/usr/bin/env python3
"""Fix training set v5: populate 'source' field with PMID + evidence source."""
import json
from pathlib import Path
from collections import defaultdict

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

# Load training set
with open(DATA/'p016_train_v5_0.json') as f: train = json.load(f)
items = train['splits']['train']

# Load evidence v3.5 (abstract-based)
with open(DATA/'from_workspace_tmp/p016_evidence_v3_5.json') as f: ev = json.load(f)
ev_by_pmid = {}
for r in ev['results']:
    pid = str(r.get('pmid',''))
    if pid and pid.isdigit(): ev_by_pmid[pid] = r

# Load Scopus evidence
with open(DATA/'from_workspace_tmp/p016_evidence_v3_5_scopus.json') as f: sc = json.load(f)
sc_by_pmid = {}
for r in sc['results']:
    pid = str(r.get('pmid',''))
    if pid and pid.isdigit(): sc_by_pmid[pid] = r

# Fix source field for each training item
fixed = 0
for it in items:
    pid = str(it.get('pmid',''))
    if not pid or not pid.isdigit():
        # Try to infer PMID from drug+disease context
        it['source'] = 'LabKG_manual_annotation'
        continue
    
    sources = []
    
    # 1. Evidence v3.5
    if pid in ev_by_pmid:
        ev_item = ev_by_pmid[pid]
        design = ev_item.get('confirmed_design','unknown')
        conclusion = ev_item.get('conclusion','unclear')
        rec = ev_item.get('recommendation','')
        sources.append(f"evidence_v3.5(design={design},conclusion={conclusion},rec={rec})")
    
    # 2. Scopus
    if pid in sc_by_pmid:
        sc_item = sc_by_pmid[pid]
        sc_design = sc_item.get('design','')
        sources.append(f"scopus(design={sc_design})")
    
    # 3. Training set build strategy
    strategy = it.get('source', '?')
    if strategy == '?':
        # Infer from label and evidence
        label = it.get('label', 0)
        if label == 1:
            strategy = 'literature_positive'
        else:
            strategy = 'negative_sampling_v5'
    
    # Compose final source string
    source_parts = [f"pmid:{pid}"]
    source_parts.extend(sources)
    source_parts.append(f"strategy:{strategy}")
    
    it['source'] = '|'.join(source_parts)
    fixed += 1

# Save fixed training set
out_path = DATA / 'p016_train_v5_1.json'
with open(out_path, 'w') as f:
    json.dump(train, f, indent=2)

# Report
sources = defaultdict(int)
for it in items:
    for part in it.get('source','').split('|'):
        if 'strategy:' in part:
            sources[part.replace('strategy:','')] += 1

print(f"Fixed {fixed}/{len(items)} training items")
print(f"Sources: {dict(sources)}")
print(f"→ {out_path}")
