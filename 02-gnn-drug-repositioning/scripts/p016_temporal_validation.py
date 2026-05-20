#!/usr/bin/env python3
"""
Temporal-split literature validation for Project-016.
Separates knowledge recovery (pre-graph-cutoff) from prospective validation (post-cutoff).
"""

import json, time
from pathlib import Path
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
TRAIN = DATA / "p016_train_v5_0.json"
VALIDATION = DATA / "literature_validation.json"

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "p016_temporal_validation"
EMAIL = "chongchong@tjh.tjmu.edu.cn"

def entrez_fetch(params):
    url = f"{ENTREZ_BASE}/efetch.fcgi?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": f"{TOOL}/1.0 ({EMAIL})"})
    time.sleep(0.35)  # Rate limit: ~3/sec
    return urlopen(req).read()

def entrez_search(params):
    url = f"{ENTREZ_BASE}/esearch.fcgi?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": f"{TOOL}/1.0 ({EMAIL})"})
    time.sleep(0.35)
    return urlopen(req).read()

# ─── Step 1: Get publication years for training PMIDs ────────────
print("Step 1: Fetching publication dates for training PMIDs...")

with open(TRAIN) as f:
    train = json.load(f)
items = train['splits']['train']

train_pmids = set()
for it in items:
    if 'pmid' in it and it['pmid']:
        train_pmids.add(it['pmid'])
    if 'evidence' in it and isinstance(it['evidence'], list):
        for ev in it['evidence']:
            if 'pmid' in ev and ev['pmid']:
                train_pmids.add(ev['pmid'])

print(f"  Training PMIDs: {len(train_pmids)}")

# Fetch publication dates
pub_years = {}
batch_size = 50
pmid_list = sorted(train_pmids)

for i in range(0, len(pmid_list), batch_size):
    batch = pmid_list[i:i+batch_size]
    try:
        xml = entrez_fetch({
            'db': 'pubmed', 'id': ','.join(batch),
            'retmode': 'xml', 'rettype': 'medline'
        })
        root = ET.fromstring(xml)
        for art in root.findall('.//PubmedArticle'):
            pmid = art.findtext('.//PMID')
            year = art.findtext('.//PubDate/Year') or art.findtext('.//ArticleDate/Year')
            if pmid and year:
                pub_years[pmid] = int(year)
    except Exception as e:
        print(f"  Error batch {i}: {e}")
        time.sleep(1)

print(f"  Fetched years for {len(pub_years)}/{len(train_pmids)} PMIDs")

# Determine cutoff
years = list(pub_years.values())
if years:
    max_year = max(years)
    min_year = min(years)
    print(f"  Year range: {min_year}–{max_year}")
    CUTOFF = max_year + 1  # Articles published the year after latest training PMID
else:
    print("  WARNING: No years found, using 2024 as cutoff")
    CUTOFF = 2024

print(f"  Temporal cutoff: {CUTOFF} (training data ends at year {max_year})")

# ─── Step 2: Re-run literature validation with temporal split ──
print(f"\nStep 2: Temporal-split literature validation (cutoff={CUTOFF})...")

with open(VALIDATION) as f:
    val = json.load(f)

top_drugs = val['predictions'][:20]

temporal_results = []
for i, pred in enumerate(top_drugs):
    drug = pred['drug']
    print(f"  {i+1:2d}. {drug}...", end=" ", flush=True)

    # Search with temporal filter (post-cutoff)
    query_post = f'{drug} AND ("temporomandibular joint" OR "TMJ" OR "TMD") AND {CUTOFF}:3000[dp]'
    try:
        xml = entrez_search({
            'db': 'pubmed', 'term': query_post,
            'retmax': '0', 'usehistory': 'n'
        })
        root = ET.fromstring(xml)
        count_post = int(root.findtext('.//Count', '0'))
    except:
        count_post = -1

    # Search without temporal filter (total)
    query_all = f'{drug} AND ("temporomandibular joint" OR "TMJ" OR "TMD")'
    try:
        xml = entrez_search({
            'db': 'pubmed', 'term': query_all,
            'retmax': '0', 'usehistory': 'n'
        })
        root = ET.fromstring(xml)
        count_all = int(root.findtext('.//Count', '0'))
    except:
        count_all = -1
        time.sleep(1)
        continue

    count_pre = max(0, count_all - count_post)

    print(f"pre={count_pre} post={count_post} (total={count_all})")

    temporal_results.append({
        'rank': i+1,
        'drug': drug,
        'score': pred['score'],
        'pubmed_total': count_all,
        'pubmed_pre_cutoff': count_pre,
        'pubmed_post_cutoff': count_post,
        'cutoff_year': CUTOFF,
    })

# ─── Step 3: Summarize ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Temporal-Split Validation Summary (Cutoff: {CUTOFF})")
print(f"{'='*60}")

pre_confirmed = sum(1 for r in temporal_results if r['pubmed_pre_cutoff'] > 0)
post_confirmed = sum(1 for r in temporal_results if r['pubmed_post_cutoff'] > 0)
any_confirmed = sum(1 for r in temporal_results if r['pubmed_total'] > 0)

print(f"\n  Pre-cutoff (< {CUTOFF}):  {pre_confirmed}/20 confirmed (knowledge recovery)")
print(f"  Post-cutoff (≥ {CUTOFF}): {post_confirmed}/20 confirmed (prospective signal)")
print(f"  Any evidence:             {any_confirmed}/20 confirmed")

# Evidence levels
post_strong = sum(1 for r in temporal_results if r['pubmed_post_cutoff'] >= 10)
post_moderate = sum(1 for r in temporal_results if 3 <= r['pubmed_post_cutoff'] < 10)
post_weak = sum(1 for r in temporal_results if 1 <= r['pubmed_post_cutoff'] < 3)
post_none = sum(1 for r in temporal_results if r['pubmed_post_cutoff'] == 0)

print(f"\n  Post-cutoff evidence breakdown:")
print(f"    Strong (≥10):   {post_strong}")
print(f"    Moderate (3–9):  {post_moderate}")
print(f"    Weak (1–2):      {post_weak}")
print(f"    None (0):        {post_none}")

# Novel candidates (post-cutoff > 0, pre-cutoff == 0)
novel = [r for r in temporal_results if r['pubmed_post_cutoff'] > 0 and r['pubmed_pre_cutoff'] == 0]
if novel:
    print(f"\n  Candidates with ONLY post-cutoff evidence (genuinely prospective):")
    for r in novel:
        print(f"    {r['drug']}: {r['pubmed_post_cutoff']} post-cutoff hit(s)")

# Save
out = {
    'method': 'temporal_split',
    'cutoff_year': CUTOFF,
    'training_pmid_years': {
        'count': len(pub_years),
        'min': min_year if years else None,
        'max': max_year if years else None,
    },
    'results': temporal_results,
    'summary': {
        'pre_cutoff_confirmed': pre_confirmed,
        'post_cutoff_confirmed': post_confirmed,
        'any_confirmed': any_confirmed,
        'post_strong': post_strong,
        'post_moderate': post_moderate,
        'post_weak': post_weak,
        'post_none': post_none,
    }
}

out_path = DATA / "temporal_validation.json"
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  → {out_path}")
