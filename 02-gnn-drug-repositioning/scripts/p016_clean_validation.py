#!/usr/bin/env python3
"""Re-run literature validation excluding all graph-construction PMIDs."""
import json, time, os, xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import quote
from pathlib import Path

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
ENTREZ = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

# ── Load graph PMIDs to exclude ──
print("Loading graph PMIDs...", flush=True)
graph_pmids = set()

with open(DATA/'p016_train_v5_0.json') as f: train = json.load(f)
for it in train['splits']['train']:
    pid = str(it.get('pmid',''))
    if pid and pid.isdigit(): graph_pmids.add(pid)

with open(DATA/'from_workspace_tmp/p016_evidence_v3_5.json') as f: ev = json.load(f)
for r in ev['results']:
    pid = str(r.get('pmid',''))
    if pid and pid.isdigit(): graph_pmids.add(pid)

print(f"  Graph PMIDs to exclude: {len(graph_pmids)}")

# ── Load top-20 predictions ──
with open(DATA/'literature_validation.json') as f: val = json.load(f)
top_drugs = [p['drug'] for p in val['predictions'][:20]]
print(f"  Top-20 drugs: {len(top_drugs)}")

# ── Re-run searches ──
results = []
for i, drug in enumerate(top_drugs):
    query = f'{drug} AND ("temporomandibular joint" OR "TMJ" OR "TMD")'
    print(f"  {i+1:2d}. {drug[:30]:30s} ...", end=" ", flush=True)

    all_pmids = set()
    excluded = set()

    # Fetch all PMIDs (paginated)
    try:
        # First get count
        xml = urlopen(Request(
            f'{ENTREZ}/esearch.fcgi?db=pubmed&term={quote(query)}&retmax=0&usehistory=y',
            headers={'User-Agent': 'p016/1.0'}
        ), timeout=15).read()
        root = ET.fromstring(xml)
        total = int(root.findtext('.//Count', '0'))
        webenv = root.findtext('.//WebEnv', '')
        querykey = root.findtext('.//QueryKey', '')

        # Fetch in batches
        batch = 200
        for start in range(0, min(total, 5000), batch):
            xml = urlopen(Request(
                f'{ENTREZ}/efetch.fcgi?db=pubmed&query_key={querykey}&WebEnv={webenv}'
                f'&retstart={start}&retmax={batch}&retmode=xml',
                headers={'User-Agent': 'p016/1.0'}
            ), timeout=30).read()
            root = ET.fromstring(xml)
            for art in root.findall('.//PubmedArticle'):
                pid = art.findtext('.//PMID')
                if pid:
                    all_pmids.add(pid)
                    if pid in graph_pmids:
                        excluded.add(pid)

            time.sleep(0.4)

    except Exception as e:
        print(f"ERROR: {e}")
        results.append({'drug': drug, 'total': -1, 'excluded': -1, 'clean': -1})
        continue

    clean_count = len(all_pmids) - len(excluded)
    evidence = 'strong' if clean_count >= 10 else ('moderate' if clean_count >= 3 else ('weak' if clean_count >= 1 else 'none'))

    print(f"total={len(all_pmids)} excluded={len(excluded)} clean={clean_count} [{evidence}]")
    results.append({
        'drug': drug, 'total': len(all_pmids), 'excluded': len(excluded),
        'clean': clean_count, 'evidence_clean': evidence
    })

# ── Summary ──
print(f"\n{'='*70}")
print(f"Excluding {len(graph_pmids)} graph-construction PMIDs from validation")
print(f"{'='*70}\n")

print(f"{'Rank':<5} {'Drug':<30} {'Raw':>5} {'Excl':>5} {'Clean':>5} {'Evidence'}")
print(f"{'-'*60}")
clean_confirmed = 0
for i, r in enumerate(results):
    flag = '✓' if r['clean'] > 0 else '✗'
    print(f"{i+1:<5} {r['drug']:<30} {r['total']:>5} {r['excluded']:>5} {r['clean']:>5} {r['evidence_clean']}")
    if r['clean'] > 0: clean_confirmed += 1

print(f"\n  Before (raw):     19/20 confirmed")
print(f"  After (excl graph): {clean_confirmed}/20 confirmed")

# Evidence breakdown
strong = sum(1 for r in results if r['evidence_clean'] == 'strong')
moderate = sum(1 for r in results if r['evidence_clean'] == 'moderate')
weak = sum(1 for r in results if r['evidence_clean'] == 'weak')
none = sum(1 for r in results if r['evidence_clean'] == 'none')
print(f"  Strong (≥10):      {strong}")
print(f"  Moderate (3-9):    {moderate}")
print(f"  Weak (1-2):        {weak}")
print(f"  None (0):          {none}")

out_path = DATA / 'literature_validation_excl_graph.json'
with open(out_path, 'w') as f:
    json.dump({
        'method': 'PubMed search excluding graph-construction PMIDs',
        'graph_pmids_excluded': len(graph_pmids),
        'results': results,
        'summary': {
            'raw_confirmed': 19,
            'clean_confirmed': clean_confirmed,
            'strong': strong, 'moderate': moderate, 'weak': weak, 'none': none
        }
    }, f, indent=2)
print(f"\n  → {out_path}")
