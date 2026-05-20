#!/usr/bin/env python3
"""Multi-source literature validation: OpenAlex + Scopus + PubMed (3-database cross-validation)."""
import json, time, os
from urllib.request import urlopen, Request
from urllib.parse import quote
from pathlib import Path

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

with open(DATA/'literature_validation.json') as f: val = json.load(f)
top_drugs = [p['drug'] for p in val['predictions'][:20]]
print(f"Top-20 drugs: {len(top_drugs)}")

# ── Load PubMed validation (already done) ──
with open(DATA/'literature_validation_excl_graph.json') as f: pub_res = json.load(f)

# ── OpenAlex ──────────────────────────────────────────────────
print("\n=== OpenAlex ===")
oa_results = []
for i, drug in enumerate(top_drugs):
    # OpenAlex search: works mentioning drug AND TMJ/TMD
    query = f'"{drug}" ("temporomandibular joint" OR TMJ OR TMD)'
    try:
        url = f'https://api.openalex.org/works?search={quote(query)}&per_page=1'
        req = Request(url, headers={'User-Agent': 'mailto:chongchong@tjh.tjmu.edu.cn'})
        data = json.loads(urlopen(req, timeout=15).read())
        count = data.get('meta',{}).get('count',0)
        time.sleep(0.15)
    except Exception as e:
        count = -1
        print(f"  ERROR: {e}")

    ev = 'strong' if count >= 10 else ('moderate' if count >= 3 else ('weak' if count >= 1 else 'none'))
    oa_results.append({'drug': drug, 'count': count, 'evidence': ev})
    print(f"  {i+1:2d}. {drug[:30]:30s} OpenAlex: {count} [{ev}]")

oa_confirmed = sum(1 for r in oa_results if r['count'] > 0)
print(f"\n  OpenAlex confirmed: {oa_confirmed}/20")

# ── Scopus ────────────────────────────────────────────────────
print("\n=== Scopus ===")
# Check if Scopus API key is available
from pathlib import Path as P
KEYCHAIN = P.home() / '.openclaw' / 'keychain' / 'scopus_api_key'
scopus_key = None
if KEYCHAIN.exists():
    with open(KEYCHAIN) as f: scopus_key = f.read().strip()
    print(f"  Scopus key: {'found' if scopus_key else 'missing'}")
else:
    print("  Scopus key not found in keychain")

sc_results = []
if scopus_key:
    for i, drug in enumerate(top_drugs):
        query = f'TITLE-ABS-KEY("{drug}") AND TITLE-ABS-KEY("temporomandibular joint" OR TMJ OR TMD)'
        try:
            url = f'https://api.elsevier.com/content/search/scopus?query={quote(query)}&count=0'
            req = Request(url, headers={
                'X-ELS-APIKey': scopus_key,
                'Accept': 'application/json',
                'User-Agent': 'p016/1.0'
            })
            data = json.loads(urlopen(req, timeout=15).read())
            count = int(data.get('search-results',{}).get('opensearch:totalResults',0))
            time.sleep(0.5)
        except Exception as e:
            count = -1
            print(f"  Scopus ERROR for {drug}: {e}")
        ev = 'strong' if count >= 10 else ('moderate' if count >= 3 else ('weak' if count >= 1 else 'none'))
        sc_results.append({'drug': drug, 'count': count, 'evidence': ev})
        print(f"  {i+1:2d}. {drug[:30]:30s} Scopus: {count} [{ev}]")
    sc_confirmed = sum(1 for r in sc_results if r['count'] > 0)
    print(f"\n  Scopus confirmed: {sc_confirmed}/20")
else:
    sc_confirmed = None

# ── Multi-source summary ──────────────────────────────────────
print(f"\n{'='*70}")
print("Multi-Source Cross-Validation Summary")
print(f"{'='*70}\n")
print(f"{'Drug':<30} {'PubMed':>8} {'OpenAlex':>10} {'Scopus':>8}  {'Consensus'}")
print(f"{'-'*70}")

cross_confirmed = 0
for i, drug in enumerate(top_drugs):
    pm = pub_res['results'][i]['clean']
    oa = oa_results[i]['count']
    sc = sc_results[i]['count'] if sc_results else '--'
    sc_str = f"{sc:>8}" if isinstance(sc, int) else f"{'--':>8}"
    pm_ok = pm > 0
    oa_ok = oa > 0
    sc_ok = sc > 0 if isinstance(sc, int) else None
    sources = sum([pm_ok, oa_ok, sc_ok if sc_ok is not None else 0])
    consensus = '✓✓✓' if sources >= 3 else ('✓✓' if sources >= 2 else ('✓' if sources >= 1 else '✗'))
    if sources >= 2: cross_confirmed += 1
    print(f"{drug:<30} {pm:>8} {oa:>10} {sc_str}  {consensus}")
    if i == 9: print(f"{'...':<30} {'...':>8} {'...':>10} {'...':>8}")

print(f"\n  PubMed confirmed:      {sum(1 for r in pub_res['results'] if r['clean']>0)}/20")
print(f"  OpenAlex confirmed:    {oa_confirmed}/20")
if sc_confirmed is not None:
    print(f"  Scopus confirmed:      {sc_confirmed}/20")
print(f"  ≥2 sources confirmed:  {cross_confirmed}/20")

# Save
with open(DATA/'multisource_validation.json', 'w') as f:
    json.dump({
        'pubmed_excl_graph': pub_res,
        'openalex': oa_results,
        'scopus': sc_results,
        'cross_confirmed_2plus': cross_confirmed
    }, f, indent=2)
print(f"\n  → multisource_validation.json")
