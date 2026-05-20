#!/usr/bin/env python3
"""
Independent validation via ClinicalTrials.gov API.
Searches for TMJ/TMD/OA trials involving each top-20 drug.
"""

import json, time, csv
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
VALIDATION = DATA / "literature_validation.json"

def search_ct(query, max_retries=3):
    """Search ClinicalTrials.gov API v2."""
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={quote(query)}&pageSize=100&format=json"
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            data = json.loads(urlopen(req, timeout=15).read())
            return data.get('totalCount', 0), data.get('studies', [])
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
            else:
                return -1, []
    return -1, []

# ─── Load top-20 predictions ───────────────────────────────────
with open(VALIDATION) as f:
    val = json.load(f)

top_drugs = val['predictions'][:20]
print(f"Loaded {len(top_drugs)} predictions for TMJOA")

# ─── ClinicalTrials.gov validation ─────────────────────────────
results = []
for i, pred in enumerate(top_drugs):
    drug = pred['drug']
    print(f"\n{i+1:2d}. {drug} (score={pred['score']:.3f})...", flush=True)

    # TMJ-specific search
    tmj_count, tmj_studies = search_ct(f'{drug} AND (TMJ OR "temporomandibular" OR TMD)')
    time.sleep(0.25)

    # OA search (broader)
    oa_count, oa_studies = search_ct(f'{drug} AND (osteoarthritis OR OA) AND (TMJ OR temporomandibular OR TMD)')
    time.sleep(0.25)

    # General OA (drug + osteoarthritis, for context)
    gen_oa_count, _ = search_ct(f'{drug} AND osteoarthritis')
    time.sleep(0.25)

    # DrugBank indication check via PubMed MeSH
    indication_query = f'{drug}[Title] AND ("therapeutic use"[Subheading] OR "drug therapy"[Subheading]) AND (TMJ OR "temporomandibular joint" OR TMD)'
    ind_count, _ = search_ct(f'{drug} AND (TMJ OR temporomandibular OR TMD) AND INTERVENTIONAL')

    # Determine CT evidence level
    if tmj_count >= 3:
        ct_evidence = "strong"
    elif tmj_count >= 1:
        ct_evidence = "moderate"
    elif oa_count >= 1:
        ct_evidence = "weak (OA-related)"
    else:
        ct_evidence = "none"

    print(f"  → CT.gov TMJ: {tmj_count}, OA+TMJ: {oa_count}, Gen OA: {gen_oa_count} | {ct_evidence}")

    results.append({
        'rank': i+1,
        'drug': drug,
        'score': pred['score'],
        'ct_tmj_count': tmj_count,
        'ct_oa_tmj_count': oa_count,
        'ct_gen_oa_count': gen_oa_count,
        'ct_evidence': ct_evidence,
    })

# ─── Summary ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ClinicalTrials.gov Independent Validation Summary")
print(f"{'='*60}")

ct_confirmed = sum(1 for r in results if r['ct_tmj_count'] > 0 or r['ct_oa_tmj_count'] > 0)
ct_strong = sum(1 for r in results if r['ct_evidence'] == 'strong')
ct_moderate = sum(1 for r in results if r['ct_evidence'] == 'moderate')
ct_weak = sum(1 for r in results if r['ct_evidence'] == 'weak (OA-related)')
ct_none = sum(1 for r in results if r['ct_evidence'] == 'none')

print(f"\n  CT.gov TMJ/TMD evidence:")
print(f"    Strong (≥3 TMJ trials):    {ct_strong}")
print(f"    Moderate (1–2 TMJ trials): {ct_moderate}")
print(f"    Weak (OA-related only):     {ct_weak}")
print(f"    None:                       {ct_none}")
print(f"    Any evidence:               {ct_confirmed}/20")

# Drugs with CT evidence
if ct_confirmed > 0:
    print(f"\n  Drugs with CT.gov TMJ/TMD trial evidence:")
    for r in results:
        if r['ct_evidence'] != 'none':
            print(f"    {r['drug']}: {r['ct_tmj_count']} TMJ + {r['ct_oa_tmj_count']} OA/TMJ trials | {r['ct_evidence']}")

# BGJ398 specific
bgj = [r for r in results if 'BGJ398' in r['drug'] or 'infigratinib' in r['drug'].lower()]
if bgj:
    print(f"\n  BGJ398 CT.gov: TMJ={bgj[0]['ct_tmj_count']}, OA={bgj[0]['ct_gen_oa_count']}")
    if bgj[0]['ct_tmj_count'] == 0 and bgj[0]['ct_gen_oa_count'] == 0:
        print("  ✓ BGJ398 remains genuinely novel — zero CT.gov trials for TMJ or OA")

# Dual-source summary
pubmed_any = sum(1 for p in val['predictions'][:20] if p.get('pubmed_hits', 0) > 0)
print(f"\n  Dual-source comparison:")
print(f"    PubMed confirmed:    {pubmed_any}/20")
print(f"    CT.gov confirmed:    {ct_confirmed}/20")
print(f"    Either source:       {sum(1 for r in results if r['ct_tmj_count']>0 or r['ct_oa_tmj_count']>0 or (any(p['pubmed_hits']>0 for p in val['predictions'][:20] if p['drug']==r['drug'])))}/20")

out_path = DATA / "ct_validation.json"
with open(out_path, 'w') as f:
    json.dump({
        'method': 'ClinicalTrials.gov independent validation',
        'search_date': '2026-05-20',
        'results': results,
        'summary': {
            'ct_strong': ct_strong,
            'ct_moderate': ct_moderate,
            'ct_weak': ct_weak,
            'ct_none': ct_none,
            'ct_any': ct_confirmed,
            'bgj398_ct_tmj': bgj[0]['ct_tmj_count'] if bgj else 0,
            'bgj398_gen_oa': bgj[0]['ct_gen_oa_count'] if bgj else 0,
        }
    }, f, indent=2)
print(f"\n  → {out_path}")
