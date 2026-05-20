#!/usr/bin/env python3
"""
PubMed E-utilities PMID harvest for Project-016
8 queries → PMID lists → deduplicate → JSON output
"""

import json
import time
import urllib.request
import urllib.parse
import os
from datetime import datetime
from pathlib import Path

# ── config ──────────────────────────────────────────────
API_KEY = None  # not in .env.apis; fall back to slower rate
EMAIL = "junxuku@yeah.net"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUT_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-chip-tmjoa/00-文献")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    ('(\"clonal hematopoiesis\"[Title/Abstract] OR \"CHIP\"[Title/Abstract]) AND (\"temporomandibular\"[Title/Abstract] OR \"TMJ\"[Title/Abstract])', 1),
    ('(\"clonal hematopoiesis\"[Title/Abstract] OR \"CHIP\"[Title/Abstract]) AND (\"osteoarthritis\"[Title/Abstract] OR \"OA\"[Title/Abstract])', 2),
    ('(\"DNMT3A\"[Title/Abstract] OR \"TET2\"[Title/Abstract] OR \"ASXL1\"[Title/Abstract]) AND (\"bone\"[Title/Abstract] OR \"osteoporosis\"[Title/Abstract] OR \"osteoblast\"[Title/Abstract])', 3),
    ('(\"menopause\"[Title/Abstract] OR \"estrogen\"[Title/Abstract] OR \"postmenopausal\"[Title/Abstract]) AND (\"temporomandibular\"[Title/Abstract] OR \"TMJ\"[Title/Abstract])', 4),
    ('(\"temporomandibular\"[Title/Abstract] OR \"TMJ\"[Title/Abstract]) AND (\"osteoarthritis\"[Title/Abstract] OR \"OA\"[Title/Abstract]) AND (\"drug\"[Title/Abstract] OR \"therapy\"[Title/Abstract] OR \"treatment\"[Title/Abstract])', 5),
    ('(\"temporomandibular\"[Title/Abstract] OR \"TMJ\"[Title/Abstract]) AND (\"osteoarthritis\"[Title/Abstract] OR \"OA\"[Title/Abstract]) AND (\"mechanism\"[Title/Abstract] OR \"pathway\"[Title/Abstract] OR \"signaling\"[Title/Abstract])', 6),
    ('(\"FGF23\"[Title/Abstract]) AND (\"osteoarthritis\"[Title/Abstract] OR \"OA\"[Title/Abstract] OR \"cartilage\"[Title/Abstract])', 7),
    ('(\"sclerostin\"[Title/Abstract] OR \"SOST\"[Title/Abstract] OR \"RANKL\"[Title/Abstract] OR \"TNFSF11\"[Title/Abstract] OR \"DKK1\"[Title/Abstract]) AND (\"osteoarthritis\"[Title/Abstract] OR \"OA\"[Title/Abstract])', 8),
]

# ── helpers ─────────────────────────────────────────────
def esearch(query, retmax=100000):
    """Return list of PMIDs for a query via esearch.fcgi"""
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "email": EMAIL,
    }
    if API_KEY:
        params["api_key"] = API_KEY

    url = f"{BASE}/esearch.fcgi?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "OpenClaw-PubMedBot/1.0"})
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    
    return data

# ── run searches ────────────────────────────────────────
results = []
all_pmids = set()
log = []

# Rate-limit: 0.4s between requests (safe for no-key: ~2.5 req/s)
DELAY = 0.4

for query, qid in QUERIES:
    entry = {
        "retrieval_id": qid,
        "query": query,
        "pmids": [],
        "count": 0,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        print(f"[{qid}/8] Searching: {query[:60]}...")
        data = esearch(query)
        
        count = data.get("esearchresult", {}).get("count", 0)
        pmids = data.get("esearchresult", {}).get("idlist", [])
        
        entry["count"] = int(count)
        entry["pmids"] = pmids
        entry["status"] = "ok"
        
        all_pmids.update(pmids)
        
        log.append({
            "qid": qid,
            "status": "ok",
            "count": int(count),
            "pmids_returned": len(pmids),
            "timestamp": entry["timestamp"],
        })
        
        print(f"  → {len(pmids)} PMIDs (count={count})")
        
    except Exception as e:
        entry["status"] = "error"
        entry["error"] = str(e)
        log.append({
            "qid": qid,
            "status": "error",
            "error": str(e),
            "timestamp": entry["timestamp"],
        })
        print(f"  → ERROR: {e}")
    
    results.append(entry)
    time.sleep(DELAY)

# ── deduplicate ─────────────────────────────────────────
unique_pmids = sorted(all_pmids, key=int)
total_unique = len(unique_pmids)

# ── year distribution (efetch summaries) ────────────────
# To get year distribution, we need to fetch summaries for all PMIDs.
# We'll batch in chunks of 200 to stay within URL limits.
year_dist = {}
BATCH = 200

print(f"\nFetching year data for {total_unique} unique PMIDs in batches of {BATCH}...")

for i in range(0, total_unique, BATCH):
    batch = unique_pmids[i:i+BATCH]
    ids = ",".join(batch)
    
    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "json",
        "email": EMAIL,
    }
    if API_KEY:
        params["api_key"] = API_KEY
    
    url = f"{BASE}/esummary.fcgi?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "OpenClaw-PubMedBot/1.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            summary = json.loads(resp.read().decode())
        
        for uid, info in summary.get("result", {}).items():
            if uid == "uids":
                continue
            pubdate = info.get("pubdate", "")
            # Extract year from pubdate string like "2024 Jan", "2024", "2024 Jan 15"
            year = pubdate[:4] if pubdate and pubdate[:4].isdigit() else "unknown"
            if 2015 <= int(year) <= 2026:
                year_dist[year] = year_dist.get(year, 0) + 1
            elif year != "unknown":
                year_dist["other"] = year_dist.get("other", 0) + 1
    except Exception as e:
        log.append({
            "phase": "year_fetch",
            "batch": f"{i}-{i+len(batch)}",
            "error": str(e),
        })
        print(f"  Batch {i}-{i+len(batch)} error: {e}")
    
    time.sleep(DELAY)
    if (i // BATCH + 1) % 10 == 0:
        print(f"  ...processed {min(i+BATCH, total_unique)}/{total_unique}")

# ── save outputs ────────────────────────────────────────
# 1. pmids_1535.json
output = {
    "queries": results,
    "total_unique": total_unique,
    "pmids": unique_pmids,
    "year_distribution": year_dist,
    "generated_at": datetime.now().isoformat(),
    "previous_count": 1535,
}

out_path = OUT_DIR / "pmids_1535.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# 2. search_log.json
log_path = OUT_DIR / "search_log.json"
with open(log_path, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "entries": log,
        "summary": {
            "queries_run": len(QUERIES),
            "successful": sum(1 for r in results if r["status"] == "ok"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "total_unique_pmids": total_unique,
            "previous_count": 1535,
        }
    }, f, indent=2)

# ── report ──────────────────────────────────────────────
print("\n" + "="*60)
print("PUBMED PMID HARVEST REPORT — Project-016")
print("="*60)
for r in results:
    status_icon = "✓" if r["status"] == "ok" else "✗"
    print(f"  {status_icon} Q{r['retrieval_id']}: {r['count']} hits")

print(f"\nTotal unique PMIDs: {total_unique}")
print(f"Previous count:     1535")
print(f"Difference:         {total_unique - 1535:+d}")

print("\nYear distribution (2015-2026):")
for y in sorted(year_dist.keys()):
    if y != "other":
        print(f"  {y}: {year_dist[y]}")
if "other" in year_dist:
    print(f"  other: {year_dist['other']}")

print(f"\nSaved: {out_path}")
print(f"Log:   {log_path}")
