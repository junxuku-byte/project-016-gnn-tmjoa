#!/usr/bin/env python3
"""Look up PMID by title via PubMed E-utilities, rename PDFs with correct PMID."""
import re, subprocess, json, time, shutil, os
from urllib.request import urlopen, Request
from urllib.parse import quote
from pathlib import Path

PDF_DIR = Path.home() / 'morph-lab/projects/project-016-gnn-drug-repositioning/00-文献'
TMP = PDF_DIR / 'pdf_extractions'; TMP.mkdir(exist_ok=True)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

ENTREZ = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

def lookup_pmid_by_title(title):
    """Search PubMed by title, return PMID."""
    # Clean title for search
    clean = re.sub(r'[^\w\s]', ' ', title)
    clean = re.sub(r'\s+', ' ', clean).strip()[:200]
    
    try:
        url = f"{ENTREZ}/esearch.fcgi?db=pubmed&term={quote(clean)}[ti]&retmax=2&retmode=json"
        req = Request(url, headers={'User-Agent': 'p016/1.0'})
        data = json.loads(urlopen(req, timeout=10).read())
        ids = data.get('esearchresult',{}).get('idlist',[])
        if ids:
            return ids[0]
    except Exception as e:
        pass
    return None

def extract_title(text):
    """Extract title from PDF text (first 30 lines)."""
    lines = [l.strip() for l in text.split('\n')[:30] if l.strip()]
    for line in lines:
        if 40 < len(line) < 300 and line[0].isalpha():
            skip_words = ['copyright','license','http','doi','google','scholar',
                'received:','accepted:','published','correspondence','department',
                'university','email','open access','editor:','citation:','figure',
                'table','supplement','data availab','conflict of','funding',
                'acknowledgment','reference','author contribution','ethical',
                'keyword','introduction','method','result','background',
                'send orders','reprints@','eissn','issn','current rheumatology',
                'journal of','oral science','bentham','springer','elsevier','wiley',
                'creative commons','distributed under','terms of the']
            if not any(w in line.lower() for w in skip_words):
                return line
    return lines[0] if lines else None

results = []
for i, pdf_path in enumerate(sorted(PDF_DIR.glob('*.pdf'))):
    if pdf_path.stat().st_size < 10000: continue
    
    print(f"{i+1:2d}. {pdf_path.name[:55]}...", flush=True)
    
    try:
        text = subprocess.run(['pdftotext', '-l', '2', str(pdf_path), '-'],
                             capture_output=True, text=True, timeout=15).stdout
    except: continue
    
    title = extract_title(text)
    if not title:
        print(f"    ✗ No title found")
        results.append((None, None, pdf_path, ''))
        continue
    
    # Look up PMID
    pmid = lookup_pmid_by_title(title)
    time.sleep(0.3)
    
    if pmid:
        print(f"    ✓ PMID={pmid} | {title[:80]}")
    else:
        print(f"    ✗ No PMID | {title[:80]}")
    
    results.append((pmid, title, pdf_path, text))

# Rename
renamed = 0
for pmid, title, path, text in results:
    if not pmid: continue
    old = path.name
    new = f"PMID_{pmid}.pdf"
    if old == new: continue
    
    target = PDF_DIR / new
    if target.exists():
        # Verify same file
        if target.stat().st_size == path.stat().st_size:
            path.unlink()  # Remove duplicate
            print(f"  DUP: {old} (removed, same as {new})")
            continue
    
    shutil.move(str(path), str(target))
    renamed += 1
    print(f"  RENAMED: {old[:50]} → {new}")

has_pmid = sum(1 for r in results if r[0])
print(f"\n{'='*60}")
print(f"Total: {len(results)} PDFs")
print(f"PMID found: {has_pmid}")
print(f"Renamed: {renamed}")

# Save manifest
manifest = [{'pmid': r[0], 'title': r[1], 'old_name': r[2].name} for r in results]
with open(PDF_DIR / 'pdf_manifest_v2.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print("→ pdf_manifest_v2.json")
