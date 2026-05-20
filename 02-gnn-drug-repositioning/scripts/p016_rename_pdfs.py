#!/usr/bin/env python3
"""Extract PMID+title from PDF metadata + full text search. Rename to PMID_XXXXXXXX_Keyword.pdf"""
import re, json, sys, subprocess
from pathlib import Path
from PyPDF2 import PdfReader

PDF_DIR = Path.home() / 'morph-lab/projects/project-016-gnn-drug-repositioning/00-文献'
KEY_TERMS = [
    'TMJOA','TMJ','TMD','osteoarthritis','cartilage','Hyaluronic','Glucosamine',
    'Chondroitin','Corticosteroid','PRP','NSAID','Metformin','Riluzole','Statin',
    'Denosumab','FGF18','FGF23','FGFR','BGJ398','MMP13','BMP','Wnt','TGF',
    'Dextrose','Prolotherapy','Capsaicin','Buprenorphine','RCT','Meta',
    'Systematic','Review','Cohort','Animal','Mouse','Rat','Injection',
    'Arthrocentesis','GNN','Graph','Knowledge','Drug','Repositioning',
    'Osteocyte','Chondrocyte','MSC','StemCell','single-cell','transcriptomic',
    'Nell-1','FCSC','training','immunity','CKD','bone','osteoporosis',
    'Tenoxicam','Bupivacaine','Morphine','Diclofenac','Ketorolac',
    'Ibuprofen','Celecoxib','Naproxen','Gabapentin','Amitriptyline',
    'Botox','Collagen','Platelet','Plasma','Laser','Ultrasound',
]
LOWER_TERMS = [t.lower() for t in KEY_TERMS]

def gen_keyword(title):
    title_lower = title.lower()
    keywords = []
    for term in LOWER_TERMS:
        if term in title_lower and term not in keywords:
            k = term.upper() if term == term.upper() else term.title()
            keywords.append(k)
    if not keywords:
        words = [w for w in re.findall(r'[A-Za-z]{5,}', title) if w.lower() not in
                 ('the','with','from','into','after','during','using','based','their','which','through','between','among','effect','effects','treatment','patients','study','results','analysis','role')]
        keywords = words[:2]
    return '_'.join(keywords[:3])

def extract_pmid_from_text(text):
    for pat in [r'PMID[:\s]*(\d{8})', r'PubMed\s*ID[:\s]*(\d{8})',
                r'pmid[:\s]*(\d{8})', r'pubmed[:\s]*(\d{8})']:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1)
    return None

results = []
for i, pdf_path in enumerate(sorted(PDF_DIR.glob('*.pdf'))):
    if pdf_path.stat().st_size < 10000:
        continue
    
    print(f"  {i+1}: {pdf_path.name[:60]}...", flush=True)
    
    try:
        reader = PdfReader(str(pdf_path))
        if len(reader.pages) == 0: continue
        
        # Strategy 1: PDF metadata
        meta = reader.metadata
        title = None
        if meta and meta.title:
            title = str(meta.title).strip()
            # Clean Google Scholar prefix
            title = re.sub(r'^[a-f0-9]{40,}\s*', '', title)
        
        # Strategy 2: Search ALL pages for PMID
        pmid = None
        for page in reader.pages[:3]:  # Check first 3 pages
            text = page.extract_text()
            if text:
                pmid = extract_pmid_from_text(text)
                if pmid: break
                # Try to find title if not from metadata
                if not title:
                    # Look for longest line in first page that seems like a title
                    for line in text.split('\n')[:15]:
                        line = line.strip()
                        if 40 < len(line) < 300 and not any(w in line.lower() for w in
                            ['copyright','license','permission','http','doi','google','scholar',
                             'reproduced','distributed','transmit','exceeds','prohibited','watermark']):
                            alpha_r = sum(1 for c in line if c.isalpha() or c.isspace()) / max(len(line),1)
                            if alpha_r > 0.7 and line[0].isalpha():
                                title = line
                                break
        
        if not pmid:
            # Strategy 3: Try pdfinfo (command line)
            try:
                out = subprocess.run(['pdfinfo', str(pdf_path)], capture_output=True, text=True, timeout=5)
                for line in out.stdout.split('\n'):
                    if line.startswith('Title:'):
                        t = line.replace('Title:','').strip()
                        if t and len(t) > 20:
                            title = t
            except: pass
        
        if title:
            title = title.strip()
            if len(title) > 200:
                title = title[:200]
        
        kw = gen_keyword(title) if title else 'Unknown'
        old = pdf_path.name
        
        if pmid:
            new = f"PMID_{pmid}_{kw}.pdf"
        elif title:
            clean = re.sub(r'[^a-zA-Z0-9_\-]', '', title.replace(' ','_'))[:80]
            new = f"NOPMID_{clean}.pdf"
        else:
            new = f"NOPMID_UNKNOWN_{i}.pdf"
        
        results.append((pmid or 'NONE', title or '?', kw, old, new, pdf_path))
        print(f"    PMID={pmid or 'NONE'}  Kw={kw}")
        
    except Exception as e:
        print(f"    ✗ Error: {e}", flush=True)

# Print summary and execute renames
print(f"\n{'='*60}")
print(f"Summary: {len(results)} PDFs")
has_pmid = sum(1 for r in results if r[0] != 'NONE')
print(f"  With PMID: {has_pmid}")
print(f"  Without PMID: {len(results) - has_pmid}")
print()

exec_renames = '--execute' in sys.argv

for pmid, title, kw, old, new, path in sorted(results, key=lambda x: x[1]):
    flag = 'PMID' if pmid != 'NONE' else 'NOPM'
    print(f"  [{flag}] {old[:55]:55s} → {new}")
    
    if exec_renames and old != new:
        target = PDF_DIR / new
        if not target.exists():
            path.rename(target)
            print(f"    RENAMED")
        else:
            print(f"    SKIP: target exists")

if not exec_renames:
    print(f"\nRun with --execute to apply renames.")
    print(f"  python3 scripts/p016_rename_pdfs.py --execute")
else:
    final = list(PDF_DIR.glob('PMID_*.pdf'))
    print(f"\nRenamed. PMID-named PDFs: {len(final)}")
