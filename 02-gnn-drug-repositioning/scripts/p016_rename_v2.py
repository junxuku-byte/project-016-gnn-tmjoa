#!/usr/bin/env python3
"""Extract PMID from full PDF text, rename to PMID_XXXXXXXX_Keyword.pdf"""
import re, subprocess, json, shutil
from pathlib import Path

PDF_DIR = Path.home() / 'morph-lab/projects/project-016-gnn-drug-repositioning/00-文献'
TMP = PDF_DIR / 'pdf_extractions'; TMP.mkdir(exist_ok=True)

KEY_TERMS = [
    'TMJOA','TMJ','TMD','osteoarthritis','cartilage','Hyaluronic','Glucosamine',
    'Chondroitin','Corticosteroid','PRP','NSAID','Metformin','Riluzole','Statin',
    'Denosumab','FGF18','FGF23','FGFR','BGJ398','MMP13','BMP','Wnt','TGF','Nell1',
    'Dextrose','Prolotherapy','Capsaicin','Buprenorphine','Diclofenac','Tenoxicam',
    'RCT','Meta','Systematic','Review','Cohort','Animal','Mouse','Rat','Rabbit',
    'Arthrocentesis','Injection','MSC','StemCell','Chondrocyte','Osteocyte',
    'GNN','Graph','Knowledge','Drug','Repositioning','scRNA','transcriptomic',
    'CKD','bone','osteoporosis','Collagen','MMP','ADAMTS','NFkB','MAPK',
    'Fmoc','Hydrogel','Peptide','Nanoparticle','Exosome',
    'Botox','Gabapentin','Amitriptyline','Ibuprofen','Celecoxib',
    'Trained','Immunity','immunoregulation','Condylectomy',
]

results = []
for pdf_path in sorted(PDF_DIR.glob('*.pdf')):
    sz = pdf_path.stat().st_size
    if sz < 10000:
        print(f"  SKIP (small): {pdf_path.name}")
        continue
    
    print(f"  Processing: {pdf_path.name[:60]}...", end=" ", flush=True)
    
    try:
        text = subprocess.run(['pdftotext', str(pdf_path), '-'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        print(f"ERROR: {e}")
        continue
    
    # Find PMID in full text
    pmid = None
    for pat in [r'PMID[:\s]*(\d{8})', r'PubMed\s*(?:ID|PMID)[:\s]*(\d{8})',
                r'pubmed\s*[\/:]?\s*(\d{8})']:
        for m in re.finditer(pat, text, re.IGNORECASE):
            pid = m.group(1)
            if pid and pid[0] in '1234':
                pmid = pid
                break
        if pmid: break
    
    if not pmid:
        # Search for DOI and try to extract PMID
        m = re.search(r'10\.\d{4,}/[^\s\)]+', text)
        if m:
            doi = m.group(0).rstrip('.,;')
        else:
            doi = None
        
        # Search Google Scholar watermark for original source
        m = re.search(r'pubmed\s*[\/:]?\s*(\d{8})', text, re.IGNORECASE)
        if m: pmid = m.group(1)
    
    # Extract title
    lines = [l.strip() for l in text.split('\n')[:30] if l.strip()]
    title = None
    for line in lines:
        if 30 < len(line) < 300 and line[0].isalpha():
            skip = any(w in line.lower() for w in [
                'copyright','license','http','doi','google','scholar','received:','accepted:',
                'published','correspondence','department','university','email','open access',
                'editor:','citation:','figure','table','supplement','data availab',
                'conflict of','funding','acknowledgment','reference','author contribution',
                'ethical','keyword','introduction','method','result','background',
                'send orders','reprints@','eissn','issn','current rheumatology',
                'journal of','oral science','bentham','springer','elsevier','wiley'])
            if not skip:
                title = line
                break
    
    # Generate keywords
    kw = []
    if title:
        tl = title.lower()
        for term in KEY_TERMS:
            if term.lower() in tl and term not in kw:
                kw.append(term)
    if not kw:
        kw = ['Paper']
    
    keyword = '_'.join(kw[:3])
    old = pdf_path.name
    
    if pmid:
        new = f"PMID_{pmid}_{keyword}.pdf"
    elif title:
        clean = re.sub(r'[^a-zA-Z0-9_\-]', '', title.replace(' ','_'))[:70]
        new = f"NOPMID_{clean}.pdf"
    else:
        new = f"UNREADABLE_{hash(text[:100])}.pdf"
    
    # Save extraction
    with open(TMP / f'FULLTEXT_{pmid or "NOPMID"}_{pdf_path.stem[:30]}.txt', 'w') as f:
        f.write(text)
    
    results.append((pmid, title, keyword, old, new, pdf_path, len(text)))
    flag = 'PMID' if pmid else 'NOPM'
    short_title = (title or '?')[:60]
    print(f"[{flag}] pmid={pmid or '?'} → {new}")
    print(f"       Title: {short_title}")

# Execute renames
renamed = 0
skipped = 0
for pmid, title, kw, old, new, path, sz in results:
    if old == new: continue
    target = PDF_DIR / new
    if target.exists():
        # Same content? Skip
        if target.stat().st_size == sz:
            skipped += 1
            continue
        # Different content, add suffix
        new = new.replace('.pdf', f'_v2.pdf')
        target = PDF_DIR / new
    shutil.move(str(path), str(target))
    renamed += 1
    print(f"  RENAMED: {old[:50]} → {new}")

# Final stats
has_pmid = sum(1 for r in results if r[0])
print(f"\n{'='*60}")
print(f"Total: {len(results)} PDFs")
print(f"  With PMID: {has_pmid}")
print(f"  Without PMID: {len(results)-has_pmid}")
print(f"  Renamed: {renamed}")
print(f"  Skipped (dup): {skipped}")

# Save manifest
manifest = [{'pmid': r[0], 'title': r[1], 'keywords': r[2], 'filename': r[4],
             'chars': r[6]} for r in results]
with open(PDF_DIR / 'pdf_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"→ pdf_manifest.json")
