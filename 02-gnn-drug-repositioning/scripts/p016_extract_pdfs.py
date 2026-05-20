#!/usr/bin/env python3
"""
Project-016 关键文献PDF精读提取
提取6篇已下载PDF的全文文本
"""

import pdfplumber
import json
from pathlib import Path

PDF_DIR = Path("/Users/junxuku/.openclaw/workspace/.tmp/p016_fulltext_pdfs")
OUTPUT_DIR = Path("/Users/junxuku/.openclaw/workspace/.tmp/p016_pdf_extractions")
OUTPUT_DIR.mkdir(exist_ok=True)

# 关键文献映射
KEY_PAPERS = {
    "PMID_29244893.pdf": {
        "pmid": "29244893",
        "title": "Effectiveness of Intra-Articular Injections of Sodium Hyaluronate or Corticosteroids for TMD",
        "priority": "HIGH",
        "question": "HA vs CS efficacy for TMD"
    },
    "PMID_40273050_PLoS.pdf": {
        "pmid": "40273050",
        "title": "FGF18 induces chondrogenesis and anti-osteoarthritic effects in a mouse model for TMJ degeneration",
        "priority": "HIGH",
        "question": "FGF18 as novel TMJOA therapeutic target"
    },
    "PMID_30814387_Unpaywall.pdf": {
        "pmid": "30814387",
        "title": "Long-term effectiveness of arthrocentesis with and without hyaluronic acid injection",
        "priority": "HIGH",
        "question": "HA add-on value to arthrocentesis"
    },
    "PMID_37608244_OA.pdf": {
        "pmid": "37608244",
        "title": "Hyaluronic acid injections for pain relief and functional improvement in TMD patients",
        "priority": "HIGH",
        "question": "HA umbrella review TMD"
    },
    "PMID_28879245.pdf": {
        "pmid": "28879245",
        "title": "Effects of High-Dose Capsaicin on TMD Subjects: A Randomized Clinical Study",
        "priority": "MEDIUM",
        "question": "Capsaicin efficacy for TMD"
    },
    "PMID_36890529.pdf": {
        "pmid": "36890529",
        "title": "Does intra-articular injection of tenoxicam after arthrocentesis heal outcomes of TMD",
        "priority": "MEDIUM",
        "question": "Tenoxicam post-arthrocentesis TMD"
    },
}

def extract_pdf_text(pdf_path, max_pages=20):
    """提取PDF文本，限制页数"""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = min(len(pdf.pages), max_pages)
            for i, page in enumerate(pdf.pages[:total]):
                text = page.extract_text()
                if text:
                    text_parts.append(f"\n--- Page {i+1} ---\n{text}")
        return "\n".join(text_parts)
    except Exception as e:
        return f"[ERROR extracting PDF: {e}]"

def main():
    results = {}
    
    for filename, meta in KEY_PAPERS.items():
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            # Try alternative names
            alt_patterns = [
                f"PMID{meta['pmid']}*.pdf",
                f"*{meta['pmid']}*.pdf"
            ]
            found = False
            for pattern in alt_patterns:
                matches = list(PDF_DIR.glob(pattern))
                if matches:
                    pdf_path = matches[0]
                    found = True
                    break
            if not found:
                print(f"❌ PDF not found: {filename}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Extracting: {meta['title'][:50]}...")
        print(f"PMID: {meta['pmid']} | File: {pdf_path.name}")
        print(f"{'='*60}")
        
        text = extract_pdf_text(pdf_path, max_pages=15)
        
        # Save full text
        out_file = OUTPUT_DIR / f"PMID_{meta['pmid']}_fulltext.txt"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"# {meta['title']}\n")
            f.write(f"# PMID: {meta['pmid']}\n")
            f.write(f"# Priority: {meta['priority']}\n")
            f.write(f"# Question: {meta['question']}\n")
            f.write(f"# File: {pdf_path.name}\n")
            f.write(f"# Text Length: {len(text)} chars\n")
            f.write("="*60 + "\n\n")
            f.write(text)
        
        print(f"  ✅ Extracted {len(text)} chars -> {out_file.name}")
        
        # Store for analysis
        results[meta['pmid']] = {
            **meta,
            'text_length': len(text),
            'text_preview': text[:2000],
            'fulltext_path': str(out_file)
        }
    
    # Save extraction log
    with open(OUTPUT_DIR / "extraction_log.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Extraction complete: {len(results)} PDFs")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
