#!/usr/bin/env python3
"""
Project-016 关键精读文献PDF下载 - 聚焦版
只下载5篇最关键文献（TMJOA直接相关 + 结论争议）

关键文献优先级:
1. PMID 38821656 - NEGATIVE: 关节内注射=安慰剂 (已确认abstract review)
2. PMID 37608244 - HA umbrella review (方向不明，需确认)
3. PMID 30814387 - Long-term arthrocentesis±HA (已下载 ✅)
4. PMID 36890529 - Tenoxicam RCT (已下载 ✅)
5. PMID 29244893 - HA vs CS Meta-analysis (方向不明)
6. PMID 41084405 - NSAID+exercise RCT (abstract已确认positive)
"""

import json
import time
import urllib.request
from pathlib import Path

PMC_DIR = Path("/Users/junxuku/.openclaw/workspace/.tmp/p016_fulltext_pdfs")
PMC_DIR.mkdir(exist_ok=True)

CRITICAL_PAPERS = [
    ("38821656", "Intra-articular injections comparable to placebo NMA", "HIGH"),
    ("37608244", "HA injections umbrella review", "HIGH"),
    ("29244893", "HA vs CS injections meta-analysis", "MEDIUM"),
    ("41084405", "NSAID+exercise combined therapy", "MEDIUM"),
    ("28879245", "High-dose capsaicin TMD RCT", "MEDIUM"),
]

def try_pmc_pdf(pmid):
    """尝试从PMC下载PDF"""
    try:
        # 先查询PMC ID
        pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
        req = urllib.request.Request(pmc_api)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            records = data.get('records', [])
            if not records or not records[0].get('pmcid'):
                return False, "Not in PMC"
            pmcid = records[0]['pmcid']
        
        # 尝试下载PDF
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        req2 = urllib.request.Request(pdf_url, headers={
            'User-Agent': 'Mozilla/5.0 (Academic Research)',
            'Accept': 'application/pdf'
        })
        
        with urllib.request.urlopen(req2, timeout=60) as resp2:
            content = resp2.read()
            if len(content) > 10000 and content[:4] == b'%PDF':
                filename = PMC_DIR / f"PMID_{pmid}.pdf"
                with open(filename, 'wb') as f:
                    f.write(content)
                return True, f"PMC PDF ({len(content)/1024:.0f} KB)"
            else:
                return False, f"Invalid PDF ({len(content)} bytes)"
                
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"Error: {str(e)[:40]}"

def try_doi_download(pmid, title):
    """尝试通过Crossref找到DOI后下载"""
    try:
        import urllib.parse
        encoded = urllib.parse.quote(title[:80])
        url = f"https://api.crossref.org/works?query.title={encoded}&rows=1"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            items = data.get('message', {}).get('items', [])
            if not items or not items[0].get('DOI'):
                return False, "No DOI found"
            doi = items[0]['DOI']
        
        # 尝试Unpaywall
        uw_url = f"https://api.unpaywall.org/v2/{doi}?email=chongchong@tjh.tjmu.edu.cn"
        req2 = urllib.request.Request(uw_url, headers={'Accept': 'application/json'})
        
        with urllib.request.urlopen(req2, timeout=15) as resp2:
            data2 = json.loads(resp2.read().decode())
            best = data2.get('best_oa_location', {})
            if best and best.get('url_for_pdf'):
                pdf_url = best['url_for_pdf']
                req3 = urllib.request.Request(pdf_url, headers={
                    'User-Agent': 'Mozilla/5.0 (Academic Research)',
                })
                
                with urllib.request.urlopen(req3, timeout=60) as resp3:
                    content = resp3.read()
                    if len(content) > 10000:
                        filename = PMC_DIR / f"PMID_{pmid}_OA.pdf"
                        with open(filename, 'wb') as f:
                            f.write(content)
                        return True, f"Unpaywall OA ({len(content)/1024:.0f} KB)"
        
        return False, "No OA PDF"
        
    except Exception as e:
        return False, f"DOI error: {str(e)[:40]}"

def main():
    print("=" * 60)
    print("Project-016 关键精读文献PDF下载 (聚焦版)")
    print("=" * 60)
    
    results = []
    
    for i, (pmid, title, priority) in enumerate(CRITICAL_PAPERS, 1):
        print(f"\n{i}. PMID {pmid} [{priority}] {title[:50]}...")
        
        # Strategy 1: PMC
        success, msg = try_pmc_pdf(pmid)
        if success:
            print(f"   ✅ {msg}")
            results.append({"pmid": pmid, "status": "success", "source": "PMC", "msg": msg})
            time.sleep(1)
            continue
        print(f"   ⚠️ PMC: {msg}")
        
        # Strategy 2: DOI + Unpaywall
        success, msg = try_doi_download(pmid, title)
        if success:
            print(f"   ✅ {msg}")
            results.append({"pmid": pmid, "status": "success", "source": "Unpaywall", "msg": msg})
            time.sleep(1)
            continue
        print(f"   ⚠️ Unpaywall: {msg}")
        
        results.append({"pmid": pmid, "status": "failed"})
        print(f"   ❌ FAILED")
        time.sleep(0.5)
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'='*60}")
    print(f"完成: {success_count}/{len(CRITICAL_PAPERS)} 成功")
    
    # List all PDFs
    print(f"\n已下载PDF:")
    for pdf in sorted(PMC_DIR.glob("PMID_*.pdf")):
        size = pdf.stat().st_size / 1024
        print(f"  {pdf.name} ({size:.0f} KB)")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
