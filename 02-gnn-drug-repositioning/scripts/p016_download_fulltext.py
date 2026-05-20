#!/usr/bin/env python3
"""
Project-016 33篇精读文献PDF批量下载
策略：PMC免费全文 → Unpaywall开放获取 → Europe PMC → 其他
"""

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

PMC_DIR = Path("/Users/junxuku/.openclaw/workspace/.tmp/p016_fulltext_pdfs")
PMC_DIR.mkdir(exist_ok=True)

# 33篇需精读文献（PMID + 标题）
PAPERS = [
    ("41084405", "Comparison of NSAID therapy alone versus combined NSAID and home exercise therapy"),
    ("40273050", "FGF18 induces chondrogenesis and anti-osteoarthritic effects in a mouse model"),
    ("38867545", "Efficiency of Glucosamine in Treating TMJOA: A Meta-Analytic Umbrella Review"),
    ("39092654", "Can treatment with chondroitin and glucosamine sulphate prevent changes"),
    ("38821656", "INTRA-ARTICULAR PHARMACOLOGICAL INJECTIONS FOR TMJOA"),
    ("37608244", "Hyaluronic acid injections for pain relief and functional improvement"),
    ("36890529", "Does intra-articular injection of tenoxicam after arthrocentesis heal outcomes"),
    ("35994388", "Recent Advances in Animal Models, Diagnosis, and Treatment of TMJOA"),
    ("33775650", "A Systematic Review of Rat Models With TMJOA"),
    ("30814387", "Long-term effectiveness of arthrocentesis with and without hyaluronic acid"),
    ("29244893", "Effectiveness of Intra-Articular Injections of Sodium Hyaluronate or Corticosteroids"),
    ("28879245", "Effects of High-Dose Capsaicin on TMD Subjects"),
    # No PMID papers (will try title search)
    (None, "Injection of sodium hyaluronate compared with a corticosteroid in TMJOA"),
    (None, "Application of dental chairside evidence-based medicine for TMJOA"),
    (None, "Dextrose Prolotherapy for Knee Osteoarthritis RCT"),
    (None, "Intra-articular Na-hyaluronate chondroprotective rabbits"),
    (None, "Manual therapy for musculoskeletal conditions systematic review"),
    (None, "Peptide-mesenchymal stem cell complex rat OA"),
    (None, "Low-level laser for masticatory muscle pain"),
    (None, "Physical therapy for chronic migraine and TMD"),
    (None, "RAIN method and Graph SAGE gastric neoplasm"),
    (None, "Bayesian Meta-Analysis antipsychotic drugs"),
    (None, "Data science drug safety adverse events"),
    (None, "Heterogeneous treatment effects joint confidence intervals"),
    (None, "FlexPro MD krill oil astaxanthin HA rat OA"),
    (None, "Trehalose bone fracture healing rat sleep deprivation"),
    (None, "ICOS alleviates paclitaxel neuropathic pain mice"),
    (None, "Metformin adjunctive therapy Parkinson pilot"),
    (None, "Meta-analysis proteomics osteoblasts bone blood"),
    ("41770501", "Comparative efficacy transdermal buprenorphine versus diclofenac knee OA"),
    ("40972625", "Global burden low bone mineral density GBD 2021"),
    (None, "Comparative effectiveness oral pharmacologic knee osteoarthritis NMA"),
    ("41727032", "Discovery TDP-43 aggregation inhibitors hybrid ML framework"),
]

def download_pmc_pdf(pmid, title):
    """尝试从PMC下载PDF"""
    try:
        # 1. 检查是否有PMC ID
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC?format=pdf"
        # 先通过Europe PMC API获取PMC ID
        search_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMCID:{pmid}"
        req = urllib.request.Request(search_url, headers={'Accept': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            # 简化处理：直接尝试PMC PDF URL
            pass
        
        # 尝试直接下载PMC PDF (如果文章在PMC中)
        pmc_pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}/pdf/main.pdf"
        # 实际上应该用Europe PMC API查询PMC ID
        
        # 使用Europe PMC直接下载
        europepmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmid}&blobtype=pdf"
        
        # 更简单的方式：Europe PMC REST API
        api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMCID:PMC{pmid}/fullText"
        
        # 实际上对于PubMed ID，用PubMed Central API查询
        pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
        
        req = urllib.request.Request(pmc_api)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            records = data.get('records', [])
            if records and records[0].get('pmcid'):
                pmcid = records[0]['pmcid']
                # 下载PDF
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
                req2 = urllib.request.Request(pdf_url, headers={
                    'User-Agent': 'Mozilla/5.0 (ResearchBot; Academic Project)',
                    'Accept': 'application/pdf'
                })
                
                with urllib.request.urlopen(req2, timeout=30) as resp2:
                    content = resp2.read()
                    if len(content) > 1000 and content[:4] == b'%PDF':
                        filename = PMC_DIR / f"{pmid}_{title[:40].replace(' ', '_')}.pdf"
                        with open(filename, 'wb') as f:
                            f.write(content)
                        return True, f"PMC PDF ({len(content)} bytes)"
            return False, "Not in PMC or no PDF available"
    except Exception as e:
        return False, f"PMC error: {str(e)[:50]}"

def download_unpaywall(doi, title):
    """尝试Unpaywall查找开放获取PDF"""
    try:
        if not doi:
            return False, "No DOI"
        
        url = f"https://api.unpaywall.org/v2/{doi}?email=chongchong@tjh.tjmu.edu.cn"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            
            # 找最佳开放获取URL
            best_oa = data.get('best_oa_location', {})
            if best_oa and best_oa.get('url_for_pdf'):
                pdf_url = best_oa['url_for_pdf']
                req2 = urllib.request.Request(pdf_url, headers={
                    'User-Agent': 'Mozilla/5.0 (ResearchBot; Academic Project)',
                })
                
                with urllib.request.urlopen(req2, timeout=30) as resp2:
                    content = resp2.read()
                    if len(content) > 1000:
                        filename = PMC_DIR / f"DOI_{doi.replace('/', '_')[:20]}_{title[:30].replace(' ', '_')}.pdf"
                        with open(filename, 'wb') as f:
                            f.write(content)
                        return True, f"Unpaywall OA ({len(content)} bytes)"
            
            return False, "No OA location found"
    except Exception as e:
        return False, f"Unpaywall error: {str(e)[:50]}"

def search_doi_by_title(title):
    """通过标题搜索DOI"""
    try:
        encoded = urllib.parse.quote(title[:100])
        url = f"https://api.crossref.org/works?query.title={encoded}&rows=1"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            items = data.get('message', {}).get('items', [])
            if items:
                return items[0].get('DOI')
    except Exception:
        pass
    return None

def main():
    print("=" * 60)
    print("Project-016 33篇精读文献PDF下载")
    print(f"保存目录: {PMC_DIR}")
    print("=" * 60)
    
    results = []
    downloaded = 0
    failed = 0
    
    for i, (pmid, title) in enumerate(PAPERS, 1):
        print(f"\n{i:2}/33 [{pmid or 'NO_PMID'}] {title[:60]}...")
        
        success = False
        source = ""
        
        # Strategy 1: PMC (for PMID papers)
        if pmid:
            success, source = download_pmc_pdf(pmid, title)
            if success:
                downloaded += 1
                print(f"  ✅ PMC: {source}")
                results.append({"pmid": pmid, "title": title, "status": "downloaded", "source": source})
                time.sleep(1)
                continue
            else:
                print(f"  ⚠️ PMC: {source}")
        
        # Strategy 2: Search DOI and Unpaywall (for all papers)
        doi = search_doi_by_title(title)
        if doi:
            print(f"  🔍 DOI found: {doi}")
            success, source = download_unpaywall(doi, title)
            if success:
                downloaded += 1
                print(f"  ✅ Unpaywall: {source}")
                results.append({"pmid": pmid, "title": title, "status": "downloaded", "source": source, "doi": doi})
                time.sleep(1)
                continue
            else:
                print(f"  ⚠️ Unpaywall: {source}")
        else:
            print(f"  ⚠️ No DOI found")
        
        # Failed
        failed += 1
        print(f"  ❌ FAILED")
        results.append({"pmid": pmid, "title": title, "status": "failed", "doi": doi})
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"下载完成: {downloaded}/33 成功, {failed}/33 失败")
    
    # 保存下载日志
    with open(PMC_DIR / "download_log.json", "w") as f:
        json.dump({
            "metadata": {"date": time.strftime("%Y-%m-%d %H:%M"), "total": 33, "downloaded": downloaded, "failed": failed},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    # 列出已下载文件
    print(f"\n已下载PDF文件:")
    for pdf_file in sorted(PMC_DIR.glob("*.pdf")):
        size = pdf_file.stat().st_size / 1024
        print(f"  {pdf_file.name[:60]}... ({size:.1f} KB)")
    
    return 0

if __name__ == "__main__":
    import sys
    import urllib.parse
    sys.exit(main())
