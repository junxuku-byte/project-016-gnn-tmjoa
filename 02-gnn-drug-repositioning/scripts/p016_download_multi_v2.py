#!/usr/bin/env python3
"""
Project-016 精简多源PDF下载器
跳过Google Scholar（CN网络阻塞），聚焦可靠来源
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

PMC_DIR = Path("/Users/junxuku/.openclaw/workspace/.tmp/p016_fulltext_pdfs")

PAPERS = [
    ("38821656", "Intra-articular pharmacological injections for temporomandibular joint osteoarthritis"),
    ("29244893", "Effectiveness of intra-articular injections sodium hyaluronate corticosteroids TMD"),
    ("41084405", "Comparison of NSAID therapy versus combined NSAID and home exercise temporomandibular"),
    ("28879245", "Effects of high-dose capsaicin on TMD subjects randomized clinical study"),
    ("40273050", "FGF18 induces chondrogenesis anti-osteoarthritic mouse model TMJ"),
    ("39092654", "Treatment with chondroitin and glucosamine sulphate temporomandibular"),
    ("35994388", "Recent advances in animal models diagnosis treatment temporomandibular joint osteoarthritis"),
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/pdf,*/*',
}

def get_doi(pmid):
    """通过Europe PMC获取DOI"""
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMCID:PMC{pmid}?format=json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            result = data.get('resultList', {}).get('result', [{}])[0]
            return result.get('doi')
    except Exception:
        pass
    
    # Fallback: ESummary
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            result = data.get('result', {}).get(pmid, {})
            return result.get('doi')
    except Exception:
        pass
    return None

def try_europepmc_ft(pmid):
    """Europe PMC全文XML转文本"""
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMCID:PMC{pmid}/fullText"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            # 如果有全文XML，提取文本
            if 'fullText' in data:
                return True, "EuropePMC_fullText_XML", data.get('fullText', '')[:2000]
    except Exception:
        pass
    return False, "", ""

def try_journal_oa(doi, pmid):
    """尝试通过DOI直接访问期刊OA版本"""
    if not doi:
        return False, "No DOI"
    
    # 尝试常见的OA域名模式
    oa_patterns = [
        f"https://doi.org/{doi}",
    ]
    
    for url in oa_patterns:
        try:
            req = urllib.request.Request(url, headers=HEADERS, method='HEAD')
            with urllib.request.urlopen(req, timeout=15, allow_redirects=True) as resp:
                final_url = resp.geturl()
                # 如果是Elsevier/Wiley/Springer等付费墙，会重定向到登陆页
                if any(x in final_url for x in ['sciencedirect', 'wiley', 'springer', 'tandfonline']):
                    return False, f"Paywall: {final_url[:40]}"
                return False, f"Redirect: {final_url[:40]}"
        except Exception:
            continue
    
    return False, "No OA access"

def try_doi_org_pdf(doi, pmid):
    """尝试doi.org解析后的PDF链接"""
    if not doi:
        return False, "No DOI"
    
    try:
        url = f"https://doi.org/{doi}"
        req = urllib.request.Request(url, headers={
            **HEADERS,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            final_url = resp.geturl()
            html = resp.read().decode('utf-8', errors='ignore')
            
            # Look for PDF links in HTML
            import re
            pdf_links = re.findall(r'href="([^"]+\.pdf[^"]*)"', html, re.I)
            pdf_links += re.findall(r'content="([^"]+\.pdf[^"]*)"', html, re.I)
            
            for pdf_url in pdf_links:
                if pdf_url.startswith('/'):
                    base = '/'.join(final_url.split('/')[:3])
                    pdf_url = base + pdf_url
                elif pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url
                
                try:
                    req2 = urllib.request.Request(pdf_url, headers=HEADERS)
                    with urllib.request.urlopen(req2, timeout=30) as resp2:
                        content = resp2.read()
                        if len(content) > 50000 and content[:4] == b'%PDF':
                            filename = PMC_DIR / f"PMID_{pmid}_DOI.pdf"
                            with open(filename, 'wb') as f:
                                f.write(content)
                            return True, f"DOI_direct ({len(content)/1024:.0f} KB)"
                except:
                    continue
            
            return False, f"{len(pdf_links)} PDF links but no valid download"
    except Exception as e:
        return False, f"DOI access: {str(e)[:40]}"

def main():
    print("=" * 60)
    print("精简多源PDF下载 (跳过Google Scholar)")
    print("=" * 60)
    
    results = []
    
    for i, (pmid, title) in enumerate(PAPERS, 1):
        print(f"\n[{i}/{len(PAPERS)}] PMID {pmid}")
        print(f"  {title[:60]}...")
        
        # Get DOI first
        doi = get_doi(pmid)
        if doi:
            print(f"  DOI: {doi}")
        
        # Try Europe PMC full text
        success, msg, text = try_europepmc_ft(pmid)
        if text:
            print(f"  ✅ Europe PMC XML全文可用 (text length: {len(text)})")
            # Save text
            filename = PMC_DIR / f"PMID_{pmid}_EuropePMC.txt"
            with open(filename, 'w') as f:
                f.write(text)
            results.append({"pmid": pmid, "status": "text_only", "source": "EuropePMC_XML"})
            time.sleep(0.5)
            continue
        
        # Try DOI direct PDF
        if doi:
            success, msg = try_doi_org_pdf(doi, pmid)
            if success:
                print(f"  ✅ {msg}")
                results.append({"pmid": pmid, "status": "pdf", "source": msg})
                time.sleep(1)
                continue
            else:
                print(f"  ❌ DOI: {msg}")
        
        print(f"  ❌ 所有策略失败")
        results.append({"pmid": pmid, "status": "failed"})
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'='*60}")
    pdf_count = len(list(PMC_DIR.glob("*.pdf")))
    txt_count = len(list(PMC_DIR.glob("*.txt")))
    print(f"PDF文件: {pdf_count}")
    print(f"TXT全文: {txt_count}")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())