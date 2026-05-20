#!/usr/bin/env python3
"""
Project-016 多源PDF下载器 - 综合版
尝试多个来源获取全文PDF
"""

import json
import time
import urllib.request
import urllib.error
import urllib.parse
import re
from pathlib import Path

PMC_DIR = Path("/Users/junxuku/.openclaw/workspace/.tmp/p016_fulltext_pdfs")
PMC_DIR.mkdir(exist_ok=True)

# 优先级排序的关键文献 (PMID, 标题)
TARGET_PAPERS = [
    ("38821656", "Intra-articular pharmacological injections for temporomandibular joint osteoarthritis"),
    ("37608244", "Hyaluronic acid injections for pain relief TMD umbrella review"),
    ("29244893", "Effectiveness of intra-articular injections sodium hyaluronate corticosteroids TMD"),
    ("41084405", "Comparison of NSAID therapy versus combined NSAID and home exercise temporomandibular"),
    ("28879245", "Effects of high-dose capsaicin on TMD subjects randomized clinical study"),
    ("36890529", "Does intra-articular injection of tenoxicam after arthrocentesis temporomandibular"),
    ("30814387", "Long-term effectiveness of arthrocentesis with and without hyaluronic acid"),
    ("40273050", "FGF18 induces chondrogenesis anti-osteoarthritic mouse model TMJ"),
    ("39092654", "Treatment with chondroitin and glucosamine sulphate temporomandibular"),
    ("35994388", "Recent advances in animal models diagnosis treatment temporomandibular joint osteoarthritis"),
]

def get_pmcid(pmid):
    """通过PMID获取PMC ID"""
    try:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            records = data.get('records', [])
            if records and records[0].get('pmcid'):
                return records[0]['pmcid']
    except Exception:
        pass
    return None

def try_ncbi_pmc(pmid, pmcid):
    """尝试1: NCBI PMC直接下载"""
    if not pmcid:
        return False, "No PMC ID"
    try:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*',
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
            if len(content) > 50000 and content[:4] == b'%PDF':
                return save_pdf(pmid, content, "NCBI_PMC")
            return False, f"Invalid content ({len(content)} bytes)"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"Error: {str(e)[:40]}"

def try_europepmc(pmid):
    """尝试2: Europe PMC API"""
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMCID:PMC{pmid}/fullText"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            # Europe PMC有时返回XML全文
            if 'fullText' in data:
                return False, "XML text only (no PDF)"
            return False, "No PDF available"
    except Exception as e:
        return False, f"Europe PMC: {str(e)[:40]}"

def try_europepmc_pdf(pmid):
    """尝试3: Europe PMC PDF渲染"""
    try:
        # Europe PMC PDF endpoint
        url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmid}&blobtype=pdf"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Academic Research)',
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
            if len(content) > 50000 and content[:4] == b'%PDF':
                return save_pdf(pmid, content, "EuropePMC")
            return False, f"Invalid ({len(content)} bytes)"
    except Exception as e:
        return False, f"Europe PMC PDF: {str(e)[:40]}"

def try_doi_and_unpaywall(pmid, title):
    """尝试4: Crossref DOI + Unpaywall"""
    try:
        # Search Crossref
        encoded = urllib.parse.quote(title[:80])
        url = f"https://api.crossref.org/works?query.title={encoded}&rows=1"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            items = data.get('message', {}).get('items', [])
            if not items or not items[0].get('DOI'):
                return False, "No DOI"
            doi = items[0]['DOI']
        
        # Unpaywall
        uw_url = f"https://api.unpaywall.org/v2/{doi}?email=chongchong@tjh.tjmu.edu.cn"
        req2 = urllib.request.Request(uw_url)
        with urllib.request.urlopen(req2, timeout=15) as resp2:
            data2 = json.loads(resp2.read().decode())
            
            # Try best OA location
            for loc in [data2.get('best_oa_location'), *data2.get('oa_locations', [])]:
                if loc and loc.get('url_for_pdf'):
                    pdf_url = loc['url_for_pdf']
                    req3 = urllib.request.Request(pdf_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Academic Research)',
                    })
                    with urllib.request.urlopen(req3, timeout=60) as resp3:
                        content = resp3.read()
                        if len(content) > 50000:
                            return save_pdf(pmid, content, "Unpaywall")
            
            # Try landing page
            for loc in [data2.get('best_oa_location'), *data2.get('oa_locations', [])]:
                if loc and loc.get('url') and not loc.get('url_for_pdf'):
                    return False, f"OA landing page but no PDF: {loc['url'][:50]}"
            
            return False, "No OA PDF"
    except Exception as e:
        return False, f"Unpaywall: {str(e)[:40]}"

def try_google_scholar(pmid, title):
    """尝试5: Google Scholar搜索PDF链接"""
    try:
        encoded = urllib.parse.quote(f"{title} filetype:pdf")
        url = f"https://scholar.google.com/scholar?q={encoded}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            # Look for PDF links
            pdf_links = re.findall(r'href="([^"]+\.pdf[^"]*)"', html)
            pdf_links += re.findall(r'\[PDF\].*?href="([^"]+)"', html)
            
            for pdf_url in pdf_links[:3]:
                try:
                    if pdf_url.startswith('/'):
                        pdf_url = f"https://scholar.google.com{pdf_url}"
                    req2 = urllib.request.Request(pdf_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Academic Research)',
                    })
                    with urllib.request.urlopen(req2, timeout=30) as resp2:
                        content = resp2.read()
                        if len(content) > 50000 and content[:4] == b'%PDF':
                            return save_pdf(pmid, content, "GoogleScholar")
                except:
                    continue
            return False, f"No valid PDF in {len(pdf_links)} links"
    except Exception as e:
        return False, f"Google Scholar: {str(e)[:40]}"

def try_researchgate(pmid, title):
    """尝试6: ResearchGate搜索"""
    try:
        # ResearchGate search
        encoded = urllib.parse.quote(title[:60])
        url = f"https://www.researchgate.net/search?q={encoded}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            # Look for publication links
            pub_links = re.findall(r'href="(/publication/[^"]+)"', html)
            if pub_links:
                pub_url = f"https://www.researchgate.net{pub_links[0]}"
                return False, f"RG page found (manual download needed): {pub_url}"
            return False, "No RG result"
    except Exception as e:
        return False, f"ResearchGate: {str(e)[:40]}"

def try_pubmed_linkout(pmid):
    """尝试7: PubMed LinkOut找全文链接"""
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&id={pmid}&cmd=prlinks&retmode=json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            links = data.get('linksets', [{}])[0].get('urllinks', [])
            for link in links:
                url = link.get('url', {}).get('value', '')
                if url:
                    # Try to find PDF
                    try:
                        req2 = urllib.request.Request(url, headers={
                            'User-Agent': 'Mozilla/5.0 (Academic Research)',
                        })
                        with urllib.request.urlopen(req2, timeout=30) as resp2:
                            content = resp2.read()
                            if len(content) > 50000 and content[:4] == b'%PDF':
                                return save_pdf(pmid, content, "PubMedLinkout")
                    except:
                        continue
            return False, f"{len(links)} links but no direct PDF"
    except Exception as e:
        return False, f"Linkout: {str(e)[:40]}"

def save_pdf(pmid, content, source):
    """保存PDF文件"""
    filename = PMC_DIR / f"PMID_{pmid}_{source}.pdf"
    with open(filename, 'wb') as f:
        f.write(content)
    return True, f"{source} ({len(content)/1024:.0f} KB)"

def download_paper(pmid, title):
    """综合下载策略"""
    print(f"\n{'='*60}")
    print(f"PMID {pmid}: {title[:60]}...")
    print(f"{'='*60}")
    
    strategies = [
        ("NCBI PMC", lambda: try_ncbi_pmc(pmid, get_pmcid(pmid))),
        ("Europe PMC", lambda: try_europepmc_pdf(pmid)),
        ("Unpaywall", lambda: try_doi_and_unpaywall(pmid, title)),
        ("Google Scholar", lambda: try_google_scholar(pmid, title)),
        ("PubMed Linkout", lambda: try_pubmed_linkout(pmid)),
        ("ResearchGate", lambda: try_researchgate(pmid, title)),
    ]
    
    for name, strategy in strategies:
        print(f"\n  🔍 {name}...", end=" ")
        success, msg = strategy()
        if success:
            print(f"✅ {msg}")
            return True, msg
        else:
            print(f"❌ {msg}")
        time.sleep(0.5)
    
    return False, "All strategies failed"

def main():
    print("=" * 60)
    print("Project-016 多源PDF下载器")
    print("尝试来源: NCBI PMC → Europe PMC → Unpaywall → Google Scholar → PubMed Linkout → ResearchGate")
    print("=" * 60)
    
    results = []
    
    for i, (pmid, title) in enumerate(TARGET_PAPERS, 1):
        print(f"\n\n[{i}/{len(TARGET_PAPERS)}] 开始下载...")
        success, msg = download_paper(pmid, title)
        results.append({"pmid": pmid, "title": title, "success": success, "msg": msg})
        time.sleep(1)
    
    # Summary
    print(f"\n\n{'='*60}")
    print("下载汇总")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"\n成功: {success_count}/{len(TARGET_PAPERS)}")
    print(f"失败: {len(TARGET_PAPERS) - success_count}/{len(TARGET_PAPERS)}")
    
    print(f"\n✅ 已下载:")
    for r in results:
        if r['success']:
            print(f"  PMID {r['pmid']}: {r['msg']}")
    
    print(f"\n❌ 失败:")
    for r in results:
        if not r['success']:
            print(f"  PMID {r['pmid']}: {r['msg']}")
    
    # Save log
    with open(PMC_DIR / "download_multi_log.json", "w") as f:
        json.dump({"results": results, "timestamp": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    
    # List all PDFs
    print(f"\n📂 PDF文件目录: {PMC_DIR}")
    for pdf in sorted(PMC_DIR.glob("*.pdf")):
        size = pdf.stat().st_size / 1024
        print(f"  {pdf.name} ({size:.0f} KB)")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
