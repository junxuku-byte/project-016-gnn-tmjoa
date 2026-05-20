#!/usr/bin/env python3
"""
Project-016 openFDA 药物适应症查询
免费REST API：https://api.fda.gov/drug/label.json
无需API key
"""

import json
import time
import urllib.request
import urllib.parse
from collections import Counter
from datetime import datetime

OPENFDA_API = "https://api.fda.gov/drug/label.json"

# FDA已批准用于关节炎/疼痛/骨代谢的药物通用名
TARGET_DRUGS = [
    "CELECOXIB", "DICLOFENAC", "IBUPROFEN", "NAPROXEN", "MELOXICAM",
    "INDOMETHACIN", "KETOPROFEN", "PIROXICAM", "SULINDAC", "TOLMETIN",
    "DICLOFENAC", "ETODOLAC", "FENOPROFEN", "FLURBIPROFEN", "KETOROLAC",
    "MECLOFENAMATE", "NABUMETONE", "OXAPROZIN", "TIAPROFENIC ACID",
    "PREDNISONE", "PREDNISOLONE", "METHYLPREDNISOLONE", "TRIAMCINOLONE",
    "DEXAMETHASONE", "BETAMETHASONE", "HYDROCORTISONE", "CORTISONE",
    "DULOXETINE", "PREGABALIN", "GABAPENTIN", "TRAMADOL", "TAPENTADOL",
    "AMITRIPTYLINE", "NORTRIPTYLINE", "DESIPRAMINE", "IMIPRAMINE",
    "CARBAMAZEPINE", "PHENYTOIN", "VALPROIC ACID", "LAMOTRIGINE",
    "TOPIRAMATE", "LEVETIRACETAM", "ZONISAMIDE", "OXCARBAZEPINE",
    "LIDOCAINE", "CAPSAICIN", "MENTHOL", "CAMPHOR",
    "METHOTREXATE", "LEFLUNOMIDE", "HYDROXYCHLOROQUINE", "SULFASALAZINE",
    "AZATHIOPRINE", "CYCLOSPORINE", "MYCOPHENOLATE",
    "ETANERCEPT", "INFLIXIMAB", "ADALIMUMAB", "CERTOLIZUMAB", "GOLIMUMAB",
    "RITUXIMAB", "TOCILIZUMAB", "ABATACEPT", "ANAKINRA",
    "TOFACITINIB", "BARICITINIB", "UPADACITINIB",
    "APREMILAST", "TOFACITINIB", "RUXOLITINIB",
    "DENOSUMAB", "ALENDRONATE", "RISEDRONATE", "IBANDRONATE", "ZOLEDRONIC ACID",
    "TERIPARATIDE", "RALOXIFENE", "Bazedoxifene", "LASOFOXIFENE",
    "CALCITONIN", "CALCITRIOL", "DOXERCALCIFEROL", "PARICALCITOL",
    "CALCIUM", "VITAMIN D", "VITAMIN D2", "VITAMIN D3", "CHOLECALCIFEROL",
    "ATORVASTATIN", "ROSUVASTATIN", "SIMVASTATIN", "PRAVASTATIN",
    "METFORMIN", "ROSIGLITAZONE", "PIOGLITAZONE",
    "GLUCOSAMINE", "CHONDROITIN", "METHYLSULFONYLMETHANE", "MSM",
    "HYALURONIC ACID", "SODIUM HYALURONATE",
    "PLATELET-RICH PLASMA", "PRP",
    "BUPRENORPHINE", "METHADONE", "NALOXONE", "NALTREXONE",
    "MORPHINE", "OXYCODONE", "HYDROCODONE", "CODEINE", "FENTANYL",
    "OXYMORPHONE", "HYDROMORPHONE", "TAPENTADOL",
]

def fda_search_indications(drug_name):
    """查询openFDA药物标签中的适应症"""
    try:
        encoded = urllib.parse.quote(drug_name)
        # 搜索药物名称在标签中
        url = f"{OPENFDA_API}?search=openfda.generic_name:{encoded}&limit=1"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            results = data.get('results', [])
            
            if not results:
                return []
            
            label = results[0]
            indications = []
            
            # 提取适应症信息
            if 'indications_and_usage' in label:
                ind_text = ' '.join(label['indications_and_usage'])
                indications.append(ind_text[:500])
            
            if 'purpose' in label:
                purpose_text = ' '.join(label['purpose'])
                indications.append(purpose_text[:500])
            
            return indications
            
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []  # 药物未找到
        print(f"  HTTP {e.code} for {drug_name}")
        return []
    except Exception as e:
        print(f"  Error {drug_name}: {e}")
        return []
    finally:
        time.sleep(0.5)  # Rate limit

def extract_disease_matches(text):
    """从适应症文本中提取匹配的疾病关键词"""
    text_lower = text.lower()
    matches = []
    
    disease_keywords = {
        "Osteoarthritis": ["osteoarthritis", "degenerative arthritis", "degenerative joint disease"],
        "Rheumatoid arthritis": ["rheumatoid arthritis", "rheumatoid"],
        "Osteoporosis": ["osteoporosis", "bone loss", "postmenopausal osteoporosis"],
        "Chronic pain": ["pain", "chronic pain", "moderate pain", "severe pain"],
        "TMD": ["temporomandibular", "tmj", "jaw pain"],
        "Periodontitis": ["periodontitis", "periodontal", "gum disease"],
        "Fibromyalgia": ["fibromyalgia"],
        "Migraine": ["migraine", "migraines"],
        "Ankylosing spondylitis": ["ankylosing spondylitis"],
        "Psoriatic arthritis": ["psoriatic arthritis"],
        "Gout": ["gout", "hyperuricemia"],
        "Lupus": ["lupus", "systemic lupus erythematosus", "sle"],
    }
    
    for disease, keywords in disease_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                matches.append(disease)
                break
    
    return list(set(matches))

def main():
    print("=" * 60)
    print("Project-016 openFDA 药物适应症查询")
    print("=" * 60)
    
    results = []
    
    for i, drug in enumerate(TARGET_DRUGS, 1):
        print(f"\n{i:3}/{len(TARGET_DRUGS)} 🔍 {drug}...", end=" ")
        
        indications = fda_search_indications(drug)
        
        if indications:
            all_text = ' '.join(indications)
            diseases = extract_disease_matches(all_text)
            
            if diseases:
                print(f"✅ {', '.join(diseases)}")
                for disease in diseases:
                    results.append({
                        "drug": drug.title(),
                        "disease": disease,
                        "confidence": 0.5,  # FDA批准 = 高置信度
                        "source": "openFDA",
                        "indication_snippet": all_text[:200],
                    })
            else:
                print("no disease match")
        else:
            print("not found")
    
    print(f"\n{'='*60}")
    print(f"✅ openFDA查询完成: {len(results)} 个关联")
    
    # 去重
    seen = set()
    unique_results = []
    for r in results:
        pair = (r['drug'], r['disease'])
        if pair not in seen:
            seen.add(pair)
            unique_results.append(r)
    
    print(f"唯一drug-disease对: {len(unique_results)}")
    
    # 统计
    drug_dist = Counter(r['drug'] for r in unique_results)
    disease_dist = Counter(r['disease'] for r in unique_results)
    
    print(f"\n药物分布:")
    for d, c in drug_dist.most_common(10):
        print(f"  {d}: {c}")
    
    print(f"\n疾病分布:")
    for dis, c in disease_dist.most_common():
        print(f"  {dis}: {c}")
    
    # 保存
    samples = []
    for r in unique_results:
        samples.append({
            "pmid": "openFDA",
            "title": f"FDA: {r['drug']} for {r['disease']}",
            "drug": r['drug'],
            "disease": r['disease'],
            "year": 2024,
            "journal_if": 5.0,  # FDA批准证据等级高
            "design": "FDA_Approved_Indication",
            "conclusion": "positive",
            "total_n": None,
            "final_score": r['confidence'],
            "label": r['confidence'],
            "weight": 0.5,  # FDA批准较高权重
            "is_external": True,
            "external_source": f"openFDA drug label: {r['indication_snippet'][:100]}",
        })
    
    output = {
        "metadata": {
            "version": "v3_9_openfda",
            "created_at": datetime.now().isoformat(),
            "total_associations": len(results),
            "unique_pairs": len(samples),
            "source": "openFDA REST API (https://api.fda.gov)",
            "drugs_queried": len(TARGET_DRUGS),
        },
        "samples": samples,
    }
    
    with open(".tmp/p016_external_openfda_v3_9.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ openFDA样本保存: .tmp/p016_external_openfda_v3_9.json")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
