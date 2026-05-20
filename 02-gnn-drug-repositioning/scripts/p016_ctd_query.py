#!/usr/bin/env python3
"""
Project-016 CTD (Comparative Toxicogenomics Database) 药物-疾病关联提取
来源: http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz
免费、无需注册、TSV格式

筛选条件:
- 目标疾病: Osteoarthritis, Rheumatoid Arthritis, Osteoporosis, Pain, 
            Temporomandibular Joint Disorders, Periodontitis, Fibromyalgia, Migraine
- DirectEvidence = therapeutic → 高置信度 (label=0.5)
- DirectEvidence = marker/mechanism → 中置信度 (label=0.3)
- InferenceScore > 5 → 低置信度 (label=0.2)

输出: .tmp/p016_external_ctd_v3_9.json
"""

import gzip
import io
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

CTD_URL = "http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"

# 目标疾病关键词（CTD中的疾病名称）
TARGET_DISEASES = {
    "Osteoarthritis": ["osteoarthritis"],
    "Rheumatoid arthritis": ["rheumatoid arthritis", "arthritis, rheumatoid"],
    "Osteoporosis": ["osteoporosis", "bone density conservation agents"],
    "Chronic pain": ["pain", "chronic pain", "pain management"],
    "TMD": ["temporomandibular joint disorders", "temporomandibular joint dysfunction syndrome", "tmj"],
    "Periodontitis": ["periodontitis", "periodontal diseases", "gingivitis"],
    "Fibromyalgia": ["fibromyalgia", "fibrositis"],
    "Migraine": ["migraine disorders", "migraine", "headache"],
    "Ankylosing spondylitis": ["ankylosing spondylitis", "spondylitis, ankylosing"],
    "Psoriatic arthritis": ["arthritis, psoriatic", "psoriatic arthritis"],
    "Gout": ["gout", "hyperuricemia"],
    "Systemic lupus erythematosus": ["lupus erythematosus, systemic", "sle"],
    "Sjogren syndrome": ["sjogren syndrome", "sjogren"],
    "Scleroderma": ["scleroderma", "systemic sclerosis"],
    "Behcet disease": ["behcet syndrome", "behcet"],
}

# 药物名称映射（CTD名称 → 训练集统一名称）
DRUG_NAME_MAP = {
    # 常用NSAIDs
    "Ibuprofen": "NSAIDs",
    "Diclofenac": "NSAIDs",
    "Naproxen": "NSAIDs",
    "Celecoxib": "NSAIDs",
    "Meloxicam": "NSAIDs",
    "Indomethacin": "NSAIDs",
    "Ketoprofen": "NSAIDs",
    "Piroxicam": "NSAIDs",
    "Etodolac": "NSAIDs",
    "Flurbiprofen": "NSAIDs",
    "Nabumetone": "NSAIDs",
    "Oxaprozin": "NSAIDs",
    "Sulindac": "NSAIDs",
    "Tolmetin": "NSAIDs",
    "Tenoxicam": "NSAIDs",
    "Lornoxicam": "NSAIDs",
    "Ketorolac": "NSAIDs",
    "Meclofenamate": "NSAIDs",
    "Tiaprofenic acid": "NSAIDs",
    
    # 糖皮质激素
    "Prednisone": "Corticosteroid",
    "Prednisolone": "Corticosteroid",
    "Methylprednisolone": "Corticosteroid",
    "Triamcinolone": "Corticosteroid",
    "Dexamethasone": "Corticosteroid",
    "Betamethasone": "Corticosteroid",
    "Hydrocortisone": "Corticosteroid",
    "Cortisone": "Corticosteroid",
    "Deflazacort": "Corticosteroid",
    "Fludrocortisone": "Corticosteroid",
    
    # 补充剂
    "Glucosamine": "Glucosamine",
    "Chondroitin sulfates": "Chondroitin",
    "Chondroitin": "Chondroitin",
    "Hyaluronic acid": "Hyaluronic acid",
    "Sodium hyaluronate": "Hyaluronic acid",
    "Collagen": "Collagen",
    "Vitamin D": "Vitamin D",
    "Vitamin D3": "Vitamin D",
    "Cholecalciferol": "Vitamin D",
    "Ergocalciferol": "Vitamin D",
    "Calcitriol": "Vitamin D",
    "Omega-3 fatty acids": "Omega-3",
    "Eicosapentaenoic acid": "Omega-3",
    "Docosahexaenoic acid": "Omega-3",
    "Curcumin": "Curcumin",
    "Resveratrol": "Resveratrol",
    "Quercetin": "Quercetin",
    "EGCG": "EGCG",
    "Epigallocatechin gallate": "EGCG",
    "Genistein": "Quercetin",  # 类似物
    "Silibinin": "Quercetin",  # 类似物
    "S-Adenosylmethionine": "SAMe",
    "Ginger": "Ginger",
    "Capsaicin": "Capsaicin",
    "Palmitoylethanolamide": "Palmitoylethanolamide",
    "Methylsulfonylmethane": "MSM",
    "Avocado soybean unsaponifiables": "Avocado-soybean unsaponifiables",
    
    # 生物制剂/靶向
    "Tofacitinib": "Tofacitinib",
    "Baricitinib": "Tofacitinib",  # JAK inhibitor family
    "Anakinra": "Anakinra",
    "Tanezumab": "Tanezumab",
    "Denosumab": "Denosumab",
    "Alendronate": "Bisphosphonate",
    "Risedronate": "Bisphosphonate",
    "Ibandronate": "Bisphosphonate",
    "Zoledronic acid": "Bisphosphonate",
    "Pamidronate": "Bisphosphonate",
    "Etidronate": "Bisphosphonate",
    "Clodronate": "Bisphosphonate",
    "Teriparatide": "Growth factor",
    
    # 疼痛管理
    "Duloxetine": "Duloxetine",
    "Pregabalin": "Pregabalin",
    "Gabapentin": "Gabapentin",
    "Tramadol": "Tramadol",
    "Tapentadol": "Tramadol",
    "Amitriptyline": "Amitriptyline",
    "Nortriptyline": "Nortriptyline",
    "Fluoxetine": "Fluoxetine",
    "Sertraline": "Sertraline",
    "Venlafaxine": "Venlafaxine",
    "Milnacipran": "Milnacipran",
    "Desipramine": "Desipramine",
    "Imipramine": "Imipramine",
    "Bupropion": "Bupropion",
    "Mirtazapine": "Mirtazapine",
    
    # 抗癫痫/神经性疼痛
    "Carbamazepine": "Carbamazepine",
    "Phenytoin": "Phenytoin",
    "Valproic acid": "Valproic acid",
    "Lamotrigine": "Lamotrigine",
    "Topiramate": "Topiramate",
    "Levetiracetam": "Levetiracetam",
    "Zonisamide": "Zonisamide",
    "Oxcarbazepine": "Oxcarbazepine",
    "Lacosamide": "Lacosamide",
    "Pregabalin": "Pregabalin",
    
    # 肌肉松弛/局部
    "Baclofen": "Baclofen",
    "Tizanidine": "Tizanidine",
    "Cyclobenzaprine": "Cyclobenzaprine",
    "Methocarbamol": "Methocarbamol",
    "Carisoprodol": "Carisoprodol",
    "Orphenadrine": "Orphenadrine",
    "Lidocaine": "Lidocaine",
    "Ketamine": "Ketamine",
    
    # 阿片类
    "Buprenorphine": "Buprenorphine",
    "Methadone": "Methadone",
    "Morphine": "Morphine",
    "Codeine": "Codeine",
    "Oxycodone": "Oxycodone",
    "Hydrocodone": "Hydrocodone",
    "Fentanyl": "Fentanyl",
    "Oxymorphone": "Oxymorphone",
    "Hydromorphone": "Hydromorphone",
    "Meperidine": "Meperidine",
    "Pentazocine": "Pentazocine",
    "Nalbuphine": "Nalbuphine",
    "Butorphanol": "Butorphanol",
    
    # 其他
    "Metformin": "Metformin",
    "Rapamycin": "Rapamycin",
    "Sirolimus": "Rapamycin",
    "Everolimus": "Rapamycin",
    "Statin": "Statin",
    "Atorvastatin": "Statin",
    "Rosuvastatin": "Statin",
    "Simvastatin": "Statin",
    "Pravastatin": "Statin",
    "Lovastatin": "Statin",
    "Fluvastatin": "Statin",
    "Pitavastatin": "Statin",
    "Botulinum toxins": "Botulinum toxin",
    "Botulinum toxin type a": "Botulinum toxin",
    "Botulinum toxin type b": "Botulinum toxin",
    "Onabotulinumtoxina": "Botulinum toxin",
    "Ozone": "Ozone",
    "Ozone therapy": "Ozone",
    "Ozone preparation": "Ozone",
    "Prolotherapy": "Prolotherapy",
    "Dextrose": "Prolotherapy",
    
    # 免疫抑制剂
    "Methotrexate": "Methotrexate",
    "Leflunomide": "Leflunomide",
    "Hydroxychloroquine": "Hydroxychloroquine",
    "Sulfasalazine": "Sulfasalazine",
    "Azathioprine": "Azathioprine",
    "Cyclosporine": "Cyclosporine",
    "Mycophenolate mofetil": "Mycophenolate",
    "Mycophenolic acid": "Mycophenolate",
    
    # TNF抑制剂
    "Etanercept": "Etanercept",
    "Infliximab": "Infliximab",
    "Adalimumab": "Adalimumab",
    "Certolizumab pegol": "Certolizumab",
    "Golimumab": "Golimumab",
    
    # 其他生物制剂
    "Rituximab": "Rituximab",
    "Tocilizumab": "Tocilizumab",
    "Abatacept": "Abatacept",
    "Sarilumab": "Sarilumab",
    "Anakinra": "Anakinra",
    "Canakinumab": "Canakinumab",
    "Rilonacept": "Rilonacept",
    "Ustekinumab": "Ustekinumab",
    "Secukinumab": "Secukinumab",
    "Ixekizumab": "Ixekizumab",
    "Brodalumab": "Brodalumab",
    "Guselkumab": "Guselkumab",
    "Tildrakizumab": "Tildrakizumab",
    "Risankizumab": "Risankizumab",
    "Apremilast": "Apremilast",
    
    # JAK抑制剂
    "Tofacitinib": "Tofacitinib",
    "Baricitinib": "Baricitinib",
    "Upadacitinib": "Upadacitinib",
    "Filgotinib": "Filgotinib",
    "Peficitinib": "Peficitinib",
    
    # 其他靶向药
    "Ruxolitinib": "Ruxolitinib",
    "Tofacitinib": "Tofacitinib",
}

def download_ctd():
    """下载CTD chemicals-diseases文件"""
    print(f"Downloading CTD data from {CTD_URL}...")
    req = urllib.request.Request(CTD_URL)
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()

def parse_ctd(data_gz):
    """解析CTD TSV并筛选目标疾病"""
    results = []
    
    with gzip.GzipFile(fileobj=io.BytesIO(data_gz)) as f:
        # 跳过注释行
        for line in f:
            line = line.decode('utf-8').strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 10:
                continue
            
            chemical_name = parts[0]
            disease_name = parts[3]
            disease_id = parts[4]
            direct_evidence = parts[5]
            inference_score = parts[7]
            pubmed_ids = parts[9]
            
            # 疾病匹配
            matched_disease = None
            disease_lower = disease_name.lower()
            for std_name, keywords in TARGET_DISEASES.items():
                for kw in keywords:
                    if kw in disease_lower:
                        matched_disease = std_name
                        break
                if matched_disease:
                    break
            
            if not matched_disease:
                continue
            
            # 药物名称映射
            mapped_drug = None
            for ct_name, std_name in DRUG_NAME_MAP.items():
                if ct_name.lower() == chemical_name.lower():
                    mapped_drug = std_name
                    break
            
            # 如果没有精确匹配，尝试部分匹配
            if not mapped_drug:
                for ct_name, std_name in DRUG_NAME_MAP.items():
                    if ct_name.lower() in chemical_name.lower() or chemical_name.lower() in ct_name.lower():
                        mapped_drug = std_name
                        break
            
            if not mapped_drug:
                continue  # 跳过不在我们药物列表中的
            
            # 确定置信度
            if direct_evidence == "therapeutic":
                confidence = 0.5
                weight = 0.5
            elif direct_evidence == "marker/mechanism":
                confidence = 0.3
                weight = 0.4
            elif inference_score and inference_score.strip():
                try:
                    score = float(inference_score)
                    if score > 8:
                        confidence = 0.3
                        weight = 0.3
                    elif score > 5:
                        confidence = 0.2
                        weight = 0.2
                    else:
                        continue  # 跳过低分推断
                except:
                    continue
            else:
                continue
            
            results.append({
                "drug": mapped_drug,
                "disease": matched_disease,
                "confidence": confidence,
                "weight": weight,
                "direct_evidence": direct_evidence,
                "inference_score": inference_score,
                "pubmed_count": len(pubmed_ids.split("|")) if pubmed_ids else 0,
                "ctd_chemical": chemical_name,
                "ctd_disease": disease_name,
            })
    
    return results

def main():
    print("=" * 60)
    print("Project-016 CTD 药物-疾病关联提取")
    print("=" * 60)
    
    # 下载
    raw_data = download_ctd()
    print(f"✅ Downloaded {len(raw_data)/1024/1024:.1f} MB")
    
    # 解析
    print("Parsing CTD data...")
    results = parse_ctd(raw_data)
    
    print(f"\n✅ Parsed: {len(results)} target drug-disease associations")
    
    # 统计
    drug_dist = Counter(r['drug'] for r in results)
    disease_dist = Counter(r['disease'] for r in results)
    evidence_dist = Counter(r['direct_evidence'] for r in results)
    
    print(f"\n药物分布 (Top 15):")
    for d, c in drug_dist.most_common(15):
        print(f"  {d:25}: {c}")
    
    print(f"\n疾病分布:")
    for dis, c in disease_dist.most_common():
        print(f"  {dis:25}: {c}")
    
    print(f"\n证据类型:")
    for ev, c in evidence_dist.most_common():
        print(f"  {ev}: {c}")
    
    # 去重（同一drug-disease对取最高置信度）
    best_by_pair = {}
    for r in results:
        pair = (r['drug'], r['disease'])
        if pair not in best_by_pair or r['confidence'] > best_by_pair[pair]['confidence']:
            best_by_pair[pair] = r
    
    unique_results = list(best_by_pair.values())
    print(f"\n去重后: {len(unique_results)} 唯一drug-disease对")
    
    # 转换为训练样本
    samples = []
    for r in unique_results:
        samples.append({
            "pmid": "CTD",
            "title": f"CTD: {r['drug']} for {r['disease']}",
            "drug": r['drug'],
            "disease": r['disease'],
            "year": 2024,
            "journal_if": 3.0,
            "design": f"CTD_{r['direct_evidence'] or 'Inference'}",
            "conclusion": "positive",
            "total_n": None,
            "final_score": r['confidence'],
            "label": r['confidence'],
            "weight": r['weight'],
            "is_external": True,
            "external_source": f"CTD: {r['ctd_chemical']} → {r['ctd_disease']} (evidence={r['direct_evidence']}, score={r['inference_score']})",
            "pubmed_count": r['pubmed_count'],
        })
    
    # 保存
    output = {
        "metadata": {
            "version": "v3_9_ctd",
            "created_at": datetime.now().isoformat(),
            "total_associations": len(results),
            "unique_pairs": len(samples),
            "source": "CTD (Comparative Toxicogenomics Database)",
            "url": CTD_URL,
        },
        "statistics": {
            "drug_distribution": dict(drug_dist.most_common(20)),
            "disease_distribution": dict(disease_dist.most_common(15)),
            "evidence_distribution": dict(evidence_dist),
        },
        "samples": samples,
    }
    
    with open(".tmp/p016_external_ctd_v3_9.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ CTD样本保存: .tmp/p016_external_ctd_v3_9.json")
    print(f"   唯一drug-disease对: {len(samples)}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
