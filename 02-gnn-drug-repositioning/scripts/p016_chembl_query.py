#!/usr/bin/env python3
"""
Project-016 ChEMBL + OpenTargets 外部药物-疾病关联查询
免费REST API，无需key
"""

import json
import time
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime

# ChEMBL API base
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# 目标药物列表（从训练集中提取）
TARGET_MOLECULES = [
    # 小分子药物（有ChEMBL ID或可通过名称查询）
    ("Metformin", "CHEMBL1431"),
    ("Rapamycin", "CHEMBL123"),
    ("Resveratrol", "CHEMBL165"),
    ("Curcumin", "CHEMBL116438"),
    ("Quercetin", "CHEMBL50"),
    ("Fisetin", "CHEMBL31520"),
    ("EGCG", "CHEMBL472076"),
    ("Glucosamine", "CHEMBL181398"),
    ("Chondroitin", None),  # 可能无单独条目
    ("Diclofenac", "CHEMBL139"),
    ("Ibuprofen", "CHEMBL521"),
    ("Celecoxib", "CHEMBL118"),
    ("Naproxen", "CHEMBL154"),
    ("Dexamethasone", "CHEMBL80"),
    ("Triamcinolone", "CHEMBL1451"),
    ("Prednisone", "CHEMBL635"),
    ("Hydrocortisone", "CHEMBL70"),
    ("Duloxetine", "CHEMBL808"),
    ("Pregabalin", "CHEMBL1059"),
    ("Gabapentin", "CHEMBL940"),
    ("Tramadol", "CHEMBL1066"),
    ("Capsaicin", "CHEMBL294199"),
    ("Vitamin D3", "CHEMBL1042"),
    ("Alendronate", "CHEMBL946"),
    ("Atorvastatin", "CHEMBL1480"),
    ("Rosuvastatin", "CHEMBL1499"),
    ("Simvastatin", "CHEMBL1224"),
    ("Omega-3", None),  # 混合物，可能无单一ID
    ("Palmitoylethanolamide", "CHEMBL469292"),
    ("Tofacitinib", "CHEMBL2218934"),
    ("Anakinra", None),  # 生物制剂
    ("Buprenorphine", "CHEMBL1308"),
    ("Methadone", "CHEMBL651"),
    ("Morphine", "CHEMBL70"),
    ("Codeine", "CHEMBL685"),
    ("Oxycodone", "CHEMBL1201730"),
    ("Amitriptyline", "CHEMBL72"),
    ("Nortriptyline", "CHEMBL425"),
    ("Fluoxetine", "CHEMBL41"),
    ("Sertraline", "CHEMBL808"),
    ("Venlafaxine", "CHEMBL637"),
    ("Milnacipran", "CHEMBL91937"),
    ("Carbamazepine", "CHEMBL108"),
    ("Phenytoin", "CHEMBL87"),
    ("Valproic acid", "CHEMBL109"),
    ("Lamotrigine", "CHEMBL741"),
    ("Topiramate", "CHEMBL910"),
    ("Levetiracetam", "CHEMBL1286"),
    ("Clonidine", "CHEMBL193"),
    ("Baclofen", "CHEMBL701"),
    ("Tizanidine", "CHEMBL1158"),
    ("Cyclobenzaprine", "CHEMBL774"),
    ("Methocarbamol", "CHEMBL1201307"),
    ("Carisoprodol", "CHEMBL1237046"),
    ("Lidocaine", "CHEMBL79"),
    ("Ketamine", "CHEMBL198743"),
]

def chembl_get_drug_indications(chembl_id):
    """从ChEMBL获取药物的适应症/疾病关联"""
    if not chembl_id:
        return []
    
    url = f"{CHEMBL_API}/drug_indication.json?molecule_chembl_id={chembl_id}"
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            return data.get('drug_indications', [])
    except Exception as e:
        print(f"  ❌ ChEMBL {chembl_id}: {e}")
        return []
    finally:
        time.sleep(0.3)  # Rate limit

def chembl_search_molecule(name):
    """通过名称搜索ChEMBL分子"""
    encoded = urllib.parse.quote(name)
    url = f"{CHEMBL_API}/molecule.json?q={encoded}"
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            molecules = data.get('molecules', [])
            if molecules:
                return molecules[0].get('molecule_chembl_id')
    except Exception as e:
        print(f"  ❌ Search {name}: {e}")
    finally:
        time.sleep(0.3)
    return None

def main():
    print("=" * 60)
    print("Project-016 ChEMBL 外部药物-疾病关联查询")
    print("=" * 60)
    
    results = []
    cache = {}
    
    for drug_name, chembl_id in TARGET_MOLECULES:
        print(f"\n🔍 {drug_name}...", end=" ")
        
        # 如果没有预置ID，先搜索
        if not chembl_id:
            chembl_id = chembl_search_molecule(drug_name)
            if chembl_id:
                print(f"found {chembl_id}", end=" ")
            else:
                print("not found")
                continue
        
        # 查询适应症
        indications = chembl_get_drug_indications(chembl_id)
        
        if indications:
            print(f"{len(indications)} indications")
            for ind in indications:
                disease_name = ind.get('mesh_heading', ind.get('efo_term', 'Unknown'))
                phase = ind.get('max_phase_for_ind', 'unknown')
                
                # 疾病名称映射
                disease_mapped = None
                disease_lower = disease_name.lower()
                if any(k in disease_lower for k in ['osteoarthritis', 'degenerative joint']):
                    disease_mapped = "Osteoarthritis"
                elif any(k in disease_lower for k in ['rheumatoid arthritis']):
                    disease_mapped = "Rheumatoid arthritis"
                elif any(k in disease_lower for k in ['osteoporosis']):
                    disease_mapped = "Osteoporosis"
                elif any(k in disease_lower for k in ['pain', 'chronic pain']):
                    disease_mapped = "Chronic pain"
                elif any(k in disease_lower for k in ['periodontitis', 'periodontal']):
                    disease_mapped = "Periodontitis"
                elif any(k in disease_lower for k in ['temporomandibular', 'tmj']):
                    disease_mapped = "TMD"
                elif any(k in disease_lower for k in ['fibromyalgia']):
                    disease_mapped = "Fibromyalgia"
                elif any(k in disease_lower for k in ['migraine']):
                    disease_mapped = "Migraine"
                
                if disease_mapped:
                    # 根据临床阶段赋分
                    confidence = 0.3  # default
                    if phase == 4:
                        confidence = 0.5
                    elif phase == 3:
                        confidence = 0.4
                    elif phase == 2:
                        confidence = 0.3
                    elif phase == 1:
                        confidence = 0.2
                    
                    results.append({
                        "drug": drug_name,
                        "disease": disease_mapped,
                        "confidence": confidence,
                        "phase": phase,
                        "source": "ChEMBL",
                        "chembl_id": chembl_id,
                        "indication": disease_name,
                    })
        else:
            print("no indications")
    
    print(f"\n{'='*60}")
    print(f"✅ ChEMBL查询完成: {len(results)} 个drug-disease关联")
    
    # 统计
    from collections import Counter
    drug_dist = Counter(r['drug'] for r in results)
    disease_dist = Counter(r['disease'] for r in results)
    
    print(f"\n涉及药物: {len(drug_dist)}种")
    for d, c in drug_dist.most_common(10):
        print(f"  {d}: {c}")
    
    print(f"\n涉及疾病: {len(disease_dist)}种")
    for dis, c in disease_dist.most_common():
        print(f"  {dis}: {c}")
    
    # 转换为训练样本格式
    samples = []
    seen = set()
    for r in results:
        pair = (r['drug'], r['disease'])
        if pair in seen:
            continue
        seen.add(pair)
        
        samples.append({
            "pmid": "ChEMBL",
            "title": f"ChEMBL: {r['drug']} for {r['disease']}",
            "drug": r['drug'],
            "disease": r['disease'],
            "year": 2024,
            "journal_if": 3.0,
            "design": "ChEMBL_Indication",
            "conclusion": "positive",
            "total_n": None,
            "final_score": r['confidence'],
            "label": r['confidence'],
            "weight": 0.4,
            "is_external": True,
            "external_source": f"ChEMBL drug_indication (phase {r['phase']})",
            "chembl_id": r['chembl_id'],
        })
    
    # 保存
    output = {
        "metadata": {
            "version": "v3_9_chembl",
            "created_at": datetime.now().isoformat(),
            "total_associations": len(results),
            "unique_pairs": len(samples),
            "source": "ChEMBL REST API (https://www.ebi.ac.uk/chembl)",
            "api": "drug_indication endpoint",
        },
        "samples": samples,
    }
    
    with open(".tmp/p016_external_chembl_v3_9.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ ChEMBL外部样本保存: .tmp/p016_external_chembl_v3_9.json")
    print(f"   唯一drug-disease对: {len(samples)}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
