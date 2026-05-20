#!/usr/bin/env python3
"""
Project-016 负样本池扩展 v3.8 — 七大来源策略
目标：正:负 = 1:5（当前1:2.5，需再补充~1500对）

七大来源：
1. 文献标题共现 = 0（重新分析v2核心文献）
2. 已知有害药物（糖皮质激素→骨坏死，化疗→骨代谢抑制）
3. 药物类别排斥（抗生素/抗病毒/精神类/心血管→与TMJOA无关）
4. 药物副作用数据库映射（FDA不良事件: 关节痛为副作用的药物）
5. 反向机制（抑制成骨/促进破骨的药物 = TMJOA恶化）
6. 跨适应症明确阴性（某药物在OA试验中失败的III期临床）
7. 随机药物-疾病组合（基线噪声，低权重）
"""

import json
import random
from collections import Counter
from datetime import datetime

random.seed(42)

# 加载v3.7训练集
with open('.tmp/p016_train_v3_7.json') as f:
    v37 = json.load(f)

all_existing = v37['splits']['train'] + v37['splits']['val'] + v37['splits']['test']
pos_pairs = set((s['drug'], s['disease']) for s in all_existing if s['label'] > 0)
neg_pairs = set((s['drug'], s['disease']) for s in all_existing if s['label'] == 0)
known_drugs = set(s['drug'] for s in all_existing)
known_diseases = set(s['disease'] for s in all_existing)

existing_neg_count = sum(1 for s in all_existing if s['label'] == 0)
total_pos = sum(1 for s in all_existing if s['label'] > 0)
target_neg = int(total_pos * 5)  # 1:5
needed = max(0, target_neg - existing_neg_count)

print(f"当前状态: 正={total_pos}, 负={existing_neg_count}, 比例=1:{existing_neg_count/total_pos:.1f}")
print(f"目标: 1:5 = 负{target_neg}, 还需补充: {needed}")
print()

# ──────────────────────────────────────────
# Source 1: 重新分析v2核心文献标题共现
# ──────────────────────────────────────────
print("Source 1: v2核心文献标题共现分析")
try:
    with open('.tmp/p016_core_v2.json') as f:
        v2 = json.load(f)
    v2_papers = v2.get('papers', v2.get('results', []))
    
    # 提取v2文献中的药物和疾病
    drug_mentions = Counter()
    disease_mentions = Counter()
    cooccurrence = Counter()
    
    for p in v2_papers:
        title = (p.get('title', '') + ' ' + p.get('abstract', '')).lower()
        
        drugs_found = []
        diseases_found = []
        
        drug_kws = {
            "PRP": ["platelet-rich", "prp"],
            "Hyaluronic acid": ["hyaluronic", "hyaluronate"],
            "Corticosteroid": ["corticosteroid", "triamcinolone", "dexamethasone"],
            "NSAIDs": ["nsaid", "ibuprofen", "diclofenac", "celecoxib"],
            "Glucosamine": ["glucosamine"],
            "Chondroitin": ["chondroitin"],
            "Collagen": ["collagen"],
            "Botulinum toxin": ["botulinum toxin", "botox"],
            "Metformin": ["metformin"],
            "Resveratrol": ["resveratrol"],
            "Curcumin": ["curcumin"],
            "Omega-3": ["omega-3", "fish oil"],
            "Vitamin D": ["vitamin d"],
            "Statin": ["statin", "atorvastatin"],
            "Duloxetine": ["duloxetine"],
            "Pregabalin": ["pregabalin"],
            "Gabapentin": ["gabapentin"],
            "Tramadol": ["tramadol"],
            "Capsaicin": ["capsaicin"],
            "Ozone": ["ozone"],
            "Laser therapy": ["laser", "low-level laser"],
            "PEMF": ["pulsed electromagnetic"],
        }
        
        disease_kws = {
            "TMJOA": ["temporomandibular joint osteoarthritis", "tmj osteoarthritis"],
            "TMJ": ["temporomandibular joint", "tmj"],
            "TMD": ["temporomandibular disorder", "tmd"],
            "Osteoarthritis": ["osteoarthritis"],
            "Knee OA": ["knee osteoarthritis"],
            "Osteoporosis": ["osteoporosis"],
        }
        
        for drug, kws in drug_kws.items():
            if any(kw in title for kw in kws):
                drugs_found.append(drug)
        
        for dis, kws in disease_kws.items():
            if any(kw in title for kw in kws):
                diseases_found.append(dis)
        
        for d in drugs_found:
            drug_mentions[d] += 1
        for dis in diseases_found:
            disease_mentions[dis] += 1
        for d in drugs_found:
            for dis in diseases_found:
                cooccurrence[(d, dis)] += 1
    
    # 共现=0 = 负样本候选
    s1_candidates = []
    all_possible_v2 = set()
    for d in drug_mentions:
        for dis in disease_mentions:
            all_possible_v2.add((d, dis))
    
    for pair in all_possible_v2:
        if cooccurrence[pair] == 0:
            if pair not in pos_pairs and pair not in neg_pairs:
                s1_candidates.append({
                    "drug": pair[0], "disease": pair[1],
                    "strategy": "title_cooccurrence_zero",
                    "hardness": "medium",
                    "reason": f"Zero co-occurrence in {len(v2_papers)} v2 core papers"
                })
    
    print(f"  v2文献: {len(v2_papers)}篇")
    print(f"  药物提及: {len(drug_mentions)}种")
    print(f"  疾病提及: {len(disease_mentions)}种")
    print(f"  共现=0候选: {len(s1_candidates)}")
    
except Exception as e:
    print(f"  v2分析失败: {e}")
    s1_candidates = []

# ──────────────────────────────────────────
# Source 2: 已知有害/抑制骨代谢的药物
# ──────────────────────────────────────────
print("\nSource 2: 骨代谢有害药物")

bone_harmful_drugs = [
    # 糖皮质激素 - 明确导致骨坏死/骨质疏松
    "Prednisone", "Prednisolone", "Hydrocortisone", "Betamethasone",
    "Methylprednisolone", "Dexamethasone", "Triamcinolone",
    # 化疗药物 - 抑制骨代谢
    "Cyclophosphamide", "Methotrexate_high_dose", "Doxorubicin",
    "Cisplatin", "5-Fluorouracil", "Paclitaxel",
    # 抗癫痫 - 影响维生素D代谢
    "Phenytoin", "Carbamazepine", "Phenobarbital", "Valproic acid",
    # 质子泵抑制剂 - 影响钙吸收
    "Omeprazole", "Esomeprazole", "Lansoprazole", "Pantoprazole",
    # 噻唑烷二酮 - 增加骨折风险
    "Rosiglitazone", "Pioglitazone",
    # 芳香酶抑制剂 - 增加骨流失
    "Anastrozole", "Letrozole", "Exemestane",
    # GnRH激动剂 - 低雌激素→骨流失
    "Leuprolide", "Goserelin",
    # 抗凝药 - 华法林影响骨代谢
    "Warfarin", "Heparin",
    # 铝制剂 - 抑制成骨
    "Aluminum hydroxide",
    # 锂 - 影响甲状腺功能→骨代谢
    "Lithium",
    # 甲氨蝶呤（长期低剂量也影响骨）
    "Methotrexate",
]

tmj_diseases = ["TMJOA", "TMJ", "TMD", "Osteoarthritis", "Knee OA", "Hip OA", "Osteoporosis"]

s2_candidates = []
for drug in bone_harmful_drugs:
    for disease in tmj_diseases:
        pair = (drug, disease)
        if pair not in pos_pairs and pair not in neg_pairs:
            s2_candidates.append({
                "drug": drug, "disease": disease,
                "strategy": "bone_harmful",
                "hardness": "hard",
                "reason": f"{drug} is known to impair bone metabolism / cause osteonecrosis"
            })

print(f"  骨有害药物: {len(bone_harmful_drugs)}种")
print(f"  候选负样本: {len(s2_candidates)}")

# ──────────────────────────────────────────
# Source 3: 药物类别排斥（与TMJOA完全无关）
# ──────────────────────────────────────────
print("\nSource 3: 无关药物类别")

unrelated_drugs = [
    # 抗生素（绝大多数与TMJOA无关）
    "Amoxicillin", "Azithromycin", "Ciprofloxacin", "Clarithromycin",
    "Doxycycline", "Levofloxacin", "Metronidazole", "Moxifloxacin",
    # 抗病毒
    "Acyclovir", "Valacyclovir", "Oseltamivir", "Remdesivir",
    # 抗真菌
    "Fluconazole", "Itraconazole", "Voriconazole",
    # 心血管（部分可能相关，但大部分是无关的）
    "Amlodipine", "Losartan", "Enalapril", "Metoprolol",
    "Atenolol", "Digoxin", "Furosemide", "Spironolactone",
    # 降糖（除Metformin外）
    "Insulin", "Glipizide", "Glyburide", "Sitagliptin",
    # 精神类（部分可能用于慢性疼痛，但非TMJOA治疗）
    "Haloperidol", "Risperidone", "Olanzapine", "Quetiapine",
    "Sertraline", "Fluoxetine", "Escitalopram", "Mirtazapine",
    # 抗组胺
    "Cetirizine", "Loratadine", "Fexofenadine", "Diphenhydramine",
    # 消化系统
    "Ranitidine", "Famotidine", "Domperidone", "Metoclopramide",
    "Loperamide", "Senna", "Bisacodyl",
    # 呼吸系统
    "Salbutamol", "Ipratropium", "Montelukast", "Budesonide",
    # 泌尿系统
    "Tamsulosin", "Finasteride", "Sildenafil", "Tadalafil",
    # 眼科
    "Timolol", "Latanoprost", "Brimonidine",
    # 皮肤科
    "Isotretinoin", "Adapalene", "Tretinoin",
    # 麻醉（非镇痛）
    "Propofol", "Ketamine_anesthesia", "Etomidate", "Midazolam",
]

s3_candidates = []
for drug in unrelated_drugs:
    for disease in tmj_diseases:
        pair = (drug, disease)
        if pair not in pos_pairs and pair not in neg_pairs:
            s3_candidates.append({
                "drug": drug, "disease": disease,
                "strategy": "unrelated_drug_class",
                "hardness": "soft",
                "reason": f"{drug} is not a treatment for musculoskeletal/joint conditions"
            })

print(f"  无关药物: {len(unrelated_drugs)}种")
print(f"  候选负样本: {len(s3_candidates)}")

# ──────────────────────────────────────────
# Source 4: FDA不良事件 - 关节痛为副作用
# ──────────────────────────────────────────
print("\nSource 4: 关节痛副作用药物")

arthralgia_drugs = [
    # 他汀类（已部分包含在正样本中，但这里标记为副作用）
    "Atorvastatin", "Rosuvastatin", "Simvastatin",
    # 双膦酸盐（已部分包含，但注意：某些情况下可能导致颌骨坏死）
    "Alendronate", "Zoledronic acid", "Risedronate",
    # 芳香酶抑制剂
    "Letrozole", "Anastrozole",
    # 维A酸类
    "Isotretinoin", "Acitretin",
    # 氟喹诺酮类
    "Ciprofloxacin", "Levofloxacin", "Moxifloxacin",
    # 抗癫痫（部分）
    "Carbamazepine", "Phenytoin",
    # 免疫检查点抑制剂（癌症免疫治疗）
    "Pembrolizumab", "Nivolumab", "Ipilimumab",
    # TNF抑制剂（ Paradoxical: 用于RA但可能引起新发自身免疫）
    "Infliximab", "Adalimumab", "Etanercept",
    # JAK抑制剂（用于RA，但可能增加感染风险）
    "Tofacitinib", "Baricitinib",
    # 其他
    "Teriparatide", "Denosumab",  # 注意：这两个是正样本药物，谨慎处理
]

s4_candidates = []
for drug in arthralgia_drugs:
    # 跳过已知的正样本药物
    if drug in known_drugs:
        continue
    for disease in tmj_diseases:
        pair = (drug, disease)
        if pair not in pos_pairs and pair not in neg_pairs:
            s4_candidates.append({
                "drug": drug, "disease": disease,
                "strategy": "arthralgia_side_effect",
                "hardness": "medium",
                "reason": f"{drug} has arthralgia/joint pain as reported adverse effect"
            })

print(f"  关节痛副作用药物: {len(arthralgia_drugs)}种（排除已知正样本后）")
print(f"  候选负样本: {len(s4_candidates)}")

# ──────────────────────────────────────────
# Source 5: 反向机制（抑制软骨修复/促进炎症）
# ──────────────────────────────────────────
print("\nSource 5: 反向机制药物")

cartilage_inhibitory = [
    # COX-2选择性抑制剂争议（长期使用可能影响软骨修复）
    "Rofecoxib",  # 已撤市，但可用于训练
    # 某些NSAIDs高剂量长期使用
    "Indomethacin_high_dose",
    # 糖皮质激素（再次强调）
    "Corticosteroid_systemic_longterm",
    # 环孢素
    "Cyclosporine",
    # 某些化疗
    "Doxorubicin", "Cisplatin",
]

s5_candidates = []
for drug in cartilage_inhibitory:
    for disease in tmj_diseases:
        pair = (drug, disease)
        if pair not in pos_pairs and pair not in neg_pairs:
            s5_candidates.append({
                "drug": drug, "disease": disease,
                "strategy": "cartilage_inhibitory",
                "hardness": "hard",
                "reason": f"{drug} may inhibit cartilage repair or promote joint degeneration"
            })

print(f"  软骨抑制药物: {len(cartilage_inhibitory)}种")
print(f"  候选负样本: {len(s5_candidates)}")

# ──────────────────────────────────────────
# Source 6: OA试验失败的III期临床药物
# ──────────────────────────────────────────
print("\nSource 6: OA临床试验失败药物")

failed_oa_trials = [
    # 已被证实对OA无效或撤市的药物
    "Rofecoxib",  # 撤市
    "Lumiracoxib",  # 撤市
    "Tanezumab",  # 部分有效但副作用大（注意：已在正样本中）
    "Fasinumab",  # 抗NGF，因关节安全停药
    "Fulranumab",  # 抗NGF，失败
    "GSK2831781",  # 抗IL-18，OA III期失败
    "Lorentixont",  # 抗炎，OA失败
    "M40403",  # SOD模拟物，OA失败
    "Diacerein",  # 部分有效但胃肠道副作用
    "Avocado-soybean unsaponifiables",  # 注意：已在正样本中
]

s6_candidates = []
for drug in failed_oa_trials:
    if drug in known_drugs:
        continue  # 已在训练集中，跳过
    for disease in tmj_diseases:
        pair = (drug, disease)
        if pair not in pos_pairs and pair not in neg_pairs:
            s6_candidates.append({
                "drug": drug, "disease": disease,
                "strategy": "failed_oa_trial",
                "hardness": "hard",
                "reason": f"{drug} failed in OA clinical trials or was withdrawn"
            })

print(f"  失败药物: {len(failed_oa_trials)}种（排除已知正样本后）")
print(f"  候选负样本: {len(s6_candidates)}")

# ──────────────────────────────────────────
# Source 7: 随机噪声基线（大量低权重负样本）
# ──────────────────────────────────────────
print("\nSource 7: 随机噪声基线")

# 更多随机药物
random_drugs = [
    "Aspirin", "Acetaminophen", "Paracetamol", "Ibuprofen", "Naproxen",
    "Ketoprofen", "Flurbiprofen", "Diclofenac", "Celecoxib", "Etoricoxib",
    "Lornoxicam", "Tenoxicam", "Piroxicam", "Meloxicam",
    "Amitriptyline", "Nortriptyline", "Venlafaxine", "Milnacipran",
    "Topiramate", "Lamotrigine", "Oxcarbazepine", "Levetiracetam",
    "Lidocaine", "Benzocaine", "Tetracaine", "Bupivacaine",
    "Methylphenidate", "Modafinil", "Atomoxetine",
    "Baclofen", "Tizanidine", "Cyclobenzaprine", "Orphenadrine",
    "Methocarbamol", "Carisoprodol",
    "Naloxone", "Naltrexone", "Buprenorphine", "Methadone",
    "Codeine", "Morphine", "Oxycodone", "Hydrocodone", "Fentanyl",
    "Tapentadol", "Alfentanil", "Sufentanil",
    "Ketamine", "Esketamine",
    "Nitrous oxide", "Xenon",
    "Clonidine", "Guanfacine", "Tizanidine",
    "Ziconotide",
    "Caffeine", "Theophylline",
    "Nicotine", "Varenicline", "Bupropion",
    "Sodium valproate", "Lamotrigine", "Gabapentin", "Pregabalin",
    "Levetiracetam", "Topiramate", "Zonisamide",
    "Riluzole", "Edaravone",
]

s7_candidates = []
for drug in random_drugs:
    if drug in known_drugs:
        continue
    for disease in tmj_diseases:
        pair = (drug, disease)
        if pair not in pos_pairs and pair not in neg_pairs:
            s7_candidates.append({
                "drug": drug, "disease": disease,
                "strategy": "random_baseline",
                "hardness": "soft",
                "reason": f"Random drug-disease pair for baseline noise ({drug} not known for {disease})"
            })

print(f"  随机药物: {len(random_drugs)}种")
print(f"  候选负样本: {len(s7_candidates)}")

# ──────────────────────────────────────────
# 合并与采样
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("合并所有候选")

all_candidates = s1_candidates + s2_candidates + s3_candidates + s4_candidates + s5_candidates + s6_candidates + s7_candidates

# 去重
seen = set()
unique_candidates = []
for c in all_candidates:
    pair = (c['drug'], c['disease'])
    if pair not in seen:
        seen.add(pair)
        unique_candidates.append(c)

print(f"\n总候选（去重前）: {len(all_candidates)}")
print(f"总候选（去重后）: {len(unique_candidates)}")

# 按硬度分层采样
hard_candidates = [c for c in unique_candidates if c['hardness'] == 'hard']
medium_candidates = [c for c in unique_candidates if c['hardness'] == 'medium']
soft_candidates = [c for c in unique_candidates if c['hardness'] == 'soft']

print(f"\n分层统计:")
print(f"  硬: {len(hard_candidates)}")
print(f"  中: {len(medium_candidates)}")
print(f"  软: {len(soft_candidates)}")

# 采样策略：按硬度优先，但不超过needed
selected = []
selected.extend(hard_candidates)  # 全部硬负样本
selected.extend(medium_candidates)  # 全部中负样本

remaining = needed - len(selected)
if remaining > 0 and soft_candidates:
    selected.extend(random.sample(soft_candidates, min(remaining, len(soft_candidates))))

print(f"\n采样后新增负样本: {len(selected)}")
print(f"  硬: {sum(1 for s in selected if s['hardness']=='hard')}")
print(f"  中: {sum(1 for s in selected if s['hardness']=='medium')}")
print(f"  软: {sum(1 for s in selected if s['hardness']=='soft')}")

# 转换为训练样本
neg_samples = []
for c in selected:
    weight_map = {'hard': 0.8, 'medium': 0.5, 'soft': 0.2}
    neg_samples.append({
        "pmid": "synthetic_v3_8",
        "title": f"Negative: {c['drug']} for {c['disease']}",
        "drug": c['drug'], "disease": c['disease'],
        "year": 2024, "journal_if": 2.0,
        "design": c['strategy'], "conclusion": "negative",
        "total_n": None, "final_score": -0.5,
        "label": 0.0, "weight": weight_map[c['hardness']],
        "is_synthetic": True, "hardness": c['hardness'],
        "strategy": c['strategy'],
        "is_repositioning_star": c['drug'] in {"Metformin", "Rapamycin", "Resveratrol", "Curcumin", "Quercetin"},
    })

# 保存
output = {
    "metadata": {
        "version": "v3_8",
        "created_at": datetime.now().isoformat(),
        "total_candidates": len(unique_candidates),
        "selected_samples": len(neg_samples),
        "needed": needed,
        "strategies": {
            "title_cooccurrence": len(s1_candidates),
            "bone_harmful": len(s2_candidates),
            "unrelated_class": len(s3_candidates),
            "arthralgia_side_effect": len(s4_candidates),
            "cartilage_inhibitory": len(s5_candidates),
            "failed_trial": len(s6_candidates),
            "random_baseline": len(s7_candidates),
        },
    },
    "candidates": unique_candidates,
    "selected_negatives": neg_samples,
}

with open('.tmp/p016_negative_pool_v3_8.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 负样本池v3.8已保存: .tmp/p016_negative_pool_v3_8.json")
print(f"   新增负样本: {len(neg_samples)}")
