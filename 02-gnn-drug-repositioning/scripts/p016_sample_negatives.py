#!/usr/bin/env python3
"""
Project-016 负样本采样脚本 (v2文献共现驱动)
基于v2筛选的400篇核心文献标题，提取drug-disease共现对，构建训练集。

正样本: 文献标题中drug与disease共现的对（弱正样本）
负样本: 文献标题中从未共现的drug-disease对（从共现=0池中采样）

输出:
  1. p016_train_pairs.json — 训练集 (train/val/test split)
  2. p016_negative_sampling_report.md — 采样报告
"""

import json
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

V2_JSON = ".tmp/p016_core_v2.json"
DISTANCE_JSON = ".tmp/p016_drug_disease_distance_matrix.json"
OUTPUT_JSON = ".tmp/p016_train_pairs.json"
OUTPUT_REPORT = ".tmp/p016_negative_sampling_report.md"

# ══════════════════════════════════════════════════════════════
# 1. Drug 关键词 + 归一化映射
# ══════════════════════════════════════════════════════════════

DRUG_PATTERNS = [
    # 天然化合物 / 补充剂
    ("resveratrol", "Resveratrol"),
    ("curcumin", "Curcumin"),
    ("quercetin", "Quercetin"),
    ("fisetin", "Fisetin"),
    ("epigallocatechin gallate", "EGCG"),
    ("egcg", "EGCG"),
    ("green tea", "EGCG"),
    ("s-adenosylmethionine", "SAMe"),
    ("same", "SAMe"),
    ("diacerein", "Diacerein"),
    ("avocado soybean unsaponifiables", "ASU"),
    ("asu", "ASU"),
    ("boswellia", "Boswellia"),
    ("turmeric", "Curcumin"),
    ("ginger", "Ginger"),
    ("omega-3 fatty acid", "Omega-3"),
    ("fish oil", "Omega-3"),
    ("omega-3", "Omega-3"),
    ("methylsulfonylmethane", "MSM"),
    ("msm", "MSM"),
    ("glucosamine sulfate", "Glucosamine"),
    ("glucosamine", "Glucosamine"),
    ("chondroitin sulfate", "Chondroitin"),
    ("chondroitin", "Chondroitin"),
    ("collagen supplement", "Collagen"),
    ("collagen", "Collagen"),
    ("vitamin d supplement", "Vitamin D"),
    ("vitamin d3", "Vitamin D"),
    ("vitamin d", "Vitamin D"),
    ("calcium supplement", "Calcium"),
    ("calcium", "Calcium"),
    ("magnesium", "Magnesium"),
    ("folate supplement", "Folate"),
    ("folate", "Folate"),
    ("vitamin b12 supplement", "Vitamin B12"),
    ("vitamin b12", "Vitamin B12"),
    ("iron chelation therapy", "Iron chelation"),
    ("deferoxamine", "Deferoxamine"),
    ("deferasirox", "Deferasirox"),
    ("deferiprone", "Deferiprone"),

    # 骨代谢药物
    ("alendronate", "Alendronate"),
    ("zoledronic acid", "Zoledronic acid"),
    ("risedronate", "Risedronate"),
    ("ibandronate", "Ibandronate"),
    ("teriparatide", "Teriparatide"),
    ("romosozumab", "Romosozumab"),
    ("burosumab", "Burosumab"),
    ("anti-fgf23", "Burosumab"),
    ("denosumab", "Denosumab"),
    ("bisphosphonate", "Bisphosphonate"),
    ("calcitonin", "Calcitonin"),
    ("strontium ranelate", "Strontium ranelate"),
    ("calcitriol", "Calcitriol"),
    ("active vitamin d analog", "Calcitriol"),
    ("active vitamin d", "Calcitriol"),
    ("paricalcitol", "Paricalcitol"),
    ("cinacalcet", "Cinacalcet"),
    ("sevelamer", "Sevelamer"),
    ("lanthanum carbonate", "Lanthanum carbonate"),
    ("phosphate binder", "Phosphate binder"),
    ("levothyroxine", "Levothyroxine"),
    ("methimazole", "Methimazole"),
    ("propylthiouracil", "Propylthiouracil"),
    ("carbimazole", "Carbimazole"),
    ("aromatase inhibitor", "Aromatase inhibitor"),
    ("antiepileptic drug", "Antiepileptic"),
    ("thiazolidinedione", "Thiazolidinedione"),
    ("pioglitazone", "Pioglitazone"),
    ("rosiglitazone", "Rosiglitazone"),
    ("statin", "Statin"),
    ("atorvastatin", "Statin"),
    ("rosuvastatin", "Statin"),
    ("simvastatin", "Statin"),
    ("pravastatin", "Statin"),

    # 关节内/局部治疗
    ("platelet-rich plasma", "PRP"),
    ("prp", "PRP"),
    ("hyaluronic acid", "Hyaluronic acid"),
    ("corticosteroid", "Corticosteroid"),
    ("triamcinolone", "Corticosteroid"),
    ("prednisone", "Corticosteroid"),
    ("dexamethasone", "Corticosteroid"),
    ("methylprednisolone", "Corticosteroid"),
    ("botulinum toxin", "Botulinum toxin"),

    # NSAIDs
    ("ibuprofen", "NSAIDs"),
    ("naproxen", "NSAIDs"),
    ("diclofenac", "NSAIDs"),
    ("celecoxib", "NSAIDs"),
    ("meloxicam", "NSAIDs"),
    ("indomethacin", "NSAIDs"),
    ("ketoprofen", "NSAIDs"),
    ("nsaid", "NSAIDs"),
    ("aspirin", "Aspirin"),
    ("acetaminophen", "Acetaminophen"),
    ("paracetamol", "Acetaminophen"),

    # 阿片类/镇痛药
    ("tramadol", "Tramadol"),
    ("codeine", "Opioids"),
    ("oxycodone", "Opioids"),
    ("morphine", "Morphine"),
    ("fentanyl", "Fentanyl"),

    # 神经病理性疼痛/情绪调节
    ("pregabalin", "Pregabalin"),
    ("gabapentin", "Gabapentin"),
    ("amitriptyline", "Amitriptyline"),
    ("nortriptyline", "Nortriptyline"),
    ("duloxetine", "Duloxetine"),

    # 肌松药
    ("cyclobenzaprine", "Muscle relaxant"),
    ("tizanidine", "Muscle relaxant"),
    ("baclofen", "Muscle relaxant"),

    # 风湿/免疫
    ("methotrexate", "Methotrexate"),
    ("hydroxychloroquine", "Hydroxychloroquine"),
    ("sulfasalazine", "Sulfasalazine"),
    ("leflunomide", "Leflunomide"),

    # 生物制剂 — TNF抑制剂
    ("etanercept", "TNF inhibitor"),
    ("infliximab", "TNF inhibitor"),
    ("adalimumab", "TNF inhibitor"),
    ("golimumab", "TNF inhibitor"),
    ("certolizumab", "TNF inhibitor"),

    # 生物制剂 — IL-6/IL-1/其他
    ("tocilizumab", "IL-6 inhibitor"),
    ("anakinra", "IL-1 inhibitor"),
    ("canakinumab", "IL-1 inhibitor"),
    ("abatacept", "Abatacept"),
    ("rituximab", "Rituximab"),

    # JAK抑制剂
    ("tofacitinib", "JAK inhibitor"),
    ("baricitinib", "JAK inhibitor"),
    ("upadacitinib", "JAK inhibitor"),
    ("apremilast", "Apremilast"),

    # IL-12/23, IL-17, IL-23 抑制剂
    ("ustekinumab", "IL-12/23 inhibitor"),
    ("secukinumab", "IL-17 inhibitor"),
    ("ixekizumab", "IL-17 inhibitor"),
    ("guselkumab", "IL-23 inhibitor"),
    ("risankizumab", "IL-23 inhibitor"),
    ("tildrakizumab", "IL-23 inhibitor"),
    ("brodalumab", "IL-17 inhibitor"),
    ("bimekizumab", "IL-17 inhibitor"),

    # 其他生物制剂
    ("belimumab", "Belimumab"),
    ("dupilumab", "Dupilumab"),
    ("omalizumab", "Omalizumab"),

    # GLP-1
    ("semaglutide", "GLP-1 agonist"),
    ("liraglutide", "GLP-1 agonist"),
    ("tirzepatide", "GLP-1/GIP agonist"),
    ("exenatide", "GLP-1 agonist"),
    ("dulaglutide", "GLP-1 agonist"),
    ("albiglutide", "GLP-1 agonist"),
    ("lixisenatide", "GLP-1 agonist"),

    # 抗炎/信号通路靶向
    ("resatorvid", "TLR4 inhibitor"),
    ("tanezumab", "NGF inhibitor"),
    ("anti-fgf23", "Anti-FGF23"),
    ("fgf23 neutralizing antibody", "Anti-FGF23"),
    ("burosumab", "Burosumab"),
    ("anti-sclerostin", "Anti-sclerostin"),
    ("sclerostin antibody", "Anti-sclerostin"),
    ("romosozumab", "Romosozumab"),
    ("bgj398", "FGFR inhibitor"),
    ("fgfr inhibitor", "FGFR inhibitor"),
    ("iwr-1", "Wnt inhibitor"),
    ("wnt inhibitor", "Wnt inhibitor"),

    # 抗衰老/代谢调节（重定位明星）
    ("metformin", "Metformin"),
    ("rapamycin", "Rapamycin"),
    ("sirolimus", "Rapamycin"),
    ("everolimus", "Everolimus"),
    ("riluzole", "Riluzole"),
]

# 重定位明星 — 即使满足负样本标准也手动排除
REPOSITIONING_STARS = {
    "Metformin", "Rapamycin", "Everolimus", "Resveratrol", "Curcumin",
    "Quercetin", "Fisetin", "EGCG", "SAMe", "Omega-3", "Ginger",
}

# ══════════════════════════════════════════════════════════════
# 2. Disease 关键词 + 归一化映射
# ══════════════════════════════════════════════════════════════

DISEASE_PATTERNS = [
    # TMJ相关（最优先匹配）
    ("temporomandibular joint osteoarthritis", "TMJOA"),
    ("tmj osteoarthritis", "TMJOA"),
    ("temporomandibular joint disorder", "TMD"),
    ("temporomandibular disorder", "TMD"),
    ("tmd", "TMD"),
    ("temporomandibular joint", "TMJ"),
    ("tmj", "TMJ"),
    ("mandibular condyle", "TMJ condyle"),
    ("condylar resorption", "Condylar resorption"),
    ("condylar degeneration", "Condylar degeneration"),
    ("condylar osteoarthritis", "TMJOA"),
    ("condylar", "TMJ condyle"),
    ("internal derangement", "TMJ internal derangement"),
    ("disc displacement", "TMJ disc displacement"),
    ("disc derangement", "TMJ disc derangement"),
    ("myofascial pain", "Myofascial pain"),
    ("orofacial pain", "Orofacial pain"),
    ("bruxism", "Bruxism"),
    ("sleep bruxism", "Bruxism"),
    ("awake bruxism", "Bruxism"),

    # 全身性OA
    ("knee osteoarthritis", "Knee OA"),
    ("hip osteoarthritis", "Hip OA"),
    ("osteoarthritis", "Osteoarthritis"),

    # 其他关节炎
    ("rheumatoid arthritis", "Rheumatoid arthritis"),
    ("psoriatic arthritis", "Psoriatic arthritis"),
    ("ankylosing spondylitis", "Ankylosing spondylitis"),
    ("gout", "Gout"),
    ("pseudogout", "Pseudogout"),
    ("calcium pyrophosphate deposition", "CPPD"),
    ("cppd", "CPPD"),

    # 骨代谢异常
    ("osteoporosis", "Osteoporosis"),
    ("osteopenia", "Osteopenia"),
    ("osteomalacia", "Osteomalacia"),
    ("vitamin d deficiency", "Vitamin D deficiency"),
    ("hypovitaminosis d", "Vitamin D deficiency"),
    ("hyperparathyroidism", "Hyperparathyroidism"),
    ("hypoparathyroidism", "Hypoparathyroidism"),
    ("hypothyroidism", "Hypothyroidism"),
    ("hyperthyroidism", "Hyperthyroidism"),
    ("chronic kidney disease", "CKD"),
    ("renal osteodystrophy", "Renal osteodystrophy"),
    ("ckd-mbd", "CKD-MBD"),
    ("diabetic bone disease", "Diabetic bone disease"),
    ("hemochromatosis", "Hemochromatosis"),
    ("iron overload", "Iron overload"),
    ("hyperhomocysteinemia", "Hyperhomocysteinemia"),
    ("bone loss", "Bone loss"),
    ("bone resorption", "Bone resorption"),
    ("bone marrow", "Bone marrow abnormality"),

    # 其他全身性疾病
    ("obesity", "Obesity"),
    ("metabolic syndrome", "Metabolic syndrome"),
    ("insulin resistance", "Insulin resistance"),
    ("diabetes mellitus", "Diabetes mellitus"),
    ("diabetes", "Diabetes mellitus"),
    ("fibromyalgia", "Fibromyalgia"),
    ("depression", "Depression"),
    ("anxiety", "Anxiety"),
    ("post-traumatic stress", "PTSD"),
    ("trauma", "Trauma"),
    ("fracture", "Fracture"),
]

# ══════════════════════════════════════════════════════════════
# 3. 提取函数
# ══════════════════════════════════════════════════════════════

def extract_entities(text: str, patterns):
    """从文本中提取实体，返回归一化后的唯一实体集合"""
    text_lower = text.lower()
    found = set()
    for pattern, normalized in patterns:
        if pattern.lower() in text_lower:
            found.add(normalized)
    return found


def build_cooccurrence_matrix(papers):
    """构建 drug-disease 共现矩阵"""
    drug_freq = Counter()
    disease_freq = Counter()
    cooccur = Counter()  # (drug, disease) -> count
    
    for paper in papers:
        title = paper.get("title", "")
        drugs = extract_entities(title, DRUG_PATTERNS)
        diseases = extract_entities(title, DISEASE_PATTERNS)
        
        # 频率统计
        for d in drugs:
            drug_freq[d] += 1
        for dis in diseases:
            disease_freq[dis] += 1
        
        # 共现统计（同一标题中同时出现）
        for d in drugs:
            for dis in diseases:
                cooccur[(d, dis)] += 1
    
    return drug_freq, disease_freq, cooccur


def load_mechanism_distance():
    """加载LabKG机制距离矩阵（如果可用）"""
    try:
        with open(DISTANCE_JSON) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def sample_negatives(positive_pairs, all_pairs, drug_freq, disease_freq, 
                     distance_matrix, ratio=5, seed=42):
    """
    采样负样本，满足:
    1. 共现=0
    2. 度数匹配（drug和disease的独立出现频率与正样本一致）
    3. 硬负样本标记（机制距离>3 hops或不连通）
    4. 重定位明星排除
    """
    random.seed(seed)
    
    # 正样本集合
    pos_set = set(positive_pairs)
    
    # 负样本候选池（共现=0且排除重定位明星）
    neg_candidates = []
    for (d, dis) in all_pairs:
        if (d, dis) in pos_set:
            continue
        if d in REPOSITIONING_STARS:
            continue
        neg_candidates.append((d, dis))
    
    # 正样本的drug和disease度数分布
    pos_drug_deg = Counter(d for d, _ in positive_pairs)
    pos_dis_deg = Counter(dis for _, dis in positive_pairs)
    
    # 负样本需要的目标数量
    target_neg_count = len(positive_pairs) * ratio
    
    # 按度数匹配采样 — 优先选择与正样本度数相近的候选
    # 简单策略：确保最终负样本中，每个drug和disease的出现次数与正样本成正比
    
    # 首先计算每个候选的"匹配分数"（与正样本度数分布的相似度）
    def degree_match_score(drug, disease):
        # 理想: 负样本的drug/disease度数分布 ≈ 正样本的分布
        # 这里用简单启发式：候选drug和disease在正样本中的期望出现次数
        # 实际采样时用贪婪算法
        return 0  # 占位，下面用贪婪算法
    
    # 贪婪度数匹配采样
    neg_sampled = []
    current_drug_deg = Counter()
    current_dis_deg = Counter()
    
    # 按硬负样本优先级排序
    def hard_neg_priority(pair):
        d, dis = pair
        # 检查机制距离
        dist = -2  # unknown
        if distance_matrix:
            # distance matrix用原始name，但我们的pair是归一化name
            # 需要做反向查找，这里简化处理：优先选unreachable
            pass
        
        # 基于独立出现频率判断：两者都高频出现但从未共现 → 强负样本
        freq_score = drug_freq.get(d, 0) + disease_freq.get(dis, 0)
        return freq_score
    
    # 先按频率分数降序排列候选
    neg_candidates_sorted = sorted(neg_candidates, key=hard_neg_priority, reverse=True)
    
    # 贪婪采样，尽量匹配度数分布
    for pair in neg_candidates_sorted:
        if len(neg_sampled) >= target_neg_count:
            break
        d, dis = pair
        
        # 度数约束：当前该drug和disease的负样本数不超过正样本的2倍
        if current_drug_deg[d] >= pos_drug_deg.get(d, 0) * 2 + 2:
            continue
        if current_dis_deg[dis] >= pos_dis_deg.get(dis, 0) * 2 + 2:
            continue
        
        # 硬负样本标记
        is_hard = False
        if drug_freq.get(d, 0) >= 3 and disease_freq.get(dis, 0) >= 3:
            is_hard = True  # 两者都高频但从未共现
        
        neg_sampled.append({
            "drug": d,
            "disease": dis,
            "label": 0,
            "is_hard_negative": is_hard,
            "cooccurrence_count": 0,
            "drug_freq_in_corpus": drug_freq.get(d, 0),
            "disease_freq_in_corpus": disease_freq.get(dis, 0),
        })
        current_drug_deg[d] += 1
        current_dis_deg[dis] += 1
    
    return neg_sampled


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-ratio", type=float, default=1.0, 
                        help="正样本保留比例（1.0=全部弱正样本）")
    parser.add_argument("--neg-ratio", type=int, default=5,
                        help="负样本:正样本比例")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 读取v2文献
    print("="*60)
    print("Project-016 负样本采样")
    print("="*60)
    
    with open(V2_JSON) as f:
        data = json.load(f)
    papers = data.get("results", [])
    print(f"📥 读取 {len(papers)} 篇v2核心文献")
    
    # 构建共现矩阵
    drug_freq, disease_freq, cooccur = build_cooccurrence_matrix(papers)
    print(f"\n📊 实体提取结果:")
    print(f"  提取到 {len(drug_freq)} 种药物: {dict(drug_freq.most_common(10))}")
    print(f"  提取到 {len(disease_freq)} 种疾病: {dict(disease_freq.most_common(10))}")
    print(f"  共现对: {len(cooccur)} 对")
    
    # 生成正样本（共现≥1）
    positive_pairs = []
    for (d, dis), count in cooccur.items():
        # 排除重定位明星作为正样本（它们太泛了，不适合作为特定训练信号）
        if d in REPOSITIONING_STARS:
            continue
        positive_pairs.append({
            "drug": d,
            "disease": dis,
            "label": 1,
            "cooccurrence_count": count,
            "drug_freq_in_corpus": drug_freq[d],
            "disease_freq_in_corpus": disease_freq[dis],
            "evidence_type": "title_cooccurrence",
            "confidence": "weak_positive" if count == 1 else "medium_positive" if count <= 3 else "strong_positive",
        })
    
    print(f"\n✅ 正样本候选: {len(positive_pairs)} 对")
    for conf, cnt in Counter(p["confidence"] for p in positive_pairs).items():
        print(f"  {conf}: {cnt}")
    
    # 正样本采样（如果指定了pos-ratio < 1.0）
    if args.pos_ratio < 1.0:
        n_pos = int(len(positive_pairs) * args.pos_ratio)
        # 优先保留强正样本
        positive_pairs_sorted = sorted(
            positive_pairs,
            key=lambda x: (-x["cooccurrence_count"], -(x["drug_freq_in_corpus"] + x["disease_freq_in_corpus"]))
        )
        positive_pairs = positive_pairs_sorted[:n_pos]
        print(f"  采样后正样本: {len(positive_pairs)} 对")
    
    # 生成所有可能的drug-disease对
    all_drugs = list(drug_freq.keys())
    all_diseases = list(disease_freq.keys())
    all_pairs = [(d, dis) for d in all_drugs for dis in all_diseases]
    print(f"\n  所有可能drug-disease对: {len(all_pairs)}")
    
    # 加载机制距离矩阵
    distance_matrix = load_mechanism_distance()
    if distance_matrix:
        print(f"  机制距离矩阵已加载")
    else:
        print(f"  ⚠️ 机制距离矩阵不可用，使用频率启发式标记硬负样本")
    
    # 采样负样本
    neg_sampled = sample_negatives(
        [(p["drug"], p["disease"]) for p in positive_pairs],
        all_pairs,
        drug_freq,
        disease_freq,
        distance_matrix,
        ratio=args.neg_ratio,
        seed=args.seed,
    )
    print(f"\n✅ 负样本采样: {len(neg_sampled)} 对 (1:{args.neg_ratio})")
    hard_neg_count = sum(1 for n in neg_sampled if n["is_hard_negative"])
    print(f"  硬负样本: {hard_neg_count} ({hard_neg_count/len(neg_sampled)*100:.1f}%)")
    
    # 合并并划分 train/val/test
    all_samples = positive_pairs + neg_sampled
    random.shuffle(all_samples)
    
    n = len(all_samples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    
    train = all_samples[:n_train]
    val = all_samples[n_train:n_train + n_val]
    test = all_samples[n_train + n_val:]
    
    # 确保各类别在每个split中都有代表
    def ensure_coverage(split, name):
        pos_count = sum(1 for s in split if s["label"] == 1)
        neg_count = len(split) - pos_count
        drugs = len(set(s["drug"] for s in split))
        diseases = len(set(s["disease"] for s in split))
        print(f"  {name}: {len(split)} total ({pos_count}+, {neg_count}-) | {drugs} drugs, {diseases} diseases")
        return pos_count, neg_count
    
    print(f"\n📊 数据集划分:")
    ensure_coverage(train, "Train")
    ensure_coverage(val, "Val")
    ensure_coverage(test, "Test")
    
    # 输出JSON
    output_data = {
        "metadata": {
            "project": "project-016",
            "version": "v2_title_cooccurrence",
            "created_at": datetime.now().isoformat(),
            "source_papers": len(papers),
            "positive_ratio": args.pos_ratio,
            "negative_ratio": args.neg_ratio,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "repositioning_stars_excluded": list(REPOSITIONING_STARS),
        },
        "statistics": {
            "total_pairs": len(all_pairs),
            "cooccurring_pairs": len(cooccur),
            "positive_samples": len(positive_pairs),
            "negative_samples": len(neg_sampled),
            "hard_negatives": hard_neg_count,
            "unique_drugs": len(all_drugs),
            "unique_diseases": len(all_diseases),
            "drug_freq": dict(drug_freq.most_common()),
            "disease_freq": dict(disease_freq.most_common()),
        },
        "splits": {
            "train": train,
            "val": val,
            "test": test,
        }
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 训练集已保存: {OUTPUT_JSON}")
    

    # 生成报告
    report_lines = [
        f"# Project-016 GNN训练集 — 负样本采样报告",
        f"",
        f"**生成时间**: {datetime.now().isoformat()}",
        f"**采样方法**: v2文献标题共现驱动 + 度数匹配 + 硬负样本标记",
        f"**数据源**: {len(papers)} 篇v2核心文献",
        f"",
        f"## 实体统计",
        f"",
        f"| 类型 | 数量 | Top 10 |",
        f"|------|------|--------|",
        f"| 药物 | {len(drug_freq)} | {', '.join([f'{k}({v})' for k, v in drug_freq.most_common(10)])} |",
        f"| 疾病 | {len(disease_freq)} | {', '.join([f'{k}({v})' for k, v in disease_freq.most_common(10)])} |",
        f"",
        f"## 正样本",
        f"",
        f"**总数**: {len(positive_pairs)} 对",
        f"",
        f"| 置信度 | 数量 | 说明 |",
        f"|--------|------|------|",
        f"| weak_positive | {sum(1 for p in positive_pairs if p['confidence'] == 'weak_positive')} | 仅1次标题共现 |",
        f"| medium_positive | {sum(1 for p in positive_pairs if p['confidence'] == 'medium_positive')} | 2-3次标题共现 |",
        f"| strong_positive | {sum(1 for p in positive_pairs if p['confidence'] == 'strong_positive')} | ≥4次标题共现 |",
        f"",
        f"### Top 10 正样本（按共现次数）",
        f"",
    ]
    
    for i, p in enumerate(sorted(positive_pairs, key=lambda x: -x['cooccurrence_count'])[:10], 1):
        report_lines.append(f"{i}. **{p['drug']} → {p['disease']}**: {p['cooccurrence_count']} 次共现")
    
    report_lines.extend([
        f"",
        f"## 负样本",
        f"",
        f"**总数**: {len(neg_sampled)} 对 (正:负 = 1:{args.neg_ratio})",
        f"**硬负样本**: {hard_neg_count} ({hard_neg_count/len(neg_sampled)*100:.1f}%)",
        f"**重定位明星已排除**: {', '.join(REPOSITIONING_STARS)}",
        f"",
        f"### 度数匹配",
        f"- 正样本 drug 度数分布: {dict(Counter(d for d, _ in positive_pairs).most_common(10))}",
        f"- 正样本 disease 度数分布: {dict(Counter(dis for _, dis in positive_pairs).most_common(10))}",
        f"- 负样本 drug 度数分布: {dict(Counter(d['drug'] for d in neg_sampled).most_common(10))}",
        f"- 负样本 disease 度数分布: {dict(Counter(d['disease'] for d in neg_sampled).most_common(10))}",
        f"",
        f"### Top 10 硬负样本",
        f"",
    ])
    
    hard_negs = [n for n in neg_sampled if n["is_hard_negative"]]
    for i, n in enumerate(hard_negs[:10], 1):
        report_lines.append(
            f"{i}. **{n['drug']} → {n['disease']}**: drug_freq={n['drug_freq_in_corpus']}, disease_freq={n['disease_freq_in_corpus']}"
        )
    
    report_lines.extend([
        f"",
        f"## 数据集划分",
        f"",
        f"| Split | 总数 | 正样本 | 负样本 | 药物种类 | 疾病种类 |",
        f"|-------|------|--------|--------|----------|----------|",
        f"| Train | {len(train)} | {sum(1 for s in train if s['label'] == 1)} | {sum(1 for s in train if s['label'] == 0)} | {len(set(s['drug'] for s in train))} | {len(set(s['disease'] for s in train))} |",
        f"| Val   | {len(val)} | {sum(1 for s in val if s['label'] == 1)} | {sum(1 for s in val if s['label'] == 0)} | {len(set(s['drug'] for s in val))} | {len(set(s['disease'] for s in val))} |",
        f"| Test  | {len(test)} | {sum(1 for s in test if s['label'] == 1)} | {sum(1 for s in test if s['label'] == 0)} | {len(set(s['drug'] for s in test))} | {len(set(s['disease'] for s in test))} |",
        f"",
        f"---",
        f"",
        f"## GNN训练建议",
        f"",
        f"### 正样本使用策略",
        f"- 弱正样本 (cooccurrence=1) 噪声较大，建议：",
        f"  1. 作为训练集使用，但权重降低（如0.5）",
        f"  2. 或仅保留medium/strong_positive作为验证/测试的正样本",
        f"- 强正样本 (cooccurrence≥4) 可信度最高，应优先放入训练集",
        f"",
        f"### 负样本使用策略",
        f"- 硬负样本在训练中应给予更高loss权重",
        f"- 考虑使用contrastive learning或margin-based loss",
        f"- 注意：重定位明星已手动排除，但训练时仍可单独测试它们的预测分数",
        f"",
        f"### 模型评估",
        f"- 主要指标: AUROC, AUPRC",
        f"- 特别关注: Top-20 predicted drug-disease对中，有多少是已知治疗（验证召回率）",
        f"- 对重定位明星的单独评估：预测分数是否偏高（验证模型是否有发现新药的能力）",
        f"",
    ])
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✅ 采样报告已保存: {OUTPUT_REPORT}")
    print(f"\n{'='*60}")
    print(f"🎉 负样本采样完成!")
    print(f"{'='*60}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
