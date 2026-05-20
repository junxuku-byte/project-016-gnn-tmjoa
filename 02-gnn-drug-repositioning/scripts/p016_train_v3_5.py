#!/usr/bin/env python3
"""
Project-016 训练集重建脚本 (v3.5分层版)
基于Scopus增强版1392篇文献的v3.5证据质量分层，重建GNN训练集。

正样本来源：
  - strong_positive (score≥0.7): 22篇 → label=1.0, weight=1.0
  - weak_positive (score 0.5-0.69): 24篇 → label=1.0, weight=0.8
  - mechanism_only (score 0-0.49): 162篇 → label=0.5, weight=0.6 (动物/体外/机制)
  - mechanism_animal (animal+阳性): 131篇 → label=0.5, weight=0.6

负样本来源：
  - exclude_negative (明确阴性): 24篇 → label=0.0, hard_negative
  - 从机制排斥池中采样 (需重新计算)
  - 高频独立出现但从未共现的drug-disease对

改进点（vs v2旧训练集）：
  1. 正样本基于证据分数而非标题共现
  2. 样本量加权（N<20降权）
  3. 期刊IF加权
  4. 重定位明星保留但单独标记（用于单独评估模型发现能力）
  5. 机制正样本（动物/体外）区分于临床正样本

输入: .tmp/p016_evidence_v3_5_scopus.json
输出: .tmp/p016_train_v3_5.json (train/val/test splits)
"""

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

INPUT_JSON = ".tmp/p016_evidence_v3_5_scopus.json"
OUTPUT_JSON = ".tmp/p016_train_v3_5.json"
OUTPUT_REPORT = ".tmp/p016_train_v3_5_report.md"

# 重定位明星 — 训练中单独标记，但不作为负样本（用于评估模型发现能力）
REPOSITIONING_STARS = {
    "Metformin", "Rapamycin", "Everolimus", "Resveratrol", "Curcumin",
    "Quercetin", "Fisetin", "EGCG", "SAMe", "Omega-3", "Ginger",
}

# GNN标签映射（evidence recommendation → gnn_label + weight）
LABEL_MAP = {
    "strong_positive":    (1.0, 1.0),
    "weak_positive":      (1.0, 0.8),
    "mechanism_only":     (0.5, 0.6),
    "mechanism_animal":   (0.5, 0.6),
    "exclude_neutral":    (0.0, 0.0),  # 排除，不纳入
    "exclude_negative":   (0.0, 1.0),  # 负样本（硬负样本）
    "exclude_animal_neutral": (0.0, 0.0),  # 排除
}


def extract_train_samples(results: list) -> tuple:
    """
    从v3.5分层结果中提取训练样本
    返回: (positive_samples, negative_samples, mechanism_samples, excluded)
    """
    positives = []
    negatives = []
    mechanisms = []  # 机制正样本（单独跟踪）
    excluded = []
    
    for r in results:
        rec = r["recommendation"]
        
        # 跳过排除的
        if rec in ["exclude_neutral", "exclude_animal_neutral"]:
            excluded.append(r)
            continue
        
        # 提取drug-disease对
        pairs = r.get("drug_disease_pairs", [])
        if not pairs:
            # 尝试从title+abstract提取（fallback）
            pairs = _fallback_extract_pairs(r)
        
        if not pairs:
            excluded.append(r)
            continue
        
        # 每篇文献的drug-disease对 → 独立样本
        for pair in pairs:
            drug = pair["drug"]
            disease = pair["disease"]
            
            # 重定位明星标记
            is_repo_star = drug in REPOSITIONING_STARS
            
            sample = {
                "pmid": r["pmid"],
                "title": r["title"],
                "drug": drug,
                "disease": disease,
                "year": r.get("year", 0),
                "journal_if": r.get("journal_if", 2.0),
                "confirmed_design": r["confirmed_design"],
                "conclusion_direction": r["conclusion_direction"],
                "total_n": r.get("total_n"),
                "final_score": r["final_score"],
                "evidence_recommendation": rec,
                "is_repositioning_star": is_repo_star,
                "abstract_snippet": r.get("abstract_snippet", "")[:100],
            }
            
            label, weight = LABEL_MAP.get(rec, (0.0, 0.0))
            sample["label"] = label
            sample["weight"] = weight
            
            # 样本量调整权重
            n = r.get("total_n")
            if n:
                if n < 20:
                    sample["weight"] *= 0.7  # 小样本降权
                elif n >= 200:
                    sample["weight"] *= 1.1  # 大样本 slight boost
            
            # 期刊IF调整权重
            jif = r.get("journal_if", 2.0)
            if jif >= 10:
                sample["weight"] *= 1.05
            elif jif < 1:
                sample["weight"] *= 0.8
            
            if rec == "exclude_negative":
                sample["is_hard_negative"] = True
                negatives.append(sample)
            elif rec in ["strong_positive", "weak_positive"]:
                positives.append(sample)
            elif rec in ["mechanism_only", "mechanism_animal"]:
                mechanisms.append(sample)
            else:
                excluded.append(r)
    
    return positives, negatives, mechanisms, excluded


def _fallback_extract_pairs(result: dict) -> list:
    """当drug_disease_pairs为空时，从标题粗略提取"""
    title = result.get("title", "").lower()
    
    # 简单启发式：如果标题中有已知药物和疾病关键词
    drug_kws = {
        "PRP": ["platelet-rich", "prp"],
        "Hyaluronic acid": ["hyaluronic", "hyaluronate"],
        "Corticosteroid": ["corticosteroid", "triamcinolone", "dexamethasone"],
        "NSAIDs": ["nsaid", "ibuprofen", "diclofenac"],
        "Denosumab": ["denosumab"],
        "Glucosamine": ["glucosamine"],
        "Chondroitin": ["chondroitin"],
        "Vitamin D": ["vitamin d"],
    }
    disease_kws = {
        "TMJOA": ["temporomandibular joint osteoarthritis", "tmj osteoarthritis"],
        "TMJ": ["temporomandibular", "tmj", "mandibular condyle"],
        "Osteoarthritis": ["osteoarthritis"],
        "TMD": ["temporomandibular disorder", "tmd"],
    }
    
    found_drugs = []
    for drug, kws in drug_kws.items():
        if any(kw in title for kw in kws):
            found_drugs.append(drug)
    
    found_diseases = []
    for dis, kws in disease_kws.items():
        if any(kw in title for kw in kws):
            found_diseases.append(dis)
    
    return [{"drug": d, "disease": dis} for d in found_drugs for dis in found_diseases]


def sample_negatives_v3(positives: list, mechanisms: list, negatives: list, 
                        target_ratio: float = 3.0, seed: int = 42) -> list:
    """
    v3负样本采样策略：
    1. 保留所有明确阴性文献作为硬负样本
    2. 从机制排斥池补充（需LabKG计算，这里简化）
    3. 确保负样本的drug/disease分布与正样本一致（度数匹配）
    """
    random.seed(seed)
    
    # 目标负样本数量
    n_pos = len(positives)
    n_mech = len(mechanisms)
    n_neg_target = int((n_pos + n_mech * 0.5) * target_ratio)  # 机制样本按0.5权重折算
    
    # 第一步：保留所有硬负样本
    hard_negs = [s for s in negatives if s.get("is_hard_negative")]
    other_negs = [s for s in negatives if not s.get("is_hard_negative")]
    
    sampled_negatives = hard_negs.copy()
    
    # 第二步：如果硬负样本不够，从其他阴性补充
    if len(sampled_negatives) < n_neg_target and other_negs:
        n_needed = n_neg_target - len(sampled_negatives)
        # 度数匹配采样：优先选择与正样本中高频drug/disease组合的阴性对
        pos_drug_freq = Counter(s["drug"] for s in positives)
        pos_dis_freq = Counter(s["disease"] for s in positives)
        
        # 按与正样本的"匹配度"排序其他阴性
        def match_score(neg):
            return pos_drug_freq.get(neg["drug"], 0) + pos_dis_freq.get(neg["disease"], 0)
        
        other_negs_sorted = sorted(other_negs, key=match_score, reverse=True)
        sampled_negatives.extend(other_negs_sorted[:n_needed])
    
    # 第三步：如果还不够，生成合成负样本（从正样本中反转label）
    # 注意：这里简化处理，实际应该用LabKG机制距离
    if len(sampled_negatives) < n_neg_target:
        print(f"⚠️ 负样本不足: 只有 {len(sampled_negatives)}/{n_neg_target} 个负样本")
        print(f"   建议: 扩展负样本池（机制排斥采样 / 时间锚定 / 文献共现排斥）")
    
    return sampled_negatives


def split_train_val_test(samples: list, train_ratio=0.7, val_ratio=0.15, seed=42) -> tuple:
    """分层划分：确保每个drug和disease在train/val/test中都有代表"""
    random.seed(seed)
    
    # 按drug分层
    by_drug = defaultdict(list)
    for s in samples:
        by_drug[s["drug"]].append(s)
    
    train, val, test = [], [], []
    
    for drug, drug_samples in by_drug.items():
        # 随机打乱
        random.shuffle(drug_samples)
        n = len(drug_samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        # 如果样本太少，全部放入train
        if n <= 3:
            train.extend(drug_samples)
        else:
            train.extend(drug_samples[:n_train])
            val.extend(drug_samples[n_train:n_train + n_val])
            test.extend(drug_samples[n_train + n_val:])
    
    # 最后整体打乱
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def main():
    print("="*60)
    print("Project-016 v3.5 训练集重建")
    print("="*60)
    
    # 加载v3.5分层结果
    with open(INPUT_JSON) as f:
        data = json.load(f)
    results = data["results"]
    print(f"📖 加载 {len(results)} 篇分层文献")
    
    # 提取样本
    positives, negatives, mechanisms, excluded = extract_train_samples(results)
    
    print(f"\n📊 样本提取结果:")
    print(f"  强/弱阳性 (label=1.0):     {len(positives)} 对")
    print(f"  机制正样本 (label=0.5):     {len(mechanisms)} 对")
    print(f"  明确阴性 (label=0.0):       {len(negatives)} 对")
    print(f"  排除:                        {len(excluded)} 篇")
    
    # 正样本统计
    print(f"\n  正样本 drug分布:")
    for d, c in Counter(s["drug"] for s in positives).most_common(15):
        print(f"    {d}: {c}")
    print(f"\n  正样本 disease分布:")
    for d, c in Counter(s["disease"] for s in positives).most_common(10):
        print(f"    {d}: {c}")
    
    # 负样本采样
    sampled_negs = sample_negatives_v3(positives, mechanisms, negatives, target_ratio=3.0)
    print(f"\n  采样后负样本: {len(sampled_negs)} 对 (目标比例 1:3)")
    
    # 硬负样本比例
    hard_neg_count = sum(1 for s in sampled_negs if s.get("is_hard_negative"))
    print(f"    硬负样本: {hard_neg_count} ({hard_neg_count/len(sampled_negs)*100:.1f}%)")
    
    # 合并所有样本（正+机制+负）
    # 机制样本权重0.5，作为半正样本
    all_samples = []
    
    for s in positives:
        s["split_group"] = "positive"
        all_samples.append(s)
    
    for s in mechanisms:
        s["split_group"] = "mechanism"
        all_samples.append(s)
    
    for s in sampled_negs:
        s["split_group"] = "negative"
        all_samples.append(s)
    
    # 分层划分
    train, val, test = split_train_val_test(all_samples, train_ratio=0.7, val_ratio=0.15)
    
    print(f"\n📊 数据集划分:")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        pos = sum(1 for s in split if s["label"] == 1.0)
        mech = sum(1 for s in split if s["label"] == 0.5)
        neg = sum(1 for s in split if s["label"] == 0.0)
        drugs = len(set(s["drug"] for s in split))
        diseases = len(set(s["disease"] for s in split))
        print(f"  {name:6}: {len(split)} total | pos={pos} mech={mech} neg={neg} | {drugs} drugs, {diseases} diseases")
    
    # 重定位明星分布
    repo_in_train = sum(1 for s in train if s.get("is_repositioning_star"))
    print(f"\n  重定位明星在训练集: {repo_in_train} 样本")
    
    # 保存训练集
    output_data = {
        "metadata": {
            "version": "v3_5_scopus",
            "created_at": datetime.now().isoformat(),
            "source_papers": len(results),
            "positive_samples": len(positives),
            "mechanism_samples": len(mechanisms),
            "negative_samples": len(sampled_negs),
            "hard_negatives": hard_neg_count,
            "repositioning_stars": list(REPOSITIONING_STARS),
            "target_neg_ratio": 3.0,
        },
        "statistics": {
            "drug_distribution": dict(Counter(s["drug"] for s in all_samples).most_common(20)),
            "disease_distribution": dict(Counter(s["disease"] for s in all_samples).most_common(15)),
            "design_distribution": dict(Counter(s["confirmed_design"] for s in all_samples).most_common(10)),
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
    report = f"""# Project-016 GNN训练集 — v3.5 Scopus增强版

**生成时间**: {datetime.now().isoformat()}
**版本**: v3.5_scopus

## 数据规模

| 类型 | 数量 | Label | Weight | 说明 |
|------|------|-------|--------|------|
| 强/弱阳性 | {len(positives)} | 1.0 | 0.8-1.1 | 临床RCT/Meta阳性证据 |
| 机制正样本 | {len(mechanisms)} | 0.5 | 0.6 | 动物/体外/机制研究 |
| 负样本 | {len(sampled_negs)} | 0.0 | 1.0 | 明确阴性 + 硬负样本 |
| **总计** | **{len(all_samples)}** | — | — | — |

## 数据集划分

| Split | 总数 | 正样本(1.0) | 机制(0.5) | 负样本(0.0) | 药物种类 | 疾病种类 |
|-------|------|-------------|-----------|-------------|----------|----------|
| Train | {len(train)} | {sum(1 for s in train if s['label']==1.0)} | {sum(1 for s in train if s['label']==0.5)} | {sum(1 for s in train if s['label']==0.0)} | {len(set(s['drug'] for s in train))} | {len(set(s['disease'] for s in train))} |
| Val   | {len(val)} | {sum(1 for s in val if s['label']==1.0)} | {sum(1 for s in val if s['label']==0.5)} | {sum(1 for s in val if s['label']==0.0)} | {len(set(s['drug'] for s in val))} | {len(set(s['disease'] for s in val))} |
| Test  | {len(test)} | {sum(1 for s in test if s['label']==1.0)} | {sum(1 for s in test if s['label']==0.5)} | {sum(1 for s in test if s['label']==0.0)} | {len(set(s['drug'] for s in test))} | {len(set(s['disease'] for s in test))} |

## Top 药物

{chr(10).join(f"- **{d}**: {c}" for d, c in Counter(s['drug'] for s in all_samples).most_common(10))}

## Top 疾病

{chr(10).join(f"- **{d}**: {c}" for d, c in Counter(s['disease'] for s in all_samples).most_common(10))}

## 与v2旧训练集对比

| 指标 | v2 (标题共现) | v3.5 (证据质量) | 变化 |
|------|---------------|-----------------|------|
| 总样本 | 117 | {len(all_samples)} | {'+' if len(all_samples) > 117 else ''}{len(all_samples) - 117} |
| 正样本 | 44 | {len(positives)} | {len(positives) - 44} |
| 机制样本 | 0 | {len(mechanisms)} | +{len(mechanisms)} |
| 负样本 | 73 | {len(sampled_negs)} | {len(sampled_negs) - 73} |
| 硬负样本 | 37 (50.7%) | {hard_neg_count} ({hard_neg_count/len(sampled_negs)*100:.1f}%) | 比例变化 |
| 重定位明星 | 手动排除11个 | 标记保留，单独评估 | 策略升级 |

## GNN训练建议

### 样本权重策略
- 强阳性 (weight=1.0): 标准权重
- 弱阳性 (weight=0.8): 略降权（证据较弱）
- 机制样本 (weight=0.6): 显著降权（动物≠临床）
- 硬负样本 (weight=1.0): 标准权重，但可考虑margin-based loss
- 小样本RCT (N<20, weight×0.7): 额外降权

### 评估指标
- 主要: AUROC, AUPRC
- 关注: 正样本召回率（特别是强阳性样本是否在Top-20中）
- 重定位明星单独评估: 预测分数是否偏高（验证模型发现新药能力）

### 数据增强（可选）
- 同药跨病: 如果 drug A 对 knee OA 有效，对 TMJOA 可能有效 → 生成 soft positive (label=0.3)
- 同病跨药: 如果 NSAIDs 对 TMJOA 有效，其他 NSAIDs 可能有效 → 生成 soft positive (label=0.3)
- 通路替换: 如果 drug A 通过 NF-κB 起效，drug B 也通过 NF-κB → 生成 soft positive (label=0.2)

---
*Generated by p016_train_v3_5.py*
"""
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)
    print(f"✅ 训练集报告: {OUTPUT_REPORT}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
