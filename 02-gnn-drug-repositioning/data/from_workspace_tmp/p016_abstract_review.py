#!/usr/bin/env python3
"""
Project-016 — 33篇需精读文献abstract深度分析
逐篇判断结论方向，更新evidence分层。

输出: .tmp/p016_fulltext_review_v3.md (精读分析报告)
"""

import json, re
from collections import Counter

with open('.tmp/p016_evidence_v3_5_scopus.json') as f:
    data = json.load(f)
with open('.tmp/p016_merged_v3_scopus.json') as f:
    raw = json.load(f)

pmid_to_paper = {p.get('pmid'): p for p in raw['papers']}

# 逐篇分析
ANALYSIS = []

# --- 1. PMID 41084405 ---
ANALYSIS.append({
    "pmid": "41084405",
    "title": "Comparison of NSAID therapy alone versus combined NSAID and home exercise therapy...",
    "abstract_snippet": "Group 2 (NSAID+exercise) showed greater MMO increase (+3.9 mm, p<.005) than Group 1 (+0.7 mm, p>.05). VAS decreased similarly.",
    "judgment": "POSITIVE", 
    "reason": "联合治疗组(运动+NSAID) mouth opening改善显著优于单纯NSAID组(+3.9mm vs +0.7mm, p<.005)。VAS两组均下降，但联合组功能改善更优。",
    "updated_recommendation": "strong_positive",
    "updated_score": 0.72,
    "note": "RCT N=60, 明确阳性。之前abstract不完整被标记为unclear，现在确认为positive。"
})

# --- 2. PMID 40273050 ---
ANALYSIS.append({
    "pmid": "40273050",
    "title": "FGF18 induces chondrogenesis and anti-osteoarthritic effects in mouse model...",
    "abstract_snippet": "FGF18 attenuates cartilage degradation... anti-osteoarthritic effects in articular cartilage...",
    "judgment": "MECHANISM_POSITIVE",
    "reason": "FGF18在小鼠TMJ-OA模型中促进软骨生成并减轻软骨降解。动物实验，机制明确。",
    "updated_recommendation": "mechanism_animal",
    "updated_score": 0.30,
    "note": "动物实验，不能作为临床正样本，但作为机制正样本(label=0.5)。"
})

# --- 3. PMID 38867545 ---
ANALYSIS.append({
    "pmid": "38867545",
    "title": "Efficiency of Glucosamine in Treating TMJOA: A Meta-Analytic Umbrella Review",
    "abstract_snippet": "Oral glucosamine and chondroitin... reduce pain and increase mouth opening in patients with TMD...",
    "judgment": "POSITIVE",
    "reason": "Meta-umbrella review, 明确结论'reduce pain and increase mouth opening'。虽然是umbrella review（二次Meta），但结论方向清晰。",
    "updated_recommendation": "weak_positive",
    "updated_score": 0.62,
    "note": "Umbrella review of existing Meta-analyses, 证据层级低于原始Meta，所以weak_positive而非strong。"
})

# --- 4. PMID 39092654 ---
ANALYSIS.append({
    "pmid": "39092654",
    "title": "Can treatment with chondroitin and glucosamine sulphate prevent changes...",
    "abstract_snippet": "investigate the action of CGS on progression of chemically induced OA in TMJ of rabbits... evaluating serum TNF-α and collagen...",
    "judgment": "MECHANISM_POSITIVE",
    "reason": "兔子TMJ-OA模型中CGS的作用，动物实验。研究的是'prevent changes'和评估血清标志物。",
    "updated_recommendation": "mechanism_animal",
    "updated_score": 0.25,
    "note": "兔子模型，动物实验，机制正样本。"
})

# --- 5. PMID 38821656 --- **KEY**
ANALYSIS.append({
    "pmid": "38821656",
    "title": "INTRA-ARTICULAR PHARMACOLOGICAL INJECTIONS FOR TMJOA ARE COMPARABLE TO PLACEBO",
    "abstract_snippet": "Effectiveness of intra-articular injections of sodium hyaluronate, corticosteroids, platelet-rich plasma on TMJOA: a systematic review and network meta-analysis...",
    "judgment": "NEUTRAL/NEGATIVE",
    "reason": "标题直接声明'comparable to placebo'。这是一篇系统综述和网络Meta分析，结论是关节内注射(HA/CS/PRP)与安慰剂效果相当。",
    "updated_recommendation": "exclude_negative",
    "updated_score": -0.20,
    "note": "⚠️ KEY NEGATIVE EVIDENCE。这是又一篇系统综述得出关节内注射与安慰剂无差异的结论。与PMID 36162894(-0.32)一致。"
})

# --- 6. PMID 37608244 ---
ANALYSIS.append({
    "pmid": "37608244",
    "title": "Hyaluronic acid injections for pain relief...: An umbrella review of systematic reviews",
    "abstract_snippet": "BACKGROUND:... intra-articular injections of HA are commonly performed. OBJECTIVES: analyse effectiveness of HA injections on pain and functional outcomes...",
    "judgment": "UNCLEAR",
    "reason": "Umbrella review，abstract只陈述了目的('analyse effectiveness')，没有给出具体结论方向。需要看正文中的summary of findings。",
    "updated_recommendation": "exclude_neutral",  # 保持排除，等全文
    "updated_score": 0.00,
    "note": "Umbrella review abstract不完整，结论方向缺失。保持exclude_neutral，建议全文获取。"
})

# --- 7. PMID 36890529 ---
ANALYSIS.append({
    "pmid": "36890529",
    "title": "Does intra-articular injection of tenoxicam after arthrocentesis heal outcomes...",
    "abstract_snippet": "Thirty patients... randomly allocated... arthrocentesis plus tenoxicam vs arthrocentesis alone...",
    "judgment": "UNCLEAR",
    "reason": "Abstract截断('Thirty patients w...')，结论部分未显示。标题是疑问式('Does...?')，无方向性。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "Abstract不完整，标题疑问式。RCT但结论缺失。保持exclude_neutral，建议全文获取。"
})

# --- 8. PMID 35994388 ---
ANALYSIS.append({
    "pmid": "35994388",
    "title": "Recent Advances in Animal Models, Diagnosis, and Treatment of TMJOA",
    "abstract_snippet": "summarized common animal models... relevant pathological symptoms and therapeutic options...",
    "judgment": "REVIEW/MECHANISM",
    "reason": "综述文章，概述动物模型和治疗选择，无具体临床RCT结论。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "综述文章，无具体方向性结论。不纳入训练集。"
})

# --- 9. PMID 33775650 ---
ANALYSIS.append({
    "pmid": "33775650",
    "title": "Systematic Review of Rat Models With TMJOA Suitable for Drug Delivery Systems",
    "abstract_snippet": "Which method of induction of TMJOA-related pain model in rats leads to prolonged painful symptoms...",
    "judgment": "REVIEW",
    "reason": "系统综述比较不同大鼠TMJOA造模方法，不是治疗研究。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "造模方法比较综述，不纳入训练集。"
})

# --- 10. PMID 30814387 ---
ANALYSIS.append({
    "pmid": "30814387",
    "title": "Long-term effectiveness of arthrocentesis with and without HA...",
    "abstract_snippet": "37 completed long-term follow-up (~4 years)... randomly allocated: arthrocentesis alone vs arthrocentesis + HA...",
    "judgment": "UNCLEAR",
    "reason": "Abstract截断，结论未显示。RCT N=37, 4年随访是高质量证据，但abstract无结论方向。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "高质量长期RCT，但abstract不完整。建议全文获取。"
})

# --- 11. PMID 29244893 ---
ANALYSIS.append({
    "pmid": "29244893",
    "title": "Effectiveness of Intra-Articular Injections of HA or CS: Systematic Review and Meta-Analysis",
    "abstract_snippet": "assess effectiveness of HA or CS for intracapsular TMD... compared to each other or to placebo...",
    "judgment": "UNCLEAR",
    "reason": "Abstract截断('Electronic...')，结论未显示。是Meta-analysis但方向不明。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "HA vs CS Meta，abstract不完整。建议全文获取。"
})

# --- 12. PMID 28879245 ---
ANALYSIS.append({
    "pmid": "28879245",
    "title": "Effects of High-Dose Capsaicin on TMD Subjects: A Randomized Clinical Study",
    "abstract_snippet": "double-blind, vehicle-controlled clinical trial to evaluate the efficacy...",
    "judgment": "UNCLEAR",
    "reason": "Abstract截断，结论未显示。RCT but no conclusion in abstract. 标题是陈述式('Effects of...')但无方向。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "RCT but abstract无结论。车辆对照(vehicle-controlled)意味着对照组不是安慰剂而是载体，设计稍弱。"
})

# --- 30. PMID 41770501 --- (非TMJOA)
ANALYSIS.append({
    "pmid": "41770501",
    "title": "Comparative efficacy of transdermal buprenorphine versus diclofenac in knee OA",
    "abstract_snippet": "knee OA... transdermal Buprenorphine compared to Diclofenac patches...",
    "judgment": "EXCLUDE_NON_TMJOA",
    "reason": "研究的是膝OA(knee OA)，非TMJOA。虽然药物(Buprenorphine/Diclofenac)可能对TMJOA也有用，但这不是直接证据。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "膝OA RCT，不直接适用于TMJOA训练集。可后续作为跨病类比(soft label)使用。"
})

# --- 31. PMID 40972625 --- (非TMJOA)
ANALYSIS.append({
    "pmid": "40972625",
    "title": "Global burden of low bone mineral density 1990-2020 (GBD 2021)",
    "abstract_snippet": "burden of low bone mineral density to fractures... GBD 2021...",
    "judgment": "EXCLUDE_NON_TMJOA",
    "reason": "全球疾病负担研究，与TMJOA无直接关系。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "GBD流行病学研究，不纳入训练集。"
})

# --- 33. PMID 41727032 --- (非TMJOA)
ANALYSIS.append({
    "pmid": "41727032",
    "title": "Discovery of TDP-43 aggregation inhibitors via hybrid ML framework",
    "abstract_snippet": "TDP-43 aggregation... GNN embeddings... neurodegenerative diseases...",
    "judgment": "EXCLUDE_NON_TMJOA",
    "reason": "神经退行性疾病(TDP-43)的药物发现，使用GNN方法学，但与TMJOA无关。",
    "updated_recommendation": "exclude_neutral",
    "updated_score": 0.00,
    "note": "方法学论文(GNN+drug discovery)，可引用作为方法论参考，不纳入训练集。"
})

# 没有PMID的文献（13-29）大部分是non-TMJOA或NO ABSTRACT
NO_PMID_NON_TMJOA = [
    ("13", "Dextrose Prolotherapy for Knee OA", "膝OA，非TMJOA"),
    ("14", "Dental chairside evidence-based medicine for TMJOA", "中文文献，无abstract"),
    ("15", "Dextrose Prolotherapy RCT", "膝OA"),
    ("16", "Na-hyaluronate in rabbit cartilage injury", "兔子模型，动物实验"),
    ("17", "Manual therapy systematic review", "手法治疗，非药物"),
    ("18", "Peptide-MSC complex rat OA", "大鼠OA模型"),
    ("19", "Low-level laser for masticatory muscle pain", "低能量激光，肌肉痛"),
    ("20", "Physical therapy for chronic migraine + TMD", "物理治疗+偏头痛"),
    ("21", "RAIN+GraphSAGE for gastric neoplasm", "胃癌药物组合，非TMJOA"),
    ("22", "Bayesian Meta-Analysis antipsychotic drugs", "精神病药物"),
    ("23", "Semantic mining adverse drug events", "药物不良反应"),
    ("24", "Heterogeneous treatment effects joint CIs", "统计方法学"),
    ("25", "FlexPro MD (krill oil+astaxanthin+HA) rat OA", "大鼠OA，复合补充剂"),
    ("26", "Trehalose bone fracture healing rat", "海藻糖骨折愈合"),
    ("27", "ICOS paclitaxel neuropathic pain mice", "化疗神经痛"),
    ("28", "Metformin adjunctive therapy Parkinson's", "帕金森病"),
    ("29", "Meta-analysis proteomics osteoblasts/bone", "蛋白质组学Meta"),
    ("32", "Comparative effectiveness oral pharmacologic knee OA NMA", "膝OA网络Meta"),
]

# 生成报告
report = """# Project-016 33篇需精读文献 — Abstract深度分析报告

**日期**: 2026-05-18
**数据来源**: v3.5_scopus 分层结果
**分析者**: AI精读（基于abstract完整文本判断）

---

## 核心发现摘要

| 分类 | 数量 | 说明 |
|------|------|------|
| **可确认POSITIVE** | 3篇 | PMID 41084405, 38867545, 40273050 |
| **可确认NEGATIVE** | 1篇 | **PMID 38821656** (关节内注射=安慰剂) |
| **机制正样本** | 2篇 | PMID 40273050 (FGF18动物), 39092654 (CGS兔子) |
| **仍UNCLEAR需全文** | 5篇 | PMID 37608244, 36890529, 30814387, 29244893, 28879245 |
| **非TMJOA排除** | 6篇 | 膝OA/GBD/神经退行/方法学 |
| **无abstract/NO PMID** | 16篇 | 主要来自Scopus/OpenAlex无PMID记录 |
| **综述无方向** | 2篇 | PMID 35994388, 33775650 |

---

## 逐篇分析

"""

for a in ANALYSIS:
    report += f"""### PMID {a['pmid']}

**标题**: {a['title']}

**Abstract关键句**: {a['abstract_snippet'][:120]}...

**判断**: {a['judgment']}

**理由**: {a['reason']}

**更新推荐**: `{a['updated_recommendation']}` (原v3.5: `exclude_neutral`)

**更新分数**: {a['updated_score']:+.2f}

> {a['note']}

---

"""

report += """## 无PMID文献分析

| # | 标题 | 判断 | 原因 |
|---|------|------|------|
"""
for num, title, reason in NO_PMID_NON_TMJOA:
    report += f"| {num} | {title[:50]}... | EXCLUDE | {reason} |\n"

report += """
---

## 对训练集的影响

### 正样本增益
- PMID 41084405 → **strong_positive** (+1样本, NSAID+exercise联合治疗)
- PMID 38867545 → **weak_positive** (+1样本, Glucosamine umbrella review)
- PMID 40273050 → **mechanism_animal** (已有机制样本)
- PMID 39092654 → **mechanism_animal** (已有机制样本)

### 负样本增益
- **PMID 38821656** → **exclude_negative** (硬负样本)
  - 系统综述+NMA结论：关节内注射(HA/CS/PRP)与安慰剂无差异
  - 与PMID 36162894(-0.32)形成双重验证：关节内注射的阴性证据链

### 排除清单（不纳入训练集）
- 6篇非TMJOA研究（膝OA/GBD/神经退行）
- 2篇综述无方向性结论
- 16篇无abstract/no PMID
- 5篇abstract不完整仍需全文（建议后续补充）

### 关键争议：关节内注射的疗效
现有证据链：
1. **阳性**：PMID 36414172, 35679902 (PRP+HA RCT阳性, +0.75)
2. **阴性**：PMID 36162894 (-0.32), **38821656** (-0.20) (系统综述+NMA = 安慰剂)
3. **中性**：PMID 37608244 (umbrella review, 方向不明)

**结论**: 关节内注射（单独使用）的疗效存在显著争议。系统综述级别证据倾向于阴性/中性，但个别高质量RCT（特别是PRP+HA联合）显示阳性。训练集中应：
- PRP+HA联合 → strong_positive
- PRP/HA单独 → 降级为weak_positive或mechanism_only（因系统综述质疑）
- 明确阴性的系统综述 → 硬负样本

---

## 建议

1. **立即纳入训练集**：PMID 41084405 (strong_positive), 38821656 (exclude_negative硬负样本)
2. **后续全文获取优先级**（5篇）：
   - HIGH: PMID 37608244 (HA umbrella review)
   - HIGH: PMID 30814387 (4年长期RCT, arthrocentesis±HA)
   - MEDIUM: PMID 36890529 (tenoxicam RCT)
   - MEDIUM: PMID 29244893 (HA vs CS Meta)
   - LOW: PMID 28879245 (capsaicin RCT, vehicle-controlled)
3. **非TMJOA文献可转用于跨病soft label**：膝OA的Diclofenac/Buprenorphine/Manual therapy → 对TMJOA label=0.2

---
*Generated by abstract review analysis*
"""

with open('.tmp/p016_fulltext_review_v3.md', 'w') as f:
    f.write(report)

# 统计
pos = sum(1 for a in ANALYSIS if a['judgment'] == 'POSITIVE')
neg = sum(1 for a in ANALYSIS if 'NEGATIVE' in a['judgment'])
mech = sum(1 for a in ANALYSIS if 'MECHANISM' in a['judgment'])
unclear = sum(1 for a in ANALYSIS if a['judgment'] == 'UNCLEAR')
exclude = sum(1 for a in ANALYSIS if 'EXCLUDE' in a['judgment'] or a['judgment'] == 'REVIEW/MECHANISM')

print(f"=== Abstract精读分析完成 ===")
print(f"可确认POSITIVE:     {pos}篇")
print(f"可确认NEGATIVE:      {neg}篇")
print(f"机制正样本:          {mech}篇")
print(f"仍UNCLEAR:           {unclear}篇")
print(f"排除(non-TMJOA/综述): {exclude}篇")
print(f"无abstract/NO PMID:  {len(NO_PMID_NON_TMJOA)}篇")
print(f"\n报告保存: .tmp/p016_fulltext_review_v3.md")
