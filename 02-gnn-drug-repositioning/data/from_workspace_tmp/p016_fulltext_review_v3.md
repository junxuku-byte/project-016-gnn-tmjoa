# Project-016 33篇需精读文献 — Abstract深度分析报告

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

### PMID 41084405

**标题**: Comparison of NSAID therapy alone versus combined NSAID and home exercise therapy...

**Abstract关键句**: Group 2 (NSAID+exercise) showed greater MMO increase (+3.9 mm, p<.005) than Group 1 (+0.7 mm, p>.05). VAS decreased simi...

**判断**: POSITIVE

**理由**: 联合治疗组(运动+NSAID) mouth opening改善显著优于单纯NSAID组(+3.9mm vs +0.7mm, p<.005)。VAS两组均下降，但联合组功能改善更优。

**更新推荐**: `strong_positive` (原v3.5: `exclude_neutral`)

**更新分数**: +0.72

> RCT N=60, 明确阳性。之前abstract不完整被标记为unclear，现在确认为positive。

---

### PMID 40273050

**标题**: FGF18 induces chondrogenesis and anti-osteoarthritic effects in mouse model...

**Abstract关键句**: FGF18 attenuates cartilage degradation... anti-osteoarthritic effects in articular cartilage......

**判断**: MECHANISM_POSITIVE

**理由**: FGF18在小鼠TMJ-OA模型中促进软骨生成并减轻软骨降解。动物实验，机制明确。

**更新推荐**: `mechanism_animal` (原v3.5: `exclude_neutral`)

**更新分数**: +0.30

> 动物实验，不能作为临床正样本，但作为机制正样本(label=0.5)。

---

### PMID 38867545

**标题**: Efficiency of Glucosamine in Treating TMJOA: A Meta-Analytic Umbrella Review

**Abstract关键句**: Oral glucosamine and chondroitin... reduce pain and increase mouth opening in patients with TMD......

**判断**: POSITIVE

**理由**: Meta-umbrella review, 明确结论'reduce pain and increase mouth opening'。虽然是umbrella review（二次Meta），但结论方向清晰。

**更新推荐**: `weak_positive` (原v3.5: `exclude_neutral`)

**更新分数**: +0.62

> Umbrella review of existing Meta-analyses, 证据层级低于原始Meta，所以weak_positive而非strong。

---

### PMID 39092654

**标题**: Can treatment with chondroitin and glucosamine sulphate prevent changes...

**Abstract关键句**: investigate the action of CGS on progression of chemically induced OA in TMJ of rabbits... evaluating serum TNF-α and co...

**判断**: MECHANISM_POSITIVE

**理由**: 兔子TMJ-OA模型中CGS的作用，动物实验。研究的是'prevent changes'和评估血清标志物。

**更新推荐**: `mechanism_animal` (原v3.5: `exclude_neutral`)

**更新分数**: +0.25

> 兔子模型，动物实验，机制正样本。

---

### PMID 38821656

**标题**: INTRA-ARTICULAR PHARMACOLOGICAL INJECTIONS FOR TMJOA ARE COMPARABLE TO PLACEBO

**Abstract关键句**: Effectiveness of intra-articular injections of sodium hyaluronate, corticosteroids, platelet-rich plasma on TMJOA: a sys...

**判断**: NEUTRAL/NEGATIVE

**理由**: 标题直接声明'comparable to placebo'。这是一篇系统综述和网络Meta分析，结论是关节内注射(HA/CS/PRP)与安慰剂效果相当。

**更新推荐**: `exclude_negative` (原v3.5: `exclude_neutral`)

**更新分数**: -0.20

> ⚠️ KEY NEGATIVE EVIDENCE。这是又一篇系统综述得出关节内注射与安慰剂无差异的结论。与PMID 36162894(-0.32)一致。

---

### PMID 37608244

**标题**: Hyaluronic acid injections for pain relief...: An umbrella review of systematic reviews

**Abstract关键句**: BACKGROUND:... intra-articular injections of HA are commonly performed. OBJECTIVES: analyse effectiveness of HA injectio...

**判断**: UNCLEAR

**理由**: Umbrella review，abstract只陈述了目的('analyse effectiveness')，没有给出具体结论方向。需要看正文中的summary of findings。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> Umbrella review abstract不完整，结论方向缺失。保持exclude_neutral，建议全文获取。

---

### PMID 36890529

**标题**: Does intra-articular injection of tenoxicam after arthrocentesis heal outcomes...

**Abstract关键句**: Thirty patients... randomly allocated... arthrocentesis plus tenoxicam vs arthrocentesis alone......

**判断**: UNCLEAR

**理由**: Abstract截断('Thirty patients w...')，结论部分未显示。标题是疑问式('Does...?')，无方向性。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> Abstract不完整，标题疑问式。RCT但结论缺失。保持exclude_neutral，建议全文获取。

---

### PMID 35994388

**标题**: Recent Advances in Animal Models, Diagnosis, and Treatment of TMJOA

**Abstract关键句**: summarized common animal models... relevant pathological symptoms and therapeutic options......

**判断**: REVIEW/MECHANISM

**理由**: 综述文章，概述动物模型和治疗选择，无具体临床RCT结论。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> 综述文章，无具体方向性结论。不纳入训练集。

---

### PMID 33775650

**标题**: Systematic Review of Rat Models With TMJOA Suitable for Drug Delivery Systems

**Abstract关键句**: Which method of induction of TMJOA-related pain model in rats leads to prolonged painful symptoms......

**判断**: REVIEW

**理由**: 系统综述比较不同大鼠TMJOA造模方法，不是治疗研究。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> 造模方法比较综述，不纳入训练集。

---

### PMID 30814387

**标题**: Long-term effectiveness of arthrocentesis with and without HA...

**Abstract关键句**: 37 completed long-term follow-up (~4 years)... randomly allocated: arthrocentesis alone vs arthrocentesis + HA......

**判断**: UNCLEAR

**理由**: Abstract截断，结论未显示。RCT N=37, 4年随访是高质量证据，但abstract无结论方向。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> 高质量长期RCT，但abstract不完整。建议全文获取。

---

### PMID 29244893

**标题**: Effectiveness of Intra-Articular Injections of HA or CS: Systematic Review and Meta-Analysis

**Abstract关键句**: assess effectiveness of HA or CS for intracapsular TMD... compared to each other or to placebo......

**判断**: UNCLEAR

**理由**: Abstract截断('Electronic...')，结论未显示。是Meta-analysis但方向不明。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> HA vs CS Meta，abstract不完整。建议全文获取。

---

### PMID 28879245

**标题**: Effects of High-Dose Capsaicin on TMD Subjects: A Randomized Clinical Study

**Abstract关键句**: double-blind, vehicle-controlled clinical trial to evaluate the efficacy......

**判断**: UNCLEAR

**理由**: Abstract截断，结论未显示。RCT but no conclusion in abstract. 标题是陈述式('Effects of...')但无方向。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> RCT but abstract无结论。车辆对照(vehicle-controlled)意味着对照组不是安慰剂而是载体，设计稍弱。

---

### PMID 41770501

**标题**: Comparative efficacy of transdermal buprenorphine versus diclofenac in knee OA

**Abstract关键句**: knee OA... transdermal Buprenorphine compared to Diclofenac patches......

**判断**: EXCLUDE_NON_TMJOA

**理由**: 研究的是膝OA(knee OA)，非TMJOA。虽然药物(Buprenorphine/Diclofenac)可能对TMJOA也有用，但这不是直接证据。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> 膝OA RCT，不直接适用于TMJOA训练集。可后续作为跨病类比(soft label)使用。

---

### PMID 40972625

**标题**: Global burden of low bone mineral density 1990-2020 (GBD 2021)

**Abstract关键句**: burden of low bone mineral density to fractures... GBD 2021......

**判断**: EXCLUDE_NON_TMJOA

**理由**: 全球疾病负担研究，与TMJOA无直接关系。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> GBD流行病学研究，不纳入训练集。

---

### PMID 41727032

**标题**: Discovery of TDP-43 aggregation inhibitors via hybrid ML framework

**Abstract关键句**: TDP-43 aggregation... GNN embeddings... neurodegenerative diseases......

**判断**: EXCLUDE_NON_TMJOA

**理由**: 神经退行性疾病(TDP-43)的药物发现，使用GNN方法学，但与TMJOA无关。

**更新推荐**: `exclude_neutral` (原v3.5: `exclude_neutral`)

**更新分数**: +0.00

> 方法学论文(GNN+drug discovery)，可引用作为方法论参考，不纳入训练集。

---

## 无PMID文献分析

| # | 标题 | 判断 | 原因 |
|---|------|------|------|
| 13 | Dextrose Prolotherapy for Knee OA... | EXCLUDE | 膝OA，非TMJOA |
| 14 | Dental chairside evidence-based medicine for TMJOA... | EXCLUDE | 中文文献，无abstract |
| 15 | Dextrose Prolotherapy RCT... | EXCLUDE | 膝OA |
| 16 | Na-hyaluronate in rabbit cartilage injury... | EXCLUDE | 兔子模型，动物实验 |
| 17 | Manual therapy systematic review... | EXCLUDE | 手法治疗，非药物 |
| 18 | Peptide-MSC complex rat OA... | EXCLUDE | 大鼠OA模型 |
| 19 | Low-level laser for masticatory muscle pain... | EXCLUDE | 低能量激光，肌肉痛 |
| 20 | Physical therapy for chronic migraine + TMD... | EXCLUDE | 物理治疗+偏头痛 |
| 21 | RAIN+GraphSAGE for gastric neoplasm... | EXCLUDE | 胃癌药物组合，非TMJOA |
| 22 | Bayesian Meta-Analysis antipsychotic drugs... | EXCLUDE | 精神病药物 |
| 23 | Semantic mining adverse drug events... | EXCLUDE | 药物不良反应 |
| 24 | Heterogeneous treatment effects joint CIs... | EXCLUDE | 统计方法学 |
| 25 | FlexPro MD (krill oil+astaxanthin+HA) rat OA... | EXCLUDE | 大鼠OA，复合补充剂 |
| 26 | Trehalose bone fracture healing rat... | EXCLUDE | 海藻糖骨折愈合 |
| 27 | ICOS paclitaxel neuropathic pain mice... | EXCLUDE | 化疗神经痛 |
| 28 | Metformin adjunctive therapy Parkinson's... | EXCLUDE | 帕金森病 |
| 29 | Meta-analysis proteomics osteoblasts/bone... | EXCLUDE | 蛋白质组学Meta |
| 32 | Comparative effectiveness oral pharmacologic knee ... | EXCLUDE | 膝OA网络Meta |

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
