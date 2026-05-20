# Project-016 Dual-Reviewer 交叉审稿报告

**论文**: GNN-Based Mechanism-Constrained Drug Repositioning for TMJOA (paper_full_draft_v2.md)  
**目标期刊**: Briefings in Bioinformatics (IF ~9)  
**审稿日期**: 2026-05-20  
**审稿流程**: R1 初审(分工) → R2 细审(对调交叉验证)  
**审稿人**: Sonnet + GPT-5.5 (OpenRouter)

---

## 最终判定: 🔴 Major Revision

**交叉验证一致性**: R1 Sonnet(方法学) + GPT(叙事) 独立判定 Major Revision；R2 对调角色后双方确认并升级了多处关键问题。无模型偏见——两模型在几乎所有重大问题上的判断一致。

---

## 🔴 Major Issues (P0 — Must Fix)

### M1: 核心内部矛盾 — 同质GNN 0.849 vs 0.68 (R1确认，R2升级)
**发现者**: R2-Sonnet叙事 / R2-GPT方法学 (独立发现)  
**位置**: Results §四层图驱动归纳性能 + Discussion §机制约束实现归纳泛化

**中文逻辑**: 
- Results 报告 True Homogeneous GNN 归纳 LDO AUC = **0.849** ± 0.097
- Discussion 声称"inductive homogeneous GNN AUC fell to **0.68**. Adding mechanism layers raised to 0.85"
- 两者直接矛盾。Discussion 的 0.68 → 0.85 叙事错误地将训练范式转换(transductive→inductive)带来的收益归因于"添加机制层"——因为在正确的归纳设置下，不加机制层的同质GNN同样达到 0.849。

**英文可粘贴措辞**:
> "The Discussion attributes a 17-point AUC gain (0.68 → 0.85) to 'adding mechanism layers,' but the True Homogeneous GNN — which has no mechanism layers — achieves 0.849 under the same inductive LDO protocol. The observed gain is primarily attributable to the transition from transductive to inductive training, not to mechanism constraints."

**修改建议**:
1. 删除 Discussion 中"0.68→机制层→0.85"的因果叙事
2. 改为："Transitioning from transductive to inductive training raised AUC from 0.68 to 0.85; all five GNN architectures — including a homogeneous baseline without mechanism encoding — achieved equivalent inductive AUC (0.849–0.851), indicating that the four-layer topology enables but does not differentiate this gain."

---

### M2: 五模型全等AUC，消融实验不支持任何架构贡献 (R1确认，R2升级)
**发现者**: R1-Sonnet方法学 (M1) + R2-GPT方法学 (M5) + R2-Sonnet叙事 (M1')  
**位置**: Results §四层图驱动归纳性能

**中文逻辑**:
五种模型 AUC 全部在 0.849–0.851 之间（±0.002），即：
- HeteroGNN 不比 Homogeneous 好
- Attention 不比简单 sum 好
- Class weighting 无效果
- Layer-type encoding 无效果

论文声称的"异质GNN贡献""attention机制优先"在数据上无支撑。但这不是致命问题——可以重新定位贡献。

**英文可粘贴措辞**:
> "All five GNN variants achieved identical AUC (0.849–0.851), indicating that (1) heterogeneous message passing, (2) type-level attention, (3) class-weighted loss, and (4) layer-type encoding contribute no measurable predictive benefit beyond the four-layer graph topology itself. Claims about 'attention autonomously prioritizing informative signaling layers' are not supported by performance data."

**修改建议**:
1. 重新定义核心贡献：**图的构建方法论 + 归纳评估协议**，而非GNN架构创新
2. 将 Attention weights 解释为"descriptive post-hoc analysis"而非"contributory mechanism"
3. 加 graph ablation (删pathway层、删disease层、随机化TP/PD边) 来隔离图结构的贡献

---

### M3: 循环验证 — 训练数据 = 验证源 (R1双确认，R2双确认)
**发现者**: R1-Sonnet方法学 (M2) + R1-GPT叙事 (M2) + R2-GPT方法学 (M2) + R2-Sonnet叙事 (M3')  
**位置**: Methods §图构建 + Methods §评估协议 + Results §文献验证

**中文逻辑**:
- DT边来自753篇TMJOA PubMed文献
- PD边来自同一LabKG文献池
- 验证用 `drug_name AND ("TMJ" OR "TMD")` PubMed搜索
- Dextrose prolotherapy = 218 PubMed hits → Top 1 预测

这不是"验证"，这是"重新发现图谱已编码的知识"。唯一例外（BGJ398）是1个案例，不足以确立外部有效性。

**英文可粘贴措辞**:
> "The validation scheme is epistemically closed: both graph edge construction and post-hoc literature validation draw from the same PubMed corpus. The 19/20 confirmation rate reflects internal consistency (the model recovers graph-encoded evidence) rather than independent validation. Dextrose prolotherapy, which has 218 existing TMJOA PubMed hits and was never explicitly annotated as a TMJOA treatment in the graph, ranking first is not a successful prediction — it is a clear demonstration of literature leakage through graph topology."

**修改建议**:
1. 时间分割验证：用 YYYY 年前的文献建图，用 YYYY 年后的文献验证
2. 或用独立来源验证：ClinicalTrials.gov、DrugBank 适应症
3. Discussion 加章节：明确区分"known literature recovery" vs "novel hypothesis generation"

---

### M4: 基线比较不满足 IF~9 期刊要求 (R1双确认，R2双确认)
**发现者**: R1-Sonnet方法学 (M3) + R1-GPT叙事 (M5) + R2-GPT方法学 (M3) + R2-Sonnet叙事 (M5')  
**位置**: Methods §Baselines + Results §与基线比较

**中文逻辑**:
仅 RWR + Node2Vec 作为基线，对 Briefings in Bioinformatics 完全不够。需加：
- 异质GNN: R-GCN, HAN, HGT  
- 知识图谱嵌入: DistMult, ComplEx, RotatE
- 药物重定位专用: KGCN, DRKG-style link prediction

**英文可粘贴措辞**:
> "The study compares only RWR and Node2Vec against the proposed heterogeneous GNN. For a methods paper targeting Briefings in Bioinformatics (IF ~9), baselines should include heterogeneous graph methods (R-GCN, HAN, HGT), knowledge graph embedding models (DistMult, ComplEx, RotatE), and drug repositioning-specific methods (KGCN), evaluated under matching inductive protocols where feasible."

**修改建议**: 添加至少3-5个强基线

---

### M5: 数据整理方法不可复现 (R1确认，R2确认)
**发现者**: R1-Sonnet方法学 (M4) + R2-GPT方法学 (M4)  
**位置**: Methods §图构建

**中文逻辑**:
"420 LabKG cards from 753 PubMed articles" — 无检索式、筛选标准、标注协议；"Each target was linked to all pathways for which it has an established regulatory role" — 极主观；ChEMBL 版本未给。

**修改建议**:
- 提供 PubMed 检索式 + 日期范围 + 纳入排除标准  
- ChEMBL 版本/日期 + 置信度阈值  
- Pathway 本体来源 (KEGG/Reactome/WikiPathways?)  
- 边证据表 (PMID/source ID)  
- 公开图文件或补充材料

---

### M6: 可能特征泄漏 — degree/betweenness (R2-GPT方法学独立发现)
**发现者**: R2-GPT方法学 (M6)  
**位置**: Methods §节点特征 + §评估协议

**中文逻辑**:
如果 degree 和 betweenness 在全图上计算，再在5折LDO中使用，则测试药物的特征包含了训练时的图连接信息。应每折重新计算特征。

**修改建议**: 报告是否 per-fold 重新计算；若不是，修正

---

### M7: 临床意义结构性缺失 (R1确认，R2升级)
**发现者**: R1-GPT叙事 (M3) + R2-Sonnet叙事 (M4')  
**位置**: Introduction + Discussion

**中文逻辑**:
- Introduction 直接跳入图结构，无 TMJOA 治疗缺口背景
- Discussion 中 Metformin 案例(AMPK缺pathway→TMJOA边)是文中最有临床洞察力的发现，却被埋在 Limitations 里

**修改建议**:
- Introduction 加 3-4 句 TMJOA clinical unmet need
- 将 Metformin 案例提升到正文作为"架构可证伪性"的验证

---

## 🟡 Minor Issues

| # | 问题 | 位置 | 建议 |
|---|------|------|------|
| m1 | Cohen's d=3.3 缺乏上下文（比较什么分布？| Results | 明确定义效应量分子/分母 + 置换次数≥1000 |
| m2 | LDO折数与药物数未报告 | Methods | 报告每折测试药物数 + AUC 95% CI |
| m3 | "counterintuitive empirical finding" 营销化 | Introduction | → "unexpected empirical observation" |
| m4 | "zero TMJOA literature" 需限定 | Abstract/Results | → "no PubMed-indexed TMJOA-specific reports" |
| m5 | p<0.01 突兀出现 | Conclusion | 移除或说明比较对象 |
| m6 | "full mechanism path traceability" 过强 | Abstract | → "mechanism path-level traceability" |
| m7 | "causally links" 过强 | Introduction | → "mechanistically links" |
| m8 | 15 diseases 中 TMJOA 仅 1/15，但论文全聚焦TMJOA | Methods | 说明为何不是 multi-disease 评估 |
| m9 | AUC 非最佳指标(正负比 1:35) | Methods | 加 AUPRC |
| m10 | "95% validation" ≠ "independent literature" | Conclusion | 删除 "independent" 修饰 |
| m11 | Metformin 低分讨论与核心叙事关系断裂 | Discussion | 移到正文作为机制完整性验证 |
| m12 | Node2Vec transductive 0.90 "superficially outperforming" | Results | → "achieved higher transductive AUC" (去除修辞) |

---

## ✅ Strengths (双方确认)

1. **评估诚实性**: 主动披露 0.849 = 0.849 = 0.851 的平推现象，展现了方法学诚信
2. **归纳协议推广价值**: LDO + 指出 transductive leakage 对整个领域有正面影响
3. **BGJ398 候选机制路径**: FGF-FGFR-MMP13 轴的生物学解释具体且可验证
4. **四层图 Schema**: drug→target→pathway→disease 是直观、可解释、可复用的设计
5. **解读性 > 性能的定位策略**正确（当性能相等时，用可解释性区分）
6. **Metformin 案例**: 说明架构依赖机制完整性——是可证伪的科学论证

---

## 📋 决策摘要

| 维度 | R1-Sonnet | R1-GPT | R2-Sonnet | R2-GPT | 共识 |
|------|-----------|--------|-----------|--------|------|
| 判定 | Major Rev | Major Rev | Major Rev | Major Rev | **Major Rev** |
| P0 问题数 | 4 | 5 | 5 | 6 | **7 shared** |
| 核心矛盾 | 缺平图基线 | 叙事过度概括 | **数据矛盾** | **数据矛盾** | 🔴 |
| 循环验证 | ✓ | ✓ | ✓ | ✓ | 🔴 |
| 基线不足 | ✓ | ✓ | ✓ | ✓ | 🔴 |
| 不可复现 | ✓ | — | ✓ | — | 🔴 |
| 临床意义 | — | ✓ | ✓ | — | 🔴 |

**结论**: 如果修回版能解决 M1(内部矛盾) + M2(重新定位贡献) + M3(循环验证) + M4(加基线) + M5(可复现性)，该论文在 Briefings in Bioinformatics 有较强竞争力。当前不宜直接投稿。

---

*Dual-Reviewer v1.1.0 | 2026-05-20 | 油油*
