# Project-016 投稿前修改清单 (Modification Checklist)

**基于**: Dual-Reviewer R1+R2 交叉验证审稿 (2026-05-20)  
**目标期刊**: Briefings in Bioinformatics (IF ~9)  
**v4 状态**: 全部完成 (20/20 项已解决，2026-05-20)

---

## P0 — 阻塞投稿，必须修改

| # | 问题 | 状态 | 操作 |
|---|------|:--:|------|
| P0-1 | **Results/Discussion 内部矛盾**: 同质GNN 0.849 vs 0.68 | ✅ | Discussion 改为 "The Graph Is the Message" — 归因于图拓扑而非架构，删除 0.68→0.85 错误因果 |
| P0-2 | **重新定位核心贡献**: 五模型全等 0.849–0.851 | ✅ | 全篇 reframe 为 graph construction methodology + inductive evaluation。删除架构优势声称。 |
| P0-3 | **循环验证**: 图谱数据和验证源同来自PubMed | ✅ | 新增 "Interpreting Literature Validation: Recovery vs. Discovery" 章节。区分 known recovery + novel hypothesis。建议时间分割。 |
| P0-4 | **添加强基线**: RWR+Node2Vec不够 | ✅ | KG baselines: DistMult 0.821 / ComplEx 0.830 / RotatE 0.804 / TransE 0.824 |
| P0-5 | **数据整理方法不可复现** | ✅ | Methods 补: ChEMBL v34, KEGG R112, Reactome v89, PubMed检索细节 + Supplementary Tables |

---

## P1 — 投稿前应修改

| # | 问题 | 状态 | 操作 |
|---|------|:--:|------|
| P1-1 | **消融实验不支撑机制贡献** | ✅ | 5种ablation完成: TP随机化−13 AUC, layer-only最优, 通路层可删 |
| P1-2 | **特征泄漏风险**: degree/betweenness | ✅ | Methods 明确 per-fold 重算特征(不含held-out drugs) |
| P1-3 | **临床意义缺失**: Introduction 无 TMJOA 背景 | ✅ | Introduction 新增 TMJOA 疾病负担+治疗缺口段落 |
| P1-4 | **竞争定位**: 未承认已有 heterogeneous KG 工作 | ✅ | Introduction 新增 competitive positioning 段落，引用 Hetionet/DRKG/PharmKG |
| P1-5 | **Dextrose prolotherapy 循环信号** | ✅ | Discussion "Interpreting Literature Validation" 正面讨论 Dextrose 为文献泄漏 |

---

## P2 — 投稿时可优化

| # | 问题 | 状态 | 操作 |
|---|------|:--:|------|
| P2-1 | Cohen's d=3.3 缺乏上下文 | ✅ | Results 明确公式: (μ_observed − μ_null) / σ_observed |
| P2-2 | LDO 折数/药物数未详细报告 | ✅ | Methods 报告 per-fold 药物数(24–28) + 测试对数(360–420) |
| P2-3 | AUC 非最佳指标(正负比 1:35) | ✅ | 所有 Tables 添加 AUPRC 列 |
| P2-4 | "counterintuitive" → "unexpected" | ✅ | 全篇替换为 "unexpected empirical observation" |
| P2-5 | "zero TMJOA literature" → 加限定 | ✅ | → "no prior PubMed-indexed TMJOA-specific literature" |
| P2-6 | "causally links" → "mechanistically links" | ✅ | 全篇替换 |
| P2-7 | "full mechanism path" → "mechanism path-level" | ✅ | 全篇替换 |
| P2-8 | "independent literature" 删除 | ✅ | Conclusion 删除 "independent" |
| P2-9 | Metformin 案例从 Limitations 移至正文 | ✅ | 已作为 Case 3，标题 "Graph Coverage as a Falsifiable Constraint" |
| P2-10 | 15 diseases vs 仅评TMJOA | ✅ | Methods 保留完整疾病列表，Discussion Limitations 说明 TMJOA-centric |

---

## 工作量

| 状态 | 项数 | 
|------|------|
| ✅ 已完成 | 20/20 |

---

*v4 更新: 2026-05-20 09:30 CST*

---

## 修回后预期

解决 P0+P1 后，该论文在 Briefings in Bioinformatics 有**较强竞争力**：
- 核心卖点（机制约束图构建 + 归纳评估协议 + Node2Vec泄漏揭露）本身有方法学贡献价值
- BGJ398 案例是好的假设生成实例
- 重新定位后比"新GNN架构"论文更诚实、更不易被审稿人攻击

---

*Dual-Reviewer v1.1.0 | 2026-05-20*
