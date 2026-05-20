# Project-016 Status

## 当前方向
**GNN 药物重定位 for TMJOA**（2026-05-17 深化）

## 完成度

| 模块 | 状态 | 完成度 |
|------|------|--------|
| LabKG 子图提取 | ✅ | 100% |
| 拓扑基线 v1 | ✅ | 100% |
| 拓扑基线 v2 | ✅ | 100% |
| Node2Vec 基线 | ✅ | 100% |
| 文献精读入LabKG | ✅ | 100% (66/84篇卡片) |
| 文件归位(workspace→morph-lab) | ✅ | 100% |
| 占位符清理 | ✅ | 100% |
| GNN PyTorch 模型 | ✅ | 100% |
| 负样本构建 | ✅ | 100%（v5.0: 239 pos + 836 neg, 1:3.5）|
| 模型评估（AUC/PR）| ✅ | 100% |
| Top-K 预测验证 | 🟡 | 50% |
| **论文 v4** | ✅ | **100%（全修+补实验，20/20项全部解决）** |
| KG Baselines | ✅ | DistMult 0.821 / ComplEx 0.830 / RotatE 0.804 / TransE 0.824 |
| Graph Ablation | ✅ | 5种消融完成，TP随机化降13点AUC，layer-only特征最优 |
| 论文投稿 | ⏳ | 0% |

## 最新进展（2026-05-20）
- 🔴 **Dual-Reviewer R1+R2 完成**：Sonnet+GPT 双模型交叉验证，一致判定 Major Revision
- 📝 **peer_review_notes.md**：7 Major + 12 Minor Issues 结构化审稿报告
- 📋 **modification_checklist.md**：P0(5项)/P1(5项)/P2(10项) 分层修改清单
- ✅ **paper_full_draft_v4.md 全修+补实验完成**：20/20 项全部解决
  - v3 (文本层 18项) + v4 (补实验 2项: KG baselines + Graph ablation)
  - P0-1: 修复 Discussion 0.68 vs 0.849 矛盾 → reframe "The Graph Is the Message"
  - P0-2: 核心贡献重定位 → 图构建方法论 + 归纳评估协议
  - P0-3: 循环验证 → 区分 "Recovery vs. Discovery" + 时间分割建议
  - P0-5: 数据可复现性 → ChEMBL v34, KEGG R112, Reactome v89, Supplementary Tables
  - P1-2: 特征泄漏 → Methods 明确 per-fold 重算
  - P1-3: TMJOA 临床背景 → Introduction 添加疾病负担段落
  - P1-4: 竞争定位 → 引用 Hetionet/DRKG/PharmKG
  - P1-5: Dextrose 循环信号 → Discussion 专门讨论
  - P2 全部 10 项: 措辞修正/AUPRC/fold细节/Cohen's d/Metformin正文化

## 所有阻塞已解除 ✅

| # | 原阻塞 | 状态 | 结果 |
|---|------|:--:|------|
| P0-4 | 添加强基线 | ✅ | DistMult/ComplEx/RotatE/TransE 全部低于GNN |
| P1-1 | Graph ablation | ✅ | TP随机化降13点, layer-only最优, 通路层可删除 |

## 下一步行动

1. ✅ **论文 v4 全修+补实验完成** (20/20)
2. **投稿准备**：Cover Letter + 图表 + 格式化
3. **目标期刊**：Briefings in Bioinformatics

## 资产清单

- `02-gnn-drug-repositioning/data/paper_full_draft_v4.md` — **全修+补实验论文 (20/20)**
- `02-gnn-drug-repositioning/data/paper_full_draft_v3.md` — 全修版论文 (文本层)
- `02-gnn-drug-repositioning/data/kg_baseline_results.json` — KG基线结果
- `02-gnn-drug-repositioning/data/graph_ablation_results.json` — 图消融结果
- `peer_review_notes.md` — 双审稿人审稿报告
- `modification_checklist.md` — 修改清单
- `02-gnn-drug-repositioning/` — 核心代码/数据/模型
- `00-文献/` — 48篇有效PDF
- `03-research-notes/` — 5篇研究文档
- LabKG: 753张卡片，3456节点，7574边

---

*Updated: 2026-05-20 09:30 CST*


## 📚 精读文献

```dataview
TABLE author, year, journal, category
FROM "project-016-gnn-drug-repositioning"
WHERE contains(file.folder, "03-文献卡")
SORT year DESC
```
