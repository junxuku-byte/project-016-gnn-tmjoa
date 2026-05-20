# Project-016 Status

## 当前方向
**GNN 药物重定位 for TMJOA**（投稿就绪）

## 完成度

| 模块 | 状态 | 完成度 |
|------|------|--------|
| 文献检索与证据筛选 | ✅ | 100%（1392篇→400篇→v3.5分层） |
| PDF 文献整理 | ✅ | 100%（23篇确认，PMID命名，去重归档） |
| LabKG 子图 | ✅ | 100% |
| 训练集构建 | ✅ | 100%（v5.1, 1075条全溯源） |
| 四层 GNN 模型 | ✅ | 100% |
| 架构消融(5模型) | ✅ | 100% |
| 图消融(5变体) | ✅ | 100% |
| KG 基线(4方法) | ✅ | 100% |
| 分子特征 ECFP4 | ✅ | 100% |
| 图补全实验 | ✅ | 100% |
| 多源文献验证 | ✅ | 100%（PubMed†/OpenAlex/Scopus三库18/20） |
| **论文 v8** | ✅ | **100%（双审稿人三轮审核通过）** |
| 投稿基础设施 | ✅ | 100%（Cover Letter + 补充表 + reproduce.sh + GitHub） |

## 最终资产

| 类别 | 文件 | 
|------|------|
| 论文 | `paper_full_draft_v8.md` |
| Cover Letter | `cover_letter.md` |
| 复现脚本 | `reproduce.sh` |
| GitHub | https://github.com/junxuku-byte/project-016-gnn-tmjoa |
| 训练集 | `p016_train_v5_1.json` (1075条, source字段全溯源) |
| 补充表 | S1-S4 (检索策略/TP映射/边溯源/全验证) |
| 文献 | 23篇 PDF (PMID命名) + LITERATURE_CARDS.md |
| 审稿报告 | peer_review_notes.md + modification_checklist.md |

## 核心实验结果

| 实验 | 结果 |
|------|------|
| 架构消融 | 5模型全等0.85，ΔAUC≤0.002 |
| 图消融 | TP随机化−13 AUC |
| 特征消融 | layer-only最优0.870 |
| 平图对照 | 0.947 = identity leakage |
| 等效性检验 | Bootstrap 95% CI [−0.065, +0.325] |
| KG基线 | DistMult/ComplEx/RotatE/TransE 0.80−0.83 |
| ECFP分子指纹 | +2.5 AUC (0.883→0.908) |
| 多源验证 | PubMed†/OpenAlex/Scopus 18/20 |
| 排除建图文库 | 442 PMID移除后验证率不变 |

## 投稿信息

- **目标期刊**: Briefings in Bioinformatics (IF ~9)
- **通讯作者**: 李昊森 (chongchong@tjh.tjmu.edu.cn)
- **单位**: 华中科技大学同济医学院同济医院口腔医学中心
- **论文标题**: Mechanism Graph Curation, Not GNN Complexity, Drives Inductive Drug Repositioning

---

*Updated: 2026-05-20 21:35 CST*
