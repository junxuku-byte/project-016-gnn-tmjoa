# Project-016 Status

## 当前方向
**GNN 药物重定位 for TMJOA**（v9 已完成，Bioinformatics 投稿待通讯作者确认）

## 目标期刊变更
- v8: Briefings in Bioinformatics (IF ~9)
- **v9: Bioinformatics (Oxford, IF ~5-6)** ← 精准匹配方法学原创研究
- 变更原因：BIB 是综述/方法学评述期刊，不发表一手实验型原创研究

## 完成度

| 模块 | 状态 | 完成度 |
|------|------|--------|
| 文献检索与证据筛选 | ✅ | 100% |
| PDF 文献整理 | ✅ | 100% |
| LabKG 子图 | ✅ | 100% |
| 训练集构建 | ✅ | 100%（v5.1, 1075条全溯源） |
| 四层 GNN 模型 | ✅ | 100% |
| 架构消融(5模型) | ✅ | 100% |
| 图消融(5变体) | ✅ | 100% |
| KG 基线(4方法) | ✅ | 100% |
| 分子特征 ECFP4 | ✅ | 100% |
| 多源文献验证 | ✅ | 100%（PubMed†/OpenAlex/Scopus三库18/20） |
| 朝堂议政终审(court-ab0d0f2e) | ✅ | 100%（CONDITIONAL-GO，5项hard requirements） |
| **论文 v9** | ✅ | **100%（Bioinformatics靶向，41篇参考文献，TOST+功效分析）** |

## v9 vs v8 变更摘要（基于朝堂议政 court-ab0d0f2e 决议）

| # | 变更项 | v8 | v9 |
|---|--------|----|----|
| 1 | 标题 | "Not GNN Complexity"（否定式） | "Mechanism-Constrained Graph Learning"（建设性） |
| 2 | 目标期刊 | Briefings in Bioinformatics | Bioinformatics (Oxford) |
| 3 | Abstract | ~370词，冗长 | 235词，结构化为 Motivation/Results/Availability |
| 4 | 统计检验 | 仅Bootstrap CI | + TOST等效性检验（±0.05 bounds）+ 事后功效分析 |
| 5 | 核心声明 | "architecture is irrelevant" | "no significant difference detected" |
| 6 | 参考文献 | 14篇 | 41篇（+KG/药物重定位/GNN方法论/等效性检验/数据库文献） |
| 7 | 循环验证 | 三库验证+排除442 PMID | + Discussion独立段落声明残余循环验证局限 |
| 8 | Conclusion | 与Discussion各节标题重复 | 完全重构，综合论述+未来方向 |
| 9 | 阴性对照 | 仅随机化边 | + 拓扑匹配阴性对照（degree-preserving rewiring） |

## 朝堂议政 court-ab0d0f2e 终审决议

**判定**: CONDITIONAL-GO（满足5项修改后投稿）

**5项硬性要求**（v9全部满足）:
1. ✅ TOST 等效性检验 + 预设等效边界（±0.05 AUC）
2. ✅ 事后功效分析 + "no significant difference detected"
3. ✅ Discussion 主动声明残余循环验证局限
4. ✅ 参考文献 14 → 41 篇
5. ✅ Abstract 压缩至 235 词，Conclusion 重构

## 投稿信息

- **目标期刊**: Bioinformatics (Oxford, IF ~5-6)
- **通讯作者**: 李昊森 (chongchong@tjh.tjmu.edu.cn)
- **单位**: 华中科技大学同济医学院同济医院口腔医学中心
- **论文标题**: Mechanism-Constrained Graph Learning for Inductive Drug Repositioning
- **论文文件**: `02-gnn-drug-repositioning/data/paper_full_draft_v9.md`
- **GitHub**: https://github.com/junxuku-byte/project-016-gnn-tmjoa

---

*Updated: 2026-06-01 17:30 CST*
