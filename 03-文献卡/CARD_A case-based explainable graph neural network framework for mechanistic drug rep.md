---
title: "A case-based explainable graph neural network framework for mechanistic drug repositioning"
author: "Adriana Carolina Gonzalez-Cavazos et al."
year: 2026
journal: "Bioinformatics"
category: "general"
project: "[[project-016-gnn-drug-repositioning]]"
tags: ["药物重定位", "图神经网络", "可解释人工智能", "案例推理", "知识图谱", "罕见病"]
---
# A case-based explainable graph neural network framework for mechanistic drug repositioning

## 元数据
- **作者**: Adriana Carolina Gonzalez-Cavazos et al.
- **年份**: 2026
- **期刊**: Bioinformatics
- **分类**: #general

## 方法
- 基于案例推理（CBR）的链接预测模块：通过检索与查询药物机制相似的药物案例，构建查询特定子图，并使用关系图卷积网络（R-GCN）生成疾病节点嵌入，预测候选疾病。
- 重要路径识别模块：通过异质路径强制掩码和节点度评分，学习边缘权重，识别连接药物与预测疾病的最具机制相关性的多跳路径。
- 使用删除、插入和稳定性测试评估解释的忠实性和鲁棒性。

## 关键发现
- DBR-X在链接预测任务中优于R-GCN、CompGCN和非参数CBR等基线模型，在MRR和Hits@K指标上均取得最高分。
- DBR-X生成的解释路径在ROC-AUC和边缘命中率上优于CBR+GNNExplainer、CBR+PGExplainer和CBR+PaGE等基线。
- DBR-X的解释在删除和插入测试中表现出高忠实性，在随机扰动下具有高稳定性（Pearson相关系数约0.85）。
- 通过三个罕见病案例（杜氏肌营养不良症、蒂莫西综合征、卡穆拉蒂-恩格尔曼病）验证了DBR-X预测和解释路径的生物学合理性。

## 关键洞察
> DBR-X通过结合案例推理和路径强制掩码学习，在药物重定位中实现了高预测性能与可解释性的平衡，其解释路径具有生物学合理性，尤其适用于罕见病的治疗候选药物发现。

---
*来源: Universal Pipeline | project-016-gnn-drug-repositioning*