---
title: "A physics-informed graph neural network to approximate docking-based binding affinity for DYRK2 in Alzheimer’s drug repurposing"
author: "Veysel Gider et al."
year: 2026
journal: "Scientific Reports"
category: "general"
project: "[[project-016-gnn-drug-repositioning]]"
tags: ["physics-informed graph neural network", "binding affinity prediction", "DYRK2", "Alzheimer's disease", "drug repurposing", "docking-derived scores", "Coulomb potential", "Lennard-Jones potential"]
---
# A physics-informed graph neural network to approximate docking-based binding affinity for DYRK2 in Alzheimer’s drug repurposing

## 元数据
- **作者**: Veysel Gider et al.
- **年份**: 2026
- **期刊**: Scientific Reports
- **分类**: #general

## 方法
- Developed PhysDual-GCN, a dual-branch GNN that processes ligand molecular graphs and a sequence-based graph of DYRK2.
- Explicitly incorporated Coulomb and Lennard-Jones interaction terms as differentiable physical energy components.
- Trained on docking-derived scores (AutoDock Vina, Smina, QVina, CB-DOCK) with strict ligand-level separation to avoid circularity.
- Compared predictions against classical docking tools (SeamDock, AutoDock, Vina, QVina, CB-DOCK) and AI-based method DeepPurpose.

## 关键发现
- PhysDual-GCN achieved low absolute errors (MAE=0.31 kcal/mol; RMSE=0.44 kcal/mol) relative to reference docking scores.
- Correctly identified stronger binders such as donepezil (−10.8 kcal/mol) and brexpiprazole (−10.0 kcal/mol).
- Outperformed classical docking tools and DeepPurpose in approximating docking-derived binding affinities.
- Integration of physical interaction terms enhanced interpretability and provided a computationally efficient surrogate for classical docking workflows.

## 关键洞察
> PhysDual-GCN demonstrates that integrating explicit physical energy terms (Coulomb and Lennard-Jones) into a GNN framework can improve interpretability and computational efficiency for approximating docking-based binding affinities, even with very limited ligand data (n=4). However, due to the small dataset and reliance on docking-derived pseudo-labels (with inherent uncertainty of ±0.5–1.5 kcal/mol), the results indicate agreement with computational references rather than generalizable predictive performance. The approach establishes a foundation for future large-scale, experimentally validated studies in AD drug repurposing.

---
*来源: Universal Pipeline | project-016-gnn-drug-repositioning*