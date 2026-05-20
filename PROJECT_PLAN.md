# Project-016 GNN 药物重定位【2026-05-17方向深化】

> 原方向：CHIP-TMJOA 假说构建（文献学习）
> 新方向：图神经网络（GNN）驱动的 TMJOA 药物重定位（计算验证）
> 深化逻辑：从"假说提出"推进到"计算方法验证 + 预测输出"

---

## 项目定位

**类型**：纯计算 / AI 独立落地项目
**核心方法**：知识图谱（KG）+ 图神经网络（GNN）链接预测
**目标产出**：预测 TMJOA 潜在新药靶点对 + 验证论文

---

## 科学问题

基于现有 LabKG（10,000+ 实体 / 14,000+ 关系），利用 GNN 的链接预测能力：

1. **已知药物-靶点对**：训练 GNN 学习药物-靶点-疾病三元组的拓扑模式
2. **未知链接预测**：预测"未标注药物-靶点"的潜在关联
3. **TMJOA 特异性过滤**：限制预测结果在 TMJOA 相关子网络
4. **回溯验证**：文献/分子对接验证 Top-K 预测

---

## 技术路线

```
Phase 1: LabKG 子图提取（已完成 ✅）
  ↓ 提取药物-靶点-疾病子网络
Phase 2: GNN 链接预测（进行中 ⏳）
  ↓ Node2Vec 基线 → GNN（GCN/GAT）→ 拓扑基线 v2
Phase 3: 预测验证（待启动）
  ↓ Top-K 预测 → 文献回溯 → 分子对接（可选）
Phase 4: 论文撰写（待启动）
  ↓ "Graph Neural Network-Based Drug Repositioning for TMJOA"
```

---

## 当前资产

| 资产 | 状态 | 路径 |
|------|------|------|
| LabKG 全量 | ✅ | 10,357 节点 / 14,626 边 |
| TMJOA 子网络 | ✅ | 已提取 |
| 拓扑基线 v2 | ✅ | `results/topology_predictions_v2_tmj_fixed.json` |
| GNN PyTorch 模型 | ✅ | `models/gnn_pure_pytorch.py` |
| Node2Vec 基线 | ✅ | `models/node2vec_baseline.py` |
| 核心文献 PDF | ✅ | 20 篇精读（CHIP/OA/TMJ） |

---

## 时间线（更新）

| 阶段 | 时间 | 产出 |
|------|------|------|
| Phase 2.5 模型优化 | Week 1 | GNN 训练收敛，AUC > 0.85 |
| Phase 3 预测验证 | Week 2 | Top-20 预测 + 文献回溯 |
| Phase 4 论文写作 | Week 3-4 | 完整论文初稿 |
| Phase 5 投稿 | Week 5 | 目标期刊确认 + 投稿 |

---

## 目标期刊

- 首选：*Briefings in Bioinformatics*（IF ~9，方法学友好）
- 备选：*Journal of Cheminformatics*（IF ~5）
- 兜底：*Scientific Reports*（IF ~4）

---

## 风险提示

1. **负样本问题**：KG 中缺失链接 ≠ 真实负样本，需构建可靠负样本
2. **冷启动问题**：新药物/新靶点无节点特征，预测受限
3. **过拟合风险**：小图（TMJOA 子网络）容易过拟合，需正则化

---

## 变更记录

- 2026-05-16: 立项（CHIP-TMJOA 假说构建）
- 2026-05-17: 方向深化 → GNN 药物重定位（计算方法验证）

---

*Updated: 2026-05-17 14:57 CST | YouYou*
