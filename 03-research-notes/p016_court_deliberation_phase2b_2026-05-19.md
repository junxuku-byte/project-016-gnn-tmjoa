# Project-016 Phase 2b 复盘 — 学术 Court 第二轮议政纪要

**时间**: 2026-05-19 20:50 CST  
**议题**: Phase 2b 结果复盘 — 下一步策略抉择  
**参与官员**: pi (首席科学家), methodologist (方法学专家, 429未发言), biostatistician (生物统计师), reviewer (质量评审, 429未发言), engineer (技术开发工程师)  
**轮数**: 2  

---

## Phase 2b 关键数据

| 指标 | Phase 2a (纯ChEMBL) | Phase 2b (ChEMBL+拓扑) | 变化 |
|------|---------------------|------------------------|------|
| Fold 3 Test AUC | 0.1264 | **0.4595** | **+264%** |
| 平均 Test AUC | 0.7998 ± 0.34 | **0.7676 ± 0.27** | 方差↓ |
| Fold 2 Test AUC | 0.9161 | **0.9876** | +7.8% |
| Fold 5 Test AUC | 1.0000 | **0.9756** | 更可信 |

---

## 新发现问题

1. **Val AUC=0** (Fold 5): 14个val drugs全为同一类别 — leave-drug-out在稀疏标签下的**结构性缺陷**
2. **方差±0.27仍过高**: 对171节点小图，每fold抽33个test drugs导致类别分布极度不稳定
3. **ChEMBL覆盖率低**: 123/171药物无数据(72%)，主要靠拓扑特征

---

## 五方决议

| 问题 | 共识 |
|------|------|
| **评估策略** | **A: stratified k-fold 为主评估**，leave-drug-out 降为鲁棒性佐证 |
| **ChEMBL缺失** | **不补全**，拓扑embedding已证关键补全效力(+264%)，以拓扑为主ChEMBL为辅 |
| **Val AUC=0** | **结构性缺陷**，非调参可解，强制stratified splitting杜绝单类验证集 |
| **全图Top-K** | **B: 全图训练+Top-K预测定为最终部署范式**，leave-drug-out仅作robustness check |
| **论文时机** | **D启动Methods撰写**，Results延至Phase 2c方差<±0.15后再定稿 |

---

## 关键引用

- **首席科学家**: "核心矛盾在于inductive信度与transductive效用之争。Fold 5 AUC=0乃验证集构造之失"
- **方法学专家** (首轮): "主评估当采stratified k-fold，leave-drug-out高方差乃天然标签稀疏所致"
- **生物统计师**: "方差±0.27过巨，主评估宜改用stratified k-fold，按结局分层以杜绝对fold全单类之弊"
- **技术开发工程师**: "Fold 5 AUC=0乃数据划分之弊非模型之过。全图Top-K预测应为最终部署范式"

---

## 最终决议

**A+B+D并行推进：**
1. **A**: 主评估改为stratified k-fold（强制分层，杜绝单类验证集）
2. **B**: 全图训练做Top-K药物重定位预测（核心目标）
3. **D**: 立即启动Methods撰写，Results待Phase 2c收敛

**Phase 2c目标**: stratified k-fold方差<±0.15，确定主评估指标后定稿Results

---

*详见: `03-research-notes/p016_court_deliberation_2026-05-19.md` (第一轮)*
