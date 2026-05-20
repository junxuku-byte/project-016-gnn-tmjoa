# Project-016 GNN 训练诊断 — 学术 Court 议政纪要

**时间**: 2026-05-19 20:26 CST
**议题**: GNN 药物重定位训练结果诊断与改进
**参与官员**: pi (首席科学家), methodologist (方法学专家), reviewer (质量评审, 429 未发言), engineer (技术开发), biostatistician (生物统计师)
**轮数**: 2

---

## 核心问题诊断

**过拟合根因共识**（5/5 官员一致）：
1. **随机特征记忆**: `torch.randn * 0.1` 无信息量，GNN 退化为拓扑记忆装置
2. **Transductive 信息泄露**: 179 节点小图中 train/val/test 共享同一图，节点 ID 即泄露
3. **BCE 饱和**: 1:3.5 负样本比导致 logits 被压制，分数偏低（最高 0.20）
4. **参数过剩**: 131K 参数对 1075 样本，记忆容量远超泛化需求

> AUC=1.0000 被定性为 **伪象 (spurious perfection)**，不可信，不可入文。

---

## 五方投票

| 选项 | 说明 | 支持官员 |
|------|------|---------|
| **A. 引入 ChEMBL/MeSH 真实特征重训** | 药物指纹 + 疾病语义嵌入 | **pi, methodologist, engineer, biostatistician** |
| **B. 改为 inductive 评估** | leave-drug-out / 时间分割交叉验证 | **pi, methodologist, engineer, biostatistician** |
| C. 调整负样本比例 | 微调 1:3.5 | 无人独立支持（附属于 A+B） |
| D. 先写论文再改进 | 当前结果直接投稿 | **全票反对** — 伪完美必遭拒稿 |

---

## 最终决议

**A+B 并行重训为唯一正道。**

执行顺序：
1. 引入 ChEMBL 药物指纹 / MeSH disease embedding 作为 node features
2. 重构 inductive 评估框架（leave-drug-out 或 leave-disease-out 交叉验证）
3. 重训练 → 获得可信 AUC/AP
4. 再做 Top-K 预测 + 文献回溯
5. 最后写论文

---

## 关键引用

- **首席科学家**: "随机初始化之节点，模型唯凭图结构记忆标签，此所谓'结构过拟合'——AUC三全实乃伪象"
- **方法学专家**: "当前结果不可直接入文。应优先引入真实特征，再设inductive验证"
- **技术开发工程师**: "随机特征记忆与transductive陷阱二毒并发"
- **生物统计师**: "随机特征致模型记节点身份，非习得药理关联，AUC遂虚高"

---

**下一步**: 启动 Phase 2 重训 — ChEMBL 特征工程 + Inductive 评估框架
