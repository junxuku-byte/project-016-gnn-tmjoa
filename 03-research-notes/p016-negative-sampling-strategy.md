# Project-016 负样本策略设计文档

**版本**: v1 (2026-05-18)
**项目**: GNN药物重定位 / TMJOA靶向预测
**目标**: 为GNN训练提供高质量负样本，解决"正样本稀疏+假阴性"问题

---

## 1. 问题分析

### 1.1 TMJOA的特殊性
- **罕见病**: FDA批准的TMJOA专用药物 ≈ 0
- **正样本来源**: 已发表研究中尝试过的off-label用药（NSAIDs, hyaluronic acid, PRP, corticosteroid injections等）
- **骨代谢共病层**: 大量全身骨代谢异常与TMJOA共存，可作为药物重定位的"桥梁"

### 1.2 负样本设计的核心矛盾
| 矛盾 | 说明 |
|------|------|
| 随机负采样太简单 | 模型容易过拟合，学到的只是"哪些drug和disease不可能同时出现"，而非机制性排斥 |
| 拓扑负采样太严格 | 知识图谱中TMJOA节点度数低，远距离节点池小，可能产生采样偏差 |
| 时间分割不适用 | TMJOA药物研究时间线短，缺乏长期追踪数据 |
| **假阴性风险** | 许多drug-disease对在现有文献中未报道，但未来可能被验证（重定位的本质） |

### 1.3 我们的优势
- **LabKG已构建三层网络**: disease → mechanism → drug，负采样可利用中间机制层过滤
- **骨代谢共病层**: 提供额外的disease节点（osteoporosis, vitamin D deficiency, CKD-MBD等），扩展负样本池
- **文献评分系统**: v2筛选的400篇文献提供了"edge可信度"的定量基础

---

## 2. 三级负采样策略

### Strategy 1: 机制排斥负采样 (Mechanism-Exclusion Negative Sampling)
**核心思想**: 如果两个drug和disease在机制层没有共享通路（或机制层距离>3 hops），则作为负样本。

**实现**:
1. 在LabKG中定义三类节点:
   - Disease nodes: TMJOA, 骨代谢异常 diseases
   - Mechanism nodes: 炎症通路, 骨代谢通路, 细胞过程
   - Drug nodes: 已知药物, 候选重定位药物
2. 计算 drug ↔ disease 的**机制路径距离**:
   - drug → target protein → signaling pathway → disease phenotype
3. 设定阈值:
   - **正样本**: 存在 ≤2 hops 的机制路径（drug-target-disease 或 drug-pathway-disease）
   - **负样本**: 机制路径距离 ≥4 hops（经过至少2个中间机制节点仍不连通）
   - **硬负样本**: 机制路径距离 = 3 hops（接近但不连通，模型需要学习微妙区分）

**公式**:
```
score_neg(drug, disease) = 1 / (1 + exp(-α * (d_mechanism - β)))
```
其中 `d_mechanism` 是机制层最短路径长度，`α` 控制陡峭程度，`β` 是阈值（建议=3）。

**优势**: 基于生物学机制而非随机运气，减少假阴性。

---

### Strategy 2: 文献共现排斥负采样 (Co-occurrence Exclusion)
**核心思想**: 在PubMed全库中从未共同出现在同一篇文献标题/摘要中的drug-disease对，且各自单独出现频率均>100次，作为负样本。

**实现**:
1. 对每个候选 (drug_i, disease_j):
   - 搜索 PubMed: `drug_i[Title/Abstract] AND disease_j[Title/Abstract]`
   - 如果返回 0 篇，进入下一步
   - 分别搜索 `drug_i` 和 `disease_j` 各自出现次数
   - 如果两者均 >100 次（说明各自都是研究热点，只是从未被放在一起研究），则标记为**强负样本**
   - 如果任一 <10 次（可能是冷门drug或冷门disease，零共现可能只是数据稀疏），则**丢弃**（避免false negative due to sparsity）

**优势**: 利用大规模文献证据，排除"只是因为还没人研究过"的drug-disease对。

**与Strategy 1的交集**: 如果某对同时满足机制排斥 AND 文献共现排斥，则为**硬负样本**（high confidence negative）。

---

### Strategy 3: 时间锚定负采样 (Temporal Anchor Negative Sampling)
**核心思想**: 利用药物上市时间和疾病研究发表时间，构造"早drug-晚disease"对，如果后续文献未报道其关联，则作为时间验证的负样本。

**实现**:
1. 定义时间线:
   - **早期药物** (t < 2010): 已上市多年，机制研究充分
   - **晚期疾病发现** (t > 2015): TMJOA的某些亚型或机制新发现
2. 构造候选对:
   - 选择2010年前已上市的drug（如Metformin 1957, Aspirin 1899）
   - 选择2015年后新发表的TMJOA机制研究（如特定通路发现）
   - 检查 2015-2026 年间是否有文献报道该drug对该TMJOA机制/亚型的治疗
   - 如果**零报道** → 时间锚定负样本

**优势**: 时间是最好的验证。一个药物如果上市50年都未被报道对某疾病有效，其概率极低（但并非不可能，这也是重定位的价值所在）。

**与Strategy 1&2的交集**: 三重验证 → **超硬负样本**。

---

## 3. 负样本比例与采样策略

### 3.1 正样本定义
| 正样本类型 | 来源 | 预计数量 |
|-----------|------|---------|
| 已验证TMJOA治疗 | 临床试验、RCT、case series | ~15-20 |
| 已验证OA治疗（类比） | Knee/Hip OA RCT → 外推 | ~30-50 |
| 骨代谢药-共病 | 已用于治疗coexisting bone metabolism disorder | ~20-30 |
| **正样本总计** | | **~70-100** |

正样本稀少 → 需要**数据增强**:
- 同药不同disease（如果drug对knee OA有效，对hip OA也有效 → 对TMJOA可能有效）
- 同disease不同药（如果NSAID有效，其他NSAID也可能有效）
- 通路替换（如果drug A通过NF-κB起效，drug B也通过NF-κB → 可能同效）

### 3.2 负样本比例
```
正:负 = 1 : 5 ~ 1 : 10
```
- 对于70个正样本，需要 350-700 个负样本
- 其中:
  - 机制排斥负样本: 40%
  - 文献共现排斥负样本: 40%
  - 时间锚定负样本: 20%
- 硬负样本（同时满足2+策略）: 标记为hard negative，GNN训练中给予更高权重

### 3.3 采样时的阶层平衡
```python
# 伪代码
negative_pool = {
    'mechanism_exclusion': sample_mechanism_exclusion(n=280, hard_ratio=0.3),
    'cooccurrence_exclusion': sample_cooccurrence_exclusion(n=280, hard_ratio=0.3),
    'temporal_anchor': sample_temporal_anchor(n=140, hard_ratio=0.5),
}

# 确保 drug 和 disease 的度数分布与正样本一致
drug_degree_pos = Counter(drug for drug, disease in positive_pairs)
disease_degree_pos = Counter(disease for drug, disease in positive_pairs)

negative_pairs = []
for strategy, candidates in negative_pool.items():
    # 按度数匹配采样
    sampled = degree_matching_sample(
        candidates,
        target_drug_degrees=drug_degree_pos,
        target_disease_degrees=disease_degree_pos,
        n_target=n_per_strategy
    )
    negative_pairs.extend(sampled)
```

---

## 4. 假阴性风险控制

### 4.1 为什么假阴性是药物重定位的致命问题
- 重定位的本质就是"发现已有drug对新disease有效"
- 如果负样本中包含了太多未来会被验证为正的药对，模型学到的只是"已知知识"，无法泛化到新发现

### 4.2 我们的三层防御

| 防御层 | 方法 | 效果 |
|--------|------|------|
| 第一层 | 机制排斥距离阈值β=3（而非β=2） | 避免把"机制上可能相关但未直接连接"的drug-disease对误判为负 |
| 第二层 | 文献共现频率阈值>100 | 排除因研究稀疏导致的零共现 |
| 第三层 | 时间锚定使用2010年前drug | 50+年未被发现的重定位极不可能，但保留Metformin等"重定位明星"的例外处理 |

### 4.3 例外处理清单
以下drug即使满足负样本标准，也应**手动排除**（它们是重定位的高概率候选）:
- **Metformin**: 已被重定位到多种炎症/衰老相关疾病
- **Rapamycin/Sirolimus**: mTOR抑制剂，广泛重定位研究
- **Resveratrol**: 天然化合物，多靶点抗炎，重定位常客
- **Curcumin**: 同上
- **Denosumab**: 已批准用于骨质疏松，TMJOA骨代谢层高度相关
- **Bisphosphonates**: 同上

---

## 5. 实施路线图

### Step 1: LabKG机制层完整化（本周）
- [ ] 从v2 400篇文献中提取所有drug-target-disease-mechanism四元组
- [ ] 计算drug-disease机制路径距离矩阵
- [ ] 标记正样本（已验证的drug-disease对）

### Step 2: 文献共现验证（本周）
- [ ] 批量PubMed API查询候选drug-disease对的共现次数
- [ ] 筛选零共现且各自高频的候选对

### Step 3: 时间锚定（下周）
- [ ] 整理drug上市时间线
- [ ] 整理TMJOA机制研究发表时间线
- [ ] 交叉验证2015-2026文献

### Step 4: 负样本集构建（下周）
- [ ] 按1:5~1:10比例采样
- [ ] 度数匹配
- [ ] 硬负样本标记
- [ ] 例外处理（手动排除重定位明星）

### Step 5: GNN训练（Phase 2）
- [ ] 正样本 + 负样本 → 二分类训练
- [ ] 评估指标: AUROC, AUPRC, top-k recall@20
- [ ] Top-20预测 → 文献回溯验证

---

## 6. 关键风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| LabKG机制层不完整 | 高 | v2文献导入优先处理mechanism extraction；允许β=3的宽松阈值 |
| PubMed API限速 | 中 | 批量查询+缓存；优先处理高频drug/disease |
| 正样本过少导致模型bias | 高 | 数据增强（同药/同disease/通路替换）；考虑few-shot learning |
| 假阴性混入 | 中 | 三层防御+例外清单；训练后人工审查top predicted "negatives" |

---

**下一步**: 实施Step 1 — LabKG机制层完整化。开始从v2 400篇文献中抽取drug-target-disease-mechanism四元组，构建机制路径距离矩阵。
