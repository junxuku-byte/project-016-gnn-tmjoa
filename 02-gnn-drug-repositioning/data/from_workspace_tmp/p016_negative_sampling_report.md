# Project-016 GNN训练集 — 负样本采样报告

**生成时间**: 2026-05-18T18:01:24.920534
**采样方法**: v2文献标题共现驱动 + 度数匹配 + 硬负样本标记
**数据源**: 400 篇v2核心文献
**正:负比例**: 1:5

## 实体统计

| 类型 | 数量 | Top 10 |
|------|------|--------|
| 药物 | 27 | PRP(18), Hyaluronic acid(15), Denosumab(6), Corticosteroid(6), Resveratrol(5), Vitamin D(5), Statin(4), Burosumab(4), Curcumin(3), Anti-FGF23(2) |
| 疾病 | 20 | TMJ(261), Osteoarthritis(257), TMJOA(163), TMJ condyle(31), Osteoporosis(22), Knee OA(15), Bone loss(11), Bone resorption(6), Rheumatoid arthritis(4), TMD(4) |

## 正样本

**总数**: 44 对

| 置信度 | 数量 | 说明 |
|--------|------|------|
| weak_positive | 27 | 仅1次标题共现 |
| medium_positive | 8 | 2-3次标题共现 |
| strong_positive | 9 | ≥4次标题共现 |

### Top 10 正样本（按共现次数）

1. **PRP → TMJ**: 17 次共现 (strong_positive)
2. **Hyaluronic acid → TMJ**: 15 次共现 (strong_positive)
3. **PRP → Osteoarthritis**: 15 次共现 (strong_positive)
4. **Hyaluronic acid → Osteoarthritis**: 15 次共现 (strong_positive)
5. **PRP → TMJOA**: 15 次共现 (strong_positive)
6. **Hyaluronic acid → TMJOA**: 13 次共现 (strong_positive)
7. **Corticosteroid → Osteoarthritis**: 6 次共现 (strong_positive)
8. **Corticosteroid → TMJ**: 6 次共现 (strong_positive)
9. **Corticosteroid → TMJOA**: 5 次共现 (strong_positive)
10. **Vitamin D → Osteoarthritis**: 3 次共现 (medium_positive)

## 负样本

**总数**: 73 对 (正:负 = 1:5)
**硬负样本**: 37 (50.7%)
**重定位明星已排除**: Metformin, Everolimus, Quercetin, Rapamycin, Fisetin, EGCG, Curcumin, Resveratrol, Ginger, Omega-3, SAMe

### 度数匹配
- 正样本 drug 度数分布: {'Hyaluronic acid': 4, 'Chondroitin': 3, 'Denosumab': 3, 'Vitamin D': 3, 'Glucosamine': 3, 'PRP': 3, 'MSM': 3, 'NSAIDs': 3, 'Statin': 3, 'Corticosteroid': 3}
- 正样本 disease 度数分布: {'Osteoarthritis': 16, 'TMJ': 13, 'TMJOA': 9, 'Osteoporosis': 2, 'TMD': 1, 'Knee OA': 1, 'Bone resorption': 1, 'TMJ disc displacement': 1}
- 负样本 drug 度数分布: {'Hyaluronic acid': 10, 'Vitamin D': 8, 'PRP': 8, 'Corticosteroid': 8, 'Denosumab': 8, 'Statin': 5, 'ASU': 3, 'Calcium': 2, 'Collagen': 2, 'JAK inhibitor': 2}
- 负样本 disease 度数分布: {'TMJOA': 10, 'TMJ': 10, 'Osteoarthritis': 7, 'Osteoporosis': 6, 'Bone resorption': 4, 'Knee OA': 4, 'TMJ disc displacement': 4, 'TMD': 4, 'Iron overload': 2, 'Rheumatoid arthritis': 2}

### Top 10 硬负样本

1. **PRP → Bone resorption**: drug_freq=18, disease_freq=6
2. **PRP → Rheumatoid arthritis**: drug_freq=18, disease_freq=4
3. **Corticosteroid → Knee OA**: drug_freq=6, disease_freq=15
4. **PRP → Osteoporosis**: drug_freq=18, disease_freq=22
5. **Hyaluronic acid → Condylar resorption**: drug_freq=15, disease_freq=3
6. **Hyaluronic acid → Knee OA**: drug_freq=15, disease_freq=15
7. **Statin → TMJ disc displacement**: drug_freq=4, disease_freq=3
8. **Hyaluronic acid → TMD**: drug_freq=15, disease_freq=4
9. **Statin → Osteoporosis**: drug_freq=4, disease_freq=22
10. **Hyaluronic acid → Osteoporosis**: drug_freq=15, disease_freq=22

## 数据集划分

| Split | 总数 | 正样本 | 负样本 | 药物种类 | 疾病种类 |
|-------|------|--------|--------|----------|----------|
| Train | 81 | 30 | 51 | 23 | 18 |
| Val | 17 | 5 | 12 | 11 | 11 |
| Test | 19 | 9 | 10 | 12 | 11 |

---

## GNN训练建议

### 正样本使用策略
- 弱正样本 (cooccurrence=1) 噪声较大，建议：
  1. 作为训练集使用，但权重降低（如0.5）
  2. 或仅保留medium/strong_positive作为验证/测试的正样本
- 强正样本 (cooccurrence≥4) 可信度最高，应优先放入训练集

### 负样本使用策略
- 硬负样本在训练中应给予更高loss权重
- 考虑使用contrastive learning或margin-based loss
- 注意：重定位明星已手动排除，但训练时仍可单独测试它们的预测分数

### 模型评估
- 主要指标: AUROC, AUPRC
- 特别关注: Top-20 predicted drug-disease对中，有多少是已知治疗（验证召回率）
- 对重定位明星的单独评估：预测分数是否偏高（验证模型是否有发现新药的能力）

---

## 关键发现

1. **PRP和Hyaluronic acid是最高频正样本** — 与TMJ/TMJOA的共现次数分别达17和13次，反映了它们在临床研究中的 prominence
2. **Corticosteroid和Denosumab有多个正样本关联** — 支持了它们在OA/骨代谢中的已知作用
3. **硬负样本占50.7%** — 说明负样本池质量高，模型需要学习微妙的机制区分
4. **数据集规模小但覆盖广** — 27种药物×20种疾病，540种可能对中筛选出117对训练样本
5. **数据增强空间** — 可考虑通过通路替换、疾病类比等方式扩展正样本
