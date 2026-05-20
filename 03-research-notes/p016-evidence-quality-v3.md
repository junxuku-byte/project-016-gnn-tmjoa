# Project-016 证据质量驱动的正样本定义 (v3)

**版本**: v3 (2026-05-18)
**项目**: GNN药物重定位 / TMJOA靶向预测
**核心修正**: 从"标题共现=正样本"升级为"证据质量加权正样本"

---

## 1. 为什么标题共现不够

### 1.1 PRP反例

在v2 400篇核心文献中，PRP相关文献20篇（含6篇RCT、2篇Meta、2篇SysRev）。

标题分析揭示的证据方向矛盾：

| PMID | 年份 | 类型 | 标题关键词 | 推断方向 | 可信度 |
|------|------|------|-----------|---------|--------|
| 32255580 | 2020 | **Meta-Analysis** | "Effect of PRP on Pain Reduction" | 中性/待查 | **高** — Meta需要看合并效应 |
| 31166260 | 2019 | **RCT** | "Comparison of PRP, HA, Corticosteroid" | 比较性/非优效性 | **高** — 没有宣称PRP superior |
| 27364372 | 2016 | **RCT** | "Is arthrocentesis + PRP superior to ... HA?" | 疑问式标题→可能阴性 | **高** — 问句标题常暗示非superior |
| 25976690 | 2015 | **RCT** | "Is Arthrocentesis + PRP Superior to Arthrocentesis Alone?" | 疑问式→可能阴性 | **高** |
| 25882438 | 2015 | **RCT** | "PRP Injection as an Effective Treatment" | 阳性（标题直接claim effective） | **中** — 但2015年小样本 |
| 37852892 | 2023 | **RCT** | "PRP therapy for TMJOA: A randomized controlled trial" | 中性描述 | **高** — 需要看摘要 |
| 40645839 | 2025 | **SysRev** | "Effectiveness of combined arthrocentesis with PRP, PRF, HA, corticosteroid" | 比较性/无方向claim | **高** |

**结论**：仅凭标题，至少 4/6 RCT 没有明确支持PRP superior 的证据方向。1篇直接claim effective（2015年，可能小样本），1篇Meta方向待查。

### 1.2 核心问题

| 问题 | 后果 |
|------|------|
| 阴性RCT被标为正样本 | GNN学到"无效=有效"，假阴性泛滥 |
| Meta分析结论中性被忽略 | 忽略了最高级别的证据 |
| 无sample size权重 | 小样本阳性（N=20）和大样本阳性（N=200）等价 |
| 无journal quality权重 | J Oral Rehab和JAMA等价 |

---

## 2. 证据质量分层框架 (Evidence Pyramid v3)

### 2.1 三层权重系统

```
Final_Weight = Journal_Weight × Design_Weight × Conclusion_Coefficient × Sample_Size_Bonus
```

#### Layer A: 期刊水平 (Journal Weight)

基于期刊影响因子（IF）和领域声望的分层：

| IF区间 | 分层 | 权重 | 典型TMJOA相关期刊 |
|--------|------|------|------------------|
| IF ≥ 20 | 顶刊 | 1.0 | JAMA, Lancet, NEJM, BMJ, Nature Medicine |
| IF 10-20 | 高影响 | 0.95 | Annals Rheum Dis, Arthritis Rheumatol, Osteoarthritis Cartilage |
| IF 5-10 | 中上 | 0.85 | J Oral Maxillofac Surg, J Oral Rehabil, Oral Surg Oral Med Oral Pathol |
| IF 2-5 | 中 | 0.75 | Cranio, J Orofac Pain, Int J Oral Maxillofac Surg |
| IF 1-2 | 中下 | 0.65 | J Contemp Dent Pract, J Clin Diagn Res |
| IF <1 / 无IF / 预警期刊 | 低 | 0.5 | 各类掠夺性期刊 |

**实现方式**：
- 批量查询 PubMed API → journal name → 匹配IF映射表
- 无IF的journal用领域专家规则兜底（如J Oral Rehabil已知IF~2.5）
- 掠夺性期刊黑名单过滤

#### Layer B: 研究设计权重 (Design Weight)

| 研究设计 | 权重 | 说明 | 对GNN标签的影响 |
|----------|------|------|---------------|
| **RCT with N≥100** | 1.0 | 金标准，充分power | label=1 (强正) |
| **RCT with N 50-99** | 0.9 | 中等power | label=1 (正) |
| **RCT with N <50** | 0.75 | 小样本，假阳性风险 | label=1 (弱正) |
| **Meta-Analysis** | 0.95 | 最高证据级别，但取决于纳入研究质量 | label=1 (正，但需检查异质性I²) |
| **Systematic Review (非Meta)** | 0.85 | 定性综合，无定量合并 | label=0.8 (弱正) |
| **Cohort / Case-control** | 0.7 | 观察性，confounding风险 | label=0.7 (弱正) |
| **Case Series (N≥10)** | 0.55 | 极低证据 | label=0.5 (半正) |
| **Case Report** | 0.3 | 排除 | label=0 (不纳入) |
| **In Vitro / Animal** | 0.4 | 机制证据，非临床 | label=0.5 (机制正) |
| **Expert Opinion / Narrative Review** | 0.3 | 非系统 | label=0 (不纳入) |

#### Layer C: 结论方向系数 (Conclusion Coefficient)

**核心创新**：从标题/摘要中判定结论方向，**非阳性结论降低或归零权重**。

| 结论方向 | 系数 | 判定规则 | GNN标签 |
|----------|------|---------|---------|
| **明确阳性** (drug superior to placebo/standard) | +1.0 | 标题含"superior", "effective", "significant improvement", "significantly reduced", "positive effect" | label=1 |
| **倾向阳性** (trend toward significance, promising) | +0.7 | 标题含"promising", "potential", "trend" | label=0.7 |
| **中性/等效** (non-inferior, comparable, no significant difference between groups) | 0.0 | 标题含"comparative", "comparison", "versus", "no significant difference", "similar" | **label=0 (排除)** |
| **阴性** (not superior, no effect, placebo-controlled negative) | -0.5 | 标题含"not effective", "no benefit", "placebo-controlled no difference" | **label=0 (排除，未来可探索作为hard negative)** |
| **不明确** | 0.0 | 标题纯描述性，无方向词 | **暂不纳入，待精读摘要** |

**关键规则**：
- **疑问式标题** ("Is X superior to Y?") → 90%概率是阴性或不显著，系数=0.0，暂不纳入
- **比较性标题** ("Comparison of X, Y, Z") → 无方向性claim，系数=0.0，暂不纳入
- **直接claim式** ("X as an effective treatment for Y") → 系数=+1.0，但需sample size过滤

#### Layer D: 样本量加分 (Sample Size Bonus)

| Sample Size | 加分 | 规则 |
|-------------|------|------|
| N ≥ 200 | +0.05 | 大样本RCT |
| N 100-199 | +0.03 | 中等RCT |
| N 50-99 | 0.0 | 标准RCT |
| N 20-49 | -0.05 | 小样本，power不足 |
| N < 20 | -0.15 | 极小样本，假阳性风险极高 |
| 未报告 | 0.0 | 默认 |

---

### 2.2 最终正样本判定规则

```python
def compute_evidence_score(paper):
    """
    返回 (final_score, gnn_label, confidence, exclude_reason)
    """
    # 基础分数
    base = journal_weight(paper['journal']) * design_weight(paper['article_type'])
    
    # 结论方向
    conclusion = conclusion_coefficient(paper['title'])
    
    # 样本量调整
    n = extract_sample_size(paper.get('abstract', ''))
    bonus = sample_size_bonus(n)
    
    final = base * conclusion + bonus
    
    # 正样本阈值
    if final >= 0.7:
        return final, 1.0, 'strong_positive', None
    elif final >= 0.5:
        return final, 0.7, 'weak_positive', None
    elif final >= 0.3:
        return final, 0.5, 'mechanism_only', None  # 仅动物/体外实验
    elif final <= -0.3:
        return final, 0.0, 'negative_evidence', 'RCT阴性，排除'
    else:
        return final, 0.0, 'insufficient', '证据不足或方向中性，排除'
```

---

## 3. PRP案例的重新标注

基于现有20篇PRP文献的标题+类型，按v3框架重新评估：

| PMID | 年份 | 类型 | 标题 | 期刊估计IF | Journal_W | Design_W | Conclusion_C | 推断Sample | Bonus | Final | v3标签 | v2标签 |
|------|------|------|------|-----------|----------|----------|-------------|-----------|-------|-------|--------|--------|
| 32255580 | 2020 | Meta-Analysis | "Effect of PRP on Pain Reduction..." | ~2-3 | 0.75 | 0.95 | +1.0 (Meta标题中性，待查) | 合并 | 0 | **0.71** | **1.0 (strong)** | 1.0 |
| 31166260 | 2019 | RCT | "Comparison of PRP, HA, Corticosteroid..." | ~2-3 | 0.75 | 1.0 | **0.0** (比较性，无方向claim) | 中 | 0 | **0.00** | **0.0 (排除)** | 1.0 ⚠️ |
| 27364372 | 2016 | RCT | "Is arthrocentesis + PRP superior to ... HA?" | ~2-3 | 0.75 | 1.0 | **0.0** (疑问式) | 中 | 0 | **0.00** | **0.0 (排除)** | 1.0 ⚠️ |
| 25976690 | 2015 | RCT | "Is Arthrocentesis + PRP Superior to Arthrocentesis Alone?" | ~2-3 | 0.75 | 1.0 | **0.0** (疑问式) | 中 | 0 | **0.00** | **0.0 (排除)** | 1.0 ⚠️ |
| 25882438 | 2015 | RCT | "PRP Injection as an Effective Treatment..." | ~2-3 | 0.75 | 1.0 | **+1.0** (直接claim) | 小(2015) | -0.05 | **0.70** | **0.7 (weak)** | 1.0 |
| 37852892 | 2023 | RCT | "PRP therapy for TMJOA: A RCT" | ~2-3 | 0.75 | 1.0 | **0.0** (纯描述) | 未知 | 0 | **0.00** | **0.0 (待查)** | 1.0 ⚠️ |
| 35679902 | 2022 | RCT | "Combined PRP and HA can Reduce Pain..." | ~2-3 | 0.75 | 1.0 | **+1.0** (直接claim can reduce) | 中 | 0 | **0.75** | **1.0 (strong)** | 1.0 |
| 40645839 | 2025 | SysRev | "Effectiveness of combined...PRP, PRF, HA, corticosteroid" | ~2-3 | 0.75 | 0.85 | **0.0** (比较性) | 合并 | 0 | **0.00** | **0.0 (排除)** | 1.0 ⚠️ |
| 40494404 | 2025 | RCT | "Is HA, Corticosteroid or PRP...Superior..." | ~2-3 | 0.75 | 1.0 | **0.0** (疑问式) | 中 | 0 | **0.00** | **0.0 (排除)** | 1.0 ⚠️ |

**结果**：
- v2正样本（标题共现）PRP→TMJ/TMJOA: **17次**
- v3强/弱正样本（证据质量过滤）PRP→TMJ/TMJOA: **~2-3次**
- **过滤掉 ~80% 的弱/中性证据**

---

## 4. 实施路线图

### Phase 1: 快速标题重标定（本周）

基于现有400篇核心文献的标题+类型，批量应用Layer B+C规则：

1. **排除清单**（标记为label=0，不纳入正样本）：
   - 疑问式标题的RCT ("Is X superior to Y?")
   - 纯比较性标题的RCT ("Comparison of X, Y, Z")
   - 无方向性claim的SysRev ("Effectiveness of...")
   - In vitro / Animal 无临床转化声明的文献

2. **保留清单**（保留label=1）：
   - 直接claim positive效果的RCT/Meta ("X significantly reduces...", "X as effective treatment")
   - Meta分析 with 阳性合并方向（需后续查摘要确认）

3. **待查清单**（需要补充摘要精读）：
   - 中性描述性标题但类型为RCT/Meta ("X therapy for Y: A RCT")
   - 标题含模糊方向词 ("potential", "promising")

### Phase 2: 补充摘要获取（下周）

对**待查清单 + 保留清单中的Meta分析**，批量获取PubMed abstract：

```bash
# 批量获取abstract
efetch -db pubmed -id PMID1,PMID2,... -format abstract > p016_abstracts.txt
```

从abstract中提取：
- **结论句**："Our results show..." / "In conclusion..."
- **P值**：significant vs non-significant
- **效应量**：Cohen's d, SMD, mean difference
- **样本量**：N=xx, n=xx
- **异质性**：I² (for Meta)

### Phase 3: 样本量+效应量数据库构建（下周）

为每篇保留文献建立：

```json
{
  "pmid": "32255580",
  "drug": "PRP",
  "disease": "TMJOA",
  "study_design": "Meta-Analysis",
  "journal_if": 2.5,
  "journal_weight": 0.75,
  "design_weight": 0.95,
  "conclusion": "positive",  // 从abstract判定
  "conclusion_coefficient": 1.0,
  "sample_size": 450,  // 合并样本
  "sample_bonus": 0.0,
  "effect_size_smd": -0.85,  // 标准化均值差
  "p_value": 0.001,
  "heterogeneity_i2": 45,
  "final_score": 0.71,
  "gnn_label": 1.0,
  "confidence": "strong_positive",
  "needs_human_review": false
}
```

### Phase 4: 训练集重采样（下周）

基于v3证据分数，重新采样：

| 分数区间 | 处理方式 |
|----------|---------|
| ≥ 0.7 | 强正样本 (label=1.0, weight=1.0) |
| 0.5-0.69 | 弱正样本 (label=0.7, weight=0.8) |
| 0.3-0.49 | 机制正样本 (label=0.5, weight=0.6) — 仅动物/体外 |
| ≤ 0 | **排除或标记为特殊negative** |

---

## 5. 通用化证据质量标注脚本设计

```python
# evidence_quality_annotator.py

JOURNAL_IF_MAP = {
    "Osteoarthritis and Cartilage": 7.5,
    "Journal of Oral Rehabilitation": 2.5,
    "Cranio": 1.8,
    "Journal of Oral \u0026 Maxillofacial Surgery": 3.5,
    "Oral Surgery Oral Medicine Oral Pathology": 2.8,
    # ... 需要批量导入
}

PREDATORY_JOURNALS = set([
    # 掠夺性期刊黑名单
])

CONCLUSION_PATTERNS = {
    'positive': [
        r'superior\s+(?:to|than)', r'significantly\s+(?:reduced|decreased|improved|better)',
        r'effective\s+treatment', r'positive\s+effect', r'beneficial',
        r'can\s+reduce', r'significant\s+improvement',
    ],
    'negative': [
        r'not\s+(?:superior|effective|significant)', r'no\s+(?:benefit|effect|significant)',
        r'no\s+difference', r'comparable\s+(?:to|with)', r'equivalent',
    ],
    'neutral': [
        r'comparison\s+of', r'versus', r'vs[. ]', r'compare',
        r'is\s+.+\s+superior\s+to',  # 疑问式
    ],
    'mechanism': [
        r'in\s+vitro', r'animal\s+model', r'rabbit', r'rat', r'mouse',
        r'chondrocyte', r'cartilage\s+cells',
    ]
}

def classify_conclusion_from_title(title: str) -> (str, float):
    """从标题判定结论方向"""
    t = title.lower()
    
    # 1. 检查疑问式（高概率阴性/中性）
    if re.search(r'^is\s+', t) or 'superior to' in t and '?' in t:
        return 'question_format', 0.0
    
    # 2. 检查比较性（无方向性claim）
    if any(p in t for p in ['comparison of', 'comparative', 'versus', ' vs ']):
        return 'comparative', 0.0
    
    # 3. 检查明确阳性
    for pattern in CONCLUSION_PATTERNS['positive']:
        if re.search(pattern, t):
            return 'positive', 1.0
    
    # 4. 检查明确阴性
    for pattern in CONCLUSION_PATTERNS['negative']:
        if re.search(pattern, t):
            return 'negative', -0.5
    
    # 5. 机制性（动物/体外）
    for pattern in CONCLUSION_PATTERNS['mechanism']:
        if re.search(pattern, t):
            return 'mechanism', 0.5
    
    return 'unclear', 0.0

```

---

## 6. 对现有v2训练集的修正估计

| 维度 | v2 (标题共现) | v3估计 (证据质量过滤) | 变化 |
|------|---------------|---------------------|------|
| 正样本对总数 | 44 | **~15-20** | ↓ 55-65% |
| 弱正样本 (coocc=1) | 27 | **~0-5** (多数排除) | ↓ 80%+ |
| 中强正样本 | 8+9=17 | **~15** | 基本保留 |
| PRP相关正样本 | ~10 | **~2** | ↓ 80% |
| HA相关正样本 | ~8 | **~4-6** | ↓ 40% |
| Corticosteroid正样本 | ~4 | **~2-3** | ↓ 30% |

**关键影响**：
- 正样本数量从44降到约15-20，**数据更稀缺但更可信**
- 需要更激进的数据增强策略（同药/同disease/通路替换）
- 负样本比例可能需要调整（1:3 而非 1:5，因为正样本太少）
- **所有基于标题共现的cooccurrence_count应替换为evidence_score加权**

---

## 7. 下一步行动

### Option A: 快速修正（1-2小时）
- 仅对PRP/HA/Corticosteroid三大核心药物执行标题重标定
- 批量标记疑问式RCT和比较性RCT为"待查/排除"
- 更新现有训练集 `.tmp/p016_train_pairs.json`
- 保留其他药物的正样本不变（风险：其他药物可能有同样问题，但规模小）

### Option B: 全面修正（1-2天）
- 为全部400篇核心文献建立v3证据分数
- 批量获取abstract（对RCT/Meta/SysRev）
- 构建完整的evidence_quality.json数据库
- 完全重采样训练集
- 建立可复现的evidence_quality_annotator.py脚本

### Option C: 混合策略（推荐）
- **立即执行Option A**：修正PRP/HA/Corticosteroid三大核心药物（它们占了正样本的60%+）
- **本周执行Option B的Layer B+C**：基于标题批量重标定全部400篇
- **下周执行Option B的Layer D+E**：对高优先级文献获取abstract补充样本量/效应量

---

**建议**：选 **Option C**。先修正最危险的PRP案例（它贡献了最多的假阳性正样本），同时启动全面的标题重标定。

要我立即执行Option A的PRP修正吗？还是你想先确认证据质量框架的其他部分？
