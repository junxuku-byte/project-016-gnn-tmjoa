# Project-016 关键文献精读分析报告

**日期**: 2026-05-18
**分析师**: 油油（AI学术助手）
**文献数**: 4篇（5篇下载，1篇内容错误排除）

---

## 文献1: PMID 40273050 — FGF18 mouse TMJOA

| 属性 | 内容 |
|------|------|
| **标题** | FGF18 induces chondrogenesis and anti-osteoarthritic effects in a mouse model for TMJ degeneration |
| **期刊** | PLoS ONE 2025 | 
| **研究设计** | 动物实验（小鼠TMJ退变模型） |
| **样本量** | 32只小鼠，4组（Control/FGF18 × Degeneration/Normal）|
| **干预** | 关节内注射rmFGF18（5µg/week，3周）vs 生理盐水 |
| **核心发现** | ① FGF18显著增加纤维软骨厚度 ② 显著增加蛋白聚糖分布 ③ 显著增加软骨细胞增殖（Edu+）④ Noggin（BMP拮抗剂）减少 ⑤ TIMP1（MMP13抑制剂）增加 ⑥ MMP13表达降低 |
| **机制** | FGF18→FGFR3→促进软骨生成+抑制降解 |
| **证据强度** | **⭐⭐⭐⭐ 强阳性**（TMJOA直接+机制明确） |

### 训练集更新建议
- **新增正样本**: `FGF18` × `TMJOA` (label=0.7, weight=1.0, tmj_relevance=direct, source=PMID40273050_animal)
- **新增正样本**: `FGF18` × `TMJ` (label=0.7, weight=1.0, tmj_relevance=direct)
- **机制标签**: chondrogenesis, anti-MMP13, TIMP1-up, Noggin-down, FGFR3-signaling

---

## 文献2: PMID 29244893 — HA vs CS Meta-analysis for TMD

| 属性 | 内容 |
|------|------|
| **标题** | Effectiveness of Intra-Articular Injections of Sodium Hyaluronate or Corticosteroids for Intracapsular TMD |
| **期刊** | J Oral Facial Pain Headache 2018 |
| **研究设计** | **系统综述 + Meta分析** |
| **纳入RCT** | 7项（其中4项可进行Meta分析）|
| **关键结论** | ① NaH与CS短期/长期疼痛改善**无显著差异** ② NaH vs Placebo：1项研究显示应答者数量显著多于安慰剂，另一项显示与CS无显著差异 ③ 证据质量：**极低**（6项研究偏倚风险不清楚，1项高风险）|
| **证据强度** | **⭐⭐ 弱阳性/中性**（Meta分析=无显著差异 vs CS，但vs安慰剂有微弱证据） |

### 训练集更新建议
- **Hyaluronic acid × TMD/TMJOA**: 当前标签可能过高（如原为1.0）→ 建议降至 **label=0.5, weight=0.8**
- 原因：Meta分析显示HA vs CS无显著差异，vs安慰剂仅有1项RCT支持
- **保留正样本但降级**：从strong_positive → weak_positive/mechanism_only

---

## 文献3: PMID 30814387 — Long-term arthrocentesis±HA

| 属性 | 内容 |
|------|------|
| **标题** | Long-term effectiveness of arthrocentesis with and without hyaluronic acid injection for treatment of temporomandibular joint osteoarthritis |
| **期刊** | J Oral Sci 2019 |
| **研究设计** | **RCT**（长期随访~4年） |
| **样本量** | 40人，37人完成随访（A组arthrocentesis alone n=17, AS组+HA n=20）|
| **关键结论** | ① **两组VAS疼痛均显著下降**（A: 64→16, AS: 63→25, P<0.001） ② 两组间VAS下降**无显著差异**（P未报告，但数值接近） ③ 最大开口度均显著增加，组间无显著差异（P=0.223） ④ **关节音未显著改善** |
| **证据强度** | **⭐⭐⭐ 中等阳性**（RCT但HA未显示优于单独arthrocentesis） |

### 训练集更新建议
- **HA × TMJOA**: 当前标签可能过高 → 建议 **label=0.6, weight=0.9**（RCT但无 superiority）
- **Arthrocentesis × TMJOA**: 应标记为正样本 **label=0.7, weight=1.0**（明确有效）
- 注意：此研究**不支持**"HA add-on value"，HA未显示优于单独arthrocentesis

---

## 文献4: PMID 28879245 — High-Dose Capsaicin TMD RCT

| 属性 | 内容 |
|------|------|
| **标题** | Effects of High-Dose Capsaicin on TMD Subjects: A Randomized Clinical Study |
| **期刊** | JDR Clin Transl Res 2016 |
| **研究设计** | **RCT**（双盲、vehicle-controlled） |
| **样本量** | TMD组+健康对照组（具体N未在摘要中明确） |
| **干预** | 8% topical capsaicin cream（2小时涂抹）vs vehicle control |
| **关键结论** | ① **capsaicin组TMD患者1周后疼痛显著降低**（vs vehicle, P显著） ② QST：热痛阈值2小时后下降（两组均下降，1周内恢复） ③ **压力痛阈值和机械敏感性无影响** |
| **证据强度** | **⭐⭐⭐ 中等阳性**（RCT有效但样本量可能较小，外用非关节内） |

### 训练集更新建议
- **Capsaicin × TMD**: 应标记为正样本 **label=0.6, weight=0.8**（RCT支持但外用途径不同）
- 注意：这是**外用**capsaicin（8% cream），不是关节内注射

---

## 文献5: PMID 37608244 — ❌ 内容错误

| 属性 | 内容 |
|------|------|
| **实际内容** | Comparison of Ultrasound- vs Landmark-Guided Injections for Musculoskeletal Pain: An Umbrella Review |
| **问题** | Google Scholar/Unpaywall下载了错误的PDF（同名或标题相似的文献） |
| **影响** | 无法用于HA for TMD的证据评估 |

### 补救措施
- 需重新获取正确的PMID 37608244 PDF
- 可能通过：① 期刊官网OA ② 机构VPN ③ 联系作者

---

## 训练集更新汇总

| 药物 | 疾病 | 当前标签 | 新标签 | 理由 |
|------|------|---------|--------|------|
| FGF18 | TMJOA | **缺失** | **0.7** | 新增：PLoS ONE 2025动物模型，直接证据 |
| FGF18 | TMJ | **缺失** | **0.7** | 新增：同上 |
| Hyaluronic acid | TMJOA | ~1.0 | **0.5** | 降级：Meta分析vs CS无差异，vs安慰剂证据弱 |
| Hyaluronic acid | TMD | ~1.0 | **0.5** | 降级：同上 |
| Arthrocentesis | TMJOA | **缺失** | **0.7** | 新增：RCT明确有效（无论是否+HA） |
| Capsaicin | TMD | ~0.3 | **0.6** | 升级：RCT证实8%外用有效 |

---

## 对GNN模型的影响

1. **FGF18新增为TMJ核心正样本** → 显著提升模型对FGF18的预测分数
2. **HA降级** → 模型对HA的预测更保守，反映证据局限性
3. **Arthrocentesis新增** → 模型学习"机械清创"作为有效干预
4. **Capsaicin升级** → TRPV1靶向治疗获得模型支持

---

## 下一步

1. 将上述更新写入训练集（v4.1）
2. 重新训练GNN模型
3. 重新评估Top-20预测（特别是FGF18的排名）
4. 重新获取PMID 37608244的正确PDF

---

*报告生成时间: 2026-05-18 23:00*
