# Project-016 精读笔记: Hu 2025 Nat Commun Denosumab FSTL1

> Hu Y, Chen W, Lan S, et al. *Nature Communications* 16:11394 (2025)
> DOI: 10.1038/s41467-025-66202-z | PMID: 41407913
> 精读日期: 2026-05-16 | 来源: 全文精读

---

## 1. 核心定位

**Denosumab 在骨关节炎中的双重机制**：
1. **经典机制**: 阻断 RANKL → 破骨细胞成熟抑制 → 骨吸收减少
2. **新发现机制**: 阻断 RANKL → 成纤维样滑膜细胞（FLS）中 RANK/TRAF6/FSTL1/NF-κB 轴抑制 → 滑膜炎症减少 → 软骨降解延缓

> 滑膜炎（synovitis）是OA发展的关键早期事件，FLS是主要效应细胞。Denosumab靶向FLS中的RANKL是新机制。

---

## 2. 关键发现

### 2.1 RANK/RANKL在OA滑膜中高表达
| 发现 | 数据 |
|------|------|
| DMM小鼠滑膜 | RANK+和RANKL+细胞比例从40%→65%（8周） |
| 人类OA关节液 | RANKL显著高于健康对照 |
| OA滑膜组织 | RANK和RANKL表达显著增加，滑膜>软骨 |
| IL-1β诱导细胞 | FLS中RANK/RANKL表达最高 > 巨噬细胞 > 软骨细胞 |
| ColVI共定位 | 65% FLS表达RANKL，55% FLS表达RANK |

### 2.2 多物种治疗效果

| 模型 | 物种 | 效果 |
|------|------|------|
| DMM（创伤后OA） | 小鼠 | OARSI评分↓28-30%，滑膜炎↓，骨赘↓，疼痛阈值↑ |
| MIA（炎症性OA） | 大鼠 | 软骨降解↓，滑膜炎↓，疼痛阈值↑ |
| 老年OA | 小鼠（18月） | 软骨退化↓，滑膜增生↓ |
| ACLT | Beagle犬 | 软骨保护，滑膜炎↓，MRI滑膜炎评分↓ |
| 临床试验 | 人（单臂） | OKS/VAS/WOMAC改善，MRI滑膜炎↓ |

### 2.3 单细胞测序核心发现
- **14个细胞群**：FLS、软骨细胞、7种免疫细胞、HSCs、内皮细胞、周细胞、施万细胞、红细胞
- **FLS比例变化**：DMM后FLS增加最显著 → Denosumab后FLS减少最显著
- **FLS是通讯"hub"**：DMM时FLS→软骨细胞和巨噬细胞的通讯显著增加 → Denosumab抑制
- **FLS分泌因子**：TNF、IL-6、CCL、CSF、MIF（ outgoing 信号）
- **软骨细胞是主要接收者**：接收FLS的炎症信号

### 2.4 FSTL1 — 关键下游效应分子

**发现过程**：
1. scRNA-seq FLS差异基因：FSTL1是表达最显著上调的基因
2. FSTL1与OA严重程度相关（KL评分、WOMAC评分）
3. FSTL1促进FLS分泌：TNF-α、IL-1β、MMP-13、ADAMTS5
4. FSTL1促进软骨细胞：Col10a1、MMP-13、ADAMTS5（分解代谢）
5. FSTL1促进巨噬细胞：迁移能力↑

### 2.5 分子机制：RANK/TRAF6/FSTL1/NF-κB轴

```
RANKL（由炎症因子诱导FLS分泌）
    ↓
结合RANK受体（FLS表面）
    ↓
招募TRAF6 → 形成RANK-TRAF6复合物
    ↓
激活IκBα磷酸化 → p50/p65核转位
    ↓
NF-κB活化 → FSTL1转录上调
    ↓
FSTL1分泌 → 
    ├── TNF-α/IL-1β/MMP-13/ADAMTS5（FLS自分泌/旁分泌）
    ├── Col10a1/MMP-13/ADAMTS5（软骨细胞分解代谢）
    └── 巨噬细胞迁移/M1极化
    ↓
滑膜炎 + 软骨降解 + 骨赘形成 → OA进展
```

**Denosumab干预点**：阻断RANKL-RANK结合 → TRAF6无法招募 → NF-κB不激活 → FSTL1↓

### 2.6 机制验证实验

| 实验 | 结果 |
|------|------|
| Co-IP | TRAF6与RANK在RANKL刺激后结合 → Denosumab阻断 |
| TRAF6 siRNA | 抑制FSTL1和p-IκBα |
| NF-κB抑制剂 (BAY11-7082) | 抑制FSTL1和p50/p65核转位 |
| FSTL1 siRNA | 阻断RANKL诱导的MMP/ADAMTS/TNF-α |
| 条件培养基实验 | RANKL-CM→软骨细胞分解代谢↑ + 巨噬细胞M1极化↑ → Denosumab-CM逆转 |
| 关节腔注射RANKL | 诱导滑膜炎+早期OA → Denosumab关节腔注射逆转 |

---

## 3. 临床数据

### 3.1 单臂临床试验设计
- **入组**: 退行性膝OA患者
- **给药**: Denosumab 60mg皮下注射（单次）
- **随访**: 6个月
- **主要终点**: VAS疼痛评分 + OKS膝关节评分
- **次要终点**: WOMAC评分 + 不良事件

### 3.2 临床结果
| 指标 | 变化 |
|------|------|
| VAS | 逐渐下降，6月时显著改善 |
| OKS | 逐渐改善 |
| WOMAC | 逐渐改善 |
| MRI滑膜炎 | 1月轻度改善，2月显著改善 |
| Hoffa-synovitis评分 | 显著下降 |
| Effusion-synovitis评分 | 显著下降 |

### 3.3 患者滑膜组织学（术前3周给药）
- 对照OA滑膜：衬里细胞层明显增厚，绒毛增生
- Denosumab滑膜：衬里细胞层轻度增厚
- RANKL、FSTL1、IL-1β、TNF-α、MMP-13、ADAMTS5表达均显著降低

---

## 4. 与Project-016 CHIP-TMJOA的对接

### 4.1 直接相关性

| CHIP-TMJOA元素 | Hu 2025机制 |
|---------------|------------|
| CHIP→促炎巨噬细胞/中性粒细胞 | = RANKL→FLS→TNF/IL-6/CCL分泌 |
| CHIP→IL-17/IL-20→破骨细胞↑ | = RANKL→RANK→破骨细胞（经典） |
| 更年期→FBM↑→炎症↑ | = 滑膜FLS活化→RANKL分泌↑ |
| 髁突软骨下骨丢失 | = Denosumab保护软骨下骨（BV/TV↑） |
| 滑膜炎 | = **新发现：Denosumab直接抑制滑膜炎症** |

### 4.2 TMJOA特异性外推

**关键论证点**:
1. TMJOA同样存在滑膜炎（synovitis）——这是OA的共同早期事件
2. TMJ的FLS同样表达RANK和RANKL——人体滑膜的共性
3. Denosumab在Beagle犬（大动物）有效→人体TMJOA可能有效
4. 作者单位：**华中科技大学同济医学院附属协和医院骨科**——与你的单位一致！

### 4.3 对IIT伦理的强化

**原论证**: Denosumab是骨质疏松药，外推到TMJOA需要假设
**新论证**: 
- Hu 2025证明Denosumab直接靶向滑膜炎症（FLS）
- 滑膜炎是OA共同病理基础，不分膝/髋/肩/颞下颌
- 人体临床单臂试验已验证滑膜炎减少（MRI + 组织学）
- Denosumab在Beagle犬（更接近人类TMJ大小/负荷）验证有效

---

## 5. 知识缺口与假设

| 缺口 | 假设 |
|------|------|
| TMJ滑膜FLS是否表达RANKL？ | **假设：是**——所有滑膜FLS均有此特性 |
| Denosumab能否穿透TMJ滑膜？ | **假设：能**——Denosumab系统性分布已在滑膜证实 |
| TMJOA患者滑膜炎MRI评分？ | 需建立TMJ专用MRI滑膜炎评分系统 |
| CHIP-TMJOA患者的RANKL水平？ | **待测**：外周血/滑液RANKL是否与CHIP克隆大小相关？ |
| FSTL1在TMJOA滑液中的水平？ | **待测**：可能与OA严重程度相关 |

---

## 6. 精读结论

### 这篇论文对Project-016的价值

1. **机制升级**: Denosumab从"骨代谢药物"升级为"**滑膜抗炎药物**"——直接靶向FLS
2. **伦理升级**: IIT论证中"超适应症"程度大幅降低——已有人体滑膜炎数据
3. **靶点发现**: FSTL1是新的OA治疗靶点，可能在CHIP-TMJOA中同样上调
4. **跨域连接**: 
   - CHIP→异常训练免疫→促炎FLS→RANKL↑→FSTL1↑→滑膜炎
   - Denosumab阻断这一链条的多个节点
5. **临床路径**: 单臂试验设计可直接复制到TMJOA（MRI滑膜炎 + 疼痛评分 + 功能评分）

### 对综述写作的直接影响

综述应新增一节：
> "**Targeting Synovial Inflammation in CHIP-Driven TMJOA: Beyond Bone Metabolism**"
> 
> 核心论点：CHIP驱动的系统性炎症不仅导致软骨下骨丢失，还通过RANKL/TRAF6/FSTL1轴激活TMJ滑膜FLS，产生滑膜炎。Denosumab作为RANKL抑制剂，可双靶点阻断骨代谢异常和滑膜炎症，是CHIP-TMJOA的潜在精准治疗。

---

*精读: YouYou | 2026-05-16 15:30 CST*
*来源: Hu Y et al. Nat Commun 2025 | PMID 41407913 | 全文18页*
