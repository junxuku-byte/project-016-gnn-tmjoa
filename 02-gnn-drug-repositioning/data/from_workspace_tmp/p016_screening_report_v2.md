# Project-016 GNN药物重定位 — 文献筛选报告 (v2)

**筛选时间**: 2026-05-18T14:34:45.272646
**原始文献**: 2991 篇
**核心文献**: 400 篇
**筛选阈值**: min_score=3, max_papers=400
**筛选版本**: v2 (CHIP-free, TMJOA+骨代谢+药物靶点导向)

## 纳入/排除标准

### 排除类型
- News, Case Reports, Comment, Editorial, Published Erratum, Biography, Letter

### 评分规则
- **Tier 1 (+3/词)**: TMJOA核心机制、骨代谢异常、药物靶点
  - TMJOA: temporomandibular, tmj, tmjoa, condylar, osteoarthritis, chondrocyte apoptosis, synovial inflammation, subchondral bone resorption...
  - 骨代谢: vitamin d deficiency, osteoporosis, osteomalacia, bone marrow fat, ckd-mbd, hyperparathyroidism, diabetic bone disease...
  - 药物靶点: drug repositioning, fgf23, rankl, denosumab, wnt, mmp13, il-1β, il-38, nf-κb, metformin, rapamycin...
- **Tier 2 (+2/词)**: 相关骨病、炎症通路、细胞类型
  - bone remodeling, osteoclast, rankl, cartilage, menopause, estrogen, cytokine, il-6, il-17, macrophage, nf-κb...
- **Tier 3 (+1/词)**: 相关机制、影像、干细胞、生物信息学方法
  - adipogenesis, autophagy, senescence, msc, exosome, glp-1, nsaid, mri, cbct, radiomics, machine learning, gnn...
- **高价值类型**: RCT(+2), Meta(+2), SysRev(+2), Review(+1), JA(+1)
- **年份**: ≥2020(+2), ≥2015(+1), <2000(-2)

### 特殊保护
- **TMJOA相关文献**（分数≥1保留）— 项目核心
- **骨代谢异常文献**（分数≥1保留）— 共病筛查靶点
- **药物靶点文献**（分数≥2保留）— GNN预测候选

## 统计摘要

| 指标 | 数值 |
|------|------|
| TMJOA 相关 | 275 |
| 骨代谢相关 | 38 |
| 药物靶点 | 143 |
| TMJOA ∩ 骨代谢 | 4 |
| TMJOA ∩ 药物靶点 | 40 |
| 骨代谢 ∩ 药物靶点 | 12 |
| 三者交集 | 0 |

### 年份分布

- 2010-2014: 22 (5.5%)
- 2015-2019: 74 (18.5%)
- 2020+: 292 (73.0%)
- <2010: 12 (3.0%)

### 分数分布

- 10+: 400 (100.0%)

### 文章类型

- Journal Article: 347
- Review: 22
- Randomized Controlled Trial: 17
- Systematic Review: 11
- Meta-Analysis: 2
- Clinical Trial: 1

## Top 20 核心文献

1. **PMID 41086521** (score=27 [BONE])
   - Reduction of subchondral bone marrow lesions concurrent with early recovery of trabecular bone mineral density in knee o...
   - Sun C et al., Seminars in arthritis and rheumatism, 2025
   - T1(4): osteoarthritis, rankl, bone mineral density | T2(6): knee osteoarthritis, ros, trabecular bone | TYPE(Journal Article): +1
2. **PMID 37625369** (score=27 [TMJ])
   - Pyroptosis of chondrocytes activated by synovial inflammation accelerates TMJ osteoarthritis cartilage degeneration via ...
   - Liu X et al., International immunopharmacology, 2023
   - T1(4): osteoarthritis, cartilage degeneration, tmj | T2(6): nlrp3, ros, inflammation | TYPE(Journal Article): +1
3. **PMID 36799502** (score=27 [DRUG])
   - IL-6, IL-1β and TNF-α regulation of the chondrocyte phenotype: a possible mechanism of haemophilic cartilage destruction...
   - Zheng L et al., Hematology (Amsterdam, Netherlands), 2023
   - T1(4): il-1β, cartilage destruction, il-6 | T2(6): tnf, cartilage, il-1 | TYPE(Journal Article): +1
4. **PMID 21035559** (score=26 [TMJ+DRUG])
   - Requirement of the NF-κB pathway for induction of Wnt-5A by interleukin-1β in condylar chondrocytes of the temporomandib...
   - Ge XP et al., Osteoarthritis and cartilage, 2011
   - T1(5): osteoarthritis, nf-κb, temporomandibular | T2(5): ros, nf-κb, interleukin | TYPE(Journal Article): +1
5. **PMID 36642020** (score=25 [DRUG])
   - PD0325901, an ERK inhibitor, attenuates RANKL-induced osteoclast formation and mitigates cartilage inflammation by inhib...
   - Jiang T et al., Bioorganic chemistry, 2023
   - T1(2): nf-κb, rankl | T2(8): mapk, nf-κb, inflammation | TYPE(Journal Article): +1
6. **PMID 33967763** (score=25 [DRUG])
   - Gamabufotalin Inhibits Osteoclastgenesis and Counteracts Estrogen-Deficient Bone Loss in Mice by Suppressing RANKL-Induc...
   - Sun K et al., Frontiers in pharmacology, 2021
   - T1(2): nf-κb, rankl | T2(8): mapk, nf-κb, estrogen | TYPE(Journal Article): +1
7. **PMID 29755657** (score=25 [DRUG])
   - Regulatory effect of anti-gp130 functional mAb on IL-6 mediated RANKL and Wnt5a expression through JAK-STAT3 signaling p...
   - Miao P et al., Oncotarget, 2018
   - T1(5): jak, wnt, rankl | T2(4): jak-stat, rankl, il-6 | TYPE(Journal Article): +1
8. **PMID 34515572** (score=24 [TMJ+DRUG])
   - Regulating Fibrocartilage Stem Cells via TNF-α/Nf-κB in TMJ Osteoarthritis....
   - Bi R et al., Journal of dental research, 2022
   - T1(4): osteoarthritis, nf-κb, tmj | T2(4): nf-κb, tnf, cartilage | T3(1): fibrocartilage stem cell
9. **PMID 34307684** (score=24 [TMJ+DRUG])
   - The Antioxidant Resveratrol Protects against Chondrocyte Apoptosis by Regulating the COX-2/NF-κB Pathway in Created Temp...
   - Li W et al., BioMed research international, 2021
   - T1(5): osteoarthritis, chondrocyte apoptosis, nf-κb | T2(3): nf-κb, apoptosis, chondrocyte | TYPE(Journal Article): +1
10. **PMID 29746861** (score=24 [DRUG])
   - Lenalidomide regulates osteocytes fate and related osteoclastogenesis via IL-1β/NF-κB/RANKL signaling....
   - Qu X et al., Biochemical and biophysical research com, 2018
   - T1(3): il-1β, nf-κb, rankl | T2(6): nf-κb, osteoclast, rankl | T3(1): fat
11. **PMID 42001864** (score=23 [TMJ+DRUG])
   - Recombinant IL-38 Alleviates Temporomandibular Joint Synovial Inflammation via IL-1R1-NF-κB-IL1β Pathway....
   - Luo P et al., International dental journal, 2026
   - T1(4): nf-κb, temporomandibular, synovial inflammation | T2(4): nf-κb, inflammation, il-1 | TYPE(Journal Article): +1
12. **PMID 38031141** (score=23 [TMJ])
   - Resatorvid alleviates experimental inflammatory TMJOA by restraining chondrocyte pyroptosis and synovial inflammation....
   - Liu X et al., Arthritis research & therapy, 2023
   - T1(4): tmjoa, chondrocyte pyroptosis, tmj | T2(4): inflammation, inflammatory, pyroptosis | TYPE(Journal Article): +1
13. **PMID 33039343** (score=23 [TMJ+DRUG])
   - Effects of Intra-Articular Resveratrol Injections on Cartilage Destruction and Synovial Inflammation in Experimental Tem...
   - Yuce P et al., Journal of oral and maxillofacial surger, 2021
   - T1(5): osteoarthritis, cartilage destruction, temporomandibular | T2(2): inflammation, cartilage | T3(1): maxillofacial
14. **PMID 30585456** (score=23 [BONE])
   - [Electroacupuncture Intervention Improves Cartilage Degeneration and Subchondral Bone Osteoporosis of Knee-joint Possibl...
   - Sun GH et al., Zhen ci yan jiu = Acupuncture research, 2018
   - T1(3): cartilage degeneration, rankl, osteoporosis | T2(6): ros, cartilage, rankl | TYPE(Journal Article): +1
15. **PMID 41123324** (score=22 [DRUG])
   - Mechanical stress promotes synovial inflammation and osteoarthritis development via the NF-κB p52/IL-6 signalling pathwa...
   - Shen Y et al., Rheumatology (Oxford, England), 2026
   - T1(4): osteoarthritis, nf-κb, synovial inflammation | T2(3): nf-κb, inflammation, il-6 | T3(1): stress
16. **PMID 40699912** (score=22 [DRUG])
   - TNF-α Promotes Synovial Inflammation and Cartilage Bone Destruction in Rheumatoid Arthritis via NF-κB/YY1/miR-103a-3p Ax...
   - Yuan Y et al., FASEB journal : official publication of , 2025
   - T1(3): nf-κb, synovial inflammation, tnf-α | T2(5): nf-κb, inflammation, tnf | TYPE(Journal Article): +1
17. **PMID 36738173** (score=22 [TMJ+DRUG])
   - Integrated Cascade Nanozyme Remodels Chondrocyte Inflammatory Microenvironment in Temporomandibular Joint Osteoarthritis...
   - Zhang Z et al., Advanced healthcare materials, 2023
   - T1(3): osteoarthritis, nf-κb, temporomandibular | T2(5): mapk, ros, nf-κb | TYPE(Journal Article): +1
18. **PMID 35513247** (score=22 [TMJ])
   - Chondrocyte apoptosis in temporomandibular joint osteoarthritis promotes bone resorption by enhancing chemotaxis of oste...
   - Guo YN et al., Osteoarthritis and cartilage, 2022
   - T1(3): osteoarthritis, chondrocyte apoptosis, temporomandibular | T2(5): apoptosis, bone resorption, cartilage | TYPE(Journal Article): +1
19. **PMID 34914744** (score=22 [DRUG])
   - Metformin attenuates osteoclast-mediated abnormal subchondral bone remodeling and alleviates osteoarthritis via AMPK/NF-...
   - Guo H et al., PloS one, 2021
   - T1(3): osteoarthritis, nf-κb, metformin | T2(5): bone remodeling, nf-κb, osteoclast | TYPE(Journal Article): +1
20. **PMID 41725564** (score=21 [TMJ+DRUG])
   - Cyanidin-3-O-glucoside chloride inhibits cartilage degeneration and inflammation in TMJOA via YAP/NF-κB signaling pathwa...
   - Li D et al., Food & function, 2026
   - T1(4): cartilage degeneration, tmjoa, nf-κb | T2(3): nf-κb, inflammation, cartilage | TYPE(Journal Article): +1

---

## 分维度核心文献

### TMJOA 机制 Top 5

1. PMID 37625369: Pyroptosis of chondrocytes activated by synovial inflammation accelerates TMJ osteoarthritis cartila... (2023)
2. PMID 21035559: Requirement of the NF-κB pathway for induction of Wnt-5A by interleukin-1β in condylar chondrocytes ... (2011)
3. PMID 34515572: Regulating Fibrocartilage Stem Cells via TNF-α/Nf-κB in TMJ Osteoarthritis.... (2022)
4. PMID 34307684: The Antioxidant Resveratrol Protects against Chondrocyte Apoptosis by Regulating the COX-2/NF-κB Pat... (2021)
5. PMID 42001864: Recombinant IL-38 Alleviates Temporomandibular Joint Synovial Inflammation via IL-1R1-NF-κB-IL1β Pat... (2026)

### 骨代谢异常 Top 5

1. PMID 41086521: Reduction of subchondral bone marrow lesions concurrent with early recovery of trabecular bone miner... (2025)
2. PMID 30585456: [Electroacupuncture Intervention Improves Cartilage Degeneration and Subchondral Bone Osteoporosis o... (2018)
3. PMID 35608639: Effects of denosumab treatment on the expression of receptor activator of nuclear kappa-B ligand (RA... (2022)
4. PMID 40896957: Human sclerostin-inspired short peptides reverse osteoporosis and suppress joint degeneration in ost... (2025)
5. PMID 38582351: Correlations with clinical and radiologic findings and prevalence of osteopenia/osteoporosis in the ... (2024)

### 药物靶点 Top 5

1. PMID 36799502: IL-6, IL-1β and TNF-α regulation of the chondrocyte phenotype: a possible mechanism of haemophilic c... (2023)
2. PMID 21035559: Requirement of the NF-κB pathway for induction of Wnt-5A by interleukin-1β in condylar chondrocytes ... (2011)
3. PMID 36642020: PD0325901, an ERK inhibitor, attenuates RANKL-induced osteoclast formation and mitigates cartilage i... (2023)
4. PMID 33967763: Gamabufotalin Inhibits Osteoclastgenesis and Counteracts Estrogen-Deficient Bone Loss in Mice by Sup... (2021)
5. PMID 29755657: Regulatory effect of anti-gp130 functional mAb on IL-6 mediated RANKL and Wnt5a expression through J... (2018)

---

## 对GNN建模的启示

### 潜在药物靶点节点（从文献高频提及提取）
- **炎症通路**: IL-1β, IL-6, IL-17, TNF-α, NF-κB → 现有抗炎药可重定位
- **基质降解**: MMP13, ADAMTS5 → MMP抑制剂
- **Wnt信号**: Wnt/β-catenin, Sclerostin, DKK1 → 骨形成促进剂
- **FGF23**: 骨代谢-软骨降解桥梁 → 抗FGF23抗体 (Burosumab)
- **代谢调节**: Metformin (AMPK/mTOR), Rapamycin → 抗衰老/抗炎
- **天然化合物**: Resveratrol, Curcumin, Quercetin → 多靶点抗炎

### 负样本设计建议
1. **随机负样本**: 随机 drug-disease 对（已验证无效）
2. **拓扑距离负样本**: 知识图谱中距离 >3  hops 的 drug-disease 对
3. **时间分割**: 2015年前已上市 drug + 2015年后发现 disease，验证是否被后续研究支持

### 网络构建思路
- **疾病层**: TMJOA 表型 → 软骨退化 / 滑膜炎 / 骨吸收 子类型
- **机制层**: 炎症通路 (IL-1β→NF-κB→MMP13) + 骨代谢 (RANKL→破骨细胞) + 代谢 (mTOR→自噬)
- **药物层**: 已知OA药 (NSAIDs, HA) + 骨代谢药 (Denosumab, Bisphosphonates) + 重定位候选 (Metformin, Rapamycin)
- **连接**: 文献共现 + 通路共享 + 靶点互作
