# Project-016 PDF文献筛选报告

**日期**: 2026-05-18
**扫描目录**: 
- ~/morph-lab/projects/project-016-gnn-drug-repositioning/00-文献/ (52篇)
- ~/.openclaw/workspace/.tmp/p016_fulltext_pdfs/ (18篇，部分重复)
**总计**: ~55篇唯一PDF

---

## 一、A类：TMJOA药物重定位直接相关 ⭐⭐⭐⭐⭐

| 文件名 | 标题/主题 | 状态 | 精读优先级 |
|--------|----------|------|-----------|
| **Riluzole_OA_Repositioning_2026.pdf** | Riluzole谷氨酸抑制剂OA重定位 | ✅ 已识别 | **HIGH** |
| **Osteocyte_MMP13_TMJ_2026.pdf** | 骨细胞MMP13与TMJOA（2026新发表） | ✅ 已识别 | **HIGH** |
| **can treatment with chondroitin and g...** | Chondroitin+Glucosamine预防TMD | ✅ 已识别 | **HIGH** (PMID 39092654) |
| **efficiency of glucosamine in treatin...** | Glucosamine治疗TMJOA效率 | ✅ 已识别 | **HIGH** (PMID 38867545) |
| **recent advances in animal models di...** | TMJOA动物模型最新进展 | ✅ 已识别 | **HIGH** (PMID 35994388) |
| **fmoc ddt fos hydrogel mitigates temp...** | FMOC-DDT-FOS水凝胶缓解TMJ退化 | ✅ 已识别 | **MEDIUM** (Materials Today Bio 2026) |
| **Denosumab_OA_Risk_2026.pdf** | Denosumab与OA风险 | ✅ 已识别 | **MEDIUM** |
| **association between denosumab use an...** | Denosumab使用与OA关联 | ✅ 已识别 | **MEDIUM** |
| **metformin use and the risk of incide...** | Metformin使用与OA发病率 | ✅ 已识别 | **MEDIUM** (cloudlib_41991265 = PMID?) |
| **the use of statins can reduce the ri...** | Statins降低OA风险 | ✅ 已识别 | **MEDIUM** |
| **disease modifying interactions betwe...** | CKD-OA疾病修饰相互作用 | ✅ 已识别 | **MEDIUM** |
| **Bliddal_STEP9_NEJM_2024.pdf** | STEP9试验（Semaglutide减重对OA？） | ✅ 已识别 | **MEDIUM** |
| **therapeutic subtypes of knee osteoar...** | Knee OA治疗亚型 | ✅ 已识别 | **LOW** (Knee OA非TMJ) |
| **Koh_FrontBioeng_2025_craniofacial_bone.pdf** | 颅面骨工程 | ✅ 已识别 | **LOW** (机制研究) |

---

## 二、B类：OA/软骨机制研究（可用于特征工程）⭐⭐⭐⭐

| 文件名 | 主题 | 用途 |
|--------|------|------|
| **FGF23_Wnt_OA_2018_Mouse_Mechanism.pdf** | FGF23-Wnt/β-catenin-OA小鼠机制 | 药物靶点网络构建 |
| **FGFR_Inhibitor_Rescues_OA_2020.pdf** | FGFR抑制剂拯救OA | 药物靶点网络构建 |
| **osteocyte intrinsic mmp13 exacerbate... (x2)** | 骨细胞MMP13加剧OA | 机制特征 |
| **sgk1 triggers cartilage degradation...** | SGK1触发软骨降解 | 机制特征 |
| **fgf23 regulates wnt catenin signal...** | FGF23调控Wnt信号 | 机制特征 |
| **fgf receptor inhibitor bgj398 partia...** | FGFR抑制剂BGJ398 | 药物靶点 |
| **inhibition of fgfr signaling partially...** | FGFR信号抑制拯救OA | 药物靶点 |
| **single cell transcriptomics reveals...** | 单细胞转录组揭示OA | 特征工程参考 |
| **Embree_NatComm_2016_FCSC.pdf** | 纤维软骨干细胞（FCSC） | 再生医学 |

---

## 三、C类：CHIP遗留文献（建议归档）🔴

| 文件名 | CHIP相关主题 |
|--------|-------------|
| Jaiswal_NEJM_2017_CHIP_CVD.pdf | CHIP与心血管疾病 |
| Jaiswal_Science_2019_CHIP_review.pdf | CHIP综述 |
| Hiitola_SciAdv_2025_CHIP_RA.pdf | CHIP与类风湿关节炎 |
| Li_GeroScience_2025_CHIP_OA.pdf | CHIP与骨关节炎 |
| Wang_Cell_2024_CHIP_bone_loss.pdf | CHIP与骨丢失 |
| Kim_JEM_2021_DNMT3A_osteoporosis.pdf | DNMT3A与骨质疏松 |
| Zioni_NatComm_2023_FBM_DNMT3A.pdf | DNMT3A与骨髓 |

**建议**: 移至 `00-文献/_archive/chip-legacy/`

---

## 四、D类：GNN/方法学论文（模型设计参考）🔧

| 文件名 | 主题 |
|--------|------|
| a physics informed graph neural netw... | 物理信息图神经网络 |
| a case based explainable graph neura... | 可解释图神经网络案例 |
| application of knowledge graphs in r .pdf | 知识图在药物重定位应用 |
| node2vec_baseline.pdf | Node2Vec基线 |
| **Chen_JAdvRes_2025_trained_immunity...** | 训练免疫调节剂（可能跨项目） |

---

## 五、E类：CloudLib未知文献（需进一步识别）❓

约21篇 `cloudlib_*.pdf` 文件名无意义，仅3篇提取到DOI：

| 文件名 | DOI | 期刊 | 识别状态 |
|--------|-----|------|---------|
| cloudlib_40707728.pdf | 10.1007/s10787-025-01864- | Inflammopharmacology | 🟡 可能是statins/抗炎药 |
| cloudlib_41407913.pdf | 10.1038/s41467-025-66202-z | Nature Communications | 🟡 未识别主题 |
| cloudlib_41470005.pdf | 10.1186/s13075-025-03690- | Arthritis Res Ther 2025 | 🟡 Tang et al. 未识别 |
| cloudlib_41716349.pdf | 10.1016/j.mtbio.2026.102906 | Materials Today Bio | ✅ FMOC水凝胶（已知） |
| cloudlib_41991265.pdf | 10.1136/bmjopen-2025-1155 | BMJ Open | ✅ Metformin与OA风险 |
| 其余16篇 | N/A | N/A | 🔴 未识别，需逐篇打开 |

---

## 六、F类：.tmp目录今日下载（33篇精读候选）

| 文件名 | PMID | 状态 |
|--------|------|------|
| PMID_40273050_PLoS.pdf | 40273050 | ✅ **已精读** — FGF18强阳性 |
| PMID_36890529_BMC.pdf | 36890529 | ✅ **已精读** — Arthrocentesis±Tenoxicam |
| PMID29244893.pdf / PMID_29244893_GoogleScholar.pdf | 29244893 | ✅ **已精读** — HA vs CS Meta |
| PMID28879245.pdf / PMID_28879245_GoogleScholar.pdf | 28879245 | ✅ **已精读** — Capsaicin TMD RCT |
| PMID_30814387_Unpaywall.pdf | 30814387 | ✅ **已精读** — Arthrocentesis±HA |
| PMID_37608244_OA.pdf / PMID_37608244_GoogleScholar.pdf | 37608244 | ⚠️ **内容错误** — 下载了错误PDF |
| PMID_39092654_GoogleScholar.pdf | 39092654 | ⚠️ **未精读** — 文件名匹配，需验证内容 |
| PMID_38821656_GoogleScholar.pdf | 38821656 | ⚠️ **未精读** — Google Scholar可能获取了错误PDF |
| PMID_41084405_GoogleScholar.pdf | 41084405 | ⚠️ **未精读** — 同上 |
| DOI_10.1007_s12663-011-0... | 无PMID | 🟡 未精读 — HA vs CS注射 |
| DOI_10.1186_2045-709x-22... | 无PMID | 🟡 未精读 — 手法治疗 |
| DOI_10.1186_s12903-023-0... | 无PMID | 🟡 未精读 — 已确认为36890529内容 |
| DOI_10.1371_journal.pone... | 无PMID | 🟡 未精读 — MSC-peptide复合物 |
| DOI_10.2334_josnusd.17-0... | 无PMID | 🟡 已精读 — 30814387 |
| DOI_10.32388_5u9vf2... | 无PMID | 🟡 未精读 — RAIN方法学 |

---

## 七、精读优先级总表

### 🔴 立即精读（影响训练集标签）
1. **Riluzole_OA_Repositioning_2026.pdf** — 谷氨酸抑制剂OA重定位，可能直接用于负样本/正样本判断
2. **Osteocyte_MMP13_TMJ_2026.pdf** — TMJ特异性机制，2026新文献
3. **can treatment with chondroitin and glucosamine** (PMID 39092654) — 已下载但内容待验证
4. **efficiency of glucosamine in treatin** (PMID 38867545) — 已下载但内容待验证
5. **recent advances in animal models di** (PMID 35994388) — 已下载但内容待验证

### 🟡 本周精读（补充TMJ核心正样本）
6. fmoc ddt fos hydrogel mitigates temp — 水凝胶治疗TMJ
7. Denosumab_OA_Risk_2026 — 抗RANKL与OA
8. metformin use and the risk of incide — Metformin与OA
9. the use of statins can reduce the ri — Statins与OA
10. Bliddal_STEP9_NEJM_2024 — 减重对OA影响

### 🟢 已精读/归档/跳过
- ✅ FGF18, Capsaicin, Arthrocentesis±HA, Arthrocentesis±Tenoxicam, HA vs CS Meta
- 🔴 CHIP遗留7篇 → 归档
- 🔧 GNN方法学4篇 → 保留参考
- ❓ CloudLib 16篇未识别 → 需批量提取

---

## 八、建议行动

1. **立即执行**: 精读A类5篇（Riluzole, MMP13-TMJ, Chondroitin+Glucosamine, Glucosamine效率, Animal models）
2. **归档**: 将C类7篇CHIP文献移至 `_archive/chip-legacy/`
3. **批量识别**: 对E类16篇CloudLib用pdf_library.py批量提取DOI+标题
4. **清理重复**: .tmp目录18篇中约6篇是重复/错误下载，可删除

---

*报告生成: 2026-05-18 23:10*
*分析师: 油油*
