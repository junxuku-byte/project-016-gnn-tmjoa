# Project-016 CHIP-TMJOA：Phase 1 完成状态

## 📄 PDF 获取情况（11篇核心文献）

### ✅ 真PDF已持有（6篇）
| 文件 | 大小 | 来源 |
|------|------|------|
| Wang_Cell_2024_CHIP_bone_loss.pdf | 380KB (2页预览) | 云图书馆 |
| Kim_JEM_2021_DNMT3A_osteoporosis.pdf | **5.4MB (23页全)** | 云图书馆 |
| Zioni_NatComm_2023_FBM_DNMT3A.pdf | **7.3MB (全)** | Nature OA |
| Embree_NatComm_2016_FCSC.pdf | **2.7MB (26页全)** | Nature OA |
| Koh_FrontBioeng_2025_craniofacial_bone.pdf | **3.4MB (全)** | Frontiers OA |
| Bliddal_STEP9_NEJM_2024.pdf | 359KB (2页预览) | 云图书馆 |

### ✅ 文本内容已提取（无PDF但已理解）
| 文献 | 来源 | 关键数据 |
|------|------|---------|
| Li GeroScience 2025 | PMC全文 | HR 1.46, 7个介质蛋白 |
| Jaiswal Science 2019 | PMC全文 | CHIP基础框架 |
| Jaiswal NEJM 2017 | PMC全文 | CHIP→CVD奠基 |
| Hiitola Sci Adv 2025 | Pubmed摘要 | CHIP→RA OR 1.49-2.06 |

### ❌ 未获取（信息量少，已确认可补）
| 文献 | 原因 | 策略 |
|------|------|------|
| Koh Annu Rev Pathol 2025 | 无PMCID | 云图书馆后补 |
| 其余1篇 | OA有摘要已足够 | 不阻塞 |

## 🧠 知识合成已完成
- ✅ `01-分析项目/CHIP-TMJOA_知识合成报告.md` — 8大章节含综合假说图
- ✅ 证据链验证：更年期→FBM↑→DNMT3A→CHIP→IL-17/IL-20→破骨细胞→TMJOA
- ✅ TMJ领域零文献空白确认

## 🔄 LabKG 后台运行中
- ✅ `build_tmj_kg.py` 正在以 glm-5.1 抽取6篇PDF的三元组
- ✅ `--import-to-labkg` 将直接汇入LabKG知识网络
- ⏳ 预计运行时间：5-10分钟

## 🏗️ 项目结构
```
project-016-chip-tmjoa/
├── PROJECT_PLAN.md          # 完整立项计划
├── 立项确认报告.md          # 确认报告
├── project-state.yaml       # 状态文件
├── 00-文献/
│   ├── 6篇PDF
│   ├── kg_output/           # LabKG产出（运行中）
│   └── pdf_backlog/         # 回补清单
└── 01-分析项目/
    └── CHIP-TMJOA_知识合成报告.md  # 核心产出

---

# Phase 1.5 更新（2026-05-16 19:00 CST）

## 新增完成项

| # | 里程碑 | 说明 |
|---|--------|------|
| 7 | **Chen 2025 J Adv Res 精读** | 训练免疫调节剂，H3K18la/IL-37/SCFA靶点识别 |
| 8 | **Hu 2025 Nat Commun 精读** | Denosumab/FSTL1滑膜炎机制，Denosumab IIT可行性⭐⭐⭐⭐ |
| 9 | **朝堂议政：靶点辩论** | 首席调查官模式，10候选靶点+药物反推 |
| 10 | **CKD-MBD→OA深度学习** | FGF23-Wnt轴桥接，2篇核心文献精读+LabKG导入 |
| 11 | **常规骨代谢网络导入** | 70节点+106边+7桥接（Vit D↓/骨质疏松/甲旁亢/甲亢/铁过载等） |
| 12 | **临床前方案设计** | 动物为主+人体单臂验证路径确定 |

## PDF新增（今日获取）

| 文件 | 大小 | 来源 | 说明 |
|------|------|------|------|
| Chen_JAdvRes_2025_trained_immunity_modulators.pdf | **3.3MB** | 云图书馆 | PMID 40975126 |
| Hu_NatCommun_2025_Denosumab_FSTL1.pdf | **7.1MB** | Nature OA | PMID 41407913 |
| FGF23_Wnt_OA_2018_Mouse_Mechanism.pdf | **18.4MB** | 云图书馆 | PMID 29718273 |
| CKD_OA_Interactions_2023_Review.pdf | **2.9MB** | 云图书馆 | PMID 37562858 |
| FGFR_Inhibitor_Rescues_OA_2020.pdf | **2.1MB** | 云图书馆 | PMID 31901095 |
| Osteocyte_MMP13_TMJ_2026.pdf | **1.1MB** | 云图书馆 | PMID 41461362 |

## LabKG 当前状态

| 指标 | 数值 |
|------|------|
| **总节点** | **10,014** |
| **总边** | **14,442** |
| **Project-016节点** | **2,629** |
| **Project-016论文** | **31篇** |
| **骨代谢概念** | **84个** |

## 关键决策

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-05-16 | 临床路径翻转为"动物为主+人体单臂" | 动物建立TMJ特异性因果链，人体验证信号，成本<<10万 |
| 2026-05-16 | Denosumab列为首选候选药物 | Hu 2025直接滑膜炎↓证据+已有CKD骨质疏松适应症 |
| 2026-05-16 | FGF23纳入TMJOA骨代谢筛查套餐 | CKD早期FGF23↑早于血磷异常 |
| 2026-05-16 | 锚固术后6月排除骨吸收抑制剂 | 下颌骨坏死风险 |

## 新增文档

| 文档 | 路径 |
|------|------|
| CKD-MBD OA深度学习报告 | `01-分析项目/CKD-MBD_OA_深度文献学习.md` |
| CKD-MBD OA学习摘要 | `01-分析项目/CKD-MBD_OA_学习摘要.md` |
| FGF23 Wnt OA精读 | `01-分析项目/精读_FGF23_Wnt_OA_2018_+_CKD_OA_2023.md` |
| 动物+人体方案设计 | `01-分析项目/动物为主_人体单臂_设计方案.md` |
| 靶点辩论简报 | `01-分析项目/CHIP-TMJOA_靶点辩论简报.md` |
| Chen精读 | `01-分析项目/精读_Chen_JAdvRes_2025_Trained_Immunity.md` |
| Hu精读 | `01-分析项目/精读_Hu_NatCommun_2025_Denosumab_FSTL1.md` |

## 下一阶段

| # | 任务 | 目标日期 | 优先级 |
|---|------|---------|--------|
| 1 | 动物实验详细Protocol | 2026-05-23 | 🔴 |
| 2 | 大鼠TMJ MIA技术确认 | 2026-05-20 | 🔴 |
| 3 | FGF23 ELISA kit询价 | 2026-05-20 | 🔴 |
| 4 | 朝堂议政：动物方案审议 | 2026-05-18 | 🟡 |
| 5 | 综述初稿 | 2026-06-06 | 🟡 |
| 6 | 动物实验启动 | 2026-06-01 | 🔴 |
```
