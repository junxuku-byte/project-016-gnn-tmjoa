# 项目C GNN药物重定位 - 当前进展报告

> 2026-05-16 实时状态

---

## ⚠️ 核心瓶颈：数据稀疏

| 指标 | 当前值 | 需要值 | 差距 |
|------|--------|--------|------|
| 药物节点 | 25 | — | — |
| 疾病节点 | 46 | — | — |
| **药物-疾病训练边** | **4** | **>50** | 🔴 差10倍+ |
| 全图边 | 14,442 | — | — |

**4个drug-disease边无法训练任何监督学习模型**（train/val/test划分后每个集合<2个样本）。

---

## ✅ 已完成

### 1. LabKG数据导出
- 节点特征矩阵：(10,014, 47)
- 边列表：14,442条（含关系类型编码）
- 完整映射和元数据

### 2. 纯PyTorch GNN框架（不依赖PyG）
- `models/gnn_pure_pytorch.py`
- GCN层 + GraphSAGE层 + MLP/内积解码器
- 链接预测训练循环（含早停、AUC评估）
- **状态**：代码就绪，等数据补充后启用

### 3. Node2Vec无监督基线
- `models/node2vec_baseline.py`
- 随机游走 + SVD嵌入
- **状态**：正在运行（~2-3分钟）

### 4. 大规模文献检索
- 339篇TMJOA/OA药物相关文献已定位
- 49篇核心文献已筛选（重点：Denosumab-OA、Anakinra、NGF抑制剂、重定位药物）
- **下载中**：已下载2篇（41520765 Denosumab-OA关联、41993107 Riluzole重定位），部分新文章因casesid提取失败

---

## 🔴 阻塞项

| 阻塞 | 原因 | 解决路径 |
|------|------|---------|
| drug-disease边太少 | 现有LabKG侧重机制通路，缺药物适应症 | 从49篇下载文献中提取药物-疾病三元组 |
| 2026年新文章下载失败 | 云图书馆casesid提取对新文章不稳定 | 改用OA降级或curl直接下载 |

---

## 📋 下一步（数据增强优先）

### 阶段1：补数据（1-2天）

从下载的文献中系统提取：

```
目标：添加200+ drug-disease/药物-靶点边

来源1：下载的49篇核心文献
  ├─ TMJOA已知治疗：NSAIDs、透明质酸、糖皮质激素、PRP、手术
  ├─ OA DMOAD：Anakinra、Tanezumab、Burosumab、Romosozumab
  ├─ 重定位药物：Metformin、Statin、Curcumin、Resveratrol
  └─ 提取格式：drug → [treats/may_treat] → disease

来源2：DrugBank/CTD数据库（公开数据）
  ├─ 25个现有药物 → 查已知适应症
  └─ 补充靶点关系

来源3：已精读的核心论文
  ├─ Chen 2025：训练免疫调节剂（雷帕霉素、二甲双胍、羟氯喹）
  ├─ Hu 2025：Denosumab→FSTL1→滑膜炎
  ├─ FGF23 OA 2018：FGF23Ab→OA保护
  └─ CKD OA 2023：CKD-MBD药物
```

### 阶段2：重新训练（数据补完后1天内）

```
1. 重新导出LabKG → 新数据
2. Node2Vec基线：全图嵌入 + 候选排序
3. GNN监督学习：50+ drug-disease边后启用
4. 对比：Node2Vec vs GNN vs 随机
```

### 阶段3：推理：TMJOA候选药物排序

```
输入：TMJOA节点 + 所有药物节点
输出：Top-20候选药物 + 置信度 + 机制路径
```

---

## 💡 务实建议

**今晚（接下来2小时）**：
1. 等Node2Vec跑完，看无监督结果
2. 手动精读已下载的2篇核心文献（Denosumab-OA关联、Riluzole重定位）
3. 提取三元组直接写入LabKG

**明天**：
1. 批量下载剩余文献（处理下载失败的文章）
2. 系统提取药物-疾病-靶点关系
3. 重新导出数据 → Node2Vec/GNN重跑

**后天**：
1. 得到TMJOA候选药物排序
2. 结合机制路径验证
3. 写结果报告

---

## 文件结构

```
02-gnn-drug-repositioning/
├── data/
│   ├── node_features.npy          ✅ (10014, 47)
│   ├── edge_list.csv               ✅ 14,442 edges
│   ├── metadata.json               ✅ full mapping
│   └── metadata.pkl                ✅ pickle backup
├── models/
│   ├── gnn_pure_pytorch.py         ✅ 监督GNN（等数据）
│   └── node2vec_baseline.py        🔄 运行中
├── results/
│   └── (待生成)
└── README.md                        📄 本项目说明
```

---

## 关键发现（Node2Vec预跑前）

从LabKG现有结构已可观察：

| 药物 | 已有连接数 | 主要连接方向 |
|------|-----------|------------|
| denosumab | 较多 | rankl, tmjoa, bone_resorption, ckd |
| burosumab | 少 | fgf23, xlh |
| metformin | 少 | aging, autophagy |
| rapamycin | 少 | mtor, aging |
| curcumin | 0 | — |
| resveratrol | 0 | — |
| statin | 0 | — |

**明显缺失**：大量已知药物未连接疾病节点。这是补数据的直接目标。

---

## ✅ v2 突破：拓扑基线TMJ修复（2026-05-16 20:45）

### 关键Bug修复
- **v1问题**：TMJOA节点category="mechanism"被排除在疾病目标外 → 所有TMJ预测CN=0
- **v2修复**：扩展疾病节点识别，包含`tmjoa`/`tmjd`/`tmj_arthralgia`/`disc_displacement`
- **结果**：现有drug→tmjoa边被正确识别（Statin直连、Metformin直连、Anakinra 2跳等）

### TMJOA药物重定位TOP 10

| 排名 | 药物 | 得分 | CN | SP | 状态 | 文献支持 |
|------|------|------|-----|-----|------|----------|
| 1 | **Statin** | 1.578 | 4 | 1 | ✓已知 | MR: FinnGen+UKB OA保护 |
| 2 | **FGF23中和抗体** | 1.407 | 3 | 2 | ⭐新 | Burt 2018: 挽救OA软骨 |
| 3 | **Anakinra** | 1.268 | 2 | 2 | ⭐新 | IL-1Ra→TMJ滑膜炎缓解 |
| 4 | **Riluzole** | 1.219 | 2 | 2 | ⭐新 | CTSS/NOS1抑制, MR验证 |
| 5 | **Calcitriol** | 1.185 | 2 | 2 | ⭐新 | 活性Vit D, 骨代谢调节 |
| 6 | **Strontium ranelate** | 1.123 | 2 | 2 | ⭐新 | 双作用骨药 |
| 7 | **SGK1抑制剂** | 1.106 | 1 | 1 | ✓已知 | PMID 41470005 TMJOA特异靶点 |
| 8 | **ACD137** | 1.104 | 1 | 2 | ⭐新 | TrkA NAM, NGF/疼痛通路 |
| 9 | **Tanezumab** | 1.057 | 1 | 2 | ⭐新 | NGF抗体(Phase 3失败) |
| 10 | **Curcumin** | 1.032 | 1 | 1 | ✓已知 | 抗炎天然产物 |

### CHIP→TMJOA假说验证

```
假说链：更年期→FBM↑→DNMT3A-CHIP→IL-17/IL-20→破骨细胞↑→TMJ软骨下骨丢失→TMJOA

药物靶点对应：
├─ Statin: ↓炎症（IL-17/IL-20上游阻断）
├─ FGF23 Ab: ↓软骨下骨丢失（直接骨保护）
├─ Anakinra: ↓IL-1β（炎症级联阻断）
├─ Riluzole: ↓CTSS/NOS1（软骨降解酶抑制）
├─ Calcitriol: ↑Klotho/↓FGF23（骨代谢调节）
└─ SGK1抑制剂: ↑FoxO1/↑自噬（TMJ特异性）
```

**核心洞察**：拓扑基线同时捕获了CHIP→TMJOA假说的**上游干预点**（Statin/炎症）和**下游干预点**（FGF23 Ab/骨保护）！

### 新增关键数据

| 维度 | v1前 | v2后 | 增量 |
|------|------|------|------|
| 药物节点 | 25 | **67** | +42 |
| 疾病节点 | 46 | **55** | +9 |
| drug-disease边 | 4 | **23** | +19 |
| LabKG总节点 | 10,014 | **10,146** | +132 |
| LabKG总边 | 14,442 | **14,666** | +224 |

### 文件更新

| 文件 | 说明 |
|------|------|
| `results/topology_predictions.json` | 完整9916对分数 |
| `results/topology_predictions_v2_tmj_fixed.json` | **v2 TMJ修复版** |

---

*报告更新：2026-05-16 20:45 | 状态：拓扑基线v2完成 | 阻塞：GNN仍需>50 drug-disease边*
