#!/usr/bin/env python3
"""
Project-016 GNN药物重定位 — 核心文献筛选 (v2, CHIP-free)
从 2,991 篇 PMID 元数据 → 300-400 篇核心文献
聚焦: TMJOA机制 + 骨代谢异常 + 药物靶点

纳入标准（评分制）:
- Tier 1 关键词 (+3): TMJOA核心机制、骨代谢异常、药物靶点
- Tier 2 关键词 (+2): 相关骨病、炎症通路、细胞类型
- Tier 3 关键词 (+1): 相关机制、影像、干细胞
- 文章类型加分: RCT +2, Meta +2, SysRev +2, Review +1, JA +1
- 排除: Case Reports(-∞), Letter, Editorial, Erratum
- 年份加分: ≥2020 +2, ≥2015 +1

输出:
  1. core_pmids.json — 核心文献列表（300-400篇）
  2. screening_report.md — 筛选报告
"""

import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# 1. 关键词定义 (v2: CHIP-free, TMJOA+骨代谢+药物靶点导向)
# ══════════════════════════════════════════════════════════════

TIER1_KEYWORDS = {
    # TMJOA 核心机制
    "temporomandibular", "tmj", "tmjoa", "mandibular condyle", "condylar",
    "osteoarthritis", "osteoarthrosis", "cartilage degeneration", "cartilage destruction",
    "chondrocyte apoptosis", "chondrocyte pyroptosis", "chondrocyte hypertrophy",
    "synovial inflammation", "synovitis",
    "subchondral bone resorption", "subchondral bone sclerosis",
    "joint degeneration", "condylar resorption",
    
    # 骨代谢异常（与TMJOA共存的全身问题）
    "vitamin d deficiency", "vitamin d", "25(oh)d",
    "osteoporosis", "osteopenia", "osteomalacia",
    "bone mineral density", "bmd",
    "bone marrow fat", "fatty marrow", "marrow adiposity",
    "ckd-mineral bone disorder", "ckd-mbd", "renal osteodystrophy",
    "hyperparathyroidism", "hypothyroidism", "hyperthyroidism",
    "hypophosphatemia", "hyperphosphatemia",
    "diabetic bone disease", "glucocorticoid-induced osteoporosis",
    "hemochromatosis", "iron overload",
    "hyperhomocysteinemia", "homocysteine",
    
    # 药物靶点（可直接用于GNN预测的候选）
    "drug repositioning", "drug repurposing", "drug screening",
    "fgf23", "rankl", "denosumab", "bisphosphonate",
    "wnt", "β-catenin", "sclerostin", "dkk1",
    "mmp13", "adamts5", "mmp9",
    "il-1β", "il-6", "il-17", "il-38", "tnf-α", "tnf-alpha",
    "nf-κb", "nf-kappa", "jak", "stat",
    "metformin", "rapamycin", "mTOR",
    "resveratrol", "curcumin", "quercetin",
}

TIER2_KEYWORDS = {
    # 骨代谢通用
    "bone remodeling", "bone resorption", "bone formation", "bone loss",
    "osteoclast", "osteoblast", "osteocyte",
    "rank", "rankl", "opg",
    "cartilage", "chondrocyte", "extracellular matrix", "ecm",
    "collagen", "aggrecan", "proteoglycan",
    "subchondral bone", "cortical bone", "trabecular bone",
    "menopause", "postmenopausal", "estrogen", "oestrogen", "estradiol",
    "hormone replacement", "hrt",
    
    # 炎症通路
    "inflammation", "inflammatory", "cytokine",
    "interleukin", "il-1", "il-2", "il-4", "il-6", "il-8", "il-10", "il-12", "il-17", "il-18", "il-20", "il-38",
    "tnf", "tnf-alpha", "tnf-α",
    "macrophage", "m1 macrophage", "m2 macrophage",
    "nf-κb", "nf-kappa", "jak-stat", "mapk", "erk", "p38",
    "nlrp3", "inflammasome", "pyroptosis", "apoptosis", "autophagy",
    "oxidative stress", "ros", "reactive oxygen species",
    
    # 其他关节OA（可作为TMJOA的类比/外推）
    "knee osteoarthritis", "hip osteoarthritis", "spine", "vertebral",
    "degenerative joint disease",
}

TIER3_KEYWORDS = {
    # 相关机制
    "adipogenesis", "adipocyte", "lipid", "fat",
    "glucocorticoid", "cortisol", "stress",
    "senescence", "aging", "ageing", "cellular senescence",
    "autophagy", "mitophagy", "mitochondrial dysfunction",
    "mesenchymal stem cell", "msc", "progenitor", "fibrocartilage stem cell",
    "exosome", "extracellular vesicle",
    
    # 药物/治疗（更泛的）
    "glucagon-like peptide", "glp-1", "semaglutide",
    "anti-il", "biologics", "dmard",
    "nsaid", "cox-2 inhibitor", "celecoxib",
    "hyaluronic acid", "platelet-rich plasma", "prp",
    "stem cell therapy", "regenerative medicine",
    
    # 影像
    "mri", "cbct", "cone-beam", "texture", "radiomics", "morphology",
    "micro-ct", "μct",
    
    # 其他解剖
    " jaw ", "maxillofacial", "craniofacial", "mandible", "condyle",
    "articular disc", "disc displacement",
    
    # 生物信息学/AI（方法学相关）
    "machine learning", "deep learning", "graph neural network", "gnn",
    "network pharmacology", "bioinformatics",
    "drug-target interaction", "protein-protein interaction",
    "knowledge graph", "heterogeneous network",
}

EXCLUDE_TYPES = {
    "Case Reports", "Letter", "Editorial", "Published Erratum",
    "Comment", "News", "Biography",
}

HIGH_VALUE_TYPES = {
    "Randomized Controlled Trial": 2,
    "Meta-Analysis": 2,
    "Systematic Review": 2,
    "Clinical Trial": 1,
    "Review": 1,
    "Journal Article": 1,
}

# ══════════════════════════════════════════════════════════════
# 2. 评分函数
# ══════════════════════════════════════════════════════════════

def score_paper(paper: dict) -> tuple:
    """返回 (score, reasons_list, is_tmjoa, is_bone_metabolism, is_drug_target)"""
    title = paper.get("title", "").lower()
    journal = paper.get("journal", "").lower()
    abstract = paper.get("abstract", "").lower()
    year = paper.get("year", 0)
    atype = paper.get("article_type", "")
    
    # 搜索空间：标题 + 期刊 + 摘要（如有）
    search_text = f"{title} {journal} {abstract}"
    
    score = 0
    reasons = []
    
    # 排除
    if atype in EXCLUDE_TYPES:
        return (-1000, [f"EXCLUDE: {atype}"], False, False, False)
    
    # Tier 1 关键词
    t1_hits = [kw for kw in TIER1_KEYWORDS if kw.lower() in search_text]
    if t1_hits:
        score += len(t1_hits) * 3
        reasons.append(f"T1({len(t1_hits)}): {', '.join(t1_hits[:3])}")
    
    # Tier 2 关键词
    t2_hits = [kw for kw in TIER2_KEYWORDS if kw.lower() in search_text]
    if t2_hits:
        score += len(t2_hits) * 2
        reasons.append(f"T2({len(t2_hits)}): {', '.join(t2_hits[:3])}")
    
    # Tier 3 关键词
    t3_hits = [kw for kw in TIER3_KEYWORDS if kw.lower() in search_text]
    if t3_hits:
        score += len(t3_hits) * 1
        if len(t3_hits) <= 3:
            reasons.append(f"T3({len(t3_hits)}): {', '.join(t3_hits)}")
        else:
            reasons.append(f"T3({len(t3_hits)})")
    
    # 文章类型
    type_score = HIGH_VALUE_TYPES.get(atype, 0)
    if type_score > 0:
        score += type_score
        reasons.append(f"TYPE({atype}): +{type_score}")
    
    # 年份
    if year >= 2020:
        score += 2
        reasons.append(f"YEAR({year}): +2")
    elif year >= 2015:
        score += 1
        reasons.append(f"YEAR({year}): +1")
    elif year < 2000:
        score -= 2  # 过老文献减分
        reasons.append(f"YEAR({year}): -2")
    
    # 特殊保护标记
    is_tmjoa = ("temporomandibular" in search_text or 
                "tmj" in search_text or 
                "mandibular condyle" in search_text or
                "tmjoa" in search_text)
    
    is_bone_metabolism = any(kw in search_text for kw in [
        "vitamin d", "osteoporosis", "osteopenia", "osteomalacia",
        "bone mineral density", "bmd", "bone marrow fat", "fatty marrow",
        "ckd-mbd", "renal osteodystrophy", "hyperparathyroidism",
        "hypothyroidism", "hyperthyroidism", "hypophosphatemia",
        "diabetic bone", "glucocorticoid-induced osteoporosis",
        "hemochromatosis", "iron overload", "homocysteine"
    ])
    
    is_drug_target = any(kw in search_text for kw in [
        "drug repositioning", "drug repurposing", "drug screening",
        "denosumab", "bisphosphonate", "wnt", "sclerostin", "mmp13",
        "adamts5", "metformin", "rapamycin", "resveratrol",
        "il-1β", "il-38", "nf-κb", "jak", "stat", "tnf-α inhibitor"
    ])
    
    return (score, reasons, is_tmjoa, is_bone_metabolism, is_drug_target)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".tmp/p016_full_2991.json", help="输入元数据 JSON")
    parser.add_argument("--output-json", default=".tmp/p016_core_pmids.json", help="核心文献 JSON")
    parser.add_argument("--output-pmids", default=".tmp/p016_core_pmids.txt", help="核心 PMID 列表")
    parser.add_argument("--output-report", default=".tmp/p016_screening_report.md", help="筛选报告")
    parser.add_argument("--min-score", type=int, default=3, help="最低分数阈值")
    parser.add_argument("--max-papers", type=int, default=400, help="最大保留数量")
    args = parser.parse_args()
    
    # 读取
    with open(args.input) as f:
        data = json.load(f)
    
    papers = data.get("results", data) if isinstance(data, dict) else data
    total = len(papers)
    print(f"📥 读取 {total} 篇文献")
    
    # 评分
    scored = []
    excluded = Counter()
    for p in papers:
        score, reasons, is_tmjoa, is_bone, is_drug = score_paper(p)
        scored.append({
            **p,
            "_score": score,
            "_reasons": reasons,
            "_is_tmjoa": is_tmjoa,
            "_is_bone_metabolism": is_bone,
            "_is_drug_target": is_drug,
        })
        if score <= -100:
            excluded[p.get("article_type", "Unknown")] += 1
    
    # 排序（分数降序，同年份按 PMID 降序即最新优先）
    scored.sort(key=lambda x: (-x["_score"], -x.get("year", 0), -int(x.get("pmid", 0))))
    
    # 筛选
    core = []
    tmj_protected = []
    bone_protected = []
    drug_protected = []
    
    for p in scored:
        if p["_score"] >= args.min_score:
            core.append(p)
        elif p["_is_tmjoa"] and p["_score"] >= 1:  # TMJOA 相关适度保留
            tmj_protected.append(p)
        elif p["_is_bone_metabolism"] and p["_score"] >= 1:  # 骨代谢相关适度保留
            bone_protected.append(p)
        elif p["_is_drug_target"] and p["_score"] >= 2:  # 药物靶点保留（需一定分数）
            drug_protected.append(p)
    
    # 合并，去重
    all_core = core + tmj_protected + bone_protected + drug_protected
    seen_pmids = set()
    unique_core = []
    for p in all_core:
        pmid = p.get("pmid")
        if pmid and pmid not in seen_pmids:
            seen_pmids.add(pmid)
            unique_core.append(p)
    
    # 限制数量 — 优先保留 TMJOA、骨代谢、药物靶点
    if len(unique_core) > args.max_papers:
        priority = [p for p in unique_core if p["_is_tmjoa"] or p["_is_bone_metabolism"] or p["_is_drug_target"]]
        others = [p for p in unique_core if not p["_is_tmjoa"] and not p["_is_bone_metabolism"] and not p["_is_drug_target"]]
        
        if len(priority) <= args.max_papers:
            keep_others = args.max_papers - len(priority)
            unique_core = priority + others[:keep_others]
        else:
            unique_core = priority[:args.max_papers]
    
    # 重新按分数排序
    unique_core.sort(key=lambda x: (-x["_score"], -x.get("year", 0)))
    
    final_count = len(unique_core)
    
    # 统计
    tmj_count = sum(1 for p in unique_core if p["_is_tmjoa"])
    bone_count = sum(1 for p in unique_core if p["_is_bone_metabolism"])
    drug_count = sum(1 for p in unique_core if p["_is_drug_target"])
    
    # 交集统计
    tmj_bone = sum(1 for p in unique_core if p["_is_tmjoa"] and p["_is_bone_metabolism"])
    tmj_drug = sum(1 for p in unique_core if p["_is_tmjoa"] and p["_is_drug_target"])
    bone_drug = sum(1 for p in unique_core if p["_is_bone_metabolism"] and p["_is_drug_target"])
    all_three = sum(1 for p in unique_core if p["_is_tmjoa"] and p["_is_bone_metabolism"] and p["_is_drug_target"])
    
    type_dist = Counter(p.get("article_type", "Unknown") for p in unique_core)
    year_dist = Counter()
    for p in unique_core:
        y = p.get("year", 0)
        if y >= 2020:
            year_dist["2020+"] += 1
        elif y >= 2015:
            year_dist["2015-2019"] += 1
        elif y >= 2010:
            year_dist["2010-2014"] += 1
        else:
            year_dist["<2010"] += 1
    
    score_dist = Counter()
    for p in unique_core:
        s = p["_score"]
        if s >= 10:
            score_dist["10+"] += 1
        elif s >= 7:
            score_dist["7-9"] += 1
        elif s >= 5:
            score_dist["5-6"] += 1
        elif s >= 3:
            score_dist["3-4"] += 1
        else:
            score_dist["<3"] += 1
    
    print(f"\n{'='*60}")
    print(f"📊 筛选结果")
    print(f"{'='*60}")
    print(f"原始总数:       {total}")
    print(f"排除:           {sum(excluded.values())} ({dict(excluded)})")
    print(f"核心文献:       {final_count}")
    print(f"  TMJOA 相关:   {tmj_count}")
    print(f"  骨代谢相关:   {bone_count}")
    print(f"  药物靶点:     {drug_count}")
    print(f"  TMJOA+骨代谢: {tmj_bone}")
    print(f"  TMJOA+药物:   {tmj_drug}")
    print(f"  骨代谢+药物:  {bone_drug}")
    print(f"  三者交集:     {all_three}")
    print(f"\n年份分布:       {dict(year_dist)}")
    print(f"分数分布:       {dict(score_dist)}")
    print(f"类型分布:       {dict(type_dist)}")
    
    # 输出 JSON
    output_data = {
        "screening_metadata": {
            "source_total": total,
            "excluded": dict(excluded),
            "core_count": final_count,
            "min_score": args.min_score,
            "max_papers": args.max_papers,
            "tmjoa_count": tmj_count,
            "bone_metabolism_count": bone_count,
            "drug_target_count": drug_count,
            "intersections": {
                "tmjoa_bone": tmj_bone,
                "tmjoa_drug": tmj_drug,
                "bone_drug": bone_drug,
                "all_three": all_three,
            },
            "year_distribution": dict(year_dist),
            "score_distribution": dict(score_dist),
            "type_distribution": dict(type_dist),
        },
        "results": [
            {k: v for k, v in p.items() if not k.startswith("_")}
            for p in unique_core
        ],
    }
    
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 输出 PMID 列表
    with open(args.output_pmids, "w") as f:
        for p in unique_core:
            f.write(f"{p.get('pmid', '')}\n")
    
    # 生成报告
    report_lines = [
        f"# Project-016 GNN药物重定位 — 文献筛选报告 (v2)",
        f"",
        f"**筛选时间**: {datetime.now().isoformat()}",
        f"**原始文献**: {total} 篇",
        f"**核心文献**: {final_count} 篇",
        f"**筛选阈值**: min_score={args.min_score}, max_papers={args.max_papers}",
        f"**筛选版本**: v2 (CHIP-free, TMJOA+骨代谢+药物靶点导向)",
        f"",
        f"## 纳入/排除标准",
        f"",
        f"### 排除类型",
        f"- {', '.join(EXCLUDE_TYPES)}",
        f"",
        f"### 评分规则",
        f"- **Tier 1 (+3/词)**: TMJOA核心机制、骨代谢异常、药物靶点",
        f"  - TMJOA: temporomandibular, tmj, tmjoa, condylar, osteoarthritis, chondrocyte apoptosis, synovial inflammation, subchondral bone resorption...",
        f"  - 骨代谢: vitamin d deficiency, osteoporosis, osteomalacia, bone marrow fat, ckd-mbd, hyperparathyroidism, diabetic bone disease...",
        f"  - 药物靶点: drug repositioning, fgf23, rankl, denosumab, wnt, mmp13, il-1β, il-38, nf-κb, metformin, rapamycin...",
        f"- **Tier 2 (+2/词)**: 相关骨病、炎症通路、细胞类型",
        f"  - bone remodeling, osteoclast, rankl, cartilage, menopause, estrogen, cytokine, il-6, il-17, macrophage, nf-κb...",
        f"- **Tier 3 (+1/词)**: 相关机制、影像、干细胞、生物信息学方法",
        f"  - adipogenesis, autophagy, senescence, msc, exosome, glp-1, nsaid, mri, cbct, radiomics, machine learning, gnn...",
        f"- **高价值类型**: RCT(+2), Meta(+2), SysRev(+2), Review(+1), JA(+1)",
        f"- **年份**: ≥2020(+2), ≥2015(+1), <2000(-2)",
        f"",
        f"### 特殊保护",
        f"- **TMJOA相关文献**（分数≥1保留）— 项目核心",
        f"- **骨代谢异常文献**（分数≥1保留）— 共病筛查靶点",
        f"- **药物靶点文献**（分数≥2保留）— GNN预测候选",
        f"",
        f"## 统计摘要",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| TMJOA 相关 | {tmj_count} |",
        f"| 骨代谢相关 | {bone_count} |",
        f"| 药物靶点 | {drug_count} |",
        f"| TMJOA ∩ 骨代谢 | {tmj_bone} |",
        f"| TMJOA ∩ 药物靶点 | {tmj_drug} |",
        f"| 骨代谢 ∩ 药物靶点 | {bone_drug} |",
        f"| 三者交集 | {all_three} |",
        f"",
        f"### 年份分布",
        f"",
    ]
    for y, c in sorted(year_dist.items()):
        report_lines.append(f"- {y}: {c} ({c/final_count*100:.1f}%)")
    
    report_lines.extend([
        f"",
        f"### 分数分布",
        f"",
    ])
    for s, c in sorted(score_dist.items(), key=lambda x: -int(x[0].replace("+", "99").split("-")[0])):
        report_lines.append(f"- {s}: {c} ({c/final_count*100:.1f}%)")
    
    report_lines.extend([
        f"",
        f"### 文章类型",
        f"",
    ])
    for t, c in type_dist.most_common():
        report_lines.append(f"- {t}: {c}")
    
    report_lines.extend([
        f"",
        f"## Top 20 核心文献",
        f"",
    ])
    
    for i, p in enumerate(unique_core[:20], 1):
        reasons = " | ".join(p["_reasons"][:3])
        flags = []
        if p["_is_tmjoa"]: flags.append("TMJ")
        if p["_is_bone_metabolism"]: flags.append("BONE")
        if p["_is_drug_target"]: flags.append("DRUG")
        flag_str = f" [{'+'.join(flags)}]" if flags else ""
        report_lines.append(
            f"{i}. **PMID {p.get('pmid', 'N/A')}** (score={p['_score']}{flag_str})\n"
            f"   - {p.get('title', '')[:120]}...\n"
            f"   - {p.get('first_author', '')} et al., {p.get('journal', '')[:40]}, {p.get('year', '?')}\n"
            f"   - {reasons}"
        )
    
    # 按维度分列 Top 5
    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 分维度核心文献",
        f"",
        f"### TMJOA 机制 Top 5",
        f"",
    ])
    tmj_top = [p for p in unique_core if p["_is_tmjoa"]][:5]
    for i, p in enumerate(tmj_top, 1):
        report_lines.append(
            f"{i}. PMID {p.get('pmid', 'N/A')}: {p.get('title', '')[:100]}... ({p.get('year', '?')})"
        )
    
    report_lines.extend([
        f"",
        f"### 骨代谢异常 Top 5",
        f"",
    ])
    bone_top = [p for p in unique_core if p["_is_bone_metabolism"]][:5]
    for i, p in enumerate(bone_top, 1):
        report_lines.append(
            f"{i}. PMID {p.get('pmid', 'N/A')}: {p.get('title', '')[:100]}... ({p.get('year', '?')})"
        )
    
    report_lines.extend([
        f"",
        f"### 药物靶点 Top 5",
        f"",
    ])
    drug_top = [p for p in unique_core if p["_is_drug_target"]][:5]
    for i, p in enumerate(drug_top, 1):
        report_lines.append(
            f"{i}. PMID {p.get('pmid', 'N/A')}: {p.get('title', '')[:100]}... ({p.get('year', '?')})"
        )
    
    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 对GNN建模的启示",
        f"",
        f"### 潜在药物靶点节点（从文献高频提及提取）",
        f"- **炎症通路**: IL-1β, IL-6, IL-17, TNF-α, NF-κB → 现有抗炎药可重定位",
        f"- **基质降解**: MMP13, ADAMTS5 → MMP抑制剂",
        f"- **Wnt信号**: Wnt/β-catenin, Sclerostin, DKK1 → 骨形成促进剂",
        f"- **FGF23**: 骨代谢-软骨降解桥梁 → 抗FGF23抗体 (Burosumab)",
        f"- **代谢调节**: Metformin (AMPK/mTOR), Rapamycin → 抗衰老/抗炎",
        f"- **天然化合物**: Resveratrol, Curcumin, Quercetin → 多靶点抗炎",
        f"",
        f"### 负样本设计建议",
        f"1. **随机负样本**: 随机 drug-disease 对（已验证无效）",
        f"2. **拓扑距离负样本**: 知识图谱中距离 >3  hops 的 drug-disease 对",
        f"3. **时间分割**: 2015年前已上市 drug + 2015年后发现 disease，验证是否被后续研究支持",
        f"",
        f"### 网络构建思路",
        f"- **疾病层**: TMJOA 表型 → 软骨退化 / 滑膜炎 / 骨吸收 子类型",
        f"- **机制层**: 炎症通路 (IL-1β→NF-κB→MMP13) + 骨代谢 (RANKL→破骨细胞) + 代谢 (mTOR→自噬)",
        f"- **药物层**: 已知OA药 (NSAIDs, HA) + 骨代谢药 (Denosumab, Bisphosphonates) + 重定位候选 (Metformin, Rapamycin)",
        f"- **连接**: 文献共现 + 通路共享 + 靶点互作",
        f"",
    ])
    
    with open(args.output_report, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✅ 输出文件:")
    print(f"   JSON:   {args.output_json}")
    print(f"   PMID:   {args.output_pmids}")
    print(f"   Report: {args.output_report}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
