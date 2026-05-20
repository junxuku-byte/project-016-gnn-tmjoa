#!/usr/bin/env python3
"""
Project-016 v3.5 证据质量分层 — 适配 lit-search.py 多库搜索结果
输入: .tmp/p016_merged_v3.json (903篇, 585篇含abstract)
输出: .tmp/p016_evidence_v3_5_from_search.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

INPUT_JSON = ".tmp/p016_merged_v3.json"
OUTPUT_JSON = ".tmp/p016_evidence_v3_5_from_search.json"

# ... (reuse all functions from p016_evidence_quality_v3_5.py)

# Journal IF mapping (abbreviated - key journals only)
JOURNAL_IF_MAP = {
    "nat commun": 14.9, "nature communications": 14.9,
    "cell": 45.5, "blood": 17.5,
    "j clin invest": 11.9, "the journal of clinical investigation": 11.9,
    "j bone miner res": 5.7, "journal of bone and mineral research": 5.7,
    "arthritis rheumatol": 11.4, "arthritis \u0026 rheumatology": 11.4,
    "ann rheum dis": 17.5, "annals of the rheumatic diseases": 17.5,
    "osteoarthritis cartilage": 7.6, "osteoarthritis and cartilage": 7.6,
    "j dent res": 7.0, "journal of dental research": 7.0,
    "j oral maxillofac surg": 2.5, "journal of oral and maxillofacial surgery": 2.5,
    "j oral rehabil": 2.8, "journal of oral rehabilitation": 2.8,
    "cranio": 1.8,
    "int j oral maxillofac surg": 2.2,
    "j oral pathol med": 2.8,
    "oral dis": 3.0, "oral diseases": 3.0,
    "j contemp dent pract": 0.8,
    "j appl oral sci": 2.5,
    "clin oral investig": 2.9,
    "dentomaxillofac radiol": 1.8,
    "j orofac pain": 2.0,
    "int dent j": 2.5,
    "stem cell rev rep": 4.0,
    "front bioeng biotechnol": 4.3,
    "j cell mol med": 4.3,
    "int immunopharmacol": 3.9,
    "food funct": 4.1,
    "arch oral biol": 1.9,
    "j adv res": 10.0,
    "rmd open": 3.0,
    "endocrinology": 4.0,
    "j immunol res": 3.0,
    "j nanobiotechnology": 10.0,
}

def get_journal_if(journal_name: str) -> float:
    if not journal_name:
        return 2.0
    j_lower = journal_name.lower()
    for key, if_val in JOURNAL_IF_MAP.items():
        if key in j_lower:
            return if_val
    return 2.0

STUDY_DESIGN_PATTERNS = {
    "RCT": [r"randomized controlled trial", r"randomised controlled trial", r"randomly assigned", r"randomly allocated", r"double-blind", r"single-blind", r"placebo-controlled", r"sham-controlled", r"parallel group", r"crossover design"],
    "Meta-Analysis": [r"meta.analys\w*", r"systematic review and meta.analysis", r"pooled analysis"],
    "Systematic Review": [r"systematic review", r"narrative review", r"scoping review"],
    "Cohort": [r"cohort study", r"prospective study", r"retrospective study", r"follow-up study"],
    "Case-Control": [r"case-control", r"case control"],
    "Cross-Sectional": [r"cross-sectional", r"cross sectional"],
    "Animal/In Vitro": [r"in vivo", r"in vitro", r"animal model", r"rat\b", r"rabbit\b", r"mouse\b", r"mice\b", r"cell culture", r"chondrocytes?\s+(?:were|were\s+isolated)"],
}

def classify_study_design(abstract: str, article_type: str) -> tuple:
    a_lower = abstract.lower() if abstract else ""
    type_map = {"Randomized Controlled Trial": "RCT", "Meta-Analysis": "Meta-Analysis", "Systematic Review": "Systematic Review", "Clinical Trial": "Clinical Trial", "Review": "Review", "Journal Article": "Journal Article"}
    base_design = type_map.get(article_type, "Journal Article")
    
    if a_lower:
        for design, patterns in STUDY_DESIGN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, a_lower, re.IGNORECASE):
                    if design != base_design:
                        if design == "RCT" and base_design in ["Journal Article", "Clinical Trial"]:
                            return "RCT", "high", f"abstract confirms RCT"
                        if design == "Meta-Analysis" and base_design == "Systematic Review":
                            return "Meta-Analysis", "high", f"abstract confirms Meta"
                        if design == "Animal/In Vitro" and base_design == "Journal Article":
                            return "Animal/In Vitro", "high", f"abstract confirms animal/in vitro"
    
    if base_design in ["RCT", "Meta-Analysis", "Systematic Review"]:
        return base_design, "high", "article_type trusted"
    
    if base_design == "Journal Article" and a_lower:
        for design, patterns in STUDY_DESIGN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, a_lower, re.IGNORECASE):
                    return design, "medium", f"abstract infers {design}"
    
    return base_design, "medium", "default from article_type"

CONCLUSION_KEYWORDS = {
    "strong_positive": [r'superior\s+(?:to|than)', r'significantly\s+(?:reduced|decreased|improved|better|lower|higher)', r'significant\s+(?:reduction|decrease|improvement|benefit|effect)', r'statistically\s+significant', r'superior\s+(?:to|than)', r'more\s+effective\s+(?:than|compared\s+to)', r'positive\s+effect', r'beneficial\s+effect', r'markedly\s+(?:reduced|improved)', r'remarkably\s+(?:reduced|improved)', r'p\s*<\s*0\.05', r'p\s*=\s*0\.0\d+', r'significant\s+difference', r'significantly\s+decreased', r'significantly\s+increased', r'improved\s+significantly', r'reduced\s+significantly', r'effective\s+treatment', r'demonstrated\s+efficacy', r'showed\s+efficacy', r'proved\s+effective'],
    "weak_positive": [r'tendency\s+to', r'trend\s+toward', r'tended\s+to', r'suggested\s+', r'may\s+(?:have|be|provide)', r'might\s+(?:have|be|provide)', r'potential\s+(?:benefit|effect|role)', r'promising\s+(?:result|outcome|effect)', r'favorable\s+(?:outcome|result)', r'beneficial\s+(?:effect|role)', r'could\s+(?:reduce|improve|alleviate)', r'appeared\s+to', r'seemed\s+to'],
    "neutral": [r'no\s+significant\s+difference', r'no\s+significant\s+(?:effect|benefit|change)', r'comparable\s+(?:to|with)', r'similar\s+(?:to|between|among)', r'equivalent\s+(?:to|efficacy)', r'non.inferior', r'not\s+significantly\s+different', r'no\s+difference', r'no\s+significant\s+improvement', r'no\s+statistically\s+significant', r'p\s*>\s*0\.05', r'not\s+superior', r'not\s+better\s+than'],
    "negative": [r'not\s+effective', r'no\s+effect', r'no\s+benefit', r'not\s+significantly', r'failed\s+to\s+(?:show|demonstrate|improve)', r'did\s+not\s+(?:improve|reduce|show|demonstrate)', r'ineffective', r'worse\s+(?:than|compared)', r'adverse\s+effect', r'worsened', r'not\s+recommended'],
}

def extract_conclusion_sentences(abstract: str) -> list:
    if not abstract:
        return []
    conclusion_markers = ["in conclusion", "conclusion", "we conclude", "taken together", "these findings", "these results", "our results", "our findings", "the results", "this study", "collectively", "together,", "in summary", "to summarize", "overall,"]
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    conclusion_sents = []
    for sent in sentences:
        sent_lower = sent.lower()
        for marker in conclusion_markers:
            if marker in sent_lower:
                conclusion_sents.append(sent.strip())
                break
    if not conclusion_sents and len(sentences) >= 2:
        conclusion_sents = sentences[-2:]
    return conclusion_sents

def classify_conclusion_from_abstract(abstract: str) -> tuple:
    if not abstract:
        return "unclear", "low", [], "no abstract"
    conclusion_sents = extract_conclusion_sentences(abstract)
    if not conclusion_sents:
        return "unclear", "low", [], "no conclusion sentences found"
    
    text_to_check = " ".join(conclusion_sents).lower()
    full_text = abstract.lower()
    matched = defaultdict(list)
    
    for direction, patterns in CONCLUSION_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                matched[direction].append(pattern)
            elif direction in ["strong_positive", "neutral", "negative"]:
                if re.search(pattern, full_text, re.IGNORECASE):
                    matched[direction].append(pattern)
    
    has_strong_pos = len(matched.get("strong_positive", []))
    has_weak_pos = len(matched.get("weak_positive", []))
    has_neutral = len(matched.get("neutral", []))
    has_negative = len(matched.get("negative", []))
    
    if has_strong_pos > 0 and has_negative > 0:
        conclusion_pos = sum(1 for p in matched.get("strong_positive", []) if re.search(p, text_to_check, re.IGNORECASE))
        conclusion_neg = sum(1 for p in matched.get("negative", []) if re.search(p, text_to_check, re.IGNORECASE))
        if conclusion_pos > conclusion_neg:
            return "positive", "medium", matched["strong_positive"], "conflicting but more positive in conclusion"
        else:
            return "negative", "medium", matched["negative"], "conflicting but more negative in conclusion"
    
    if has_negative > 0:
        return "negative", "medium", matched["negative"], "negative keywords found"
    if has_strong_pos > 0:
        return "positive", "high", matched["strong_positive"], "strong positive keywords found"
    if has_neutral > 0:
        return "neutral", "medium", matched["neutral"], "neutral keywords found"
    if has_weak_pos > 0:
        return "positive", "low", matched["weak_positive"], "weak positive keywords found"
    
    for direction, patterns in CONCLUSION_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                if direction == "strong_positive":
                    return "positive", "low", [pattern], f"{direction} found in full text"
                elif direction == "neutral":
                    return "neutral", "low", [pattern], f"{direction} found in full text"
                elif direction == "negative":
                    return "negative", "low", [pattern], f"{direction} found in full text"
    
    return "unclear", "low", [], "no clear direction keywords found"

def extract_sample_size_v2(abstract: str, title: str) -> dict:
    if not abstract:
        return {}
    a_lower = abstract.lower()
    t_lower = (title or "").lower()
    text = a_lower + " " + t_lower
    patterns = [
        r'a total of (\d+) patients? (?:were )?(?:enrolled|recruited|included)',
        r'(\d+) patients? (?:were )?(?:randomly assigned|randomly allocated|divided|separated)',
        r'(?:enrolled|recruited|included) (\d+) patients?',
        r'(?:study|trial) (?:included|enrolled|recruited) (\d+) patients?',
        r'[nN]\s*=\s*(\d+)',
        r'sample size (?:of )?(\d+)',
        r'(\d+) subjects?',
        r'(\d+) participants?',
        r'(\d+) individuals?',
        r'(\d+) cases?',
        r'(\d+) (?:knees?|joints?|tmjs?)',
        r'(\d+) studies?[,.;\s]+(?:including\s+)?(?:a total of )?(\d+) patients?',
        r'from (\d+) patients?',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                nums = [int(n) for group in matches for n in group if n.isdigit()]
            else:
                nums = [int(m) for m in matches if m.isdigit()]
            if nums:
                valid_nums = [n for n in nums if 10 <= n <= 100000]
                if valid_nums:
                    return {"total_n": max(valid_nums), "source": "abstract_v2", "confidence": "high", "matched_pattern": pattern}
    all_nums = re.findall(r'\b(\d{2,5})\b', text)
    if all_nums:
        nums = [int(n) for n in all_nums]
        valid = [n for n in nums if 20 <= n <= 10000]
        if valid:
            return {"total_n": max(valid), "source": "heuristic_fallback", "confidence": "low"}
    return {}

DRUG_KEYWORDS = {
    "PRP": ["platelet-rich plasma", "platelet rich plasma", "platelet-rich fibrin", "prp"],
    "Hyaluronic acid": ["hyaluronic acid", "sodium hyaluronate", "hyaluronan"],
    "Corticosteroid": ["corticosteroid", "triamcinolone", "dexamethasone", "prednisone", "methylprednisolone"],
    "NSAIDs": ["ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam", "indomethacin", "ketoprofen", "nsaid"],
    "Denosumab": ["denosumab"],
    "Bisphosphonate": ["bisphosphonate", "alendronate", "zoledronic acid", "risedronate"],
    "Metformin": ["metformin"],
    "Rapamycin": ["rapamycin", "sirolimus", "everolimus"],
    "Resveratrol": ["resveratrol"],
    "Curcumin": ["curcumin"],
    "Quercetin": ["quercetin"],
    "Fisetin": ["fisetin"],
    "Tofacitinib": ["tofacitinib"],
    "Anakinra": ["anakinra"],
    "Tanezumab": ["tanezumab"],
    "Resatorvid": ["resatorvid"],
    "Vitamin D": ["vitamin d", "vitamin d3", "25(oh)d"],
    "Glucosamine": ["glucosamine"],
    "Chondroitin": ["chondroitin"],
    "Collagen": ["collagen"],
    "Omega-3": ["omega-3", "fish oil"],
    "Statin": ["statin", "atorvastatin", "rosuvastatin"],
    "Botulinum toxin": ["botulinum toxin", "botox"],
}

DISEASE_KEYWORDS = {
    "TMJOA": ["temporomandibular joint osteoarthritis", "tmj osteoarthritis", "tmjoa"],
    "TMJ": ["temporomandibular joint", "tmj", "mandibular condyle", "condylar"],
    "TMD": ["temporomandibular disorder", "temporomandibular joint disorder", "tmd"],
    "Osteoarthritis": ["osteoarthritis", "osteoarthrosis"],
    "Knee OA": ["knee osteoarthritis"],
    "Hip OA": ["hip osteoarthritis"],
    "Osteoporosis": ["osteoporosis"],
    "Rheumatoid arthritis": ["rheumatoid arthritis"],
}

def extract_drug_disease_pairs(title: str, abstract: str) -> list:
    text = ((title or "") + " " + (abstract or "")).lower()
    found_drugs = []
    for drug_name, patterns in DRUG_KEYWORDS.items():
        for pattern in patterns:
            if pattern.lower() in text:
                found_drugs.append(drug_name)
                break
    found_diseases = []
    for disease_name, patterns in DISEASE_KEYWORDS.items():
        for pattern in patterns:
            if pattern.lower() in text:
                found_diseases.append(disease_name)
                break
    pairs = []
    for drug in found_drugs:
        for disease in found_diseases:
            pairs.append({"drug": drug, "disease": disease})
    return pairs

REPOSITIONING_STARS = {"Metformin", "Rapamycin", "Everolimus", "Resveratrol", "Curcumin", "Quercetin", "Fisetin", "EGCG", "SAMe", "Omega-3", "Ginger"}

def compute_evidence_score(paper: dict) -> dict:
    pmid = paper.get("pmid", "")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    article_type = paper.get("article_type", "")
    journal = paper.get("journal", "")
    journal_iso = paper.get("journal_iso", "")
    
    journal_if = get_journal_if(journal_iso or journal)
    if journal_if >= 20:
        journal_weight = 1.0
    elif journal_if >= 10:
        journal_weight = 0.95
    elif journal_if >= 5:
        journal_weight = 0.85
    elif journal_if >= 2:
        journal_weight = 0.75
    elif journal_if >= 1:
        journal_weight = 0.65
    else:
        journal_weight = 0.5
    
    confirmed_design, design_conf, design_reason = classify_study_design(abstract, article_type)
    
    DESIGN_WEIGHTS = {"RCT": 1.0, "Meta-Analysis": 0.95, "Systematic Review": 0.85, "Clinical Trial": 0.9, "Cohort": 0.7, "Case-Control": 0.65, "Cross-Sectional": 0.6, "Animal/In Vitro": 0.4, "Journal Article": 0.55, "Review": 0.5}
    design_weight = DESIGN_WEIGHTS.get(confirmed_design, 0.5)
    
    conclusion_dir, conclusion_conf, conclusion_kws, conclusion_reason = classify_conclusion_from_abstract(abstract)
    
    CONCLUSION_COEFF = {"positive": 1.0, "weak_positive": 0.7, "neutral": 0.0, "negative": -0.5, "unclear": 0.0}
    conclusion_coeff = CONCLUSION_COEFF.get(conclusion_dir, 0.0)
    
    sample_size = extract_sample_size_v2(abstract, title)
    total_n = sample_size.get("total_n")
    
    if total_n:
        if total_n >= 200:
            sample_bonus = 0.05
        elif total_n >= 100:
            sample_bonus = 0.03
        elif total_n >= 50:
            sample_bonus = 0.0
        elif total_n >= 20:
            sample_bonus = -0.05
        else:
            sample_bonus = -0.15
    else:
        sample_bonus = 0.0
        total_n = None
    
    pairs = extract_drug_disease_pairs(title, abstract)
    
    base_score = journal_weight * design_weight * conclusion_coeff
    final_score = base_score + sample_bonus
    
    if final_score >= 0.7:
        gnn_label, gnn_weight, recommendation = 1.0, 1.0, "strong_positive"
    elif final_score >= 0.5:
        gnn_label, gnn_weight, recommendation = 1.0, 0.8, "weak_positive"
    elif final_score > 0:
        gnn_label, gnn_weight, recommendation = 0.5, 0.6, "mechanism_only"
    elif final_score == 0:
        gnn_label, gnn_weight, recommendation = 0.0, 0.0, "exclude_neutral"
    else:
        gnn_label, gnn_weight, recommendation = 0.0, 0.0, "exclude_negative"
    
    if confirmed_design == "Animal/In Vitro":
        if conclusion_dir == "positive":
            gnn_label = 0.5
            recommendation = "mechanism_animal"
        else:
            recommendation = "exclude_animal_neutral"
    
    needs_fulltext = False
    fulltext_reason = ""
    if confirmed_design in ["RCT", "Meta-Analysis"] and conclusion_dir == "unclear":
        needs_fulltext = True
        fulltext_reason = "RCT/Meta but conclusion unclear from abstract"
    if journal_if >= 5 and any(p["drug"] in ["PRP", "Hyaluronic acid", "Corticosteroid"] for p in pairs):
        needs_fulltext = True
        fulltext_reason += "; key drug in high-IF journal"
    if conclusion_conf == "medium" and "conflicting" in conclusion_reason:
        needs_fulltext = True
        fulltext_reason += "; conflicting keywords need resolution"
    
    return {
        "pmid": pmid, "title": title, "journal": journal, "journal_iso": journal_iso,
        "journal_if": journal_if, "year": paper.get("year", 0),
        "article_type": article_type, "confirmed_design": confirmed_design,
        "design_confidence": design_conf, "design_reason": design_reason,
        "conclusion_direction": conclusion_dir, "conclusion_confidence": conclusion_conf,
        "conclusion_keywords": conclusion_kws, "conclusion_reason": conclusion_reason,
        "sample_size": sample_size, "total_n": total_n,
        "drug_disease_pairs": pairs,
        "journal_weight": journal_weight, "design_weight": design_weight,
        "conclusion_coefficient": conclusion_coeff, "sample_bonus": sample_bonus,
        "base_score": base_score, "final_score": final_score,
        "gnn_label": gnn_label, "gnn_weight": gnn_weight,
        "recommendation": recommendation,
        "needs_fulltext_review": needs_fulltext,
        "fulltext_reason": fulltext_reason.strip("; "),
        "abstract_snippet": (abstract or "")[:200] if abstract else "",
        "tier": paper.get("tier", "C"),
        "sources": paper.get("sources", []),
    }

def main():
    print("="*60)
    print("Project-016 v3.5 证据质量分层 (lit-search multi-source)")
    print("="*60)
    
    with open(INPUT_JSON) as f:
        data = json.load(f)
    papers = data.get("papers", data.get("results", []))
    print(f"📖 读取 {len(papers)} 篇文献")
    
    # Process only papers with abstracts (585 have abstract)
    results = []
    processed_with_abstract = 0
    for i, paper in enumerate(papers, 1):
        if i % 100 == 0:
            print(f"  处理 {i}/{len(papers)}...")
        ev = compute_evidence_score(paper)
        results.append(ev)
        if paper.get("abstract"):
            processed_with_abstract += 1
    
    print(f"\n📊 证据质量分层统计:")
    
    design_dist = Counter(r["confirmed_design"] for r in results)
    print("研究设计分布:")
    for d, c in design_dist.most_common():
        print(f"  {d}: {c}")
    
    conclusion_dist = Counter(r["conclusion_direction"] for r in results)
    print("\n结论方向分布:")
    for d, c in conclusion_dist.most_common():
        print(f"  {d}: {c}")
    
    rec_dist = Counter(r["recommendation"] for r in results)
    print("\nGNN推荐分层:")
    for rec, c in rec_dist.most_common():
        print(f"  {rec}: {c}")
    
    fulltext_needed = [r for r in results if r["needs_fulltext_review"]]
    print(f"\n需要全文精读的文献: {len(fulltext_needed)} 篇")
    
    # Top 20 by score (among those with abstract)
    with_abstract_results = [r for r in results if r["abstract_snippet"]]
    results_sorted = sorted(with_abstract_results, key=lambda x: -x["final_score"])
    print(f"\n🏆 Top 15 强阳性文献 (score >= 0.5, has abstract):")
    for i, r in enumerate([x for x in results_sorted if x["final_score"] >= 0.5][:15], 1):
        pairs_str = ", ".join([f"{p['drug']}>{p['disease']}" for p in r["drug_disease_pairs"]])
        print(f"  {i}. PMID {r['pmid']} | score={r['final_score']:.2f} | {r['confirmed_design']} | {r['journal_if']:.1f}IF")
        print(f"      {r['title'][:70]}...")
        print(f"      Pairs: {pairs_str or 'none'} | N={r['total_n'] or '?'} | Conclusion: {r['conclusion_direction']}")
        print()
    
    # PRP analysis
    prp_papers = [r for r in results if any(p["drug"] == "PRP" for p in r["drug_disease_pairs"])]
    print(f"\n🔍 PRP文献详细分析 ({len(prp_papers)} 篇):")
    for r in sorted(prp_papers, key=lambda x: -x["final_score"]):
        disease_str = ", ".join(set(p["disease"] for p in r["drug_disease_pairs"]))
        print(f"  PMID {r['pmid']} | {r['year']} | {r['confirmed_design']} | score={r['final_score']:.2f} | {r['conclusion_direction']}")
        print(f"    Disease: {disease_str} | N={r['total_n'] or '?'} | IF={r['journal_if']:.1f}")
        print(f"    {r['title'][:70]}...")
        if r["needs_fulltext_review"]:
            print(f"    ⚠️ NEEDS FULLTEXT: {r['fulltext_reason']}")
        print()
    
    output = {
        "metadata": {
            "version": "v3_5_lit_search_multi_source",
            "created_at": datetime.now().isoformat(),
            "total_papers": len(results),
            "with_abstract": processed_with_abstract,
            "needs_fulltext_count": len(fulltext_needed),
        },
        "statistics": {
            "design_distribution": dict(design_dist),
            "conclusion_distribution": dict(conclusion_dist),
            "recommendation_distribution": dict(rec_dist),
        },
        "fulltext_needed": [
            {"pmid": r["pmid"], "title": r["title"], "reason": r["fulltext_reason"], "design": r["confirmed_design"], "journal_if": r["journal_if"]}
            for r in fulltext_needed
        ],
        "results": results,
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 证据分层结果已保存: {OUTPUT_JSON}")
    
    with open(".tmp/p016_needs_fulltext_v3.txt", "w") as f:
        for r in fulltext_needed:
            f.write(f"{r['pmid']}\t{r['title'][:80]}\t{r['fulltext_reason']}\n")
    print(f"   需精读PMID列表: .tmp/p016_needs_fulltext_v3.txt ({len(fulltext_needed)} 篇)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
