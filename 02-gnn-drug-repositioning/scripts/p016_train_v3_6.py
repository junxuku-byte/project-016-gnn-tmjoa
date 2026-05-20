#!/usr/bin/env python3
"""
Project-016 训练集重建 v3.6 — 完整版
基于1213篇有abstract文献 + abstract精读更新 + 扩展负样本采样
"""

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

INPUT_JSON = ".tmp/p016_evidence_v3_5_scopus.json"
RAW_JSON = ".tmp/p016_merged_v3_scopus.json"
OUTPUT_JSON = ".tmp/p016_train_v3_6.json"
OUTPUT_REPORT = ".tmp/p016_train_v3_6_report.md"

REPOSITIONING_STARS = {
    "Metformin", "Rapamycin", "Everolimus", "Resveratrol", "Curcumin",
    "Quercetin", "Fisetin", "EGCG", "SAMe", "Omega-3", "Ginger",
}

# Abstract精读更新映射
ABSTRACT_REVIEW_UPDATES = {
    "41084405": {"rec": "strong_positive", "score": 0.72},
    "38867545": {"rec": "weak_positive", "score": 0.62},
    "40273050": {"rec": "mechanism_animal", "score": 0.30},
    "39092654": {"rec": "mechanism_animal", "score": 0.25},
    "38821656": {"rec": "exclude_negative", "score": -0.20},
}

DRUG_KWS = {
    "PRP": ["platelet-rich", "platelet rich", "prp", "platelet-rich fibrin", "prf"],
    "Hyaluronic acid": ["hyaluronic acid", "hyaluronate", "hyaluronan", "sodium hyaluronate"],
    "Corticosteroid": ["corticosteroid", "triamcinolone", "dexamethasone", "prednisone", "methylprednisolone"],
    "NSAIDs": ["nsaid", "ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam", "tenoxicam", "non-steroidal"],
    "Denosumab": ["denosumab"],
    "Bisphosphonate": ["bisphosphonate", "alendronate", "zoledronic acid", "risedronate"],
    "Metformin": ["metformin"],
    "Rapamycin": ["rapamycin", "sirolimus", "everolimus"],
    "Resveratrol": ["resveratrol"],
    "Curcumin": ["curcumin"],
    "Quercetin": ["quercetin"],
    "Fisetin": ["fisetin"],
    "EGCG": ["egcg", "epigallocatechin"],
    "SAMe": ["s-adenosyl", "same"],
    "Omega-3": ["omega-3", "omega 3", "fish oil", "epa", "dha"],
    "Ginger": ["ginger"],
    "Glucosamine": ["glucosamine"],
    "Chondroitin": ["chondroitin"],
    "Collagen": ["collagen", "type ii collagen"],
    "Vitamin D": ["vitamin d", "vitamin d3", "cholecalciferol"],
    "Statin": ["statin", "atorvastatin", "rosuvastatin", "simvastatin"],
    "Botulinum toxin": ["botulinum toxin", "botox"],
    "Tanezumab": ["tanezumab"],
    "Duloxetine": ["duloxetine"],
    "Pregabalin": ["pregabalin"],
    "Gabapentin": ["gabapentin"],
    "Tramadol": ["tramadol"],
    "Capsaicin": ["capsaicin"],
    "Palmitoylethanolamide": ["palmitoylethanolamide", "pea"],
    "Avocado-soybean unsaponifiables": ["avocado", "unsaponifiable", "asu"],
    "Buprenorphine": ["buprenorphine"],
    "Ozone": ["ozone", "medical ozone"],
    "Laser therapy": ["low-level laser", "laser therapy", "photobiomodulation"],
    "PEMF": ["pulsed electromagnetic", "pemf"],
    "Prolotherapy": ["prolotherapy", "dextrose injection"],
    "IL-38": ["il-38", "interleukin-38"],
    "FGF18": ["fgf18", "fibroblast growth factor 18"],
    "Senolytics": ["senolytic", "dasatinib"],
    "Trehalose": ["trehalose"],
    "Mesenchymal stem cell": ["mesenchymal stem cell", "msc"],
    "Exosome": ["exosome"],
    "Growth factor": ["growth factor", "cgf", "concentrated growth factor", "bmp", "igf", "tgf-beta"],
}

DISEASE_KWS = {
    "TMJOA": ["temporomandibular joint osteoarthritis", "tmj osteoarthritis", "tmjoa"],
    "TMJ": ["temporomandibular joint", "tmj", "mandibular condyle", "condylar"],
    "TMD": ["temporomandibular disorder", "tmd", "myofascial pain"],
    "Osteoarthritis": ["osteoarthritis", "osteoarthrosis", "degenerative joint disease"],
    "Knee OA": ["knee osteoarthritis", "knee oa", "gonarthrosis"],
    "Hip OA": ["hip osteoarthritis", "hip oa"],
    "Osteoporosis": ["osteoporosis", "low bone mineral density", "bone loss"],
    "Rheumatoid arthritis": ["rheumatoid arthritis", "ra"],
    "Periodontitis": ["periodontitis", "periodontal disease"],
    "Oral cancer": ["oral cancer", "oral squamous cell carcinoma"],
}

def fallback_extract(title, abstract):
    text = ((title or "") + " " + (abstract or "")).lower()
    found_drugs = []
    for drug, kws in DRUG_KWS.items():
        for kw in kws:
            if kw.lower() in text:
                found_drugs.append(drug)
                break
    found_diseases = []
    for dis, kws in DISEASE_KWS.items():
        for kw in kws:
            if kw.lower() in text:
                found_diseases.append(dis)
                break
    if found_drugs and found_diseases:
        return [{"drug": d, "disease": dis} for d in found_drugs for dis in found_diseases]
    return []

def main():
    print("=" * 60)
    print("Project-016 v3.6 训练集重建")
    print("=" * 60)
    
    with open(INPUT_JSON) as f:
        data = json.load(f)
    with open(RAW_JSON) as f:
        raw = json.load(f)
    
    results = data['results']
    raw_papers = raw['papers']
    pmid_to_paper = {p.get('pmid'): p for p in raw_papers}
    
    print(f"Loading {len(results)} evidence records, {len(raw_papers)} raw papers")
    
    positives = []
    negatives = []
    mechanisms = []
    excluded = []
    
    for r in results:
        pmid = r.get('pmid', '')
        
        # Apply abstract review updates
        if pmid in ABSTRACT_REVIEW_UPDATES:
            upd = ABSTRACT_REVIEW_UPDATES[pmid]
            r = dict(r)
            r['recommendation'] = upd['rec']
            r['final_score'] = upd['score']
        
        rec = r['recommendation']
        if rec in ['exclude_neutral', 'exclude_animal_neutral']:
            excluded.append(r)
            continue
        
        # Extract pairs
        pairs = r.get('drug_disease_pairs', [])
        if not pairs:
            paper = pmid_to_paper.get(pmid, {})
            pairs = fallback_extract(paper.get('title', ''), paper.get('abstract', ''))
        if not pairs:
            if rec == 'exclude_negative':
                pairs = fallback_extract(r['title'], '')
            if not pairs:
                excluded.append(r)
                continue
        
        for pair in pairs:
            drug = pair['drug']
            disease = pair['disease']
            sample = {
                "pmid": pmid,
                "title": r['title'][:100],
                "drug": drug,
                "disease": disease,
                "year": r.get('year', 0),
                "journal_if": r.get('journal_if', 2.0),
                "design": r['confirmed_design'],
                "conclusion": r['conclusion_direction'],
                "total_n": r.get('total_n'),
                "final_score": r['final_score'],
                "is_repositioning_star": drug in REPOSITIONING_STARS,
            }
            
            if rec == 'strong_positive':
                sample.update({'label': 1.0, 'weight': 1.0, 'is_hard_positive': True})
                positives.append(sample)
            elif rec == 'weak_positive':
                sample.update({'label': 1.0, 'weight': 0.8})
                positives.append(sample)
            elif rec in ['mechanism_only', 'mechanism_animal']:
                sample.update({'label': 0.5, 'weight': 0.6})
                mechanisms.append(sample)
            elif rec == 'exclude_negative':
                sample.update({'label': 0.0, 'weight': 1.0, 'is_hard_negative': True})
                negatives.append(sample)
            else:
                excluded.append(r)
    
    print(f"\nInitial extraction:")
    print(f"  Positives:  {len(positives)}")
    print(f"  Mechanisms: {len(mechanisms)}")
    print(f"  Negatives:  {len(negatives)}")
    print(f"  Excluded:   {len(excluded)}")
    
    # Generate synthetic negatives
    random.seed(42)
    n_pos = len(positives)
    n_mech = len(mechanisms)
    target_neg = int((n_pos + n_mech * 0.5) * 5)
    
    known_drugs = list(set(s['drug'] for s in positives + mechanisms))
    known_diseases = list(set(s['disease'] for s in positives + mechanisms))
    
    disease_category = {
        "TMJOA": "tmj", "TMJ": "tmj", "TMD": "tmj",
        "Osteoarthritis": "oa", "Knee OA": "oa", "Hip OA": "oa",
        "Osteoporosis": "bone", "Periodontitis": "oral",
        "Rheumatoid arthritis": "ra", "Oral cancer": "cancer",
    }
    
    synthetic = []
    seen_pairs = set()
    for drug in known_drugs:
        drug_diseases = [s['disease'] for s in positives + mechanisms if s['drug'] == drug]
        drug_cats = set(disease_category.get(d, 'other') for d in drug_diseases)
        for disease in known_diseases:
            dc = disease_category.get(disease, 'other')
            if dc not in drug_cats and len(drug_diseases) > 0:
                key = (drug, disease)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    synthetic.append({
                        "pmid": "synthetic",
                        "title": f"Synthetic: {drug} for {disease} (no evidence)",
                        "drug": drug, "disease": disease, "year": 2024,
                        "journal_if": 2.0, "design": "Synthetic",
                        "conclusion": "negative", "total_n": None,
                        "final_score": -0.5, "label": 0.0, "weight": 0.3,
                        "is_synthetic": True,
                        "is_repositioning_star": drug in REPOSITIONING_STARS,
                    })
    
    # Sample synthetic negatives to target
    if len(synthetic) > target_neg - len(negatives):
        synthetic = random.sample(synthetic, max(0, target_neg - len(negatives)))
    
    all_negs = negatives + synthetic
    all_samples = positives + mechanisms + all_negs
    
    print(f"\nAfter synthetic negative generation:")
    print(f"  Target negatives: {target_neg}")
    print(f"  Hard negatives:   {len(negatives)}")
    print(f"  Synthetic negs:   {len(synthetic)}")
    print(f"  Total negs:       {len(all_negs)}")
    print(f"  Total samples:    {len(all_samples)}")
    
    # Stratified split
    by_drug = defaultdict(list)
    for s in all_samples:
        by_drug[s['drug']].append(s)
    
    train, val, test = [], [], []
    for drug, ds in by_drug.items():
        random.shuffle(ds)
        n = len(ds)
        if n <= 3:
            train.extend(ds)
        else:
            nt = max(1, int(n * 0.7))
            nv = max(1, int(n * 0.15))
            train.extend(ds[:nt])
            val.extend(ds[nt:nt+nv])
            test.extend(ds[nt+nv:])
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    print(f"\nSplit:")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        pos = sum(1 for s in split if s['label'] == 1.0)
        mech = sum(1 for s in split if s['label'] == 0.5)
        neg = sum(1 for s in split if s['label'] == 0.0)
        hard = sum(1 for s in split if s.get('is_hard_negative'))
        syn = sum(1 for s in split if s.get('is_synthetic'))
        repo = sum(1 for s in split if s.get('is_repositioning_star'))
        drugs = len(set(s['drug'] for s in split))
        diseases = len(set(s['disease'] for s in split))
        print(f"  {name:6}: {len(split):4} | pos={pos:3} mech={mech:3} neg={neg:3} (hard={hard}, syn={syn}) | {drugs} drugs, {diseases} diseases | repo={repo}")
    
    # Save
    output = {
        "metadata": {
            "version": "v3_6_full",
            "created_at": datetime.now().isoformat(),
            "total_samples": len(all_samples),
            "positives": len(positives),
            "mechanisms": len(mechanisms),
            "negatives": len(all_negs),
            "hard_negatives": len(negatives),
            "synthetic_negatives": len(synthetic),
        },
        "statistics": {
            "drug_distribution": dict(Counter(s['drug'] for s in all_samples).most_common(20)),
            "disease_distribution": dict(Counter(s['disease'] for s in all_samples).most_common(15)),
            "label_distribution": dict(Counter(s['label'] for s in all_samples)),
        },
        "splits": {"train": train, "val": val, "test": test},
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_JSON}")
    
    # Quick report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(f"# v3.6 Training Set Report\n\nTotal: {len(all_samples)} samples\n")
        f.write(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}\n")
    print(f"Saved: {OUTPUT_REPORT}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
