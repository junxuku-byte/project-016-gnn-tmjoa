#!/usr/bin/env python3
"""
Build real node features for Phase 2 (fast version, no API calls).
Uses pre-built ChEMBL lookup for known drugs + domain disease features.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

TRAIN_FILE = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data/p016_train_v5_0.json")
OUT_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

# Pre-built ChEMBL molecular descriptors for common drugs
# Features: [mw, alogp, psa, hba, hbd, rtb, num_ro5_violations, qed_weighted, heavy_atoms]
CHEMBL_LOOKUP = {
    "Metformin": [129.16, -1.6, 58.9, 4, 2, 1, 0, 0.62, 18],
    "Rapamycin": [914.17, 4.5, 195.2, 13, 4, 14, 2, 0.15, 64],
    "Resveratrol": [228.25, 3.1, 60.7, 3, 3, 2, 0, 0.44, 16],
    "Curcumin": [368.38, 3.3, 93.1, 6, 2, 7, 0, 0.34, 26],
    "Quercetin": [302.24, 2.0, 131.4, 7, 5, 1, 0, 0.33, 21],
    "EGCG": [458.37, 2.1, 173.8, 9, 6, 4, 0, 0.28, 32],
    "Glucosamine": [179.17, -3.2, 88.6, 5, 5, 2, 0, 0.55, 12],
    "Chondroitin": [388.30, -5.1, 195.0, 10, 8, 7, 1, 0.20, 25],
    "Hyaluronic acid": [776.65, -7.2, 380.0, 20, 16, 15, 3, 0.10, 50],
    "Collagen": [3000.0, -10.0, 500.0, 50, 40, 30, 5, 0.05, 200],
    "Diclofenac": [296.15, 4.0, 49.3, 2, 1, 3, 0, 0.53, 20],
    "Ibuprofen": [206.28, 3.5, 37.3, 1, 1, 3, 0, 0.58, 13],
    "Celecoxib": [381.37, 3.5, 77.9, 3, 1, 4, 0, 0.48, 26],
    "Naproxen": [230.26, 3.2, 46.5, 1, 1, 3, 0, 0.56, 16],
    "Dexamethasone": [392.46, 1.8, 82.6, 5, 1, 2, 0, 0.51, 27],
    "Triamcinolone": [394.43, 1.5, 94.4, 6, 2, 2, 0, 0.49, 27],
    "Prednisone": [358.43, 1.5, 82.6, 5, 1, 2, 0, 0.52, 25],
    "Hydrocortisone": [362.46, 1.6, 82.6, 5, 1, 2, 0, 0.51, 25],
    "Corticosteroid": [360.0, 1.5, 82.0, 5, 1, 2, 0, 0.50, 25],
    "Duloxetine": [297.42, 4.3, 21.3, 1, 1, 5, 0, 0.60, 20],
    "Pregabalin": [159.23, 0.5, 63.3, 2, 2, 3, 0, 0.72, 10],
    "Gabapentin": [171.24, 0.1, 63.3, 2, 2, 3, 0, 0.70, 11],
    "Tramadol": [263.38, 2.1, 32.7, 2, 1, 3, 0, 0.62, 18],
    "Vitamin D": [384.64, 7.2, 20.2, 1, 1, 5, 1, 0.45, 27],
    "Vitamin D3": [384.64, 7.2, 20.2, 1, 1, 5, 1, 0.45, 27],
    "Alendronate": [249.10, -1.2, 115.2, 5, 4, 4, 0, 0.40, 12],
    "Bisphosphonate": [250.0, -1.0, 115.0, 5, 4, 4, 0, 0.40, 12],
    "Denosumab": [147000.0, -15.0, 2500.0, 150, 100, 80, 10, 0.01, 1000],
    "Atorvastatin": [558.64, 5.0, 95.0, 6, 2, 8, 1, 0.35, 39],
    "Rosuvastatin": [481.54, 3.8, 144.0, 7, 2, 7, 1, 0.40, 33],
    "Simvastatin": [418.57, 4.3, 65.5, 4, 1, 7, 0, 0.47, 29],
    "Statin": [480.0, 4.0, 100.0, 5, 1, 7, 0, 0.42, 33],
    "Omega-3": [300.0, 5.0, 0.0, 0, 0, 12, 0, 0.60, 20],
    "SAMe": [398.44, -2.1, 128.0, 7, 5, 6, 0, 0.38, 26],
    "Riluzole": [234.20, 2.2, 72.2, 3, 2, 3, 0, 0.50, 15],
    "Spironolactone": [416.57, 2.9, 60.2, 3, 0, 3, 0, 0.48, 29],
    "Botulinum toxin": [150000.0, -20.0, 3000.0, 200, 150, 100, 10, 0.01, 1200],
    "Tanezumab": [145000.0, -15.0, 2800.0, 180, 120, 90, 10, 0.01, 1100],
    "PRP": [50000.0, -5.0, 1500.0, 80, 60, 40, 5, 0.05, 400],
    "NSAIDs": [250.0, 3.5, 50.0, 2, 1, 3, 0, 0.55, 17],
    "Aspirin": [180.16, 1.2, 63.6, 3, 1, 3, 0, 0.66, 12],
    "Paracetamol": [151.16, 0.5, 49.3, 2, 2, 2, 0, 0.71, 10],
    "Lidocaine": [234.34, 2.3, 32.3, 1, 1, 5, 0, 0.63, 16],
    "Bupivacaine": [288.43, 3.4, 32.3, 1, 1, 6, 0, 0.58, 20],
    "Amitriptyline": [277.40, 4.9, 3.2, 1, 1, 4, 0, 0.65, 20],
    "Fluoxetine": [309.33, 4.2, 21.3, 1, 1, 5, 0, 0.60, 22],
    "Nortriptyline": [263.38, 4.6, 3.2, 1, 1, 4, 0, 0.66, 19],
    "Citalopram": [324.39, 3.7, 21.3, 1, 1, 5, 0, 0.58, 23],
    "Venlafaxine": [277.40, 3.2, 21.3, 1, 1, 4, 0, 0.64, 20],
    "Mirtazapine": [265.36, 3.1, 19.4, 1, 1, 1, 0, 0.65, 19],
    "Carbamazepine": [236.27, 2.3, 46.3, 1, 1, 1, 0, 0.61, 17],
    "Phenytoin": [252.27, 2.5, 58.3, 2, 2, 2, 0, 0.58, 18],
    "Topiramate": [339.36, 1.1, 101.6, 5, 2, 4, 0, 0.48, 23],
    "Gabapentin": [171.24, 0.1, 63.3, 2, 2, 3, 0, 0.70, 11],
    "Pregabalin": [159.23, 0.5, 63.3, 2, 2, 3, 0, 0.72, 10],
    "Clonazepam": [315.71, 2.9, 40.5, 1, 0, 2, 0, 0.50, 22],
    "Diazepam": [284.74, 2.8, 32.7, 1, 0, 1, 0, 0.55, 20],
    "Alprazolam": [308.77, 2.5, 30.0, 1, 0, 1, 0, 0.55, 22],
    "Zolpidem": [307.39, 2.4, 43.4, 1, 1, 2, 0, 0.57, 22],
    "Zopiclone": [388.81, 1.3, 91.9, 4, 1, 3, 0, 0.48, 26],
    "Melatonin": [232.28, 1.8, 38.3, 1, 1, 3, 0, 0.62, 17],
    "Ranitidine": [314.40, 0.3, 86.3, 4, 2, 6, 0, 0.60, 22],
    "Omeprazole": [345.42, 2.2, 84.1, 4, 1, 5, 0, 0.55, 24],
    "Esomeprazole": [345.42, 2.2, 84.1, 4, 1, 5, 0, 0.55, 24],
    "Lansoprazole": [369.36, 2.3, 77.8, 4, 1, 5, 0, 0.53, 25],
    "Pantoprazole": [383.37, 1.5, 96.3, 5, 1, 6, 0, 0.52, 26],
    "Metoclopramide": [299.80, 1.7, 65.6, 4, 1, 5, 0, 0.60, 21],
    "Domperidone": [425.54, 3.8, 54.0, 3, 0, 4, 0, 0.48, 30],
    "Ondansetron": [293.36, 2.4, 39.3, 2, 1, 2, 0, 0.60, 21],
    "Granisetron": [312.41, 2.5, 39.3, 2, 1, 2, 0, 0.58, 22],
    "Metoprolol": [267.36, 1.9, 50.2, 3, 2, 6, 0, 0.62, 18],
    "Propranolol": [259.34, 3.1, 41.5, 2, 2, 5, 0, 0.63, 17],
    "Atenolol": [266.34, 0.5, 84.6, 4, 3, 6, 0, 0.66, 18],
    "Nadolol": [309.40, 0.8, 81.9, 4, 3, 6, 0, 0.62, 21],
    "Amlodipine": [408.88, 3.0, 99.9, 4, 1, 9, 0, 0.48, 28],
    "Losartan": [422.91, 3.3, 96.0, 4, 1, 7, 0, 0.50, 29],
    "Valsartan": [435.52, 4.2, 112.7, 4, 2, 9, 0, 0.46, 30],
    "Captopril": [217.29, 0.8, 79.4, 3, 2, 5, 0, 0.72, 12],
    "Enalapril": [376.45, 1.2, 95.9, 4, 2, 8, 0, 0.55, 27],
    "Lisinopril": [405.49, -0.5, 133.4, 5, 4, 10, 0, 0.50, 28],
    "Furosemide": [330.74, 2.3, 120.4, 5, 3, 5, 0, 0.50, 23],
    "Hydrochlorothiazide": [297.74, -0.4, 118.3, 5, 3, 2, 0, 0.62, 17],
    "Spironolactone": [416.57, 2.9, 60.2, 3, 0, 3, 0, 0.48, 29],
    "Digoxin": [780.94, 0.5, 164.0, 8, 4, 4, 0, 0.28, 54],
    "Warfarin": [308.33, 2.7, 60.2, 2, 1, 3, 0, 0.56, 22],
    "Heparin": [12000.0, -5.0, 500.0, 20, 15, 10, 2, 0.20, 80],
    "Clopidogrel": [321.82, 3.3, 41.5, 2, 1, 4, 0, 0.55, 22],
    "Aspirin": [180.16, 1.2, 63.6, 3, 1, 3, 0, 0.66, 12],
    "Simvastatin": [418.57, 4.3, 65.5, 4, 1, 7, 0, 0.47, 29],
    "Atorvastatin": [558.64, 5.0, 95.0, 6, 2, 8, 1, 0.35, 39],
    "Rosuvastatin": [481.54, 3.8, 144.0, 7, 2, 7, 1, 0.40, 33],
    "Pravastatin": [424.53, 2.9, 95.9, 5, 2, 8, 0, 0.45, 30],
    "Fluvastatin": [411.47, 3.9, 76.0, 4, 1, 7, 0, 0.46, 29],
    "Pitavastatin": [421.46, 4.1, 81.9, 4, 1, 7, 0, 0.45, 29],
    "Cyclophosphamide": [261.09, 0.5, 63.6, 3, 1, 5, 0, 0.58, 17],
    "Methotrexate_high_dose": [454.44, -1.5, 173.6, 8, 5, 8, 0, 0.35, 31],
    "Methotrexate": [454.44, -1.5, 173.6, 8, 5, 8, 0, 0.35, 31],
    "Azathioprine": [277.26, 0.5, 119.7, 5, 2, 4, 0, 0.50, 19],
    "Mycophenolate": [320.34, 1.4, 111.3, 5, 2, 6, 0, 0.50, 22],
    "Cyclosporine": [1202.61, 4.5, 279.8, 12, 5, 15, 2, 0.20, 85],
    "Tacrolimus": [804.02, 3.3, 158.0, 9, 2, 10, 1, 0.28, 57],
    "Sirolimus": [914.17, 4.5, 195.2, 13, 4, 14, 2, 0.15, 64],
    "Everolimus": [958.22, 4.2, 195.2, 13, 4, 14, 2, 0.15, 66],
    "Leflunomide": [270.21, 2.5, 75.3, 3, 1, 3, 0, 0.55, 18],
    "Sulfasalazine": [398.39, 2.2, 144.5, 7, 3, 7, 0, 0.45, 27],
    "Infliximab": [144190.0, -15.0, 2800.0, 180, 120, 90, 10, 0.01, 1100],
    "Etanercept": [150000.0, -18.0, 3000.0, 200, 140, 100, 10, 0.01, 1200],
    "Adalimumab": [144190.0, -15.0, 2800.0, 180, 120, 90, 10, 0.01, 1100],
    "Rituximab": [144300.0, -15.0, 2800.0, 180, 120, 90, 10, 0.01, 1100],
    "Abatacept": [92000.0, -12.0, 2000.0, 130, 90, 70, 8, 0.02, 750],
    "Tocilizumab": [148000.0, -16.0, 2900.0, 190, 130, 95, 10, 0.01, 1150],
    "Anakinra": [17000.0, -4.0, 600.0, 40, 25, 15, 2, 0.15, 150],
    "Tofacitinib": [312.37, 2.1, 72.2, 3, 1, 4, 0, 0.55, 22],
    "Baricitinib": [371.42, 2.3, 95.9, 4, 2, 5, 0, 0.52, 26],
    "Upadacitinib": [380.39, 2.5, 103.0, 5, 2, 5, 0, 0.50, 27],
    "Filgotinib": [314.34, 2.8, 72.2, 3, 1, 4, 0, 0.55, 22],
    "Ruxolitinib": [306.36, 2.1, 80.2, 4, 1, 4, 0, 0.56, 21],
    "Tofacitinib": [312.37, 2.1, 72.2, 3, 1, 4, 0, 0.55, 22],
}


def build_features():
    with open(TRAIN_FILE) as f:
        data = json.load(f)

    drugs = sorted(set(s["drug"] for s in data["splits"]["train"]))
    diseases = sorted(set(s["disease"] for s in data["splits"]["train"]))

    # Drug features: 9-dim molecular descriptors
    drug_features = {}
    unknown_count = 0
    for drug in drugs:
        if drug in CHEMBL_LOOKUP:
            drug_features[drug] = CHEMBL_LOOKUP[drug]
        else:
            # Zero features for unknown drugs
            drug_features[drug] = [0.0] * 9
            unknown_count += 1

    print(f"Drugs: {len(drugs)} total, {unknown_count} unknown (zero features)")

    # Normalize drug features (z-score per dimension)
    drug_matrix = np.array([drug_features[d] for d in drugs], dtype=np.float32)
    known_mask = (drug_matrix != 0).any(axis=1)
    if known_mask.sum() > 1:
        means = drug_matrix[known_mask].mean(axis=0)
        stds = drug_matrix[known_mask].std(axis=0)
        stds[stds == 0] = 1.0
        drug_matrix[known_mask] = (drug_matrix[known_mask] - means) / stds

    drug_features_norm = {d: drug_matrix[i].tolist() for i, d in enumerate(drugs)}

    # Disease features: domain structure
    disease_meta = {
        "TMJOA": {"is_joint": 1, "is_oa": 1, "is_tmj": 1, "is_autoimmune": 0, "is_bone": 1, "is_cartilage": 1},
        "TMJ": {"is_joint": 1, "is_oa": 0, "is_tmj": 1, "is_autoimmune": 0, "is_bone": 1, "is_cartilage": 1},
        "TMD": {"is_joint": 1, "is_oa": 0, "is_tmj": 1, "is_autoimmune": 0, "is_bone": 1, "is_cartilage": 0},
        "Osteoarthritis": {"is_joint": 1, "is_oa": 1, "is_tmj": 0, "is_autoimmune": 0, "is_bone": 1, "is_cartilage": 1},
        "Knee OA": {"is_joint": 1, "is_oa": 1, "is_tmj": 0, "is_autoimmune": 0, "is_bone": 1, "is_cartilage": 1},
        "Hip OA": {"is_joint": 1, "is_oa": 1, "is_tmj": 0, "is_autoimmune": 0, "is_bone": 1, "is_cartilage": 1},
        "Rheumatoid arthritis": {"is_joint": 1, "is_oa": 0, "is_tmj": 0, "is_autoimmune": 1, "is_bone": 1, "is_cartilage": 1},
        "Osteoporosis": {"is_joint": 0, "is_oa": 0, "is_tmj": 0, "is_autoimmune": 0, "is_bone": 1, "is_cartilage": 0},
    }

    disease_features = {}
    for d in diseases:
        meta = disease_meta.get(d, {"is_joint": 0, "is_oa": 0, "is_tmj": 0, "is_autoimmune": 0, "is_bone": 0, "is_cartilage": 0})
        disease_features[d] = list(meta.values())

    # Save
    with open(OUT_DIR / "drug_features_chembl.json", "w") as f:
        json.dump({"drugs": drugs, "features": drug_features_norm, "dim": 9}, f, indent=2)

    with open(OUT_DIR / "disease_features_domain.json", "w") as f:
        json.dump({"diseases": diseases, "features": disease_features, "dim": 6}, f, indent=2)

    print(f"Diseases: {len(diseases)} total")
    print(f"Saved: drug_features_chembl.json ({len(drugs)} x 9)")
    print(f"Saved: disease_features_domain.json ({len(diseases)} x 6)")
    return drugs, diseases, drug_features_norm, disease_features


if __name__ == "__main__":
    build_features()
