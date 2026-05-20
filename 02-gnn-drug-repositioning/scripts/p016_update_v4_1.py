#!/usr/bin/env python3
"""Update training set v4.1 based on fulltext review findings"""

import json
import random
from datetime import datetime

random.seed(42)

with open('.tmp/p016_train_v3_9d.json') as f:
    data = json.load(f)

all_samples = data['splits']['train'] + data['splits']['val'] + data['splits']['test']
changes = []

# 1. FGF18新增
for disease in ['TMJOA', 'TMJ']:
    sample = {
        'drug': 'FGF18',
        'disease': disease,
        'label': 0.7,
        'weight': 1.0,
        'tmj_relevance': 'direct',
        'source': 'PMID40273050_animal_model',
        'evidence_type': 'animal_experiment',
        'study_design': 'mouse_model',
        'journal': 'PLoS ONE',
        'year': 2025,
        'mechanism': ['chondrogenesis', 'anti-MMP13', 'TIMP1-up', 'Noggin-down', 'FGFR3-signaling'],
    }
    all_samples.append(sample)
    changes.append(f"ADD: FGF18 x {disease} = 0.7 [PMID40273050]")

# 2. HA降级
for s in all_samples:
    if s['drug'] == 'Hyaluronic acid' and s['disease'] in ['TMJOA', 'TMD', 'TMJ']:
        old = s['label']
        if old >= 0.8:
            s['label'] = 0.5
            s['weight'] = 0.8
            s['downgrade_reason'] = 'PMID29244893_Meta: no sig diff vs CS, weak vs placebo'
            changes.append(f"DOWNGRADE: HA x {s['disease']} {old:.1f} -> 0.5")

# 3. Arthrocentesis新增
for disease in ['TMJOA', 'TMD']:
    sample = {
        'drug': 'Arthrocentesis',
        'disease': disease,
        'label': 0.7,
        'weight': 1.0,
        'tmj_relevance': 'direct',
        'source': 'PMID30814387_RCT',
        'evidence_type': 'clinical_trial',
        'study_design': 'RCT_longterm',
        'journal': 'J Oral Sci',
        'year': 2019,
    }
    all_samples.append(sample)
    changes.append(f"ADD: Arthrocentesis x {disease} = 0.7 [PMID30814387]")

# 4. Capsaicin升级
for s in all_samples:
    if s['drug'] == 'Capsaicin' and s['disease'] in ['TMD', 'TMJ']:
        old = s['label']
        if old < 0.5:
            s['label'] = 0.6
            s['weight'] = 0.8
            s['upgrade_reason'] = 'PMID28879245_RCT: 8pct topical effective for TMD pain'
            changes.append(f"UPGRADE: Capsaicin x {s['disease']} {old:.1f} -> 0.6")

print('=== v4.1 Training Set Updates ===')
for c in changes:
    print(f'  {c}')

pos = [s for s in all_samples if s['label'] > 0]
neg = [s for s in all_samples if s['label'] == 0]
print(f'\nTotal: {len(all_samples)} | Pos: {len(pos)} | Neg: {len(neg)} | Ratio 1:{len(neg)/len(pos):.1f}')

random.shuffle(all_samples)
n = len(all_samples)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

output = {
    'metadata': {
        'version': 'v4_1_fulltext_reviewed',
        'created_at': datetime.now().isoformat(),
        'total_samples': len(all_samples),
        'positives': len(pos),
        'negatives': len(neg),
        'ratio': f'1:{len(neg)/len(pos):.1f}',
        'changes': changes,
        'new_drugs': ['FGF18', 'Arthrocentesis'],
        'description': 'Post-fulltext-review update',
    },
    'splits': {'train': all_samples[:train_end], 'val': all_samples[train_end:val_end], 'test': all_samples[val_end:]}
}

with open('.tmp/p016_train_v4_1.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\nSaved: .tmp/p016_train_v4_1.json')
