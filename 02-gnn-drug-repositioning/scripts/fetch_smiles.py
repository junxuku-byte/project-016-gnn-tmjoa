#!/usr/bin/env python3
import json, time, sys
from urllib.request import urlopen, Request
from urllib.parse import quote
from pathlib import Path

DATA = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")
TRAIN = DATA / "p016_train_v5_0.json"

print("Loading drugs...", flush=True)
with open(TRAIN) as f: train_data = json.load(f)
all_drugs = sorted(set(it['drug'] for it in train_data['splits']['train']))
print(f"Total drugs: {len(all_drugs)}", flush=True)

cache_path = DATA / "drug_smiles_cache.json"
smiles_cache = {}
if cache_path.exists():
    with open(cache_path) as f: smiles_cache = json.load(f)
    print(f"Cached: {len(smiles_cache)}", flush=True)

new = 0; fail = 0; skip = 0
for i, drug in enumerate(all_drugs):
    if drug in smiles_cache: skip += 1; continue
    clean = drug.replace('_', ' ').split('_')[0].strip()
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(clean)}/property/IsomericSMILES/JSON'
        req = Request(url, headers={'User-Agent': 'p016/1.0'})
        data = json.loads(urlopen(req, timeout=15).read())
        smiles = data['PropertyTable']['Properties'][0]['SMILES']
        smiles_cache[drug] = smiles; new += 1
    except Exception as e:
        smiles_cache[drug] = None; fail += 1
    if (new+fail) % 30 == 0:
        print(f"  fetched {new} new, {fail} failed, {skip} cached", flush=True)
    time.sleep(0.15)

with open(cache_path, 'w') as f: json.dump(smiles_cache, f, indent=2)
found = sum(1 for v in smiles_cache.values() if v)
print(f"DONE. SMILES: {found}/{len(all_drugs)} (new={new} fail={fail} cached={skip})", flush=True)
