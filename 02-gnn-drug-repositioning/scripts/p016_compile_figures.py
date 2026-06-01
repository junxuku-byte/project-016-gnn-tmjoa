#!/usr/bin/env python3
"""
P1+P2: Ablation table + Literature validation visualization data.
Outputs publication-ready JSON for Figure generation.

# ── Paths (portable) ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
"""
import json
from pathlib import Path

DATA=Path("DATA_DIR")
OUT=DATA/"paper_figures_data.json"

# ─── Ablation table ─────────────────────────────────────────────
ablation={
    'models':[
        {'name':'RWR (transductive)','auc':0.7221,'std':0.2903,'type':'baseline','interpretable':False},
        {'name':'Node2Vec (transductive)','auc':0.8973,'std':0.0684,'type':'baseline_transductive_leak','interpretable':False},
        {'name':'Node2Vec (inductive)','auc':0.6865,'std':0.1693,'type':'baseline','interpretable':False},
        {'name':'Homogeneous GNN','auc':0.6847,'std':0.0885,'type':'ablation','components':[],'interpretable':False},
        {'name':'+ Heterogeneous (A)','auc':0.7197,'std':0.2111,'type':'ablation','components':['hetero_msg'],'interpretable':True},
        {'name':'+ Topological feat (B)','auc':0.7340,'std':0.1983,'type':'ablation','components':['hetero_msg','topo_feat'],'interpretable':True},
        {'name':'+ Class weight (C)','auc':0.7515,'std':0.1960,'type':'ablation','components':['hetero_msg','topo_feat','class_weight'],'interpretable':True},
        {'name':'**+ Type attention**','auc':0.8513,'std':0.1078,'type':'final','components':['hetero_msg','topo_feat','class_weight','type_attention'],'interpretable':True},
    ],
    'permutation_test':{
        'n_permutations':100,
        'observed_mean_auc':0.7515,
        'null_mean_auc':0.46,
        'null_auc_geq_observed':0,
        'p_value':'<0.01'
    }
}

# ─── Literature validation table ────────────────────────────────
with open(DATA/"literature_validation.json") as f:
    lit=json.load(f)

lit_table={
    'target_disease':'TMJOA',
    'discovery_rate':'19/20 (95%)',
    'predictions':lit['predictions'],
    'evidence_summary':{
        'strong (≥10 hits)':sum(1 for p in lit['predictions'] if p['evidence_level']=='strong'),
        'moderate (3-9 hits)':sum(1 for p in lit['predictions'] if p['evidence_level']=='moderate'),
        'weak (1-2 hits)':sum(1 for p in lit['predictions'] if p['evidence_level']=='weak'),
        'none (0 hits)':sum(1 for p in lit['predictions'] if p['evidence_level']=='none'),
    }
}

# ─── Mechanism case studies ─────────────────────────────────────
with open(DATA/"mechanism_case_studies.json") as f:
    cases=json.load(f)

# ─── Graph statistics ───────────────────────────────────────────
with open(DATA/"four_layer_graph_full_v2.json") as f:
    g=json.load(f)

graph_stats={
    'nodes':g['metadata'],
    'edges':{
        'drug_target':len(g['drug_target_edges']),
        'target_pathway':len(g['target_pathway_edges']),
        'pathway_disease':len(g['pathway_disease_edges']),
        'total':len(g['drug_target_edges'])+len(g['target_pathway_edges'])+len(g['pathway_disease_edges'])
    },
    'density_dt':len(g['drug_target_edges'])/(g['metadata']['n_drugs']*g['metadata']['n_targets']),
    'density_tp':len(g['target_pathway_edges'])/(g['metadata']['n_targets']*g['metadata']['n_pathways']),
    'density_pd':len(g['pathway_disease_edges'])/(g['metadata']['n_pathways']*g['metadata']['n_diseases']),
}

# ─── Final results comparison table ─────────────────────────────
with open(DATA/"attn_hgnn_ldo_results.json") as f:
    attn_res=json.load(f)
with open(DATA/"node2vec_inductive_results.json") as f:
    n2v_res=json.load(f)
with open(DATA/"rwr_baseline_results.json") as f:
    rwr_res=json.load(f)
with open(DATA/"node2vec_baseline_results.json") as f:
    n2v_td_res=json.load(f)

comparison={
    'methods':[
        {'name':'Random','auc':0.50,'std':0.0,'setting':'null'},
        {'name':'RWR','auc':rwr_res['mean_auc'],'std':rwr_res['std_auc'],'setting':'transductive'},
        {'name':'Node2Vec+MLP','auc':n2v_td_res['mean_auc'],'std':n2v_td_res['std_auc'],'setting':'transductive (embedding leak)'},
        {'name':'Node2Vec+MLP (inductive)','auc':n2v_res['mean_auc'],'std':n2v_res['std_auc'],'setting':'inductive'},
        {'name':'HeteroGNN+A+B+C','auc':0.7515,'std':0.1960,'setting':'inductive'},
        {'name':'AttnHeteroGNN+A+B+C','auc':attn_res['mean_auc'],'std':attn_res['std_auc'],'setting':'inductive'},
    ]
}

result={
    'ablation':ablation,
    'literature_validation':lit_table,
    'mechanism_case_studies':cases,
    'graph_statistics':graph_stats,
    'method_comparison':comparison,
    'attention_weights':cases.get('attention_weights',{})
}

with open(OUT,'w') as f:
    json.dump(result,f,indent=2)

print("Paper figures data compiled:")
print(f"  Ablation: {len(ablation['models'])} models")
print(f"  Literature: {lit_table['discovery_rate']} validated")
print(f"  Case studies: {len(cases['case_studies'])} drugs")
print(f"  Graph: {graph_stats['nodes']}")
print(f"  Methods compared: {len(comparison['methods'])}")
print(f"\n→ {OUT}")
