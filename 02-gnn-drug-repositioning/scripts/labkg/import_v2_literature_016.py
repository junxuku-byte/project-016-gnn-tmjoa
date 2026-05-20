#!/usr/bin/env python3
"""
Project-016 LabKG网络更新 (v2文献驱动)
基于v2筛选的400篇核心文献，提取概念节点并增强LabKG。
同时计算 drug-disease 机制路径距离矩阵，为负样本策略提供数据基础。
"""

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store

PROJECT_ID = "project-016"
V2_JSON = ".tmp/p016_core_v2.json"


def nid(name: str) -> str:
    """标准化节点ID"""
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '').replace(',', '')[:80]}"


def extract_concepts_from_title(title: str) -> dict:
    """
    从文献标题中提取候选概念（基于关键词匹配，快速启发式方法）。
    返回: {'drugs': [], 'diseases': [], 'mechanisms': [], 'proteins': []}
    """
    t = title.lower()
    
    # Drug keywords (部分已知的TMJOA相关药物)
    drug_patterns = {
        "denosumab": "Denosumab",
        "bisphosphonate": "Bisphosphonate",
        "metformin": "Metformin",
        "rapamycin": "Rapamycin",
        "sirolimus": "Sirolimus",
        "resveratrol": "Resveratrol",
        "curcumin": "Curcumin",
        "quercetin": "Quercetin",
        "hyaluronic acid": "Hyaluronic acid",
        "prp": "Platelet-rich plasma",
        "platelet-rich plasma": "Platelet-rich plasma",
        "corticosteroid": "Corticosteroid",
        "nsaid": "NSAID",
        "celecoxib": "Celecoxib",
        "ibuprofen": "Ibuprofen",
        "naproxen": "Naproxen",
        "diclofenac": "Diclofenac",
        "meloxicam": "Meloxicam",
        "pregabalin": "Pregabalin",
        "gabapentin": "Gabapentin",
        "duloxetine": "Duloxetine",
        "amitriptyline": "Amitriptyline",
        "tramadol": "Tramadol",
        "oxycodone": "Oxycodone",
        "morphine": "Morphine",
        "acetaminophen": "Acetaminophen",
        "paracetamol": "Paracetamol",
        "glucosamine": "Glucosamine",
        "chondroitin": "Chondroitin",
        "collagen": "Collagen supplement",
        "vitamin d": "Vitamin D supplement",
        "vitamin d3": "Vitamin D3",
        "calcium": "Calcium supplement",
        "alendronate": "Alendronate",
        "zoledronic acid": "Zoledronic acid",
        "risedronate": "Risedronate",
        "raloxifene": "Raloxifene",
        "teriparatide": "Teriparatide",
        "romosozumab": "Romosozumab",
        "burosumab": "Burosumab",
        "calcitonin": "Calcitonin",
        "strontium ranelate": "Strontium ranelate",
        "semaglutide": "Semaglutide",
        "liraglutide": "Liraglutide",
        "tirzepatide": "Tirzepatide",
        "canakinumab": "Canakinumab",
        "anakinra": "Anakinra",
        "tocilizumab": "Tocilizumab",
        "adalimumab": "Adalimumab",
        "etanercept": "Etanercept",
        "infliximab": "Infliximab",
        "golimumab": "Golimumab",
        "certolizumab": "Certolizumab",
        "ustekinumab": "Ustekinumab",
        "secukinumab": "Secukinumab",
        "ixekizumab": "Ixekizumab",
        "guselkumab": "Guselkumab",
        "risankizumab": "Risankizumab",
        "tildrakizumab": "Tildrakizumab",
        "brodalumab": "Brodalumab",
        "bimekizumab": "Bimekizumab",
        "apremilast": "Apremilast",
        "tofacitinib": "Tofacitinib",
        "baricitinib": "Baricitinib",
        "upadacitinib": "Upadacitinib",
        "filgotinib": "Filgotinib",
        "abatacept": "Abatacept",
        "rituximab": "Rituximab",
        "belimumab": "Belimumab",
        "mepolizumab": "Mepolizumab",
        "reslizumab": "Reslizumab",
        "benralizumab": "Benralizumab",
        "dupilumab": "Dupilumab",
        "omalizumab": "Omalizumab",
        "ligelizumab": "Ligelizumab",
        "lebrikizumab": "Lebrikizumab",
        "nemolizumab": "Nemolizumab",
        "tralokinumab": "Tralokinumab",
        "fezakinumab": "Fezakinumab",
        "sarilumab": "Sarilumab",
        "sirukumab": "Sirukumab",
        "siltuximab": "Siltuximab",
        "clazakizumab": "Clazakizumab",
        "olokizumab": "Olokizumab",
        "vobarilizumab": "Vobarilizumab",
        "namilumab": "Namilumab",
        "mavrilimumab": "Mavrilimumab",
        "lenzilumab": "Lenzilumab",
        "leronlimab": "Leronlimab",
        "leronlimab": "Leronlimab",
        "bevacizumab": "Bevacizumab",
        "ranibizumab": "Ranibizumab",
        "aflibercept": "Aflibercept",
        "conbercept": "Conbercept",
        "ziv-aflibercept": "Ziv-aflibercept",
        "ramucirumab": "Ramucirumab",
        "nivolumab": "Nivolumab",
        "pembrolizumab": "Pembrolizumab",
        "atezolizumab": "Atezolizumab",
        "avelumab": "Avelumab",
        "durvalumab": "Durvalumab",
        "ipilimumab": "Ipilimumab",
        "tremelimumab": "Tremelimumab",
        "sinlimab": "Sinlimab",
        "sintilimab": "Sintilimab",
        "toripalimab": "Toripalimab",
        "camrelizumab": "Camrelizumab",
        "tislelizumab": "Tislelizumab",
        "serplulimab": "Serplulimab",
        "envafolimab": "Envafolimab",
        "ak105": "AK105",
        "ak104": "AK104",
        "kn035": "KN035",
        "cs1003": "CS1003",
        "sg001": "SG001",
        "hx008": "HX008",
        "hlx10": "HLX10",
        "hlx04": "HLX04",
        "ibi308": "IBI308",
        "ibi310": "IBI310",
        "shr1210": "SHR1210",
        "shr1316": "SHR1316",
        "js001": "JS001",
        "js003": "JS003",
        "bat1306": "BAT1306",
        "bat1706": "BAT1706",
        "gb226": "GB226",
        "gb221": "GB221",
        "f520": "F520",
        "f636": "F636",
        "m7824": "M7824",
        "m9241": "M9241",
        "pf-06801591": "PF-06801591",
        "pf-06940434": "PF-06940434",
        "rg7744": "RG7744",
        "rg7446": "RG7446",
        "rg7853": "RG7853",
        "rg7884": "RG7884",
        "rg7888": "RG7888",
    }
    
    # Disease keywords
    disease_patterns = {
        "temporomandibular joint osteoarthritis": "TMJOA",
        "tmj osteoarthritis": "TMJOA",
        "tmjoa": "TMJOA",
        "temporomandibular joint disorders": "TMD",
        "tmd": "TMD",
        "temporomandibular joint": "TMJ",
        "tmj": "TMJ",
        "mandibular condyle": "TMJ condyle",
        "condylar resorption": "Condylar resorption",
        "condylar degeneration": "Condylar degeneration",
        "condylar osteoarthritis": "Condylar OA",
        "internal derangement": "TMJ internal derangement",
        "disc displacement": "TMJ disc displacement",
        "disc derangement": "TMJ disc derangement",
        "osteoarthritis": "Osteoarthritis",
        "rheumatoid arthritis": "Rheumatoid arthritis",
        "psoriatic arthritis": "Psoriatic arthritis",
        "ankylosing spondylitis": "Ankylosing spondylitis",
        "gout": "Gout",
        "pseudogout": "Pseudogout",
        "calcium pyrophosphate deposition": "CPPD",
        "cppd": "CPPD",
        "osteoporosis": "Osteoporosis",
        "osteopenia": "Osteopenia",
        "osteomalacia": "Osteomalacia",
        "vitamin d deficiency": "Vitamin D deficiency",
        "hypothyroidism": "Hypothyroidism",
        "hyperthyroidism": "Hyperthyroidism",
        "hyperparathyroidism": "Hyperparathyroidism",
        "hypoparathyroidism": "Hypoparathyroidism",
        "ckd": "Chronic kidney disease",
        "chronic kidney disease": "Chronic kidney disease",
        "diabetes": "Diabetes mellitus",
        "diabetic": "Diabetes mellitus",
        "type 2 diabetes": "Type 2 diabetes",
        "type 1 diabetes": "Type 1 diabetes",
        "obesity": "Obesity",
        "metabolic syndrome": "Metabolic syndrome",
        "insulin resistance": "Insulin resistance",
        "fibromyalgia": "Fibromyalgia",
        "chronic pain": "Chronic pain",
        "myofascial pain": "Myofascial pain",
        "bruxism": "Bruxism",
        "sleep apnea": "Sleep apnea",
        "depression": "Depression",
        "anxiety": "Anxiety",
        "stress": "Psychological stress",
        "post-traumatic": "Post-traumatic stress",
        "trauma": "Trauma",
        "fracture": "Fracture",
        "bone marrow": "Bone marrow disorder",
        "bone loss": "Bone loss",
        "bone resorption": "Bone resorption",
    }
    
    # Mechanism keywords
    mechanism_patterns = {
        "inflammation": "Inflammation",
        "inflammatory": "Inflammation",
        "cytokine": "Cytokine signaling",
        "interleukin": "Interleukin signaling",
        "il-1": "IL-1 signaling",
        "il-6": "IL-6 signaling",
        "il-17": "IL-17 signaling",
        "il-38": "IL-38 signaling",
        "tnf": "TNF signaling",
        "tnf-α": "TNF-α signaling",
        "nf-κb": "NF-κB signaling",
        "nf-kappa": "NF-κB signaling",
        "jak": "JAK-STAT signaling",
        "stat": "JAK-STAT signaling",
        "mapk": "MAPK signaling",
        "erk": "ERK signaling",
        "p38": "p38 MAPK signaling",
        "pi3k": "PI3K-AKT signaling",
        "akt": "AKT signaling",
        "mtor": "mTOR signaling",
        "ampk": "AMPK signaling",
        "wnt": "Wnt signaling",
        "β-catenin": "β-catenin signaling",
        "beta-catenin": "β-catenin signaling",
        "sclerostin": "Sclerostin signaling",
        "dkk1": "DKK1 signaling",
        "rankl": "RANKL signaling",
        "rank": "RANK signaling",
        "opg": "OPG signaling",
        "mmp": "MMP activity",
        "mmp13": "MMP13 activity",
        "adamts": "ADAMTS activity",
        "adamts5": "ADAMTS5 activity",
        "collagen": "Collagen metabolism",
        "proteoglycan": "Proteoglycan metabolism",
        "aggrecan": "Aggrecan metabolism",
        "chondrocyte": "Chondrocyte biology",
        "apoptosis": "Apoptosis",
        "pyroptosis": "Pyroptosis",
        "autophagy": "Autophagy",
        "senescence": "Cellular senescence",
        "oxidative stress": "Oxidative stress",
        "ros": "Reactive oxygen species",
        "mitochondrial": "Mitochondrial dysfunction",
        "hypoxia": "Hypoxia",
        "hif": "HIF signaling",
        "angiogenesis": "Angiogenesis",
        "vascular": "Vascular biology",
        "nerve growth factor": "NGF signaling",
        "substance p": "Substance P signaling",
        "calcitonin gene-related peptide": "CGRP signaling",
        "trpv1": "TRPV1 signaling",
        "acid-sensing ion channel": "ASIC signaling",
        "piezo": "PIEZO signaling",
        "mechanotransduction": "Mechanotransduction",
        "mechanical loading": "Mechanical loading",
        "biomechanics": "Biomechanics",
        "bone remodeling": "Bone remodeling",
        "bone formation": "Bone formation",
        "osteoclast": "Osteoclast activity",
        "osteoblast": "Osteoblast activity",
        "osteocyte": "Osteocyte signaling",
        "subchondral bone": "Subchondral bone remodeling",
        "cartilage degeneration": "Cartilage degeneration",
        "cartilage destruction": "Cartilage destruction",
        "synovial": "Synovial biology",
        "synovitis": "Synovitis",
        "synovial fibroblast": "Synovial fibroblast",
        "macrophage": "Macrophage biology",
        "m1": "M1 macrophage polarization",
        "m2": "M2 macrophage polarization",
        "t cell": "T cell biology",
        "b cell": "B cell biology",
        "neutrophil": "Neutrophil biology",
        "mast cell": "Mast cell biology",
        "platelet": "Platelet biology",
        "extracellular vesicle": "Extracellular vesicle",
        "exosome": "Exosome signaling",
        "microrna": "miRNA regulation",
        "lncrna": "lncRNA regulation",
        "circrna": "circRNA regulation",
        "epigenetic": "Epigenetic regulation",
        "dna methylation": "DNA methylation",
        "histone": "Histone modification",
        "acetylation": "Histone acetylation",
        "methylation": "DNA/histone methylation",
        "gut microbiota": "Gut microbiota",
        "microbiome": "Microbiome",
        "lipid metabolism": "Lipid metabolism",
        "glucose metabolism": "Glucose metabolism",
        "insulin signaling": "Insulin signaling",
        "estrogen": "Estrogen signaling",
        "androgen": "Androgen signaling",
        "progesterone": "Progesterone signaling",
        "vitamin d receptor": "VDR signaling",
        "vdr": "VDR signaling",
        "pth": "Parathyroid hormone signaling",
        "fgf23": "FGF23 signaling",
        "klotho": "Klotho signaling",
        "sirtuin": "Sirtuin signaling",
        "amp-activated protein kinase": "AMPK signaling",
        "sirt1": "SIRT1 signaling",
        "nrf2": "Nrf2 signaling",
        "keap1": "Keap1-Nrf2 pathway",
        "foxo": "FOXO signaling",
        "p53": "p53 signaling",
        "p16": "p16 signaling",
        "p21": "p21 signaling",
        "rb": "Rb signaling",
        "e2f": "E2F signaling",
        "cyclin": "Cell cycle regulation",
        "cdk": "CDK regulation",
        "g1": "G1 phase",
        "s phase": "S phase",
        "g2": "G2 phase",
        "m phase": "M phase",
        "dna damage": "DNA damage response",
        "dna repair": "DNA repair",
        "telomere": "Telomere biology",
        "telomerase": "Telomerase activity",
        "senescence-associated secretory phenotype": "SASP",
        "sasp": "SASP",
        "inflammasome": "Inflammasome",
        "nlrp3": "NLRP3 inflammasome",
        "aim2": "AIM2 inflammasome",
        "nlrc4": "NLRC4 inflammasome",
        "pyrin": "Pyrin inflammasome",
        "gasdermin": "Gasdermin D",
        "gsdmd": "GSDMD",
        "caspase-1": "Caspase-1",
        "caspase-11": "Caspase-11",
        "caspase-4": "Caspase-4",
        "caspase-5": "Caspase-5",
        "il-1β": "IL-1β",
        "il-18": "IL-18",
        "hmgb1": "HMGB1",
        "hmgb1": "HMGB1",
        "s100a8": "S100A8",
        "s100a9": "S100A9",
        "s100a8/a9": "S100A8/A9",
        "calprotectin": "Calprotectin",
        "lactoferrin": "Lactoferrin",
        "lysozyme": "Lysozyme",
        "defensin": "Defensin",
        "cathelicidin": "Cathelicidin",
        "ll-37": "LL-37",
        "cramp": "CRAMP",
        "surfactant protein": "Surfactant protein",
        "collectin": "Collectin",
        "mannose-binding lectin": "MBL",
        "complement": "Complement system",
        "c1q": "C1q",
        "c3": "C3",
        "c4": "C4",
        "factor b": "Factor B",
        "properdin": "Properdin",
        "factor h": "Factor H",
        "factor i": "Factor I",
        "cd59": "CD59",
        "mcp": "MCP",
        "daf": "DAF",
        "cr1": "CR1",
        "cr2": "CR2",
        "cr3": "CR3",
        "cr4": "CR4",
        "ic3b": "iC3b",
        "c3a": "C3a",
        "c5a": "C5a",
        "c3adesarg": "C3a des Arg",
        "c5adesarg": "C5a des Arg",
        "c5b-9": "C5b-9",
        "mac": "MAC",
        "terminal complement complex": "TCC",
        "sc5b-9": "sC5b-9",
        "neurophil extracellular trap": "NET",
        "net": "NET",
        "netosis": "NETosis",
        "cit-h3": "CitH3",
        "pad4": "PAD4",
        "neutrophil elastase": "Neutrophil elastase",
        "myeloperoxidase": "MPO",
        "proteinase 3": "PR3",
        "cathepsin g": "Cathepsin G",
        "azurocidin": "Azurocidin",
        "bpi": "BPI",
        "hnp1-3": "HNP1-3",
        "hnp-1": "HNP-1",
        "hnp-2": "HNP-2",
        "hnp-3": "HNP-3",
        "hnp-4": "HNP-4",
        "defensin alpha": "α-defensin",
        "defensin beta": "β-defensin",
        "hbd-1": "HBD-1",
        "hbd-2": "HBD-2",
        "hbd-3": "HBD-3",
        "hbd-4": "HBD-4",
        "rnase 7": "RNase 7",
        "psoriasin": "Psoriasin",
        "s100a7": "S100A7",
    }
    
    concepts = {
        "drugs": [],
        "diseases": [],
        "mechanisms": [],
        "proteins": [],
    }
    
    for pattern, name in drug_patterns.items():
        if pattern in t:
            if name not in concepts["drugs"]:
                concepts["drugs"].append(name)
    
    for pattern, name in disease_patterns.items():
        if pattern in t:
            if name not in concepts["diseases"]:
                concepts["diseases"].append(name)
    
    for pattern, name in mechanism_patterns.items():
        if pattern in t:
            if name not in concepts["mechanisms"]:
                concepts["mechanisms"].append(name)
    
    return concepts


def import_v2_literature():
    """从v2 400篇核心文献中导入LabKG"""
    store = get_graph_store()
    print("="*70)
    print("Project-016 LabKG v2文献导入")
    print("="*70)
    
    # 读取v2文献
    with open(V2_JSON) as f:
        data = json.load(f)
    
    papers = data.get("results", data) if isinstance(data, dict) else data
    total = len(papers)
    print(f"📥 读取 {total} 篇v2核心文献")
    
    added_papers = 0
    added_concepts = {"drug": 0, "disease": 0, "mechanism": 0, "protein": 0}
    added_edges = 0
    
    for paper in papers:
        pmid = paper.get("pmid", "")
        if not pmid:
            continue
        
        paper_id = f"paper:{pmid}"
        
        # 添加/更新paper节点
        if paper_id not in store.graph.nodes:
            store.graph.add_node(
                paper_id,
                node_type="paper",
                title=paper.get("title", ""),
                first_author=paper.get("first_author", ""),
                year=paper.get("year", 0),
                journal=paper.get("journal", ""),
                pmid=pmid,
                doi=paper.get("doi", ""),
                project_ids=[PROJECT_ID],
                source="project-016_v2_screening",
                v2_score=paper.get("_score", 0),
            )
            added_papers += 1
        else:
            # 更新project_ids
            node_data = store.graph.nodes[paper_id]
            if PROJECT_ID not in node_data.get("project_ids", []):
                node_data["project_ids"] = list(set(node_data.get("project_ids", []) + [PROJECT_ID]))
        
        # 从标题提取概念
        concepts = extract_concepts_from_title(paper.get("title", ""))
        
        # 添加concept节点和paper→concept边
        for category, names in concepts.items():
            for name in names:
                concept_id = nid(name)
                
                # 确定concept类别
                if category == "drugs":
                    node_category = "drug"
                elif category == "diseases":
                    node_category = "disease"
                elif category == "mechanisms":
                    node_category = "mechanism"
                else:
                    node_category = "protein"
                
                # 添加concept节点（如果不存在）
                if concept_id not in store.graph.nodes:
                    store.graph.add_node(
                        concept_id,
                        node_type="concept",
                        name=name,
                        category=node_category,
                        project_ids=[PROJECT_ID],
                        source="project-016_v2_concept_extraction",
                        extracted_from=paper_id,
                    )
                    added_concepts[node_category] = added_concepts.get(node_category, 0) + 1
                else:
                    # 更新project_ids
                    node_data = store.graph.nodes[concept_id]
                    if PROJECT_ID not in node_data.get("project_ids", []):
                        node_data["project_ids"] = list(set(node_data.get("project_ids", []) + [PROJECT_ID]))
                
                # 添加paper→concept边
                if not store.graph.has_edge(paper_id, concept_id):
                    store.graph.add_edge(
                        paper_id, concept_id,
                        edge_type="mentions",
                        project_ids=[PROJECT_ID],
                        evidence=f"Extracted from title: {paper.get('title', '')[:80]}...",
                    )
                    added_edges += 1
    
    store.save()
    
    # 统计
    project_nodes = [n for n, d in store.graph.nodes(data=True) 
                     if PROJECT_ID in d.get("project_ids", [])]
    papers_in_kg = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "paper"]
    concepts_in_kg = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "concept"]
    
    print(f"\n{'='*70}")
    print(f"📊 LabKG v2导入完成")
    print(f"{'='*70}")
    print(f"新增paper节点:      {added_papers}")
    print(f"新增concept节点:    {sum(added_concepts.values())} (drug:{added_concepts.get('drug',0)} disease:{added_concepts.get('disease',0)} mechanism:{added_concepts.get('mechanism',0)} protein:{added_concepts.get('protein',0)})")
    print(f"新增paper→concept边: {added_edges}")
    print(f"\nLabKG总计:          {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016总计:    {len(project_nodes)} nodes ({len(papers_in_kg)} papers, {len(concepts_in_kg)} concepts)")
    
    return True


def compute_mechanism_distance_matrix():
    """计算 drug-disease 机制路径距离矩阵"""
    import networkx as nx
    
    store = get_graph_store()
    print("\n" + "="*70)
    print("计算 drug-disease 机制路径距离矩阵")
    print("="*70)
    
    # 获取Project-016的所有drug和disease节点
    project_nodes = [n for n, d in store.graph.nodes(data=True) 
                     if PROJECT_ID in d.get("project_ids", [])]
    
    drugs = [n for n in project_nodes 
             if store.graph.nodes[n].get("category") in ["drug", "treatment", "surgery", "procedure"]]
    diseases = [n for n in project_nodes 
              if store.graph.nodes[n].get("category") in ["disease", "pathology", "outcome", "anatomy"]]
    
    print(f"Project-016 drug节点:     {len(drugs)}")
    print(f"Project-016 disease节点:  {len(diseases)}")
    
    # 只考虑机制层节点（排除paper节点）
    mechanism_nodes = [n for n in project_nodes 
                       if store.graph.nodes[n].get("node_type") == "concept"]
    
    # 构建机制子图（排除paper节点和paper→concept边）
    mechanism_edges = [
        (u, v) for u, v, d in store.graph.edges(data=True)
        if store.graph.nodes[u].get("node_type") == "concept" 
        and store.graph.nodes[v].get("node_type") == "concept"
        and d.get("edge_type") != "mentions"  # 排除paper→concept边
    ]
    
    mech_graph = nx.Graph()
    mech_graph.add_nodes_from(mechanism_nodes)
    mech_graph.add_edges_from(mechanism_edges)
    
    print(f"机制子图:               {mech_graph.number_of_nodes()} nodes, {mech_graph.number_of_edges()} edges")
    
    # 计算drug-disease最短路径
    distance_matrix = {}
    reachable_pairs = 0
    unreachable_pairs = 0
    
    for drug in drugs:
        drug_name = store.graph.nodes[drug].get("name", drug)
        distance_matrix[drug_name] = {}
        
        for disease in diseases:
            disease_name = store.graph.nodes[disease].get("name", disease)
            
            try:
                path_length = nx.shortest_path_length(mech_graph, source=drug, target=disease)
                distance_matrix[drug_name][disease_name] = path_length
                if path_length <= 3:
                    reachable_pairs += 1
                else:
                    unreachable_pairs += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distance_matrix[drug_name][disease_name] = -1  # 不连通
                unreachable_pairs += 1
    
    total_pairs = len(drugs) * len(diseases)
    print(f"\n药物-疾病对总数:        {total_pairs}")
    print(f"路径距离 ≤3 hops:       {reachable_pairs} ({reachable_pairs/total_pairs*100:.1f}%)")
    print(f"路径距离 >3 或不连通:   {unreachable_pairs} ({unreachable_pairs/total_pairs*100:.1f}%)")
    
    # 输出距离分布
    dist_counts = Counter()
    for d_name, d_dict in distance_matrix.items():
        for dis_name, dist in d_dict.items():
            if dist == -1:
                dist_counts["unreachable"] += 1
            elif dist <= 2:
                dist_counts["1-2"] += 1
            elif dist == 3:
                dist_counts["3"] += 1
            elif dist == 4:
                dist_counts["4"] += 1
            else:
                dist_counts["5+"] += 1
    
    print(f"\n距离分布:")
    for k, v in sorted(dist_counts.items(), key=lambda x: (x[0] != "unreachable", x[0])):
        print(f"  {k}: {v} ({v/total_pairs*100:.1f}%)")
    
    # 保存距离矩阵
    output_path = ".tmp/p016_drug_disease_distance_matrix.json"
    with open(output_path, "w") as f:
        json.dump(distance_matrix, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 距离矩阵已保存: {output_path}")
    
    # 输出负样本候选（距离 >3 或不连通）
    neg_candidates = []
    for d_name, d_dict in distance_matrix.items():
        for dis_name, dist in d_dict.items():
            if dist == -1 or dist >= 4:
                neg_candidates.append({
                    "drug": d_name,
                    "disease": dis_name,
                    "mechanism_distance": dist,
                    "negative_type": "mechanism_exclusion",
                    "confidence": "high" if dist == -1 else "medium"
                })
    
    neg_output = ".tmp/p016_negative_candidates_mechanism.json"
    with open(neg_output, "w") as f:
        json.dump(neg_candidates, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 机制排斥负样本候选: {neg_output} ({len(neg_candidates)} pairs)")
    
    return distance_matrix, neg_candidates


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-import", action="store_true", help="跳过文献导入")
    parser.add_argument("--skip-distance", action="store_true", help="跳过距离矩阵计算")
    args = parser.parse_args()
    
    if not args.skip_import:
        import_v2_literature()
    
    if not args.skip_distance:
        compute_mechanism_distance_matrix()
    
    print(f"\n{'='*70}")
    print("✅ LabKG v2更新完成")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
