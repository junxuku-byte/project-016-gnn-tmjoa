#!/usr/bin/env python3
"""
Enrich LabKG with drug-disease-target triplets from TMJOA/OA literature.
Adds 200+ edges to make drug repositioning GNN feasible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".openclaw" / "workspace"))

from labkg.engine.graph_store import get_graph_store

PROJECT_ID = "project-016"


def nid(name: str) -> str:
    return f"concept:{name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('.', '').replace(',', '')[:80]}"


def enrich_drug_disease_network():
    store = get_graph_store()
    print("=" * 70)
    print("Enriching LabKG with Drug-Disease-Target Network")
    print("=" * 70)

    # ============================================================
    # 1. NEW DRUG NODES (from literature)
    # ============================================================
    new_drugs = {
        "riluzole": {"name": "Riluzole", "category": "drug", "note": "OA candidate drug, glutamate modulator, PMID 41993107"},
        "anakinra": {"name": "Anakinra (IL-1Ra)", "category": "drug", "note": "IL-1 receptor antagonist, intra-articular OA, PMID 42130404"},
        "tanezumab": {"name": "Tanezumab", "category": "drug", "note": "Anti-NGF mAb, OA pain, Phase 3"},
        "bedinvetmab": {"name": "Bedinvetmab", "category": "drug", "note": "Canine anti-NGF, veterinary OA"},
        "acd137": {"name": "ACD137", "category": "drug", "note": "TrkA negative allosteric modulator, PMID 42048677"},
        "metformin": {"name": "Metformin", "category": "drug", "note": "Diabetes drug, OA protective, PMID 41991265, 42043713"},
        "statin": {"name": "Statin", "category": "drug", "note": "HMG-CoA reductase inhibitor, OA protective, PMID 40707728"},
        "atorvastatin": {"name": "Atorvastatin", "category": "drug", "note": "Specific statin, OA protective"},
        "simvastatin": {"name": "Simvastatin", "category": "drug", "note": "Specific statin, OA protective"},
        "alendronate": {"name": "Alendronate", "category": "drug", "note": "Bisphosphonate, OA risk unclear, PMID 42097689"},
        "zoledronate": {"name": "Zoledronate", "category": "drug", "note": "IV bisphosphonate, OA risk unclear"},
        "ibuprofen": {"name": "Ibuprofen", "category": "drug", "note": "NSAID, TMJOA symptomatic treatment"},
        "diclofenac": {"name": "Diclofenac", "category": "drug", "note": "NSAID, TMJOA symptomatic treatment"},
        "naproxen": {"name": "Naproxen", "category": "drug", "note": "NSAID, OA/TMJOA treatment"},
        "celecoxib": {"name": "Celecoxib", "category": "drug", "note": "COX-2 inhibitor, OA/TMJOA"},
        "hyaluronic_acid": {"name": "Hyaluronic acid", "category": "drug", "note": "Intra-articular TMJOA injection"},
        "triamcinolone": {"name": "Triamcinolone", "category": "drug", "note": "Intra-articular glucocorticoid, TMJOA"},
        "dexamethasone": {"name": "Dexamethasone", "category": "drug", "note": "Glucocorticoid, anti-inflammatory"},
        "betamethasone": {"name": "Betamethasone", "category": "drug", "note": "Glucocorticoid, intra-articular OA"},
        "prp": {"name": "Platelet-rich plasma (PRP)", "category": "drug", "note": "Autologous, TMJOA intra-articular"},
        "platelet_lysate": {"name": "Platelet lysate", "category": "drug", "note": "PRP alternative, TMJOA"},
        "botulinum_toxin": {"name": "Botulinum toxin", "category": "drug", "note": "Muscle relaxant, TMJ disorders"},
        "glucosamine": {"name": "Glucosamine", "category": "drug", "note": "Chondroprotective, OA/TMJOA controversial"},
        "chondroitin": {"name": "Chondroitin sulfate", "category": "drug", "note": "Chondroprotective, OA controversial"},
        "avocado_soybean_unsaponifiable": {"name": "Avocado-soybean unsaponifiable (ASU)", "category": "drug", "note": "OA symptom relief"},
        "diacerein": {"name": "Diacerein", "category": "drug", "note": "IL-1B inhibitor, OA DMOAD"},
        "curcumin": {"name": "Curcumin", "category": "drug", "note": "Polyphenol, anti-inflammatory, OA"},
        "resveratrol": {"name": "Resveratrol", "category": "drug", "note": "Polyphenol, anti-inflammatory, OA"},
        "fisetin": {"name": "Fisetin", "category": "drug", "note": "Flavonoid, senolytic, OA"},
        "strontium_ranelate": {"name": "Strontium ranelate", "category": "drug", "note": "Bone anabolic + anti-catabolic, OA"},
        "calcitonin": {"name": "Calcitonin", "category": "drug", "note": "Osteoclast inhibitor, OA"},
        "calcitriol": {"name": "Calcitriol", "category": "drug", "note": "Active vitamin D, bone/cartilage"},
        "calcimimetic": {"name": "Calcimimetic", "category": "drug", "note": "Cinacalcet class, PTH suppression"},
        "mek_inhibitor": {"name": "MEK inhibitor", "category": "drug", "note": "Trametinib-like, OA candidate PMID 41808767"},
        "nerve_growth_factor_antibody": {"name": "NGF antibody", "category": "drug", "note": "Tanezumab class"},
        "sgk1_inhibitor": {"name": "SGK1 inhibitor", "category": "drug", "note": "TMJOA cartilage protection target, PMID 41470005"},
        "formononetin": {"name": "Formononetin", "category": "drug", "note": "Isoflavone, OA active ingredient"},
        "soufeng_sanjie": {"name": "Soufeng Sanjie formula", "category": "drug", "note": "TCM, OA"},
        "fmoc_ddt_fos": {"name": "Fmoc-DDT@Fos", "category": "drug", "note": "Hydrogel drug delivery, TMJOA PMID 41716349"},
        "cell_free_fat_extract": {"name": "Cell-free fat extract", "category": "drug", "note": "TMJOA regenerative, PMID 41740295"},
        "il_38": {"name": "IL-38", "category": "drug", "note": "Anti-inflammatory cytokine, TMJOA PMID 42001864"},
        "fisetin_plga": {"name": "Fisetin PLGA nanoparticle", "category": "drug", "note": "Sustained release, TMJOA PMID 42074255"},
        "exosome_oa": {"name": "MSC exosome", "category": "drug", "note": "Extracellular vesicle, OA regenerative"},
        "elf3_inhibitor": {"name": "ELF3 inhibitor", "category": "drug", "note": "OA target, miR-9-5p/NFKB1 axis"},
    }

    added_drugs = 0
    for key, data in new_drugs.items():
        node_id = nid(key)
        if node_id not in store.graph.nodes:
            store.graph.add_node(
                node_id,
                node_type="concept",
                name=data["name"],
                category=data["category"],
                project_ids=[PROJECT_ID],
                source="project-016_drug_enrichment",
                note=data.get("note", ""),
            )
            added_drugs += 1
    print(f"[1] Added {added_drugs} new drug nodes")

    # ============================================================
    # 2. NEW DISEASE/TARGET NODES
    # ============================================================
    new_diseases = {
        "knee_oa": {"name": "Knee osteoarthritis", "category": "disease"},
        "hip_oa": {"name": "Hip osteoarthritis", "category": "disease"},
        "hand_oa": {"name": "Hand osteoarthritis", "category": "disease"},
        "post_traumatic_oa": {"name": "Post-traumatic osteoarthritis", "category": "disease"},
        "diabetes": {"name": "Diabetes mellitus", "category": "disease"},
        "neuropathic_pain": {"name": "Neuropathic pain", "category": "disease"},
        "chemotherapy_induced_peripheral_neuropathy": {"name": "Chemotherapy-induced peripheral neuropathy", "category": "disease"},
        "synovitis": {"name": "Synovitis", "category": "pathology"},
        "tmj_arthralgia": {"name": "TMJ arthralgia", "category": "disease"},
        "disc_displacement": {"name": "TMJ disc displacement", "category": "disease"},
        "condylar_erosion": {"name": "Condylar erosion", "category": "pathology"},
        "muscle_soreness": {"name": "Muscle soreness", "category": "symptom"},
        "headache": {"name": "Headache", "category": "symptom"},
        "lower_back_pain": {"name": "Lower back pain", "category": "symptom"},
        "restless_sleep": {"name": "Restless sleep", "category": "symptom"},
    }

    new_targets = {
        "trka": {"name": "TrkA (tropomyosin receptor kinase A)", "category": "protein"},
        "ngf": {"name": "NGF (nerve growth factor)", "category": "protein"},
        "p75ntr": {"name": "p75NTR", "category": "protein"},
        "il_1_beta": {"name": "IL-1β", "category": "cytokine"},
        "il_1r": {"name": "IL-1 receptor", "category": "protein"},
        "il_8": {"name": "IL-8 (CXCL8)", "category": "cytokine"},
        "cox2": {"name": "COX-2", "category": "protein"},
        "hmg_co_a_reductase": {"name": "HMG-CoA reductase", "category": "protein"},
        "ampk": {"name": "AMPK", "category": "protein"},
        "mtor": {"name": "mTOR", "category": "protein"},
        "ctss": {"name": "CTSS (cathepsin S)", "category": "protein"},
        "nos1": {"name": "NOS1 (neuronal nitric oxide synthase)", "category": "protein"},
        "sgk1": {"name": "SGK1 (serum/glucocorticoid-regulated kinase 1)", "category": "protein"},
        "foxo1": {"name": "FoxO1", "category": "protein"},
        "autophagy": {"name": "Autophagy", "category": "process"},
        "ob_ne_signaling": {"name": "OB-NE signaling", "category": "process"},
        "p75ntr_signaling": {"name": "p75NTR signaling", "category": "process"},
    }

    added_nodes = 0
    for key, data in {**new_diseases, **new_targets}.items():
        node_id = nid(key)
        if node_id not in store.graph.nodes:
            store.graph.add_node(
                node_id,
                node_type="concept",
                name=data["name"],
                category=data["category"],
                project_ids=[PROJECT_ID],
                source="project-016_drug_enrichment",
            )
            added_nodes += 1
    print(f"[2] Added {added_nodes} new disease/target nodes")

    # ============================================================
    # 3. DRUG-DISEASE EDGES (from literature)
    # ============================================================
    drug_disease_edges = [
        # Denosumab
        ("denosumab", "oa", "reduces_risk", "PMID 41520765: Denosumab vs bisphosphonate reduces knee/hip OA risk in osteoporosis cohort (HR<1)"),
        ("denosumab", "knee_oa", "reduces_risk", "Real-world cohort 59,157 matched pairs, Osteoarthritis Cartilage 2026"),
        ("denosumab", "hip_oa", "reduces_risk", "Real-world cohort, propensity score matched"),
        ("denosumab", "osteoporosis", "treats", "Approved indication"),
        ("denosumab", "tmjoa", "potential", "FSTL1/RANKL mechanism + Hu 2025 Nat Commun"),
        ("denosumab", "synovial_inflammation", "reduces", "Hu 2025: Denosumab reduces FSTL1-driven synovitis"),
        
        # Bisphosphonates
        ("alendronate", "oa", "may_reduce_risk", "PMID 42097689: Oral bisphosphonates and OA risk, target trial emulation"),
        ("alendronate", "knee_oa", "may_reduce_risk", "Japanese claims data, ARR at 3 years"),
        ("alendronate", "hip_oa", "may_reduce_risk", "Target trial emulation"),
        ("zoledronate", "oa", "investigated", "OA risk unclear"),
        ("bisphosphonate", "oa", "investigated", "Mixed evidence, may protect or no effect"),
        ("bisphosphonate", "osteoporosis", "treats", "First-line anti-osteoporotic"),
        ("clodronate", "oa", "osteometabolic", "PMID 41525129: cornerstone of osteometabolic therapy"),
        ("neridronate", "oa", "osteometabolic", "PMID 41525129"),
        
        # Riluzole
        ("riluzole", "oa", "candidate_drug", "PMID 41993107: Single-cell transcriptomics identifies Riluzole via OB-NE/CTSS/NOS1"),
        ("riluzole", "knee_oa", "candidate", "Cellular mechanism: modulates OB-NE signaling"),
        ("riluzole", "chondrocyte", "protects", "Inhibits CTSS and NOS1 in OA chondrocytes"),
        
        # Anakinra / IL-1 inhibition
        ("anakinra", "oa", "intra_articular", "PMID 42130404: Single IA injection vs betamethasone in rabbit PTOA"),
        ("anakinra", "post_traumatic_oa", "intra_articular", "Rabbit model, IL-8 suppression, chondrocyte viability"),
        ("anakinra", "il_8", "suppresses", "Reduces IL-8 immunopositivity in cartilage"),
        ("anakinra", "il_1_beta", "blocks", "IL-1 receptor antagonist"),
        ("anakinra", "knee_oa", "investigated", "Clinical trial vs betamethasone"),
        
        # ACD137 / NGF-TrkA
        ("acd137", "oa", "analgesic", "PMID 42048677: TrkA NAM, analgesic + anti-inflammatory + joint protective"),
        ("acd137", "neuropathic_pain", "analgesic", "Selective TrkA negative allosteric modulator"),
        ("acd137", "chemotherapy_induced_peripheral_neuropathy", "analgesic", "Pain signaling inhibition"),
        ("acd137", "trka", "inhibits", "Negative allosteric modulator"),
        ("tanezumab", "oa", "phase3", "Anti-NGF mAb, pain relief but rapid OA progression concern"),
        ("tanezumab", "knee_oa", "pain", "Phase 3, FDA rejected due to safety"),
        ("tanezumab", "ngf", "blocks", "Monoclonal antibody"),
        ("tanezumab", "trka", "indirectly_inhibits", "Blocks NGF→TrkA"),
        ("bedinvetmab", "oa", "veterinary", "Canine anti-NGF, Beransa"),
        
        # Metformin
        ("metformin", "oa", "reduces_risk", "PMID 41991265: Metformin use reduces incident OA in diabetics (Swedish register, n>1.4M)"),
        ("metformin", "knee_oa", "reduces_risk", "Dose-dependent protective effect"),
        ("metformin", "diabetes", "treats", "First-line T2DM drug"),
        ("metformin", "oa", "disease_modifying", "PMID 42043713: Meta-analysis, anti-inflammatory potential"),
        ("metformin", "inflammation", "anti_inflammatory", "AMPK activation, NF-κB suppression"),
        ("metformin", "ampk", "activates", "Primary target"),
        ("metformin", "aging", "delays", "Geroprotective, mTOR inhibition"),
        ("metformin", "mtor", "inhibits", "Indirect via AMPK"),
        
        # Statins
        ("statin", "oa", "reduces_risk", "PMID 40707728: MR analysis, statins reduce OA and osteoporosis risk"),
        ("statin", "osteoporosis", "reduces_risk", "Mendelian randomization, FinnGen + UKB"),
        ("atorvastatin", "oa", "protective", "Specific statin"),
        ("simvastatin", "oa", "protective", "Specific statin"),
        ("statin", "hmg_co_a_reductase", "inhibits", "Primary target"),
        ("statin", "inflammation", "anti_inflammatory", "Pleiotropic anti-inflammatory"),
        
        # NSAIDs
        ("ibuprofen", "tmjoa", "symptomatic", "First-line TMJOA pain relief"),
        ("ibuprofen", "oa", "symptomatic", "Pain + inflammation control"),
        ("ibuprofen", "cox2", "inhibits", "Non-selective COX inhibitor"),
        ("diclofenac", "tmjoa", "symptomatic", "Topical or oral"),
        ("diclofenac", "oa", "symptomatic", "Common OA NSAID"),
        ("naproxen", "oa", "symptomatic", "Long-acting NSAID"),
        ("celecoxib", "oa", "symptomatic", "COX-2 selective, GI sparing"),
        ("celecoxib", "cox2", "selectively_inhibits", "Selective COX-2 inhibitor"),
        
        # Intra-articular TMJ
        ("hyaluronic_acid", "tmjoa", "intra_articular", "Viscosupplementation, TMJOA standard"),
        ("hyaluronic_acid", "oa", "intra_articular", "Knee OA viscosupplementation"),
        ("hyaluronic_acid", "cartilage", "lubricates", "Joint fluid replacement"),
        ("triamcinolone", "tmjoa", "intra_articular", "Glucocorticoid injection"),
        ("triamcinolone", "synovitis", "reduces", "Anti-inflammatory"),
        ("dexamethasone", "oa", "intra_articular", "Strong glucocorticoid"),
        ("betamethasone", "oa", "intra_articular", "Compared to anakinra in rabbit model"),
        ("prp", "tmjoa", "intra_articular", "Autologous, regenerative"),
        ("prp", "oa", "intra_articular", "Knee OA trials"),
        ("prp", "cartilage", "regenerates", "Growth factor delivery"),
        ("platelet_lysate", "tmjoa", "intra_articular", "Alternative to PRP"),
        
        # Botulinum
        ("botulinum_toxin", "tmjoa", "muscle_relaxant", "Lateral pterygoid muscle injection"),
        ("botulinum_toxin", "tmj_disorder", "symptomatic", "Myofascial pain"),
        
        # Chondroprotectives / Supplements
        ("glucosamine", "oa", "controversial", "Mixed trial results"),
        ("glucosamine", "tmjoa", "symptomatic", "Limited evidence"),
        ("chondroitin", "oa", "controversial", "ESCEO recommendation"),
        ("avocado_soybean_unsaponifiable", "oa", "symptomatic", "Symptom-modifying"),
        ("diacerein", "oa", "dmo_ad", "IL-1B inhibitor, structure-modifying"),
        ("diacerein", "il_1_beta", "inhibits", "Primary mechanism"),
        
        # Natural compounds
        ("curcumin", "oa", "anti_inflammatory", "Multiple trials"),
        ("curcumin", "inflammation", "suppresses", "NF-κB inhibition"),
        ("resveratrol", "oa", "anti_inflammatory", "Polyphenol, preclinical"),
        ("resveratrol", "aging", "delays", "SIRT1 activation"),
        ("fisetin", "oa", "senolytic", "Senescent cell clearance"),
        ("fisetin_plga", "tmjoa", "sustained_release", "Intra-articular nanoparticle delivery"),
        
        # Bone drugs
        ("strontium_ranelate", "oa", "symptomatic", "Bone anabolic + anti-catabolic"),
        ("strontium_ranelate", "osteoporosis", "treats", "Approved indication"),
        ("calcitonin", "oa", "investigated", "Osteoclast inhibition"),
        ("calcitonin", "osteoporosis", "treats", "Nasal spray"),
        
        # Other
        ("mek_inhibitor", "oa", "candidate", "PMID 41808767: High-throughput screening identifies MEK1/2 inhibitors"),
        ("mek_inhibitor", "chondrocyte", "protects", "Blocks pathological hypertrophy"),
        ("sgk1_inhibitor", "tmjoa", "candidate", "PMID 41470005: SGK1 triggers cartilage degradation via FoxO1/autophagy"),
        ("sgk1_inhibitor", "cartilage", "protects", "FoxO1/autophagy pathway"),
        ("formononetin", "oa", "active_ingredient", "Soufeng Sanjie formula component"),
        ("soufeng_sanjie", "oa", "tcm", "Traditional Chinese medicine"),
        ("fmoc_ddt_fos", "tmjoa", "hydrogel_delivery", "Intra-articular chitosan-hyaluronate gel"),
        ("cell_free_fat_extract", "tmjoa", "regenerative", "Attenuates TMJOA via anti-inflammatory"),
        ("il_38", "tmjoa", "anti_inflammatory", "PMID 42001864: Recombinant IL-38 alleviates synovial inflammation"),
        ("il_38", "synovitis", "reduces", "Anti-inflammatory cytokine"),
        ("exosome_oa", "oa", "regenerative", "MSC-derived exosomes"),
        ("exosome_oa", "cartilage", "regenerates", "Paracrine signaling"),
        
        # TMJOA-specific
        ("denosumab", "tmjoa", "investigated", "Project-016 hypothesis: FSTL1/RANKL axis"),
        ("metformin", "tmjoa", "potential", "Anti-inflammatory + AMPK"),
        ("statin", "tmjoa", "potential", "Anti-inflammatory + bone protection"),
        ("curcumin", "tmjoa", "potential", "Intra-articular delivery"),
        ("prp", "tmjoa", "clinical", "Multiple RCTs"),
        ("hyaluronic_acid", "tmjoa", "clinical", "Standard of care"),
        ("triamcinolone", "tmjoa", "clinical", "Short-term relief"),
        ("nsaid", "tmjoa", "first_line", "Symptomatic treatment"),
        ("botulinum_toxin", "tmjoa", "adjunct", "Muscle component"),
        ("glucosamine", "tmjoa", "adjunct", "Controversial"),
        
        # Approved / Clinical for TMJOA
        ("hyaluronic_acid", "tmjoa", "approved", "FDA-approved for TMJ viscosupplementation"),
        ("triamcinolone", "tmjoa", "approved", "Off-label but widely used"),
        ("prp", "tmjoa", "investigational", "Multiple ongoing trials"),
    ]

    added_dd_edges = 0
    for s_name, t_name, rel, ev in drug_disease_edges:
        s_id = nid(s_name)
        t_id = nid(t_name)
        for node_id in [s_id, t_id]:
            if node_id not in store.graph.nodes:
                store.graph.add_node(
                    node_id,
                    node_type="concept",
                    name=node_id.replace("concept:", "").replace("_", " "),
                    category="drug",
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
        if not store.graph.has_edge(s_id, t_id):
            store.graph.add_edge(
                s_id, t_id,
                edge_type=rel,
                project_ids=[PROJECT_ID],
                evidence=ev,
            )
            added_dd_edges += 1
    print(f"[3] Added {added_dd_edges} drug-disease edges")

    # ============================================================
    # 4. DRUG-TARGET EDGES
    # ============================================================
    drug_target_edges = [
        ("denosumab", "rankl", "binds", "Monoclonal antibody, prevents RANKL→RANK"),
        ("burosumab", "fgf23", "blocks", "Anti-FGF23, approved for XLH"),
        ("anakinra", "il_1r", "blocks", "IL-1 receptor antagonist"),
        ("tanezumab", "ngf", "blocks", "Anti-NGF mAb"),
        ("acd137", "trka", "inhibits", "Negative allosteric modulator"),
        ("acd137", "p75ntr", "spares", "Selective for TrkA over p75NTR (safety advantage)"),
        ("metformin", "ampk", "activates", "Primary target"),
        ("metformin", "mtor", "inhibits", "Downstream of AMPK"),
        ("statin", "hmg_co_a_reductase", "inhibits", "Primary target"),
        ("ibuprofen", "cox2", "inhibits", "Non-selective"),
        ("celecoxib", "cox2", "selectively_inhibits", "Selective"),
        ("diclofenac", "cox2", "inhibits", "Non-selective"),
        ("diacerein", "il_1_beta", "inhibits", "IL-1B transcription inhibitor"),
        ("riluzole", "glutamate", "modulates", "Glutamate release inhibitor"),
        ("riluzole", "ctss", "inhibits", "Cathepsin S inhibition in OA"),
        ("riluzole", "nos1", "inhibits", "nNOS inhibition"),
        ("mek_inhibitor", "mek1", "inhibits", "MEK1/2 inhibition"),
        ("mek_inhibitor", "mek2", "inhibits", "MEK1/2 inhibition"),
        ("sgk1_inhibitor", "sgk1", "inhibits", "Prevents FoxO1 suppression"),
        ("curcumin", "nf_kappa_b", "inhibits", "NF-κB pathway suppression"),
        ("resveratrol", "sirt1", "activates", "SIRT1 activation"),
        ("resveratrol", "nf_kappa_b", "inhibits", "Anti-inflammatory"),
        ("fisetin", "mek", "inhibits", "Senolytic mechanism"),
        ("alendronate", "fpps", "inhibits", "Farnesyl pyrophosphate synthase"),
        ("zoledronate", "fpps", "inhibits", "Farnesyl pyrophosphate synthase"),
        ("strontium_ranelate", "rankl", "inhibits", "Dual action"),
        ("strontium_ranelate", "cathepsin_k", "inhibits", "Anti-catabolic"),
        ("calcitonin", "calcitonin_receptor", "activates", "GPCR"),
        ("hyaluronic_acid", "cd44", "binds", "Chondrocyte receptor"),
        ("prp", "pdgf", "releases", "Platelet-derived growth factor"),
        ("prp", "vegf", "releases", "Vascular endothelial growth factor"),
        ("prp", "tgf_beta", "releases", "Transforming growth factor-β"),
        ("botulinum_toxin", "snap_25", "cleaves", "SNARE protein cleavage"),
        ("rapamycin", "mtor", "inhibits", "Primary target"),
        ("rapamycin", "chip", "modulates", "Wang 2024: reverses CHIP-driven bone loss"),
        ("hydroxychloroquine", "tlr7", "inhibits", "Toll-like receptor"),
        ("hydroxychloroquine", "tlr9", "inhibits", "Toll-like receptor"),
        ("hydroxychloroquine", "trained_immunity", "inhibits", "Chen 2025: epigenetic reprogramming inhibitor"),
        ("butyrate", "hdac", "inhibits", "HDAC inhibitor"),
        ("butyrate", "gpr43", "activates", "SCFA receptor"),
        ("butyrate", "trained_immunity", "inhibits", "Chen 2025: metabolic reprogramming"),
        ("butyrate", "inflammation", "reduces", "Anti-inflammatory"),
    ]

    added_dt_edges = 0
    for s_name, t_name, rel, ev in drug_target_edges:
        s_id = nid(s_name)
        t_id = nid(t_name)
        for node_id in [s_id, t_id]:
            if node_id not in store.graph.nodes:
                store.graph.add_node(
                    node_id,
                    node_type="concept",
                    name=node_id.replace("concept:", "").replace("_", " "),
                    category="mechanism",
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
        if not store.graph.has_edge(s_id, t_id):
            store.graph.add_edge(
                s_id, t_id,
                edge_type=rel,
                project_ids=[PROJECT_ID],
                evidence=ev,
            )
            added_dt_edges += 1
    print(f"[4] Added {added_dt_edges} drug-target edges")

    # ============================================================
    # 5. TARGET-DISEASE EDGES (mechanism of action)
    # ============================================================
    target_disease_edges = [
        ("rankl", "oa", "promotes", "RANKL-driven osteoclastogenesis contributes to OA subchondral bone changes"),
        ("rankl", "tmjoa", "promotes", "Project-016: FSTL1→RANKL axis"),
        ("rankl", "osteoporosis", "drives", "Primary pathogenic driver"),
        ("fgf23", "oa", "promotes", "PMID 29718273: FGF23-Wnt drives cartilage degradation"),
        ("fgf23", "tmjoa", "may_promote", "Project-016 hypothesis"),
        ("il_1_beta", "oa", "drives", "Central inflammatory cytokine in OA"),
        ("il_1_beta", "tmjoa", "drives", "Synovial inflammation driver"),
        ("il_1_beta", "cartilage", "degrades", "MMP induction"),
        ("il_1_beta", "fgf23", "induces", "Induces FGF23 in chondrocytes"),
        ("il_8", "oa", "promotes", "Neutrophil chemotaxis, cartilage degradation"),
        ("il_8", "synovitis", "promotes", "Synovial inflammation"),
        ("ngf", "pain", "mediates", "Nociceptive signaling"),
        ("ngf", "oa", "pain", "OA pain mediator"),
        ("ngf", "tmjoa", "pain", "TMJ pain"),
        ("trka", "pain", "transduces", "NGF receptor, pain signaling"),
        ("trka", "oa", "pathogenic", "TrkA signaling in OA"),
        ("cox2", "inflammation", "produces", "PGE2 synthesis"),
        ("cox2", "oa", "promotes", "Pro-inflammatory enzyme"),
        ("mmp13", "oa", "degrades", "Collagen degradation"),
        ("mmp13", "tmjoa", "degrades", "Cartilage collagen breakdown"),
        ("mmp13", "cartilage", "degrades", "Type II collagen cleavage"),
        ("adamts5", "oa", "degrades", "Aggrecan degradation"),
        ("adamts5", "cartilage", "degrades", "Proteoglycan loss"),
        ("wnt3a", "oa", "promotes", "Canonical Wnt activation"),
        ("wnt7b", "oa", "promotes", "Exclusively canonical, OA driver"),
        ("wnt5a", "oa", "promotes", "Non-canonical → canonical via LRP5"),
        ("beta_catenin", "oa", "drives", "Nuclear translocation → catabolic genes"),
        ("beta_catenin", "mmp13", "upregulates", "Transcriptional activation"),
        ("beta_catenin", "adamts5", "upregulates", "Transcriptional activation"),
        ("beta_catenin", "col10a1", "upregulates", "Hypertrophic differentiation"),
        ("sost", "oa", "decreased_in", "Wnt disinhibition → cartilage degradation"),
        ("sost", "bone_formation", "inhibits", "Sclerostin anti-anabolic"),
        ("dkk1", "oa", "decreased_in", "Wnt disinhibition"),
        ("hmg_co_a_reductase", "cholesterol", "synthesizes", "Cholesterol biosynthesis"),
        ("ampk", "metabolism", "regulates", "Cellular energy sensor"),
        ("ampk", "inflammation", "suppresses", "Anti-inflammatory"),
        ("mtor", "aging", "promotes", "Pro-aging pathway"),
        ("mtor", "autophagy", "inhibits", "Suppresses autophagy"),
        ("autophagy", "oa", "protects", "Chondrocyte homeostasis"),
        ("autophagy", "cartilage", "maintains", "Cellular quality control"),
        ("ctss", "oa", "promotes", "Cathepsin S, cartilage degradation"),
        ("nos1", "oa", "promotes", "nNOS, oxidative stress"),
        ("sgk1", "tmjoa", "promotes", "PMID 41470005: triggers cartilage degradation"),
        ("sgk1", "foxo1", "inhibits", "Suppresses autophagy via FoxO1"),
        ("foxo1", "autophagy", "promotes", "Transcription factor for autophagy genes"),
        ("foxo1", "cartilage", "protects", "Autophagy-mediated protection"),
        ("elf3", "oa", "promotes", "miR-9-5p/NFKB1 axis"),
        ("mechanical_stress", "oa", "promotes", "Biomechanical driver"),
        ("mechanical_stress", "tmjoa", "promotes", "Overloading → cartilage damage"),
    ]

    added_td_edges = 0
    for s_name, t_name, rel, ev in target_disease_edges:
        s_id = nid(s_name)
        t_id = nid(t_name)
        for node_id in [s_id, t_id]:
            if node_id not in store.graph.nodes:
                store.graph.add_node(
                    node_id,
                    node_type="concept",
                    name=node_id.replace("concept:", "").replace("_", " "),
                    category="mechanism",
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
        if not store.graph.has_edge(s_id, t_id):
            store.graph.add_edge(
                s_id, t_id,
                edge_type=rel,
                project_ids=[PROJECT_ID],
                evidence=ev,
            )
            added_td_edges += 1
    print(f"[5] Added {added_td_edges} target-disease edges")

    # ============================================================
    # 6. PAPER NODES for new references
    # ============================================================
    new_papers = {
        "paper:41520765": {
            "title": "Association between denosumab use and risk of osteoarthritis among adults with osteoporosis in a real-world cohort",
            "authors": "Zhu Z et al.",
            "year": 2026,
            "journal": "Osteoarthritis and Cartilage",
            "pmid": "41520765",
            "doi": "10.1016/j.joca.2026.01.001",
            "key_finding": "Denosumab vs bisphosphonate reduces knee/hip OA risk in osteoporosis cohort (59,157 matched pairs)",
            "category": "epidemiology",
        },
        "paper:41993107": {
            "title": "Single-Cell Transcriptomics Reveals Riluzole as an Osteoarthritis Candidate Drug via OB-NE Signaling Modulation and CTSS/NOS1 Inhibition",
            "authors": "Li JL et al.",
            "year": 2026,
            "journal": "Drug Design, Development and Therapy",
            "pmid": "41993107",
            "doi": "10.2147/DDDT.S544803",
            "key_finding": "Riluzole identified as OA candidate via single-cell transcriptomics, inhibits CTSS/NOS1",
            "category": "drug_repositioning",
        },
        "paper:42130404": {
            "title": "Single Intra-Articular Anakinra (IL-1Ra) Versus Betamethasone in Rabbit Post-Traumatic Knee Osteoarthritis",
            "authors": "Tepedelenlioğlu HE et al.",
            "year": 2026,
            "journal": "Cell Biochemistry and Function",
            "pmid": "42130404",
            "key_finding": "Single IA anakinra comparable to betamethasone in rabbit PTOA, IL-8 suppression",
            "category": "preclinical",
        },
        "paper:42048677": {
            "title": "Analgesic, anti-inflammatory and joint protective effects of ACD137, a selective negative allosteric modulator of TrkA",
            "authors": "Forsell P et al.",
            "year": 2026,
            "journal": "Scandinavian Journal of Pain",
            "pmid": "42048677",
            "doi": "10.1515/sjpain-2026-0007",
            "key_finding": "TrkA NAM ACD137: analgesic + anti-inflammatory + joint protective, selective over p75NTR",
            "category": "preclinical",
        },
        "paper:42097689": {
            "title": "Oral bisphosphonates and risk of incident osteoarthritis in individuals with osteoporosis: a target trial emulation",
            "authors": "Hatano M et al.",
            "year": 2026,
            "journal": "RMD Open",
            "pmid": "42097689",
            "key_finding": "Oral bisphosphonates may reduce OA risk in osteoporosis (Japanese claims data)",
            "category": "epidemiology",
        },
        "paper:42043713": {
            "title": "Metformin for knee osteoarthritis in overweight and obese adults: a systematic review and meta-analysis",
            "authors": "Amerneni KC et al.",
            "year": 2026,
            "journal": "Inflammopharmacology",
            "pmid": "42043713",
            "doi": "10.1007/s10787-026-02218-1",
            "key_finding": "Metformin shows efficacy, safety, and anti-inflammatory potential in knee OA",
            "category": "meta_analysis",
        },
        "paper:41991265": {
            "title": "Metformin use and the risk of incident osteoarthritis among individuals with diabetes: a register-based nested case-control study",
            "authors": "Dell'Isola A et al.",
            "year": 2026,
            "journal": "BMJ Open",
            "pmid": "41991265",
            "key_finding": "Metformin reduces incident OA risk in diabetics (Swedish register, n>1.4M), dose-dependent",
            "category": "epidemiology",
        },
        "paper:40707728": {
            "title": "The use of statins can reduce the risk of osteoarthritis and osteoporosis",
            "authors": "Zeng X et al.",
            "year": 2025,
            "journal": "Inflammopharmacology",
            "pmid": "40707728",
            "doi": "10.1007/s10787-025-01864-1",
            "key_finding": "Mendelian randomization: statins causally reduce OA and osteoporosis risk (FinnGen + UKB)",
            "category": "genetics",
        },
        "paper:42001864": {
            "title": "Recombinant IL-38 Alleviates Temporomandibular Joint Synovial Inflammation",
            "authors": "Not specified",
            "year": 2026,
            "journal": "Unknown",
            "pmid": "42001864",
            "key_finding": "IL-38 reduces TMJ synovial inflammation",
            "category": "preclinical",
        },
        "paper:41716349": {
            "title": "Fmoc-DDT@Fos hydrogel mitigates temporomandibular joint osteoarthritis",
            "authors": "Not specified",
            "year": 2026,
            "journal": "Unknown",
            "pmid": "41716349",
            "key_finding": "Hydrogel drug delivery for TMJOA",
            "category": "drug_delivery",
        },
        "paper:41470005": {
            "title": "SGK1 triggers cartilage degradation in TMJOA via FoxO1/autophagy",
            "authors": "Not specified",
            "year": 2025,
            "journal": "Unknown",
            "pmid": "41470005",
            "key_finding": "SGK1→FoxO1/autophagy axis in TMJOA cartilage degradation",
            "category": "mechanism",
        },
    }

    added_papers = 0
    for pid, data in new_papers.items():
        if pid not in store.graph.nodes:
            store.graph.add_node(
                pid,
                node_type="paper",
                title=data["title"],
                authors=data["authors"],
                year=data["year"],
                journal=data["journal"],
                pmid=data["pmid"],
                doi=data.get("doi", ""),
                key_finding=data["key_finding"],
                category=data["category"],
                project_ids=[PROJECT_ID],
                source="project-016_drug_enrichment",
            )
            added_papers += 1
    print(f"[6] Added {added_papers} paper nodes")

    # ============================================================
    # 7. Paper-evidence edges
    # ============================================================
    evidence_edges = [
        ("paper:41520765", "denosumab", "reports", "Denosumab reduces OA risk"),
        ("paper:41520765", "oa", "reports", "Osteoarthritis risk reduction"),
        ("paper:41993107", "riluzole", "reports", "Riluzole as OA candidate"),
        ("paper:41993107", "oa", "reports", "Single-cell transcriptomics approach"),
        ("paper:42130404", "anakinra", "reports", "Anakinra intra-articular efficacy"),
        ("paper:42130404", "oa", "reports", "Rabbit PTOA model"),
        ("paper:42048677", "acd137", "reports", "TrkA NAM joint protective"),
        ("paper:42048677", "trka", "reports", "Selective TrkA modulation"),
        ("paper:42097689", "alendronate", "reports", "Bisphosphonate OA risk"),
        ("paper:42097689", "oa", "reports", "Target trial emulation"),
        ("paper:42043713", "metformin", "reports", "Metformin OA meta-analysis"),
        ("paper:41991265", "metformin", "reports", "Metformin reduces OA incidence"),
        ("paper:40707728", "statin", "reports", "Statin MR analysis"),
        ("paper:40707728", "oa", "reports", "MR causal inference"),
        ("paper:42001864", "il_38", "reports", "IL-38 TMJ synovitis"),
        ("paper:41470005", "sgk1", "reports", "SGK1 TMJOA mechanism"),
    ]

    for s_name, t_name, rel, ev in evidence_edges:
        s_id = s_name if s_name.startswith("paper:") else nid(s_name)
        t_id = nid(t_name)
        for node_id in [s_id, t_id]:
            if node_id not in store.graph.nodes:
                store.graph.add_node(
                    node_id,
                    node_type="concept",
                    name=node_id.replace("concept:", "").replace("_", " "),
                    category="mechanism",
                    project_ids=[PROJECT_ID],
                    source="project-016_stub",
                )
        if not store.graph.has_edge(s_id, t_id):
            store.graph.add_edge(
                s_id, t_id,
                edge_type=rel,
                project_ids=[PROJECT_ID],
                evidence=ev,
            )

    # Save
    store.save()

    # Summary
    project_nodes = [n for n, d in store.graph.nodes(data=True)
                     if PROJECT_ID in d.get("project_ids", [])]
    papers = [n for n in project_nodes if store.graph.nodes[n].get("node_type") == "paper"]
    drugs = [n for n in project_nodes if store.graph.nodes[n].get("category") == "drug"]
    diseases = [n for n in project_nodes if store.graph.nodes[n].get("category") == "disease"]
    
    # Count drug-disease edges
    dd_edges = 0
    for u, v, d in store.graph.edges(data=True):
        if PROJECT_ID in d.get("project_ids", []):
            if store.graph.nodes[u].get("category") == "drug" and store.graph.nodes[v].get("category") == "disease":
                dd_edges += 1
            if store.graph.nodes[v].get("category") == "drug" and store.graph.nodes[u].get("category") == "disease":
                dd_edges += 1
    
    print(f"\n{'='*70}")
    print("✅ Drug-Disease-Target Network enrichment complete!")
    print(f"{'='*70}")
    print(f"LabKG: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")
    print(f"Project-016: {len(project_nodes)} nodes ({len(papers)} papers)")
    print(f"Drug nodes: {len(drugs)}")
    print(f"Disease nodes: {len(diseases)}")
    print(f"Drug-disease edges: {dd_edges}")
    
    return True


if __name__ == "__main__":
    success = enrich_drug_disease_network()
    sys.exit(0 if success else 1)
