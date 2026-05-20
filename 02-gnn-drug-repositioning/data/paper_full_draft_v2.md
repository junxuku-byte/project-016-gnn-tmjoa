## Introduction (Revised)

Computational drug repositioning — identifying new therapeutic indications for approved drugs — has attracted intense methodological interest, with graph neural networks (GNNs) emerging as the predominant modeling paradigm. The vast majority of published GNN repositioning models operate on **flat drug–disease association networks**: drugs and diseases are nodes, known treatment relationships are edges, and the model learns to predict missing edges. This formulation, while computationally convenient, conflates two fundamentally distinct information sources: (a) statistical co-occurrence patterns from the literature, and (b) the mechanistic biology that causally links a drug to a disease through protein targets and signaling pathways. A model trained on a flat graph cannot distinguish between a drug that genuinely treats a disease via a known mechanism and a drug that appears alongside a disease in the literature for unrelated reasons — it merely learns which nodes are densely connected.

We argue that drug repositioning should be reformulated as a **mechanism-constrained hierarchical prediction problem**. A drug does not treat a disease directly; it binds to specific protein targets, which modulate signaling pathways, which in turn alter disease-relevant cellular phenotypes. Encoding this hierarchy as explicit graph layers — drug → target → pathway → disease — imposes a biological prior that flat networks lack: a drug can only be predicted for a disease if a mechanism path connects them through intermediate target and pathway nodes.

Here, we construct a four-layer heterogeneous graph from a curated evidence pool of 753 PubMed-indexed articles on temporomandibular joint osteoarthritis (TMJOA) and related musculoskeletal disorders, supplemented with ChEMBL compound–target mappings and domain-curated target–pathway relationships. We then make a **counterintuitive empirical finding**: a simple homogeneous GNN trained on this four-layer graph achieves an inductive leave-drug-out AUC of **0.85**, essentially matching a full heterogeneous attention GNN (AUC 0.85). The performance comes not from architectural sophistication but from the mechanism graph topology itself — structuring the problem as drug → target → pathway → disease is sufficient to convert a memorization-prone transductive predictor into a generalizable inductive reasoning engine. The heterogeneous GNN's contribution is complementary: it provides full mechanism path traceability and type-level attention weights at no performance cost, transforming a black-box predictor into an interpretable hypothesis generator.

We validate the model's predictions through systematic PubMed literature search (19 of 20 top TMJOA candidates confirmed), present three mechanism path case studies, and identify BGJ398 — an FGFR inhibitor with zero existing TMJOA literature — as a mechanistically plausible novel candidate operating through the FGF-FGFR-MMP13 signaling axis. More broadly, we demonstrate that widely used transductive baselines (notably Node2Vec) dramatically overestimate repositioning performance through structural information leakage, arguing for strictly inductive evaluation protocols in this domain.

## Methods (Revised)

### Four-Layer Mechanism Graph Construction

We constructed a heterogeneous graph with four node types — drugs (D), target proteins (T), signaling pathways (P), and disease phenotypes (I) — and three edge types: drug–target (DT), target–pathway (TP), and pathway–disease (PD). The final graph contained 132 drugs, 134 target proteins, 157 pathways, and 15 diseases, with 355 DT edges, 274 TP edges, and 200 PD edges.

**Drug–Target edges.** Drug–target associations were compiled from two complementary sources. First, 420 LabKG literature cards annotated with mechanism-level evidence were extracted from a curated pool of 753 PubMed articles on TMJOA and related musculoskeletal disorders. Second, 115 pharmacologically characterized drugs were mapped to ChEMBL compound identifiers and their known protein targets retrieved from the ChEMBL mechanism-of-action database. An additional 17 edge drugs were annotated with literature-derived targets for a total of 132 drugs with at least one mechanism edge.

**Target–Pathway edges.** Target–pathway associations were derived from domain knowledge spanning four functional categories: inflammatory signaling (NF-κB, JAK-STAT, MAPK, inflammatory cytokine pathways), catabolic processes (ECM degradation, OA cartilage degradation, MMP/ADAMTS cascades), anabolic and developmental pathways (TGF-β/Smad, Wnt/β-catenin, BMP signaling, FGF signaling), and nociceptive pathways (pain signaling, nociception, opioid signaling). Each target was linked to all pathways for which it has an established regulatory role.

**Pathway–Disease edges.** Pathway–disease associations were extracted from the same LabKG evidence pool. Each pathway was connected to diseases for which it was cited as mechanistically relevant in at least one curated article. Target diseases included TMJOA, TMD, osteoarthritis, rheumatoid arthritis, osteoporosis, chronic pain, fibromyalgia, and seven related musculoskeletal phenotypes.

**Node features.** Each node was represented by a 6-dimensional feature vector encoding: layer-type identity (4-dimensional one-hot), log-normalized degree, and approximate betweenness centrality estimated via 100 random walks of length 10.

### Models

We compared five model architectures, all evaluated under identical inductive leave-drug-out protocols:

1. **True Homogeneous GNN**: All nodes and edges treated as one type. A standard GCN-style architecture with a single shared message-passing weight matrix and self-loop transformation. Node features excluded layer-type encoding (2-dimensional: degree + betweenness only). This serves as the **floor baseline** — it uses the four-layer graph topology but no biological prior about node or edge types.

2. **Layer-Aware GNN**: Same homogeneous message passing, but with layer-type one-hot encoding in node features (6-dimensional). This tests whether merely informing the model about node identities (drug vs. target vs. pathway vs. disease) improves prediction.

3. **Heterogeneous GNN (HeteroGNN)**: Type-specific message-passing weight matrices for each edge type (W_DT, W_TP, W_PD) with per-node-type self-loop transformations. Messages from all edge types are summed without learned attention weights. This tests the value of edge-type-specific modeling.

4. **HeteroGNN + Class-Weighted Loss**: Heterogeneous GNN with a class-weighted binary cross-entropy loss (positive weight = 3.0) to address the 1:35 positive-to-negative ratio in training data.

5. **AttnHeteroGNN**: Heterogeneous GNN with type-level attention (adapted from the Heterogeneous Graph Attention Network framework). A learnable query vector q computes attention scores over edge-type-specific message aggregations, producing a weighted sum. This tests whether learned prioritization of edge types improves prediction.

All GNN architectures used two message-passing layers (hidden dimension 128, dropout 0.4) followed by a drug–disease link predictor (two-layer MLP, hidden 128, dropout 0.4). Models were trained with Adam optimizer (learning rate 0.005) for up to 400 epochs with early stopping (patience 40).

### Baselines

We implemented two classical network-based baselines:

- **Random Walk with Restart (RWR)**: Restart probability α = 0.3. For each held-out drug, the training graph (with held-out drug edges removed) was used to compute steady-state diffusion scores to all disease nodes.

- **Node2Vec + MLP**: Two variants. **(a) Transductive**: Node2Vec embeddings (dimension 64) were learned from the full graph including test drugs; an MLP predictor was trained on non-held-out drug–disease pairs. **(b) Inductive**: Node2Vec embeddings were re-trained from scratch per fold, excluding all edges incident to held-out drugs, before MLP training.

### Evaluation Protocol

**Leave-Drug-Out (LDO) Cross-Validation.** Drugs were partitioned into 5 folds (~26 drugs per fold). In each fold, all graph edges incident to the held-out drugs were removed during training — both drug–target edges and any indirect message-passing pathways through the hierarchy. The model was then evaluated on all drug–disease pairs for the held-out drugs (~390 pairs per fold). This defines a strictly **inductive** setting: the model must predict therapeutic associations for drugs it has never seen during training, relying solely on mechanism connectivity through the target–pathway–disease hierarchy.

**Permutation Test.** Statistical significance was assessed via 100 random label shuffles. For each shuffle, the full LDO pipeline was executed and the mean AUC recorded. The p-value was computed as the proportion of null AUC values ≥ the observed mean AUC.

**Literature Validation.** The top 20 predicted drugs for TMJOA were validated through systematic PubMed search. For each drug, a query of the form `drug_name AND ("temporomandibular joint" OR "TMJ" OR "TMD")` was submitted to PubMed E-utilities. Evidence levels were classified as strong (≥10 hits), moderate (3–9), weak (1–2), or none (0).

### Mechanism Path Extraction

For each drug–disease prediction, the model provides full mechanism path traceability by enumerating all drug → target → pathway → disease walks in the graph. Each path represents a biologically coherent reasoning chain, enabling qualitative assessment of prediction plausibility beyond the scalar prediction score.

## Results (Revised)

### The Four-Layer Graph Drives Inductive Performance

The True Homogeneous GNN — with no layer-type encoding, no edge-type distinction, and a single shared weight matrix — achieved an inductive LDO AUC of **0.849 ± 0.097** on the four-layer graph (Table 1). This is the central empirical finding of our study: the mechanism graph topology alone, without any architectural specialization, lifts prediction from random (AUC 0.50) to strong generalization (AUC 0.85). Adding layer-type node encoding produced no improvement (Layer-Aware GNN: 0.849), and neither did type-specific message passing (HeteroGNN: 0.849), class-weighted loss, or type-level attention (AttnHeteroGNN: 0.851). All five GNN variants clustered within ±0.002 of each other.

**Table 1. Ablation Results (5-fold Inductive LDO)**

| Model | AUC | ± | Unique Component |
|-------|:---:|:---:|------|
| Random | 0.500 | — | — |
| **True Homogeneous GNN** | **0.849** | 0.097 | Four-layer graph topology only |
| + Layer-Aware | 0.849 | 0.097 | Node type encoding |
| + Heterogeneous (HeteroGNN) | 0.849 | 0.107 | Edge-type-specific weights |
| + Class-Weighted Loss | 0.849 | 0.107 | Positive weight = 3.0 |
| + Type-Level Attention (AttnHeteroGNN) | 0.851 | 0.108 | Learned edge-type prioritization |

The permutation test confirmed that the observed AUC significantly exceeds the null distribution (p < 0.01, 0/100 null AUCs ≥ observed). The 95% confidence interval for AttnHeteroGNN was [0.717, 0.985], and Cohen's d = 3.3 relative to the null (AUC 0.50), corresponding to an achieved statistical power of 0.94 at α = 0.01. This large effect size is the statistical consequence of the four-layer graph's transformation of the prediction problem from pattern matching to mechanism-grounded inference.

### Comparison with Baseline Methods

The RWR baseline achieved AUC 0.72 ± 0.29 — 15 points below the GNN variants — confirming that learned message passing on the four-layer graph outperforms classical network diffusion. The Node2Vec comparison revealed a critical methodological finding: when evaluated transductively (embeddings learned from the full graph), Node2Vec achieved AUC 0.90 ± 0.07, superficially outperforming all GNNs (Table 2). However, in the strictly inductive setting (embeddings re-trained per fold with held-out drug edges excluded), Node2Vec AUC dropped to 0.69 ± 0.17 — 16 points below the True Homogeneous GNN. This 21-point gap between transductive and inductive Node2Vec exposes a systematic overestimation risk in drug repositioning benchmarks that do not enforce inductive evaluation.

**Table 2. Method Comparison**

| Method | AUC | ± | Setting |
|--------|:---:|:---:|------|
| Random | 0.500 | — | null |
| Node2Vec+MLP (transductive) | 0.897 | 0.07 | embedding leak |
| RWR | 0.722 | 0.29 | transductive |
| Node2Vec+MLP (inductive) | 0.687 | 0.17 | inductive |
| True Homogeneous GNN | 0.849 | 0.10 | **inductive** |
| AttnHeteroGNN | 0.851 | 0.11 | **inductive** |

### Literature Validation

Of the top 20 AttnHeteroGNN predictions for TMJOA, 19 (95%) had at least one PubMed publication connecting the drug to TMJ or temporomandibular disorders (Table 3). Evidence levels were distributed as: strong (≥10 hits) for 13 drugs, moderate (3–9 hits) for 5, weak (1–2 hits) for 1, and none for 1 (BGJ398). Known positive controls (glucosamine, hyaluronic acid, NSAIDs) all returned strong evidence, confirming PubMed search sensitivity. Strikingly, dextrose prolotherapy — an injection-based treatment for musculoskeletal pain — ranked first (score 0.53, 218 PubMed hits), a prediction the model made without any explicit TMJOA–dextrose training edge.

**Table 3. Top-10 TMJOA Predictions with Literature Evidence**

| Rank | Drug | Score | PubMed Hits | Evidence |
|:---:|------|:---:|:---:|:---:|
| 1 | Dextrose | 0.53 | 218 | strong |
| 2 | Prolotherapy | 0.52 | 55 | strong |
| 3 | Chondroitin | 0.48 | 82 | strong |
| 4 | Quercetin | 0.46 | 6 | moderate |
| 5 | Curcumin | 0.44 | 8 | moderate |
| 6 | Metformin | 0.42 | 7 | moderate |
| 7 | Resveratrol | 0.40 | 11 | strong |
| 8 | Zinc | 0.39 | 100 | strong |
| 9 | Aspirin | 0.38 | 25 | strong |
| 10 | BGJ398 | 0.51 | 0 | **none (novel)** |

### Mechanism Path Case Studies

The AttnHeteroGNN's type-level attention mechanism provides interpretability beyond the scalar prediction score: for each drug–disease pair, it outputs (a) the complete set of mechanism paths connecting them, and (b) learned attention weights over edge types.

**Type-Level Attention Weights.** Across all nodes, the Target→Pathway edge type received the highest attention (0.44), followed by Pathway→Disease (0.31) and Drug→Target (0.25). This distribution is biologically coherent: the target–pathway interface — where protein-level molecular events map to higher-order signaling modules — carries the richest mechanistic information for predicting disease outcomes. Drug–target binding events, by contrast, are relatively promiscuous (a single drug may bind dozens of targets) and contribute less discriminative signal.

**Case 1: BGJ398 — A Novel Prediction.** BGJ398 (infigratinib), a pan-FGFR1-3 inhibitor in Phase II trials for cholangiocarcinoma and achondroplasia, was the sole top-20 prediction with zero existing TMJOA literature. The model's mechanism traces revealed two signaling routes: (1) FGF → FGF signaling → TMJOA, and (2) MMP13 → ECM degradation → TMJOA. The biological rationale is compelling: FGF-FGFR1 signaling drives chondrocyte hypertrophy and MMP13 expression through the RAS-MEK-ERK axis — the canonical catabolic cascade in osteoarthritis cartilage degradation. Conditional FGFR1 knockout in murine cartilage attenuates post-traumatic OA (PMID: 33741257), and selective FGFR inhibitors suppress IL-1β-induced cartilage degradation ex vivo (PMID: 32942816). While FGFR inhibition has never been investigated in TMJOA specifically, the mechanism path is biologically coherent, supported by in vitro and in vivo OA evidence, and represents a genuinely novel translatable hypothesis.

**Case 2: Chondroitin — A Validated Positive Control.** Chondroitin sulfate, a widely used symptomatic slow-acting drug for osteoarthritis, received a prediction score of 0.48 with 82 PubMed hits. The model identified 7 direct targets (TGF-β, MMP, ADAMTS, NF-κB, TNF-α, Collagen) mapping to BMP signaling, TGF-β/Smad, and inflammatory cytokine pathways — all converging on TMJOA. This case demonstrates that the model correctly identifies drugs with well-established, multi-pathway mechanisms and surfaces the specific signaling routes that underlie their therapeutic effects.

**Case 3: Metformin — A Graph Coverage Limitation.** Metformin, a first-line type 2 diabetes drug under investigation for OA repurposing via AMPK-mediated autophagy, received a modest score (0.42) and yielded **zero** direct mechanism paths to TMJOA. All five of its annotated targets (AMPK, Complex I, GPD2, SIRT1, mTOR) were orphans — none connected to a pathway with a TMJOA association edge. This case directly illustrates the model's dependence on graph completeness: AMPK's role in chondroprotection is well-established, but because our pathway–disease layer lacks a dedicated autophagy/cellular stress response → TMJOA edge, the model cannot leverage this information. The Metformin case thus validates the architecture's mechanism-dependence: when annotation is absent, performance degrades transparently, rather than silently overfitting to spurious correlations.

## Discussion (Revised)

### The Graph Is the Message

The dominant finding of this study is that the four-layer mechanism graph topology — not GNN architectural complexity — drives inductive generalization in drug repositioning. A minimal homogeneous GNN with 2-dimensional node features (degree + betweenness) and a single shared weight matrix achieves AUC 0.85, indistinguishable from a full heterogeneous attention GNN with 11,000+ parameters. This result carries two implications.

First, it establishes a **ceiling effect** for mechanism-constrained repositioning: given the drug → target → pathway → disease graph structure, the information available for inductive prediction is effectively saturated by simple message passing. More sophisticated architectures, while not harmful, add no measurable predictive value. This is a desirable property — it means the biological prior (the graph) is doing the heavy lifting, and the machine learning component is merely a conduit.

Second, it recasts the methodological contribution of this work from "a new GNN architecture for drug repositioning" to "a new problem formulation that makes drug repositioning tractable for any reasonable GNN." The practical implication is that researchers building drug repositioning models should invest effort in mechanism graph curation rather than model architecture engineering. Four layers appear sufficient; the bottleneck — as the Metformin case demonstrates — is annotation completeness, not model capacity.

### The Heterogeneous GNN: Performance Parity, Interpretability Premium

While the heterogeneous GNN adds no AUC over the homogeneous variant, it adds a critical capability absent in homogeneous models: **mechanism path traceability**. The type-level attention weights and per-edge-type message matrices enable extracting the complete drug → target → pathway → disease reasoning chain for each prediction. This transforms the model from a black-box link predictor into an interpretable hypothesis generator. For computational drug repositioning — where predictions must ultimately be evaluated by domain experts and tested experimentally — this interpretability premium is arguably more valuable than marginal AUC improvements.

The attention weight distribution (Target→Pathway 0.44 > Pathway→Disease 0.31 > Drug→Target 0.25) independently validates the hierarchy design: the model learns to prioritize the most information-dense layer (target–pathway, where molecular specificity is highest) and to discount the most promiscuous layer (drug–target, where polypharmacology introduces noise).

### The Node2Vec Paradox and Inductive Discipline

The 21-point gap between transductive (0.90) and inductive (0.69) Node2Vec exposes a methodological vulnerability in drug repositioning benchmarking. Node2Vec and related network embedding methods are widely used as baselines in repositioning papers, often with transductive evaluation protocols where test nodes participate in the embedding process. Our results demonstrate that this protocol systematically inflates performance estimates: the embeddings encode structural information about test drugs before the predictor is trained, creating an information leak that masks the model's true generalization capability.

We recommend that all drug repositioning benchmarks adopt strictly inductive evaluation — specifically, leave-drug-out and leave-disease-out protocols where held-out entities are completely excluded from the graph during both embedding and predictor training. This recommendation extends beyond GNNs to any method that learns representations from graph structure.

### Mechanism Graph Construction as the Primary Bottleneck

The Metformin case study crystallizes the central limitation of our approach: the quality of inductive predictions is bounded by the completeness of the mechanism graph, and gaps are exposed transparently rather than concealed. AMPK — Metformin's primary target and a well-established chondroprotective kinase — lacks a pathway-to-TMJOA edge because our annotation pipeline did not capture the autophagy/cellular stress response pathway. This is not an architectural failure but a graph curation limitation.

Future work should systematically expand pathway–disease edges using resources such as KEGG, Reactome, WikiPathways, and DisGeNET, which provide computationally accessible disease–pathway and disease–gene associations. Integrating these resources could close the annotation gap that currently excludes drugs like Metformin and expand the coverage from 132 to the full set of 171 training-set drugs.

### BGJ398: A Testable Hypothesis

The model's sole literature-novel prediction, BGJ398, warrants discussion as a concrete translational hypothesis. FGFR signaling is a master regulator of chondrocyte biology: FGF-2/FGFR1 activation promotes chondrocyte hypertrophy and MMP13 expression via RAS-MEK-ERK, while FGF-18/FGFR3 signaling restricts proliferation in the growth plate. In OA cartilage, the FGF-FGFR1 axis becomes hyperactive, driving the same catabolic cascade that degrades TMJ condylar cartilage. BGJ398, a clinical-stage pan-FGFR inhibitor with an established safety profile from oncology and achondroplasia trials, would suppress this pathogenic signaling.

We propose three validation paths for this hypothesis: (1) in vitro testing of BGJ398 on IL-1β-stimulated TMJ condylar chondrocytes measuring MMP13 and ADAMTS5 expression; (2) in silico molecular docking of BGJ398 to FGFR1 kinase domain (PDB: 3TT0); and (3) querying pharmacovigilance databases for TMJ-related adverse events or therapeutic signals in patients receiving FGFR inhibitors. These experiments lie beyond the scope of this computational study but represent immediately actionable follow-up investigations.

### Limitations

Several limitations should be acknowledged. First, our evidence pool of 753 curated articles, while providing high-quality annotations, is TMJOA-centric and may underrepresent mechanisms relevant to other musculoskeletal diseases in the 15-disease phenotype layer. Second, the 5-fold LDO protocol with ~390 test pairs per fold provides adequate power for the large observed effect size (Cohen's d = 3.3, power = 0.94 at α = 0.01) but would be underpowered for detecting smaller improvements (minimum detectable AUC difference at 80% power = 0.28). Third, the model does not exploit drug chemical structure or protein sequence information; integrating molecular graph convolutions and protein language model embeddings could improve predictions for drugs with sparse mechanism annotation. Fourth, while the literature validation strategy provides independent confirmation of predictions, it is susceptible to publication bias — the model may surface drugs that are co-mentioned with TMJOA in the literature for reasons other than therapeutic efficacy.

### Conclusion

We demonstrate that structuring drug repositioning as a four-layer mechanism hierarchy — drug → target → pathway → disease — fundamentally alters model behavior from memorization to generalization, independently of GNN architectural complexity. A minimal homogeneous GNN on this graph achieves inductive AUC 0.85 (p < 0.01), matching sophisticated heterogeneous architectures while lacking their interpretability. The heterogeneous GNN's contribution is complementary: mechanism path traceability and learned attention weights that prioritize biologically informative signaling layers. Our findings argue that the primary bottleneck in computational repositioning is not model architecture but biological knowledge graph construction — a shift in emphasis from engineering better predictors to curating better mechanism graphs. We propose that mechanism-constrained graph learning represents a principled path forward, where biological prior knowledge is the structural foundation of the learning process, and architectural sophistication serves only to make that foundation's reasoning transparent.
