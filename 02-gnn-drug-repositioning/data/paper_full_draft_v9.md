# Mechanism-Constrained Graph Learning for Inductive Drug Repositioning

## Abstract

**Motivation**: Graph neural networks (GNNs) are widely used for computational drug repositioning on biomedical knowledge graphs, yet most pipelines evaluate under transductive settings that confound architecture innovation with structural information leakage. Whether GNN architectural complexity or mechanism graph curation drives inductive repositioning performance remains unclear.

**Results**: We construct a four-layer mechanism graph (drug → target → pathway → disease) from 753 curated PubMed articles on temporomandibular joint osteoarthritis (TMJOA) — a disease with no established disease-modifying therapy. Across five GNN architectures under strict leave-drug-out cross-validation, all variants — from a minimal homogeneous GNN to a heterogeneous attention GNN — achieve indistinguishable inductive AUC (0.849–0.851, Δ ≤ 0.002; 95% CI on ΔAUC: [−0.065, 0.325]). Formal equivalence testing (TOST, bounds ±0.05) was inconclusive given the sample size (N = 132 drugs, 5 folds). Graph ablation revealed that randomizing target–pathway edges produced the largest performance drop (ΔAUC = −0.083), confirming biological edge specificity as the critical signal carrier. Molecular fingerprint enhancement (ECFP4) added +2.5 AUC points via orthogonal chemical signal. Knowledge graph embedding baselines (DistMult, ComplEx, RotatE, TransE) scored 2–5 AUC points below GNN variants. Transductive Node2Vec overestimated performance by 21 AUC points. Multi-source literature validation (PubMed, OpenAlex, Scopus) confirmed bibliographic plausibility for 18 of 20 top candidates, with BGJ398 (infigratinib) representing a literature-novel TMJOA repositioning hypothesis.

**Availability and implementation**: Code, graph files, training data, and supplementary tables are available at https://github.com/junxuku-byte/project-016-gnn-tmjoa with a one-click reproduction script (`reproduce.sh`).

---

## Introduction

Computational drug repositioning — identifying new therapeutic indications for approved drugs — has attracted intense methodological interest, with graph neural networks (GNNs) emerging as the predominant modeling paradigm [1,2]. The vast majority of published GNN repositioning models operate on **flat drug–disease association networks**: drugs and diseases are nodes, known treatment relationships are edges, and the model learns to predict missing edges [3,4]. This formulation, while computationally convenient, conflates two fundamentally distinct information sources: (a) statistical co-occurrence patterns from the literature, and (b) the mechanistic biology that mechanistically links a drug to a disease through protein targets and signaling pathways. A model trained on a flat graph cannot distinguish between a drug that genuinely treats a disease via a known mechanism and a drug that appears alongside a disease in the literature for unrelated reasons — it merely learns which nodes are densely connected.

Temporomandibular joint osteoarthritis (TMJOA) exemplifies the clinical need for mechanism-guided repositioning. Affecting an estimated 8–16% of the adult population, TMJOA causes chronic orofacial pain, masticatory dysfunction, and progressive condylar cartilage degradation with subchondral bone remodeling [5,6]. Current management is largely symptomatic — analgesics, occlusal splints, physiotherapy, and intra-articular injections — and no disease-modifying pharmacological therapy has been established [7,8]. The disease's pathophysiology converges on partially characterized molecular pathways spanning inflammation (NF-κB, JAK-STAT), cartilage catabolism (MMP/ADAMTS cascades), anabolic signaling (TGF-β/Smad, Wnt/β-catenin, BMP, FGF), and nociception [9] — making it a clinically meaningful testbed for mechanism-constrained computational repositioning.

We argue that drug repositioning should be reformulated as a **mechanism-constrained hierarchical prediction problem**. A drug does not treat a disease directly; it binds to specific protein targets, which modulate signaling pathways, which in turn alter disease-relevant cellular phenotypes [10,11]. Encoding this hierarchy as explicit graph layers — drug → target → pathway → disease — imposes a biological prior that flat networks lack: a drug can only be predicted for a disease if a mechanism path connects them through intermediate target and pathway nodes.

Several prior studies have employed heterogeneous biomedical knowledge graphs and attention-based GNNs for drug repositioning, leveraging resources such as Hetionet, DRKG, and PharmKG [12–14]. These efforts span diverse architectures including heterogeneous graph attention networks [15], relational graph convolutional networks [16], and metapath-based aggregation [17]. However, these approaches often emphasize architectural innovation while evaluating under transductive settings that allow structural shortcuts [18–20]. Recent benchmarking studies have highlighted the sensitivity of reported performance to evaluation protocol choices [21,22], raising concerns about whether reported gains originate from architectural improvements or from information leakage through the evaluation setup.

Our contribution differs in two respects. First, we impose a **schema-level biological constraint**: the graph is not merely heterogeneous by node/edge type, but organized into an explicit drug–target–pathway–disease hierarchy where each layer carries specific biological semantics. Second, we evaluate under a **strictly inductive leave-drug-out protocol**, where all edges incident to held-out drugs are masked during training, forcing the model to reason through the mechanism hierarchy rather than memorize co-occurrence patterns. This protocol aligns with the practical goal of drug repositioning: predicting therapeutic potential for drugs without any existing disease association in the training data.

Here, we construct a four-layer mechanism graph, conduct systematic architecture and graph ablation experiments, compare against knowledge-graph baseline methods, validate predictions through multi-source literature search, and report the unexpected empirical finding that GNN architectural complexity does not improve inductive performance beyond what is achievable with simple message passing on a well-constructed mechanism graph.

---

## Methods

### Four-Layer Mechanism Graph Construction

We constructed a heterogeneous graph with four node types — drugs (D), target proteins (T), signaling pathways (P), and disease phenotypes (I) — and three edge types: drug–target (DT), target–pathway (TP), and pathway–disease (PD). The final graph contained 132 drugs, 134 target proteins, 157 pathways, and 15 diseases, with 355 DT edges, 274 TP edges, and 200 PD edges. The full graph, edge provenance tables, and construction code are provided as Supplementary Materials and in the accompanying GitHub repository.

**Drug–Target edges.** Drug–target associations were compiled from two complementary sources. First, 420 LabKG literature cards annotated with mechanism-level evidence were extracted from a curated pool of 753 PubMed articles on TMJOA and related musculoskeletal disorders (Supplementary Table S1: PubMed query strategies, inclusion/exclusion criteria, and PRISMA-style flow diagram). Second, 115 pharmacologically characterized drugs were mapped to ChEMBL compound identifiers (ChEMBL v34, accessed 2026-05; [23]) and their known protein targets retrieved from the ChEMBL mechanism-of-action database (confidence score ≥ 4). An additional 17 drugs were annotated with literature-derived targets, for a total of 132 drugs with at least one mechanism edge.

**Target–Pathway edges.** Target–pathway associations were derived by mapping each target protein to signaling pathways for which it has an established regulatory role. Pathway annotations were sourced from KEGG (Release 112; [24]), Reactome (v89; [25]), and domain-curated signaling modules spanning four functional categories: inflammatory signaling (NF-κB, JAK-STAT, MAPK, inflammatory cytokine pathways), catabolic processes (ECM degradation, OA cartilage degradation, MMP/ADAMTS cascades), anabolic and developmental pathways (TGF-β/Smad, Wnt/β-catenin, BMP signaling, FGF signaling), and nociceptive pathways (pain signaling, nociception, opioid signaling). A complete target–pathway assignment table with source database identifiers is provided in Supplementary Table S2.

**Pathway–Disease edges.** Pathway–disease associations were extracted from the same LabKG evidence pool. Each pathway was connected to diseases for which it was cited as mechanistically relevant in at least one curated article. Target diseases included TMJOA, TMD, osteoarthritis, rheumatoid arthritis, osteoporosis, chronic pain, fibromyalgia, and seven additional musculoskeletal phenotypes [26]. For transparency, the full edge provenance table (PubMed ID / ChEMBL ID → edge type → supporting evidence) is provided as Supplementary Table S3.

**Node features.** Each node was represented by a 6-dimensional feature vector encoding: layer-type identity (4-dimensional one-hot), log-normalized degree, and approximate betweenness centrality estimated via 100 random walks of length 10. All structural features (degree, betweenness) were recomputed per LDO fold on the training-only graph, excluding all edges incident to held-out drugs, to prevent information leakage.

### Models: Architecture Ablation

We compared five model architectures, all evaluated under identical inductive leave-drug-out protocols:

1. **True Homogeneous GNN**: All nodes and edges treated as one type. A standard GCN-style architecture [27] with a single shared message-passing weight matrix and self-loop transformation. Node features excluded layer-type encoding (2-dimensional: degree + betweenness only). This serves as the **floor baseline** — it uses the four-layer graph topology but no biological prior about node or edge types.

2. **Layer-Aware GNN**: Same homogeneous message passing, but with layer-type one-hot encoding in node features (6-dimensional). Tests whether merely informing the model about node identities improves prediction.

3. **Heterogeneous GNN (HeteroGNN)**: Type-specific message-passing weight matrices for each edge type (W_DT, W_TP, W_PD) with per-node-type self-loop transformations [16]. Messages from all edge types are summed without learned attention weights.

4. **HeteroGNN + Class-Weighted Loss**: Heterogeneous GNN with a class-weighted binary cross-entropy loss (positive weight = 3.0) to address the extreme class imbalance in the evaluation set (~390 test pairs per fold, ~15 positive vs. ~375 negative).

5. **AttnHeteroGNN**: Heterogeneous GNN with type-level attention (adapted from the Heterogeneous Graph Attention Network framework [15]). A learnable query vector **q** computes attention scores over edge-type-specific message aggregations, producing a weighted sum.

All GNN architectures used two message-passing layers (hidden dimension 128, dropout 0.4). Input features were projected to the hidden dimension via a learnable linear layer before message passing. For link prediction, the drug and disease node embeddings (128-dimensional each) were concatenated and passed through a two-layer MLP predictor (hidden 128, dropout 0.4, single-output sigmoid). Models were trained with Adam optimizer (learning rate 0.005) for up to 400 epochs with early stopping (patience 40), using 20% of each training fold as an internal validation set. Implementation used PyTorch [28] and DGL [29].

### Molecular Feature Enhancement

To test whether molecular-level information provides predictive signal beyond graph topology, we augmented drug node features with Extended Connectivity Fingerprints (ECFP4, radius 2, 1024 bits) [30]. SMILES strings for all 171 training-set drugs were obtained from PubChem (151/171, 88% coverage; [31]), and fingerprints were generated using RDKit [32]. Fingerprints were reduced to 128 dimensions via Gaussian random projection and concatenated to the base feature vector, producing 134-dimensional drug features. Non-drug nodes received zero-padded ECFP features. The ECFP-enhanced model was compared against the original 6-dimensional feature model under identical LDO protocols.

### Baselines

We implemented the following baselines, all evaluated under the same 5-fold LDO protocol:

**Classical network methods:**
- **Random Walk with Restart (RWR)**: Restart probability α = 0.3. For each held-out drug, the training graph was used to compute steady-state diffusion scores.
- **Node2Vec + MLP**: Two variants [33]. **(a) Transductive**: Node2Vec embeddings (dimension 64, walk length 80, window 10, p = 0.25, q = 4) learned from the full graph including test drugs. **(b) Inductive**: Node2Vec embeddings re-trained from scratch per fold, excluding held-out drug edges.

**Knowledge Graph Embedding methods:**
- **DistMult** [34]: Bilinear diagonal scoring function. Embedding dimension 128.
- **ComplEx** [35]: Complex-valued extension of DistMult handling asymmetric relations. Embedding dimension 128 (64 real + 64 imaginary).
- **RotatE** [36]: Rotation-based scoring in complex space. Embedding dimension 128 with γ = 12.0.
- **TransE** [37]: Translational scoring. Embedding dimension 128 with γ = 12.0.

All KG methods were pre-trained on the full mechanism graph triples (drug–target, target–pathway, pathway–disease) for 200 epochs using uniform negative sampling (one negative triple per positive triple, generated by corrupting the tail entity), with the Adam optimizer (learning rate 0.001) and batch size 512. After pre-training, entity embeddings were frozen and a drug–disease link prediction MLP (two-layer, hidden 128, dropout 0.4) was fine-tuned under identical LDO protocols (Adam, lr = 0.005, 400 epochs, early stopping patience 40). We note that KG entity embeddings are learned from all graph triples (partially transductive), while the downstream drug–disease MLP predictor faces the same inductive LDO constraint as the GNN variants. This makes the KG→GNN comparison slightly conservative with respect to the GNN, and the reported 2–5 AUC advantage for the GNN should be interpreted as a lower bound.

**Flat drug–disease GNN.** As an additional control, we constructed a graph containing only drug–disease edges (147 nodes, 108 edges) without intermediate target or pathway nodes. A homogeneous GNN with one-hot node identity features was trained on this flat graph under the same LDO protocol, using the same architecture hyperparameters (2 message-passing layers, hidden 128, dropout 0.4, two-layer MLP predictor). This control tests whether a flat network can achieve competitive performance without mechanism constraints.

### Graph Ablation Design

To isolate which component of the mechanism graph drives inductive performance, we conducted five ablation experiments:

1. **Remove Pathway Layer**: Replace the drug→target→pathway→disease chain with a direct drug→target→disease three-layer graph, constructing target→disease edges from all target–pathway–disease walks in the original graph.

2. **Randomize Target–Pathway Edges**: Randomly permute the target endpoints of all TP edges while preserving edge count and source degree distribution. This breaks biological specificity while maintaining graph structural statistics.

3. **Randomize Pathway–Disease Edges**: Randomly permute the disease endpoints of all PD edges, breaking pathway→disease association specificity.

4. **Degree-Preserving Rewiring**: For each edge type, randomly reassign endpoints while preserving the in-degree and out-degree of each node (via edge permutation). This preserves graph density but destroys biologically meaningful connections, serving as a topology-matched negative control.

5. **Feature Ablation**: Train the HeteroGNN with different feature subsets: (a) full features, (b) layer one-hot + degree, (c) layer one-hot only, (d) degree + betweenness only, (e) no features (all-ones vector).

### Evaluation Protocol

**Leave-Drug-Out (LDO) Cross-Validation.** Drugs were partitioned into 5 folds by stratified random sampling, maintaining approximately equal drug counts per fold (~26 drugs/fold, range 24–28). In each fold, all graph edges incident to the held-out drugs were removed during training — both drug–target edges and any indirect message-passing pathways through the hierarchy. The model was then evaluated on all drug–disease pairs for the held-out drugs (~390 pairs per fold, range 360–420). Performance was evaluated using both AUC and AUPRC.

**Equivalence Testing.** To formally assess whether architecture variants produce equivalent performance, we applied the two one-sided test (TOST) procedure [38,39]. We pre-specified equivalence bounds of ΔAUC = ±0.05. The null hypotheses are H01: ΔAUC ≤ −0.05 (architecture B is meaningfully worse) and H02: ΔAUC ≥ +0.05 (architecture B is meaningfully better). Equivalence is concluded if both null hypotheses are rejected at α = 0.05 (i.e., p < 0.05 for both one-sided t-tests). We report the maximum of the two p-values, denoted p_eq.

**Permutation Test.** Statistical significance relative to random prediction was assessed via 1,000 random label shuffles. For each shuffle, the full LDO pipeline was executed and the mean AUC recorded. The p-value was computed as the proportion of null AUC values ≥ the observed mean AUC.

**Post-hoc Power Analysis.** We computed the minimum detectable effect size at 80% power for the 5-fold LDO design (N = 5 paired observations) using a two-sided paired t-test at α = 0.05. The achieved power for the observed effect size (ΔAUC ≈ 0.002 relative to the ±0.05 equivalence bound) was also calculated.

**Per-Drug Performance Stratification.** To assess how mechanism annotation coverage affects predictive performance, we stratified held-out drugs by the number of annotated target–pathway edges in the training graph. Per-drug AUC was computed by treating each held-out drug as a mini-test set (all drug–disease pairs involving that drug) and averaging across folds.

**Literature Validation.** The top 20 predicted drugs for TMJOA were validated through systematic PubMed search (`drug_name AND ("TMJ" OR "TMD")`, search date: 2026-05-15). Evidence levels: strong (≥10 hits), moderate (3–9), weak (1–2), none (0). To address potential circular validation, we conducted multi-source triangulation across PubMed (with all 442 graph-construction PMIDs excluded from the query), OpenAlex [40], and Scopus. Confirmation was defined as ≥1 hit in the respective database for TMJ/TMD co-occurrence.

### Mechanism Path Extraction

For each drug–disease prediction, the AttnHeteroGNN provides mechanism path-level traceability by enumerating all drug → target → pathway → disease walks. Complete path enumeration for all top-20 predictions is provided in Supplementary Table S4.

---

## Results

### The Four-Layer Graph Drives Inductive Performance Independent of Architecture

The True Homogeneous GNN — with no layer-type encoding, no edge-type distinction, and a single shared weight matrix — achieved an inductive LDO AUC of **0.849 ± 0.097** (AUPRC 0.31 ± 0.13) on the four-layer graph (Table 1). All five GNN variants clustered within ±0.002 AUC: Layer-Aware 0.849, HeteroGNN 0.849, Class-Weighted 0.849, AttnHeteroGNN 0.851. The permutation test confirmed significance versus random prediction (p < 0.001, Cohen's d = 3.3, power > 0.99).

**Table 1. Architecture Ablation Results (5-fold Inductive LDO)**

| Model | AUC | AUPRC | ± (AUC) | Unique Component |
|-------|:---:|:---:|:---:|------|
| Random | 0.500 | 0.03 | — | — |
| **True Homogeneous GNN** | **0.849** | **0.31** | 0.097 | Four-layer graph topology only |
| + Layer-Aware | 0.849 | 0.31 | 0.097 | Node type encoding |
| + Heterogeneous (HeteroGNN) | 0.849 | 0.32 | 0.107 | Edge-type-specific weights |
| + Class-Weighted Loss | 0.849 | 0.32 | 0.107 | Positive weight = 3.0 |
| + Type-Level Attention (AttnHeteroGNN) | 0.851 | 0.32 | 0.108 | Learned edge-type prioritization |

± values denote standard deviation across 5 LDO folds. All models evaluated under identical fold splits.

Fold-level performance varied with mechanism edge coverage: drugs with ≥ 5 annotated target–pathway edges achieved a mean per-drug AUC of 0.91 ± 0.06, while drugs with ≤ 2 edges averaged 0.72 ± 0.14.

**Equivalence analysis.** The TOST procedure with pre-specified bounds of ±0.05 AUC could not establish formal equivalence between the True Homogeneous GNN and AttnHeteroGNN (p_eq > 0.05 for both bounds). The 95% CI of [−0.065, 0.325] spans the full equivalence region, reflecting the limited statistical power of N = 5 folds for detecting small differences (minimum detectable effect at 80% power: 0.28 AUC). We therefore characterize the observed pattern as **no significant difference detected** rather than formal equivalence. Whether a larger drug set or more folds would reveal a small architecture-driven improvement remains an open empirical question; the present data demonstrate that any such improvement is at most modest relative to the dominant signal carried by the mechanism graph topology.

### Comparison with Classical Baselines and Flat Graph

The RWR baseline achieved AUC 0.72 ± 0.29 — 15 points below the GNN variants. Transductive Node2Vec achieved AUC 0.90 ± 0.07, but inductive Node2Vec dropped to 0.69 ± 0.17 — a 21-point gap exposing systematic overestimation (Table 2).

A **flat drug–disease GNN** — trained on a graph containing only drug–disease edges (147 nodes, 108 edges) without intermediate target/pathway nodes — achieved an apparently higher LDO AUC of 0.947 ± 0.027. This superficially superior result is an artifact of identity-based memorization: with only 147 nodes and one-hot features, the MLP predictor can memorize which drug indices are positive during training even when held-out drug edges are removed during message passing — conceptually identical to the transductive leakage that inflates Node2Vec AUC [18,20]. The four-layer graph forces prediction through 438 nodes with target and pathway intermediates, requiring genuine mechanism-grounded inference rather than identity-based shortcuts.

**Table 2. Classical and Flat Graph Baseline Comparison**

| Method | AUC | AUPRC | ± (AUC) | Setting |
|--------|:---:|:---:|:---:|------|
| Random | 0.500 | 0.03 | — | null |
| Node2Vec+MLP (transductive) | 0.897 | 0.42 | 0.07 | embedding leak |
| RWR | 0.722 | 0.24 | 0.29 | transductive |
| Flat drug–disease GNN | 0.947 | 0.23 | 0.03 | identity leak |
| Node2Vec+MLP (inductive) | 0.687 | 0.15 | 0.17 | inductive |
| True Homogeneous GNN | 0.849 | 0.31 | 0.10 | **inductive** |
| AttnHeteroGNN | 0.851 | 0.32 | 0.11 | **inductive** |

### Comparison with Knowledge Graph Embedding Methods

All four KG embedding methods achieved substantially lower AUC than the GNN variants under identical inductive LDO protocols (Table 3). ComplEx performed best among KG methods (AUC 0.830 ± 0.038), followed by TransE (0.824 ± 0.064), DistMult (0.821 ± 0.036), and RotatE (0.804 ± 0.074). The GNN advantage (2–5 AUC points) likely stems from learned message passing, which propagates information across the full mechanism hierarchy, compared to KG methods that score triples independently. More notably, KG methods showed substantially lower AUPRC (0.08–0.13 vs. GNN 0.31–0.32), indicating poor ranking of true positives under extreme class imbalance — a critical limitation for drug repositioning where the primary goal is surfacing a manageable number of high-confidence candidates [11].

**Table 3. Knowledge Graph Embedding Baseline Comparison**

| Method | AUC | AUPRC | ± (AUC) | Scoring |
|--------|:---:|:---:|:---:|------|
| DistMult | 0.821 | 0.110 | 0.036 | Bilinear diagonal |
| ComplEx | 0.830 | 0.109 | 0.038 | Complex bilinear |
| RotatE | 0.804 | 0.084 | 0.074 | Rotation |
| TransE | 0.824 | 0.134 | 0.064 | Translation |
| **HeteroGNN** | **0.849** | **0.32** | **0.107** | Message passing |

### Graph Ablation: Target–Pathway Specificity Is the Critical Signal

For all graph and feature ablation experiments (Tables 4, 5), we used the HeteroGNN architecture with full 6-dimensional node features (layer one-hot + degree + betweenness), trained and evaluated under the same 5-fold LDO protocol as the architecture ablation. The control AUC for these ablation experiments (0.805) is lower than the architecture ablation HeteroGNN (0.849, Table 1) because graph ablation experiments were run with a consistent random seed and fold partitioning optimized for ablation reproducibility rather than peak performance, and excluded the class-weighted loss to isolate graph-structure effects. The ECFP4 experiment (Table 6) was conducted separately and achieved a higher baseline (0.883), reflecting a different fold split that happened to be more favorable for the HeteroGNN; cross-experiment AUC comparisons should be interpreted qualitatively rather than numerically.

Graph ablation experiments (Table 4) revealed a clear hierarchy of component importance. **Randomizing target–pathway edges** produced the largest performance drop (ΔAUC = −0.083, ≈8 percentage points vs. the 0.805 control), confirming that the specific mapping of protein targets to their cognate signaling modules is the primary carrier of predictive signal. **Randomizing pathway–disease edges** produced a smaller degradation (ΔAUC = −0.017, ≈2 percentage points). Removing the pathway layer entirely — creating a direct drug→target→disease graph — preserved performance (AUC 0.83 ± 0.11), indicating that the pathway layer functions as an information relay rather than an independent information source: the target→disease composite captures the same connectivity. Degree-preserving rewiring of all edges — our topology-matched negative control — yielded AUC indistinguishable from the original graph (0.81 vs. 0.81), confirming that raw graph density alone is insufficient: it is the *biological specificity* of edges, not their existence or the graph's macroscopic structural properties, that drives prediction.

**Table 4. Graph Ablation Results (HeteroGNN, 5-fold Inductive LDO)**

| Graph Variant | AUC | AUPRC | Δ vs. Original |
|------|:---:|:---:|:---:|
| **Original (4-layer, full)** | **0.805** | **0.237** | — |
| Remove pathway layer (drug→target→disease) | 0.829 | 0.186 | +0.024 |
| Randomize target–pathway edges | 0.722 | 0.214 | **−0.083** |
| Randomize pathway–disease edges | 0.788 | 0.276 | −0.017 |
| Degree-preserving rewiring | 0.806 | 0.188 | +0.001 |

Feature ablation (Table 5) confirmed that **layer-type one-hot encoding alone** achieves the highest AUC (0.870 ± 0.103). Adding degree and betweenness features did not improve performance (Full: 0.810 ± 0.079), and in some folds, structural features degraded prediction (Layer+Degree: 0.795 ± 0.156). The all-ones baseline (no features) achieved AUC 0.775 ± 0.259 — substantially above random but highly variable, indicating that message passing on the mechanism graph provides sufficient structural context for prediction even without explicit features.

**Table 5. Feature Ablation Results (HeteroGNN, 5-fold Inductive LDO)**

| Features | AUC | AUPRC | ± (AUC) |
|------|:---:|:---:|:---:|
| Layer One-Hot Only | **0.870** | 0.271 | 0.103 |
| Degree + Betweenness Only | 0.863 | 0.225 | 0.114 |
| Full (Layer + Degree + Betweenness) | 0.810 | 0.161 | 0.079 |
| Layer + Degree | 0.795 | 0.218 | 0.156 |
| No Features (all-ones) | 0.775 | 0.208 | 0.259 |

### Molecular Features Provide Complementary Signal

ECFP4 molecular fingerprints (1024-bit, reduced to 128 dimensions via random projection) provided an orthogonal signal channel (Table 6). The ECFP-enhanced model achieved AUC 0.908 ± 0.099 — a +2.5 point gain over the original features — with reduced cross-fold variance (0.099 vs. 0.112), indicating more stable predictions. The largest gains occurred on hard folds (Fold 2: +6 points; Fold 5: +10 points), where graph connectivity alone provided weak signal.

**Table 6. Molecular Feature Enhancement (HeteroGNN, 5-fold Inductive LDO)**

| Features | AUC | AUPRC | ± (AUC) |
|------|:---:|:---:|:---:|
| Original (6-dim: layer+deg+btw) | 0.883 | 0.466 | 0.112 |
| **+ ECFP4 (134-dim)** | **0.908** | **0.488** | **0.099** |
| Δ | +0.025 | +0.022 | −0.013 |

This finding complements the graph ablation results: while graph topology is the primary performance driver, molecular similarity provides an orthogonal, complementary signal channel — particularly valuable for drugs whose mechanism annotation is sparse [41]. Importantly, ECFP features do not require target annotations and can be computed for any drug with a known structure, opening a path toward prediction for the 93 cold-start drugs in our training set that lack target edges.

### Literature Validation

To exclude the possibility of circular validation — where both training data and validation draw from the same PubMed corpus — we conducted multi-source triangulation across three independent bibliographic databases: PubMed (with graph-construction PMIDs excluded), OpenAlex, and Scopus (Table 7). After excluding all 442 PMIDs used in graph construction or evidence screening, 19 of 20 top TMJOA candidates retained literature support — identical to the pre-exclusion rate. Cross-validation across all three databases confirmed 18 of 20 candidates (90%), with 19/20 confirmed by at least two databases.

**Table 7. Multi-Source Cross-Validation of Top-20 TMJOA Predictions**

| Rank | Drug | PubMed† | OpenAlex | Scopus | Consensus |
|:---:|------|:---:|:---:|:---:|:---:|
| 1 | Dextrose | 216 | 1,010 | 65 | ✓✓✓ |
| 2 | Hypertonic dextrose | 18 | 106 | 14 | ✓✓✓ |
| 3 | Prolotherapy | 55 | 362 | 81 | ✓✓✓ |
| 4 | Chondroitin | 79 | 1,349 | 95 | ✓✓✓ |
| 5 | Dextrose prolotherapy | 47 | 216 | 41 | ✓✓✓ |
| 6 | PDRN | 3 | 24 | 4 | ✓✓✓ |
| 7 | Zinc | 97 | 6,804 | 217 | ✓✓✓ |
| 8 | BGJ398 | 0 | 19 | 0 | ✗ |
| 9 | Resveratrol | 9 | 974 | 13 | ✓✓✓ |
| 10 | Quercetin | 6 | 902 | 11 | ✓✓✓ |

† PubMed counts exclude 442 PMIDs used in graph construction or evidence screening.
Full top-20 in Supplementary Table S4.

BGJ398 shows 0 PubMed and 0 Scopus hits — confirming its status as a genuinely novel repositioning candidate with no existing TMJOA literature. The 19 OpenAlex hits for BGJ398 reflect broader full-text indexing that captures mentions in references, author affiliations, or supplementary materials unrelated to TMJOA therapeutics.

### Mechanism Path Case Studies

**Type-Level Attention Weights.** The Target→Pathway edge type received the highest attention (0.44), followed by Pathway→Disease (0.31) and Drug→Target (0.25). This distribution aligns with the ablation finding that target–pathway edge specificity is the most critical signal. These attention weights are descriptive rather than causal: they characterize the learned information flow but do not imply that attention contributes to predictive performance (which is identical across all architectures).

**Case 1: BGJ398 — A Novel Prediction.** BGJ398 (infigratinib), a pan-FGFR1-3 inhibitor in Phase II trials for cholangiocarcinoma and achondroplasia, was the sole top-20 prediction with no prior PubMed-indexed TMJOA-specific literature. Mechanism traces: (1) FGF → FGF signaling → TMJOA, and (2) MMP13 → ECM degradation → TMJOA. Conditional FGFR1 knockout in murine cartilage attenuates post-traumatic OA (PMID: 33741257); selective FGFR inhibitors suppress IL-1β-induced cartilage degradation ex vivo (PMID: 32942816). While FGFR inhibition has never been investigated in TMJOA specifically, the mechanism path is biologically coherent and represents a genuinely novel translatable hypothesis.

**Case 2: Chondroitin — A Validated Positive Control.** Chondroitin sulfate (score 0.48, 79 PubMed hits). The model identified 7 direct targets (TGF-β, MMP, ADAMTS, NF-κB, TNF-α, Collagen) converging on TMJOA through BMP, TGF-β/Smad, and inflammatory cytokine pathways — demonstrating correct identification of drugs with well-established multi-pathway mechanisms.

**Case 3: Metformin — Graph Coverage as a Falsifiable Constraint.** Metformin received a modest score (0.42) and yielded **zero** direct mechanism paths to TMJOA. Its five annotated targets (AMPK, Complex I, GPD2, SIRT1, mTOR) were orphans — none connected to a pathway with a TMJOA association edge. This validates the architecture's mechanism-dependence: when annotation is absent, performance degrades transparently rather than silently overfitting to spurious correlations.

---

## Discussion

### Graph Topology as the Primary Determinant of Inductive Performance

The dominant finding of this study — spanning architecture ablation, KG and flat-graph baseline comparison, and graph ablation — is that **the four-layer mechanism graph topology, not GNN architectural complexity, drives inductive generalization in drug repositioning**. A minimal homogeneous GNN with 2-dimensional node features achieves AUC 0.85, with no architecture variant producing a statistically significant difference. Architecture ablation reveals a strict ceiling effect: all five GNN variants differ by ≤ 0.002 AUC, indicating that the information available for inductive prediction is saturated by simple message passing on the mechanism graph.

Graph ablation sharpens this finding. The ablation that most degrades performance — randomizing target–pathway edges (AUC 0.72) — confirms that the **biological specificity** of protein→signaling module mappings, not raw graph structure, carries the predictive signal. The degree-preserving rewiring control — which maintains graph density and node degree distributions while destroying biological edge semantics — leaves AUC unchanged, ruling out the possibility that macroscopic graph properties alone explain performance. Critically, removing the explicit pathway layer (switching to drug→target→disease) preserves AUC, indicating that the pathway layer functions as an **information relay**: the target–pathway–disease composite captures the same connectivity as a direct target–disease edge.

Feature ablation further reinforces the graph-centric interpretation. Layer-type one-hot encoding alone achieves the highest AUC (0.870), and structural features (degree, betweenness) provide no additional benefit. The all-ones baseline (AUC 0.775) — where every node has identical features — still achieves strong prediction, demonstrating that the graph topology itself provides sufficient inductive bias for generalization.

### Statistical Cautions on the Architecture Equivalence Claim

We emphasize that the observed near-identity of AUC values across architectures (Δ ≤ 0.002) should be interpreted as **failure to detect a difference** at the present sample size, rather than positive proof of equivalence. The TOST procedure with ±0.05 bounds failed to establish formal equivalence (p_eq > 0.05), and the minimum detectable effect size at 80% power is 0.28 AUC — far larger than the observed 0.002 gap. A reader should not conclude from these data that all GNN architectures are identical; rather, any architecture-driven improvement, if it exists, is at most modest and is dwarfed by the signal carried by the mechanism graph topology. Future work with larger drug sets (>500) or more folds would be needed to establish formal equivalence.

### Molecular Features as Orthogonal Signal

ECFP4 molecular fingerprints provide a +2.5 AUC gain, with the largest improvements on hard folds where graph connectivity alone was weakest. This suggests that molecular similarity provides a signal channel **orthogonal to and independent of** mechanism graph topology — particularly valuable for drugs whose mechanism annotation is sparse. The gain is comparable to the gap between GNNs and the best KG embedding method, confirming that molecular features add value even when graph topology is already informative.

### The Heterogeneous GNN: Performance Parity, Interpretability Premium

While the heterogeneous GNN adds no AUC over the homogeneous variant, it adds mechanism path-level traceability — the ability to extract complete drug → target → pathway → disease reasoning chains for each prediction. The attention weight distribution (Target→Pathway 0.44 > Pathway→Disease 0.31 > Drug→Target 0.25) independently validates the graph ablation finding that target–pathway specificity is the critical signal.

### Interpreting Literature Validation: Recovery vs. Discovery

The 19/20 literature validation rate warrants careful interpretation. Because both the training graph and validation search draw from biomedical literature, a high confirmation rate could reflect mechanism-grounded associations rather than de novo biological discovery. We addressed this concern through three strategies.

**Empirical exclusion.** We removed all 442 PMIDs used in graph construction from the PubMed validation search. The confirmation rate remained unchanged (19/20), with a median exclusion of only 1 PMID per drug, demonstrating that graph-construction articles and validation-search articles occupy largely disjoint subsets of the PubMed corpus.

**Multi-source triangulation.** We replicated the validation on OpenAlex and Scopus — two bibliographic databases with independent indexing and coverage. All three databases confirmed 18 of 20 candidates (90%).

**Residual circularity.** We acknowledge that this exclusion strategy does not fully sever all training–validation information links. ChEMBL, KEGG, and Reactome annotations ultimately derive from the same biomedical literature corpus as PubMed validation searches. While the specific PMIDs differ (mechanism evidence vs. clinical co-occurrence), the underlying knowledge sources are not fully independent. Future work should employ time-split validation (building graphs from pre-2018 literature and validating on post-2018 publications) or prospective experimental validation to fully address this concern.

We distinguish two prediction classes: **known candidate recovery** (19/20, positive-control sanity check) and **novel hypothesis generation** (BGJ398, representing genuine repositioning discovery). For transparency, we define four evidence levels: Level 1 — direct TMJOA therapeutic evidence (RCT/meta-analysis); Level 2 — TMJ/cartilage/bone disease-model evidence; Level 3 — pathway-relevant mechanistic evidence; Level 4 — indirect target/pathway plausibility only. Full per-candidate mapping is in Supplementary Table S4.

### The Node2Vec Paradox and Inductive Discipline

The 21-point gap between transductive (0.90) and inductive (0.69) Node2Vec exposes a methodological vulnerability that extends beyond embedding methods: test nodes participating in the training process encode structural information before predictor training, creating an information leak. We recommend strictly inductive evaluation — leave-drug-out and leave-disease-out — for all drug repositioning benchmarks [18,20,22].

### Mechanism Graph Construction as the Primary Bottleneck

The Metformin case crystallizes the central limitation: AMPK — a well-established chondroprotective kinase and Metformin's primary target — lacks a pathway→TMJOA edge because our annotation pipeline did not capture the autophagy/cellular stress response pathway. The architecture correctly fails to predict Metformin (zero paths), demonstrating transparent degradation rather than spurious extrapolation. Future work should expand pathway–disease edges using comprehensive resources such as DisGeNET [26] and WikiPathways.

### BGJ398: A Testable Hypothesis

BGJ398 (infigratinib), a pan-FGFR inhibitor in Phase II oncology/achondroplasia trials, is the model's sole literature-novel prediction. FGFR signaling is a master regulator of chondrocyte biology: FGF-2/FGFR1 promotes chondrocyte hypertrophy and MMP13 via RAS-MEK-ERK, while the FGFR-FGF18 axis restricts proliferation. In OA cartilage, FGF-FGFR1 becomes hyperactive, driving the same catabolic cascade active in TMJ condylar cartilage. We propose three validation paths: (1) in vitro BGJ398 on IL-1β-stimulated TMJ condylar chondrocytes measuring MMP13/ADAMTS5, (2) in silico docking to FGFR1 kinase domain (PDB: 3TT0), and (3) pharmacovigilance database query for TMJ-related signals in FGFR inhibitor-treated patients.

### Limitations

Several limitations should be noted. First, the 753-article evidence pool is TMJOA-centric and may underrepresent mechanisms for other diseases in the 15-disease phenotype layer. Second, 5-fold LDO with ~390 test pairs per fold is adequately powered for detecting large effects relative to random (d = 3.3, power > 0.99) but underpowered for detecting small architectural improvements (minimum detectable ΔAUC ≈ 0.28 at 80% power) or establishing formal equivalence. Third, drug chemical structure features (beyond ECFP4) and protein sequence features are not exploited; integrating graph-based molecular convolutions and protein language model embeddings could improve predictions for sparsely annotated drugs [41]. Fourth, the homogeneous ≈ heterogeneous result is demonstrated on a single mechanism graph; external replication on independent KG resources such as Hetionet or DRKG is needed to establish generalizability. Fifth, graph ablation was conducted with the HeteroGNN architecture; consistent patterns across the homogeneous GNN would strengthen the claim that graph effects are architecture-independent. Sixth, despite our efforts to mitigate circular validation through multi-source triangulation, the training and validation corpora ultimately share a common ancestry in the biomedical literature; true external validation requires prospective experimental confirmation.

---

## Conclusion

We constructed a four-layer mechanism graph (drug → target → pathway → disease) for TMJOA drug repositioning and evaluated five GNN architectures under strict inductive leave-drug-out protocols. All architectures — from a minimal homogeneous GNN to a full heterogeneous attention GNN — achieved indistinguishable inductive performance (AUC 0.849–0.851), with no statistically significant differences detected. Graph ablation identified target–pathway edge specificity as the critical signal carrier, the degree-preserving rewiring control confirmed that biological specificity rather than graph density drives prediction, and molecular fingerprint enhancement provided an orthogonal +2.5 AUC gain through chemical similarity. The heterogeneous GNN contributed mechanism path-level interpretability without predictive advantage.

Our findings suggest that, for curated disease-specific mechanism graphs, graph construction — specifically, the quality and specificity of biological edge annotations — may be a more consequential modeling decision than GNN architectural complexity. We propose that mechanism-constrained graph learning offers a principled framework for computational drug repositioning, one in which biological prior knowledge provides the structural foundation of the learning process, and methodological rigor in evaluation — particularly inductive protocols and appropriate statistical framing — ensures that reported gains reflect genuine generalization rather than leakage artifacts.

---

## References

[1] Pushpakom S, et al. Drug repurposing: progress, challenges and recommendations. Nat Rev Drug Discov. 2019;18(1):41–58.
[2] Jarada TN, et al. A review of computational drug repositioning: strategies, approaches, opportunities, challenges, and directions. J Cheminform. 2020;12(1):46.
[3] Wang M, et al. Deep learning for drug repurposing: methods, databases, and applications. WIREs Comput Mol Sci. 2022;12(4):e1597.
[4] Bagherian M, et al. Machine learning approaches and databases for prediction of drug–target interaction: a survey paper. Brief Bioinform. 2021;22(1):247–269.
[5] Schiffman E, et al. Diagnostic criteria for temporomandibular disorders (DC/TMD). J Oral Facial Pain Headache. 2014;28(1):6–27.
[6] Wang XD, et al. Current understanding of pathogenesis and treatment of TMJ osteoarthritis. J Dent Res. 2015;94(5):666–673.
[7] de Souza RF, et al. Interventions for the management of temporomandibular joint osteoarthritis. Cochrane Database Syst Rev. 2012;(4):CD007261.
[8] Wang Z, et al. TMJ osteoarthritis: a review of current understanding and future directions. Int J Oral Sci. 2022;14(1):48.
[9] de Leeuw R, Klasser GD. Orofacial pain: guidelines for assessment, diagnosis, and management. 6th ed. Quintessence; 2018.
[10] Yella JK, et al. Changing trends in computational drug repositioning. Pharmaceuticals. 2018;11(2):57.
[11] Chen T, et al. Drug–target interaction prediction: databases, web servers and computational models. Brief Bioinform. 2016;17(4):696–712.
[12] Himmelstein DS, et al. Systematic integration of biomedical knowledge prioritizes drugs for repurposing. eLife. 2017;6:e26726.
[13] Ioannidis VN, et al. DRKG — Drug Repurposing Knowledge Graph. 2020. arXiv:2002.02035.
[14] Zheng S, et al. PharmKG: a dedicated knowledge graph benchmark for biomedical data mining. Brief Bioinform. 2021;22(4):bbaa344.
[15] Wang X, et al. Heterogeneous graph attention network. WWW. 2019:2022–2032.
[16] Schlichtkrull M, et al. Modeling relational data with graph convolutional networks. ESWC. 2018:593–607.
[17] Hu Z, et al. Heterogeneous graph transformer. WWW. 2020:2704–2710.
[18] Hamilton WL, et al. Inductive representation learning on large graphs. NeurIPS. 2017:1024–1034.
[19] Zhou J, et al. Graph neural networks: A review of methods and applications. AI Open. 2020;1:57–81.
[20] Xu K, et al. How powerful are graph neural networks? ICLR. 2019.
[21] Huang K, et al. DeepPurpose: a deep learning library for drug–target interaction prediction. Bioinformatics. 2021;36(22–23):5545–5547.
[22] Liu Y, et al. Learning interpretable cellular responses from perturbation data. Nat Methods. 2022;19(9):1093–1101.
[23] Mendez D, et al. ChEMBL: towards direct deposition of bioassay data. Nucleic Acids Res. 2019;47(D1):D930–D940.
[24] Kanehisa M, et al. KEGG: new perspectives on genomes, pathways, diseases and drugs. Nucleic Acids Res. 2017;45(D1):D353–D361.
[25] Fabregat A, et al. The Reactome pathway knowledgebase. Nucleic Acids Res. 2018;46(D1):D649–D655.
[26] Piñero J, et al. DisGeNET: a comprehensive platform integrating information on human disease-associated genes and variants. Nucleic Acids Res. 2017;45(D1):D833–D839.
[27] Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. ICLR. 2017.
[28] Paszke A, et al. PyTorch: an imperative style, high-performance deep learning library. NeurIPS. 2019:8024–8035.
[29] Wang M, et al. Deep Graph Library: a graph-centric, highly-performant package for graph neural networks. 2019. arXiv:1909.01315.
[30] Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model. 2010;50(5):742–754.
[31] Kim S, et al. PubChem 2023 update. Nucleic Acids Res. 2023;51(D1):D1373–D1380.
[32] Landrum G. RDKit: Open-source cheminformatics. 2023. https://www.rdkit.org.
[33] Grover A, Leskovec J. node2vec: Scalable feature learning for networks. KDD. 2016:855–864.
[34] Yang B, et al. Embedding entities and relations for learning and inference in knowledge bases. ICLR. 2015.
[35] Trouillon T, et al. Complex embeddings for simple link prediction. ICML. 2016:2071–2080.
[36] Sun Z, et al. RotatE: Knowledge graph embedding by relational rotation in complex space. ICLR. 2019.
[37] Bordes A, et al. Translating embeddings for modeling multi-relational data. NeurIPS. 2013:2787–2795.
[38] Schuirmann DJ. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J Pharmacokinet Biopharm. 1987;15(6):657–680.
[39] Lakens D. Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. Soc Psychol Personal Sci. 2017;8(4):355–362.
[40] Priem J, et al. OpenAlex: a fully-open index of scholarly works, authors, venues, institutions, and concepts. STI 2022.
[41] Öztürk H, et al. DeepDTA: deep drug–target binding affinity prediction. Bioinformatics. 2018;34(17):i821–i829.
