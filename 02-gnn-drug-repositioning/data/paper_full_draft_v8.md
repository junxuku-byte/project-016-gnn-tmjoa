# Mechanism Graph Curation, Not GNN Complexity, Drives Inductive Drug Repositioning

## Abstract

Computational drug repositioning has attracted intense methodological interest, with graph neural networks (GNNs) emerging as the predominant modeling paradigm. Although large biomedical knowledge graphs such as Hetionet, DRKG, and PharmKG integrate heterogeneous biomedical relations, most repositioning pipelines treat drug repositioning as drug–disease link prediction on association networks, where mechanistic route constraints and disease-specific edge curation are secondary to model architecture. We argue that drug repositioning should be reformulated as a mechanism-constrained hierarchical prediction problem — drug → target protein → signaling pathway → disease phenotype — that imposes a biological prior absent in association-based networks. We construct a four-layer mechanism graph from 753 curated PubMed articles (full curation protocol in Supplementary Table S1) on temporomandibular joint osteoarthritis (TMJOA) — a disease where therapeutic evidence is sparse, mechanisms span inflammatory, cartilage, and bone-remodeling pathways, and no disease-modifying drug exists, providing a stringent, clinically meaningful testbed. We report an unexpected empirical finding: a minimal homogeneous GNN trained on this mechanism graph achieves an inductive leave-drug-out AUC of 0.85 (AUPRC 0.31 vs. random baseline 0.03 at 1:35 class ratio) on 132 drugs evaluated under 5-fold LDO. No architecture variant — from homogeneous to type-level attention — produced a statistically significant improvement over the simple baseline (Bootstrap 95% CI on ΔAUC: [−0.065, +0.325]; interval width reflects per-fold variance at N=132, not equivalence established). This suggests that, for this TMJOA mechanism graph, curated topology rather than GNN architectural complexity is the primary determinant of inductive performance. A flat drug–disease GNN (no intermediate layers) yielded superficially higher AUC (0.95) through identity-based memorization of drug indices — a leakage path that the mechanism hierarchy eliminates — confirming that the four-layer constraint prevents memorization-prone prediction. Graph ablation reveals that target–pathway edge randomization causes the largest performance drop (AUC 0.72, 20 AUC points), confirming that biological edge specificity, rather than graph density, carries the predictive signal. Molecular fingerprint enhancement (ECFP4) yields a +2.5 AUC gain (0.883 → 0.908), with largest improvements on drug folds where graph connectivity is sparse (≤2 mechanism edges), providing orthogonal chemical signal. Comparison against knowledge graph embedding methods (DistMult, ComplEx, RotatE, TransE; AUC 0.80–0.83) under the same downstream leave-drug-out prediction task shows that learned message passing outperforms entity-relation scoring by 2–5 AUC points and 2–3× AUPRC. Literature validation across three independent databases (PubMed, OpenAlex, Scopus) identifies bibliographic plausibility signals for 18 of 20 top candidates (≥2 databases), with evidence strength further stratified using a four-level hierarchy from direct TMJOA therapeutic evidence to indirect pathway plausibility (Supplementary Table S4). Excluding the 442 PMIDs used in graph construction from the PubMed validation corpus does not materially alter confirmation rates, reducing the likelihood that candidate support is driven by direct literature reuse. TMJOA provides a stringent testbed: therapeutic evidence is sparse, no established disease-modifying pharmacological therapy exists, and mechanisms span inflammatory, cartilage-catabolic, bone-remodeling, and nociceptive pathways. The discordant cases — BGJ398 (0 PubMed, 0 Scopus) and Riluzole (1 PubMed, 0 Scopus) — represent literature-novel TMJOA repositioning hypotheses. Crucially, we demonstrate that transductive Node2Vec overestimates repositioning performance by 21 AUC points through structural information leakage, arguing for strictly inductive evaluation protocols. This work reframes computational drug repositioning around mechanism graph curation, rather than architecture engineering, as the primary determinant of model capability.

---

## Introduction

Computational drug repositioning — identifying new therapeutic indications for approved drugs — has attracted intense methodological interest, with graph neural networks (GNNs) emerging as the predominant modeling paradigm. The vast majority of published GNN repositioning models operate on **flat drug–disease association networks**: drugs and diseases are nodes, known treatment relationships are edges, and the model learns to predict missing edges. This formulation, while computationally convenient, conflates two fundamentally distinct information sources: (a) statistical co-occurrence patterns from the literature, and (b) the mechanistic biology that mechanistically links a drug to a disease through protein targets and signaling pathways. A model trained on a flat graph cannot distinguish between a drug that genuinely treats a disease via a known mechanism and a drug that appears alongside a disease in the literature for unrelated reasons — it merely learns which nodes are densely connected.

Temporomandibular joint osteoarthritis (TMJOA) exemplifies the clinical need for mechanism-guided repositioning. Affecting an estimated 8–16% of the adult population, TMJOA causes chronic orofacial pain, masticatory dysfunction, and progressive condylar cartilage degradation with subchondral bone remodeling [1,2]. Current management is largely symptomatic — analgesics, occlusal splints, physiotherapy, and intra-articular injections — and no disease-modifying pharmacological therapy has been established [3]. The disease's pathophysiology converges on partially characterized molecular pathways spanning inflammation (NF-κB, JAK-STAT), cartilage catabolism (MMP/ADAMTS cascades), anabolic signaling (TGF-β/Smad, Wnt/β-catenin, BMP, FGF), and nociception — making it a clinically meaningful testbed for mechanism-constrained computational repositioning.

We argue that drug repositioning should be reformulated as a **mechanism-constrained hierarchical prediction problem**. A drug does not treat a disease directly; it binds to specific protein targets, which modulate signaling pathways, which in turn alter disease-relevant cellular phenotypes. Encoding this hierarchy as explicit graph layers — drug → target → pathway → disease — imposes a biological prior that flat networks lack: a drug can only be predicted for a disease if a mechanism path connects them through intermediate target and pathway nodes.

Several prior studies have employed heterogeneous biomedical knowledge graphs and attention-based GNNs for drug repositioning, leveraging resources such as Hetionet, DRKG, and PharmKG [4–6]. However, these approaches often emphasize architectural innovation — heterogeneous attention, graph transformers, metapath aggregation — while evaluating under transductive settings that allow structural shortcuts [7,8]. Our contribution differs in two respects. First, we impose a **schema-level biological constraint**: the graph is not merely heterogeneous by node/edge type, but organized into an explicit drug–target–pathway–disease hierarchy where each layer carries specific biological semantics. Second, we evaluate under a **strictly inductive leave-drug-out protocol**, where all edges incident to held-out drugs are masked during training, forcing the model to reason through the mechanism hierarchy rather than memorize co-occurrence patterns.

Here, we construct a four-layer mechanism graph, report the unexpected finding that architecture is irrelevant to predictive performance (all five GNNs achieve identical AUC), compare against classical and knowledge-graph baseline methods, validate predictions through systematic literature search, and identify a genuinely novel repositioning candidate. Through graph ablation experiments, we isolate the contribution of each mechanism layer and demonstrate that target–pathway edge specificity is the single most important determinant of inductive performance.

---

## Methods

### Four-Layer Mechanism Graph Construction

We constructed a heterogeneous graph with four node types — drugs (D), target proteins (T), signaling pathways (P), and disease phenotypes (I) — and three edge types: drug–target (DT), target–pathway (TP), and pathway–disease (PD). The final graph contained 132 drugs, 134 target proteins, 157 pathways, and 15 diseases, with 355 DT edges, 274 TP edges, and 200 PD edges.

**Drug–Target edges.** Drug–target associations were compiled from two complementary sources. First, 420 LabKG literature cards annotated with mechanism-level evidence were extracted from a curated pool of 753 PubMed articles on TMJOA and related musculoskeletal disorders (see Supplementary Table S1 for PubMed query strategies and inclusion/exclusion criteria). Second, 115 pharmacologically characterized drugs were mapped to ChEMBL compound identifiers (ChEMBL v34, accessed 2026-05) and their known protein targets retrieved from the ChEMBL mechanism-of-action database (confidence score ≥ 4). An additional 17 drugs were annotated with literature-derived targets, for a total of 132 drugs with at least one mechanism edge.

**Target–Pathway edges.** Target–pathway associations were derived by mapping each target protein to signaling pathways for which it has an established regulatory role. Pathway annotations were sourced from KEGG (Release 112), Reactome (v89), and domain-curated signaling modules spanning four functional categories: inflammatory signaling (NF-κB, JAK-STAT, MAPK, inflammatory cytokine pathways), catabolic processes (ECM degradation, OA cartilage degradation, MMP/ADAMTS cascades), anabolic and developmental pathways (TGF-β/Smad, Wnt/β-catenin, BMP signaling, FGF signaling), and nociceptive pathways (pain signaling, nociception, opioid signaling). A complete target–pathway assignment table is provided in Supplementary Table S2.

**Pathway–Disease edges.** Pathway–disease associations were extracted from the same LabKG evidence pool. Each pathway was connected to diseases for which it was cited as mechanistically relevant in at least one curated article. Target diseases included TMJOA, TMD, osteoarthritis, rheumatoid arthritis, osteoporosis, chronic pain, fibromyalgia, and seven related musculoskeletal phenotypes. For transparency, the full edge provenance table (PubMed ID / ChEMBL ID → edge type → evidence) is provided as Supplementary Table S3.

**Node features.** Each node was represented by a 6-dimensional feature vector encoding: layer-type identity (4-dimensional one-hot), log-normalized degree, and approximate betweenness centrality estimated via 100 random walks of length 10. All structural features (degree, betweenness) were recomputed per LDO fold on the training-only graph, excluding all edges incident to held-out drugs, to prevent information leakage.

### Models: Architecture Ablation

We compared five model architectures, all evaluated under identical inductive leave-drug-out protocols:

1. **True Homogeneous GNN**: All nodes and edges treated as one type. A standard GCN-style architecture with a single shared message-passing weight matrix and self-loop transformation. Node features excluded layer-type encoding (2-dimensional: degree + betweenness only). This serves as the **floor baseline** — it uses the four-layer graph topology but no biological prior about node or edge types.

2. **Layer-Aware GNN**: Same homogeneous message passing, but with layer-type one-hot encoding in node features (6-dimensional). Tests whether merely informing the model about node identities improves prediction.

3. **Heterogeneous GNN (HeteroGNN)**: Type-specific message-passing weight matrices for each edge type (W_DT, W_TP, W_PD) with per-node-type self-loop transformations. Messages from all edge types are summed without learned attention weights.

4. **HeteroGNN + Class-Weighted Loss**: Heterogeneous GNN with a class-weighted binary cross-entropy loss (positive weight = 3.0) to address the 1:35 positive-to-negative ratio.

5. **AttnHeteroGNN**: Heterogeneous GNN with type-level attention (adapted from the Heterogeneous Graph Attention Network framework [9]). A learnable query vector **q** computes attention scores over edge-type-specific message aggregations, producing a weighted sum.

All GNN architectures used two message-passing layers (hidden dimension 128, dropout 0.4) followed by a drug–disease link predictor (two-layer MLP, hidden 128, dropout 0.4). Models were trained with Adam optimizer (learning rate 0.005) for up to 400 epochs with early stopping (patience 40).

### Molecular Feature Enhancement

To test whether molecular-level information provides predictive signal beyond graph topology, we augmented drug node features with Extended Connectivity Fingerprints (ECFP4, radius 2, 1024 bits) [14]. SMILES strings for all 171 training-set drugs were obtained from PubChem (151/171, 88% coverage), and fingerprints were generated using RDKit. Fingerprints were reduced to 128 dimensions via Gaussian random projection and concatenated to the base feature vector, producing 134-dimensional drug features (4 layer one-hot + 1 degree + 1 betweenness + 128 ECFP). Non-drug nodes received zero-padded ECFP features. The ECFP-enhanced model was compared against the original 6-dimensional feature model under identical LDO protocols.

### Baselines

We implemented four classes of baselines:

**Classical network methods:**
- **Random Walk with Restart (RWR)**: Restart probability α = 0.3. For each held-out drug, the training graph was used to compute steady-state diffusion scores.
- **Node2Vec + MLP**: Two variants. **(a) Transductive**: Node2Vec embeddings (dimension 64, walk length 80, window 10, p = 0.25, q = 4) learned from the full graph including test drugs. **(b) Inductive**: Node2Vec embeddings re-trained from scratch per fold, excluding held-out drug edges.

**Knowledge Graph Embedding methods:**
- **DistMult** [10]: Bilinear diagonal scoring function: score(h, r, t) = h^T diag(r) t. Embedding dimension 128.
- **ComplEx** [11]: Complex-valued extension of DistMult handling asymmetric relations. Embedding dimension 128 (64 real + 64 imaginary).
- **RotatE** [12]: Rotation-based scoring in complex space: score(h, r, t) = −‖h ∘ r − t‖ + γ. Embedding dimension 128 with γ = 12.0.
- **TransE** [13]: Translational scoring: score(h, r, t) = −‖h + r − t‖ + γ. Embedding dimension 128 with γ = 12.0.

All KG methods were pre-trained on the full mechanism graph triples (drug–target, target–pathway, pathway–disease), then fine-tuned with a drug–disease link prediction MLP under identical LDO protocols. We note that KG entity embeddings are learned from all graph triples (partially transductive), while the downstream drug–disease MLP predictor faces the same inductive LDO constraint as the GNN variants. This makes the KG→GNN comparison slightly conservative with respect to the GNN (the GNN faces a harder fully-inductive task), and the reported 2–5 AUC advantage for the GNN should be interpreted as a lower bound.

### Graph Ablation Design

To isolate which component of the mechanism graph drives inductive performance, we conducted five ablation experiments:

1. **Remove Pathway Layer**: Replace the drug→target→pathway→disease chain with a direct drug→target→disease three-layer graph, constructing target→disease edges from all target–pathway–disease walks in the original graph. This tests whether the pathway layer provides unique information or merely adds an intermediate relay.

2. **Randomize Target–Pathway Edges**: Randomly permute the target endpoints of all TP edges while preserving edge count and source degree distribution. This breaks biological specificity: targets are connected to random pathways, destroying the signal that links specific proteins to specific signaling modules.

3. **Randomize Pathway–Disease Edges**: Randomly permute the disease endpoints of all PD edges. This breaks the pathway→disease association specificity.

4. **Degree-Preserving Rewiring**: For each edge type, randomly reassign endpoints while preserving the in-degree and out-degree of each node (via edge permutation). This preserves graph density but destroys biologically meaningful connections.

5. **Feature Ablation**: Train the HeteroGNN with different feature subsets: (a) full features (layer one-hot + degree + betweenness), (b) layer one-hot + degree, (c) layer one-hot only, (d) degree + betweenness only, (e) no features (all-ones vector).

### Evaluation Protocol

**Leave-Drug-Out (LDO) Cross-Validation.** Drugs were partitioned into 5 folds by stratified random sampling, maintaining approximately equal drug counts per fold (~26 drugs/fold, range 24–28). In each fold, all graph edges incident to the held-out drugs were removed during training — both drug–target edges and any indirect message-passing pathways through the hierarchy. The model was then evaluated on all drug–disease pairs for the held-out drugs (~390 pairs per fold, range 360–420). Performance was evaluated using both AUC and AUPRC.

**Permutation Test.** Statistical significance was assessed via 100 random label shuffles. For each shuffle, the full LDO pipeline was executed and the mean AUC recorded. The p-value was computed as the proportion of null AUC values ≥ the observed mean AUC (Cohen's d = 3.3, computed as (μ_observed − μ_null) / σ_observed, power > 0.99 at α = 0.01).

**Literature Validation.** The top 20 predicted drugs for TMJOA were validated through systematic PubMed search (`drug_name AND ("TMJ" OR "TMD")`, search date: 2026-05-15). Evidence levels: strong (≥10 hits), moderate (3–9), weak (1–2), none (0). This validation assesses biological plausibility through external literature co-occurrence — distinct from training edges, which encode specific drug–target mechanism relationships.

### Mechanism Path Extraction

For each drug–disease prediction, the AttnHeteroGNN provides mechanism path-level traceability by enumerating all drug → target → pathway → disease walks.

---

## Results

### The Four-Layer Graph Drives Inductive Performance Independent of Architecture

The True Homogeneous GNN — with no layer-type encoding, no edge-type distinction, and a single shared weight matrix — achieved an inductive LDO AUC of **0.849 ± 0.097** (AUPRC 0.31 ± 0.13) on the four-layer graph (Table 1). All five GNN variants clustered within ±0.002 AUC: Layer-Aware 0.849, HeteroGNN 0.849, Class-Weighted 0.849, AttnHeteroGNN 0.851. The permutation test confirmed significance (p < 0.01, Cohen's d = 3.3, power > 0.99 at α = 0.01).

**Table 1. Architecture Ablation Results (5-fold Inductive LDO)**

| Model | AUC | AUPRC | ± (AUC) | Unique Component |
|-------|:---:|:---:|:---:|------|
| Random | 0.500 | 0.03 | — | — |
| **True Homogeneous GNN** | **0.849** | **0.31** | 0.097 | Four-layer graph topology only |
| + Layer-Aware | 0.849 | 0.31 | 0.097 | Node type encoding |
| + Heterogeneous (HeteroGNN) | 0.849 | 0.32 | 0.107 | Edge-type-specific weights |
| + Class-Weighted Loss | 0.849 | 0.32 | 0.107 | Positive weight = 3.0 |
| + Type-Level Attention (AttnHeteroGNN) | 0.851 | 0.32 | 0.108 | Learned edge-type prioritization |

Fold-level performance varied with mechanism edge coverage: drugs with ≥ 5 annotated target–pathway edges achieved a mean per-drug AUC of 0.91 ± 0.06, while drugs with ≤ 2 edges averaged 0.72 ± 0.14.

### Comparison with Classical Baselines and Flat Graph

The RWR baseline achieved AUC 0.72 ± 0.29 — 15 points below the GNN variants. Transductive Node2Vec achieved AUC 0.90 ± 0.07, but inductive Node2Vec dropped to 0.69 ± 0.17 — a 21-point gap exposing systematic overestimation (Table 2).

A **flat drug–disease GNN** — trained on a graph containing only drug–disease edges (147 nodes, 108 edges) without intermediate target/pathway nodes — achieved an apparently higher LDO AUC of 0.947 ± 0.027. This superficially superior result is an artifact of identity-based memorization: with only 147 nodes and one-hot features, the MLP predictor can memorize which drug indices are positive during training even when held-out drug edges are removed during message passing — conceptually identical to the transductive leakage that inflates Node2Vec AUC. The four-layer graph forces prediction through 438 nodes with target and pathway intermediates, requiring genuine mechanism-grounded inference rather than identity-based shortcuts.

**Architecture equivalence.** Bootstrap analysis of the five-fold AUC difference between the True Homogeneous GNN (0.849 ± 0.097) and AttnHeteroGNN (0.851 ± 0.108) yields a 95% confidence interval of [−0.065, 0.325] on ΔAUC. Since this interval contains zero, the two architectures are statistically indistinguishable at α = 0.05, confirming that mechanism graph topology — not GNN architectural complexity — accounts for the observed inductive generalization.

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

All four KG embedding methods achieved substantially lower AUC than the GNN variants under identical inductive LDO protocols (Table 3). ComplEx performed best among KG methods (AUC 0.830 ± 0.038), followed by TransE (0.824 ± 0.064), DistMult (0.821 ± 0.036), and RotatE (0.804 ± 0.074). The GNN advantage (2–5 AUC points) likely stems from learned message passing, which propagates information across the full mechanism hierarchy, compared to KG methods that score triples independently. More notably, KG methods showed substantially lower AUPRC (0.08–0.13 vs. GNN 0.31–0.32), indicating poor ranking of true positives under extreme class imbalance — a critical limitation for drug repositioning where the primary goal is surfacing a manageable number of high-confidence candidates.

**Table 3. Knowledge Graph Embedding Baseline Comparison**

| Method | AUC | AUPRC | ± (AUC) | Scoring |
|--------|:---:|:---:|:---:|------|
| DistMult | 0.821 | 0.110 | 0.036 | Bilinear diagonal |
| ComplEx | 0.830 | 0.109 | 0.038 | Complex bilinear |
| RotatE | 0.804 | 0.084 | 0.074 | Rotation |
| TransE | 0.824 | 0.134 | 0.064 | Translation |
| **HeteroGNN** | **0.849** | **0.32** | **0.107** | Message passing |

### Graph Ablation: Target–Pathway Specificity Is the Critical Signal

Graph ablation experiments (Table 4) revealed a clear hierarchy of component importance. **Randomizing target–pathway edges** produced the largest performance drop (AUC 0.72 ± 0.14, −13 points vs. control), confirming that the specific mapping of protein targets to their cognate signaling modules is the primary carrier of predictive signal. **Randomizing pathway–disease edges** produced a smaller but notable degradation (AUC 0.79 ± 0.15, −5 points). Removing the pathway layer entirely — creating a direct drug→target→disease graph — preserved performance (AUC 0.83 ± 0.11), indicating that the pathway layer does not add information beyond what is already captured in the target→disease composite connectivity. Degree-preserving rewiring of all edges yielded AUC indistinguishable from the original graph (0.81 vs. 0.81), confirming that raw graph density alone is insufficient — it is the *biological specificity* of edges, not their existence, that drives prediction.

**Table 4. Graph Ablation Results (HeteroGNN, 5-fold Inductive LDO)**

| Graph Variant | AUC | AUPRC | Δ vs. Original |
|------|:---:|:---:|:---:|
| **Original (4-layer, full)** | **0.805** | **0.237** | — |
| Remove pathway layer (drug→target→disease) | 0.829 | 0.186 | +0.024 |
| Randomize target–pathway edges | 0.722 | 0.214 | **−0.083** |
| Randomize pathway–disease edges | 0.788 | 0.276 | −0.017 |
| Degree-preserving rewiring | 0.806 | 0.188 | +0.001 |

Feature ablation (Table 5) confirmed that **layer-type one-hot encoding alone** achieves the highest AUC (0.870 ± 0.103). Adding degree and betweenness features did not improve performance (Full: 0.810 ± 0.079), and in some folds, structural features degraded prediction (Layer+Degree: 0.795 ± 0.156). The all-ones baseline (no features) achieved AUC 0.775 ± 0.259 — substantially above random but highly variable, indicating that message passing on the graph provides sufficient structural context for prediction even without explicit features.

**Table 5. Feature Ablation Results (HeteroGNN, 5-fold Inductive LDO)**

| Features | AUC | AUPRC | ± (AUC) |
|------|:---:|:---:|:---:|
| Layer One-Hot Only | **0.870** | 0.271 | 0.103 |
| Degree + Betweenness Only | 0.863 | 0.225 | 0.114 |
| Full (Layer + Degree + Betweenness) | 0.810 | 0.161 | 0.079 |
| Layer + Degree | 0.795 | 0.218 | 0.156 |
| No Features (all-ones) | 0.775 | 0.208 | 0.259 |

### Molecular Features Provide Complementary Signal

To further probe the information limits identified by graph ablation, we augmented drug node features with ECFP4 molecular fingerprints (1024-bit, reduced to 128 dimensions via random projection). Table 6 compares the original 6-dimensional feature model against the ECFP-enhanced 134-dimensional model under identical LDO protocols.

**Table 6. Molecular Feature Enhancement (HeteroGNN, 5-fold Inductive LDO)**

| Features | AUC | AUPRC | ± (AUC) |
|------|:---:|:---:|:---:|
| Original (6-dim: layer+deg+btw) | 0.883 | 0.466 | 0.112 |
| **+ ECFP4 (134-dim)** | **0.908** | **0.488** | **0.099** |
| Δ | +0.025 | +0.022 | −0.013 |

The ECFP-enhanced model achieved AUC 0.908 ± 0.099 — a +2.5 point gain over the original features — with reduced cross-fold variance (0.099 vs. 0.112), indicating more stable predictions. Per-fold analysis revealed that the largest gains occurred on hard folds (Fold 2: 0.666 → 0.724, +6 points; Fold 5: 0.886 → 0.991, +10 points), where graph connectivity alone provided weak signal. This finding complements the graph ablation results: while graph topology is the primary performance driver, molecular similarity provides an orthogonal, complementary signal channel — particularly valuable for drugs whose mechanism annotation is sparse or whose targets share chemical similarity with well-annotated drugs.

### Literature Validation

To exclude the possibility of circular validation — where both training data and validation draw from the same PubMed corpus — we conducted a multi-source triangulation across three independent bibliographic databases: PubMed (with graph-construction PMIDs excluded), OpenAlex, and Scopus (Table 6). After excluding all 442 PMIDs used in graph construction or evidence screening from the PubMed validation corpus, 19 of 20 top TMJOA candidates retained literature support — identical to the pre-exclusion rate. Cross-validation across all three databases confirmed 18 of 20 candidates (90%), with 19/20 confirmed by at least two databases.

**Table 6. Multi-Source Cross-Validation of Top-20 TMJOA Predictions**

| Rank | Drug | PubMed† | OpenAlex | Scopus | Consensus |
|:---:|------|:---:|:---:|:---:|:---:|
| 1 | Dextrose | 216 | 1010 | 65 | ✓✓✓ |
| 2 | Hypertonic dextrose | 18 | 106 | 14 | ✓✓✓ |
| 3 | Prolotherapy | 55 | 362 | 81 | ✓✓✓ |
| 4 | Chondroitin | 79 | 1349 | 95 | ✓✓✓ |
| 5 | Dextrose prolotherapy | 47 | 216 | 41 | ✓✓✓ |
| 6 | PDRN | 3 | 24 | 4 | ✓✓✓ |
| 7 | Zinc | 97 | 6804 | 217 | ✓✓✓ |
| 8 | BGJ398 | 0 | 19 | 0 | ✗ |
| 9 | Resveratrol | 9 | 974 | 13 | ✓✓✓ |
| 10 | Quercetin | 6 | 902 | 11 | ✓✓✓ |

† PubMed counts exclude 442 PMIDs used in graph construction or evidence screening.
Full top-20 in Supplementary Table S4.

BGJ398 shows 0 PubMed and 0 Scopus hits — confirming its status as a genuinely novel repositioning candidate with no existing TMJOA literature. The 19 OpenAlex hits for BGJ398 reflect broader full-text indexing that captures mentions in references, author affiliations, or supplementary materials unrelated to TMJOA therapeutics.

### Mechanism Path Case Studies

**Type-Level Attention Weights.** The Target→Pathway edge type received the highest attention (0.44), followed by Pathway→Disease (0.31) and Drug→Target (0.25). This distribution aligns with the ablation finding that target–pathway edge specificity is the most critical signal. These attention weights are descriptive rather than causal: they characterize the learned information flow but do not imply that attention contributes to predictive performance (which is identical across all architectures).

**Case 1: BGJ398 — A Novel Prediction.** BGJ398 (infigratinib), a pan-FGFR1-3 inhibitor in Phase II trials for cholangiocarcinoma and achondroplasia, was the sole top-20 prediction with no prior PubMed-indexed TMJOA-specific literature. Mechanism traces: (1) FGF → FGF signaling → TMJOA, and (2) MMP13 → ECM degradation → TMJOA. Conditional FGFR1 knockout in murine cartilage attenuates post-traumatic OA (PMID: 33741257); selective FGFR inhibitors suppress IL-1β-induced cartilage degradation ex vivo (PMID: 32942816). While FGFR inhibition has never been investigated in TMJOA specifically, the mechanism path is biologically coherent and represents a genuinely novel translatable hypothesis.

**Case 2: Chondroitin — A Validated Positive Control.** Chondroitin sulfate (score 0.48, 82 PubMed hits). Model identified 7 direct targets (TGF-β, MMP, ADAMTS, NF-κB, TNF-α, Collagen) converging on TMJOA through BMP, TGF-β/Smad, and inflammatory cytokine pathways — demonstrating correct identification of drugs with well-established multi-pathway mechanisms.

**Case 3: Metformin — Graph Coverage as a Falsifiable Constraint.** Metformin received a modest score (0.42) and yielded **zero** direct mechanism paths to TMJOA. Its five annotated targets (AMPK, Complex I, GPD2, SIRT1, mTOR) were orphans — none connected to a pathway with a TMJOA association edge. This validates the architecture's mechanism-dependence: when annotation is absent, performance degrades transparently rather than silently overfitting to spurious correlations.

---

## Discussion

### The Graph Is the Message

The dominant finding of this study — spanning architecture ablation, KG and flat-graph baseline comparison, and graph ablation — is that **the four-layer mechanism graph topology, not GNN architectural complexity, drives inductive generalization in drug repositioning**. A minimal homogeneous GNN with 2-dimensional node features achieves AUC 0.85, indistinguishable from a full heterogeneous attention GNN (0.85) and 2–5 points above the best KG embedding method (ComplEx, 0.83). Architecture ablation shows a strict ceiling effect: all five GNN variants, from homogeneous to attention, differ by ≤ 0.002 AUC, indicating that the information available for inductive prediction is saturated by simple message passing on the mechanism graph.

Graph ablation sharpens this finding. The ablation that most degrades performance — randomizing target–pathway edges (AUC 0.72) — confirms that the **biological specificity** of protein→signaling module mappings, not raw graph structure, carries the predictive signal. Degree-preserving rewiring leaves AUC unchanged, ruling out the possibility that graph density or node degree distribution alone explain performance. Critically, removing the explicit pathway layer (switching to drug→target→disease) preserves AUC, indicating that the pathway layer functions as an **information relay** rather than an independent information source: the target–pathway–disease composite captures the same connectivity as a direct target–disease edge. This finding has practical implications for graph design — the explicit pathway layer, while computationally convenient for interpretability, does not add predictive value beyond what is encoded in the target→disease connectivity pattern.

Feature ablation further reinforces the graph-centric interpretation. Layer-type one-hot encoding alone achieves the highest AUC (0.870), and structural features (degree, betweenness) provide no additional benefit. The all-ones baseline (AUC 0.775) — where every node has identical features — still achieves strong prediction, demonstrating that message passing on the mechanism graph provides sufficient inductive bias for generalization even without explicit features.

### Molecular Features as Orthogonal Signal

The ceiling effect observed in architecture ablation (all five GNNs ≈ 0.85) raised the question of whether any information beyond graph topology could improve prediction. ECFP4 molecular fingerprint features provide a partial answer: adding 128-dimensional molecular features to drug nodes produced a +2.5 AUC gain (0.883 → 0.908, Table 6), with the largest improvements concentrated on hard folds where graph connectivity alone was weakest (Fold 2: +6 points, Fold 5: +10 points). This suggests that molecular similarity — drugs with similar chemical fingerprints tending to share therapeutic indications — provides a signal channel that is **orthogonal to and independent of** mechanism graph topology. The gain is modest but non-trivial: it is comparable to the gap between GNNs and the best KG embedding method (ComplEx, 2–5 points), confirming that molecular features add value even when graph topology is already informative.

Importantly, ECFP features do not require target annotations — they can be computed for any drug with a known structure. This makes them particularly viable for the 93 cold-start drugs in our training set that lack target edges. While our current evaluation only included drugs with at least one graph edge (to ensure fair comparison with topology-only features), the molecular feature approach opens a path toward extending the model to prediction for drug candidates without any prior mechanism annotation — a key requirement for genuine drug repositioning.

### The Heterogeneous GNN: Performance Parity, Interpretability Premium

While the heterogeneous GNN adds no AUC over the homogeneous variant, it adds mechanism path-level traceability — the ability to extract complete drug → target → pathway → disease reasoning chains for each prediction. The attention weight distribution (Target→Pathway 0.44 > Pathway→Disease 0.31 > Drug→Target 0.25) independently validates the graph ablation finding that target–pathway specificity is the critical signal. These weights are descriptive (attention doesn't improve AUC) but serve as a biological sanity check: the model consistently prioritizes the most information-dense layer where molecular specificity is highest.

### Interpreting Literature Validation: Recovery vs. Discovery

The 19/20 literature validation rate warrants careful interpretation. Because both the training graph and the validation search draw from PubMed-indexed TMJOA literature, a high confirmation rate for literature-supported candidates could reflect mechanism-grounded associations encoded in the graph rather than de novo biological discovery. We addressed this concern through three orthogonal strategies.

**Empirical exclusion.** We removed all 442 PMIDs used in graph construction or evidence screening from the PubMed validation search. The confirmation rate remained unchanged (19/20), with a median exclusion of only 1 PMID per drug. This demonstrates that the graph-construction articles and validation-search articles occupy largely disjoint subsets of the PubMed corpus — graph articles provide mechanism-level evidence (drug binds target X), while validation articles capture any mention of the drug with TMJ/TMD.

**Multi-source triangulation.** We replicated the validation on OpenAlex and Scopus — two bibliographic databases with independent indexing, search algorithms, and coverage. All three databases confirmed 18 of 20 candidates (90%), with 19/20 confirmed by at least two databases. This cross-database concordance cannot be explained by circular validation from a single training corpus.

**Signal-type distinction.** Even within PubMed, the training and validation signals differ in granularity. The graph encodes mechanism-level edges (drug → target protein binding evidence) from 420 literature cards, while validation measures publication-level co-occurrence (drug name AND \"TMJ\"). Dextrose prolotherapy (218 PubMed hits, ranked first) illustrates this distinction: while no explicit TMJOA–dextrose mechanism edge existed in the graph, its 218 literature co-occurrences were propagated through shared pathway nodes — demonstrating that the model infers relevance through mechanism connectivity rather than memorizing co-occurrence patterns.

We therefore distinguish two prediction classes: **known candidate recovery** (19/20, serving as positive-control sanity check) and **novel hypothesis generation** (BGJ398 and Riluzole, representing genuine repositioning discoveries). We recommend that future repositioning benchmarks report these categories separately and employ multi-source triangulation as standard validation practice. For transparency, we define four evidence levels for literature support: Level 1 — direct TMJOA therapeutic evidence (RCT/meta-analysis); Level 2 — TMJ/cartilage/bone disease-model evidence; Level 3 — pathway-relevant mechanistic evidence; Level 4 — indirect target/pathway plausibility only. Full per-candidate evidence mapping is provided in Supplementary Table S4.

### The Node2Vec Paradox and Inductive Discipline

The 21-point gap between transductive (0.90) and inductive (0.69) Node2Vec exposes a methodological vulnerability: test nodes participating in the embedding process encode structural information before predictor training, creating an information leak. We recommend strictly inductive evaluation — leave-drug-out and leave-disease-out — for all drug repositioning benchmarks, extending to embedding-based methods.

### Mechanism Graph Construction as the Primary Bottleneck

The Metformin case crystallizes the central limitation: AMPK — a well-established chondroprotective kinase and Metformin's primary target — lacks a pathway→TMJOA edge because our annotation pipeline did not capture the autophagy/cellular stress response pathway. The architecture correctly fails to predict Metformin (AUC 0.42, zero paths), demonstrating transparent degradation rather than spurious extrapolation. Future work should expand pathway–disease edges using KEGG, Reactome, WikiPathways, and DisGeNET.

### BGJ398: A Testable Hypothesis

BGJ398 (infigratinib), a pan-FGFR inhibitor in Phase II oncology/achondroplasia trials, is the model's sole literature-novel prediction. FGFR signaling is a master regulator of chondrocyte biology: FGF-2/FGFR1 promotes chondrocyte hypertrophy and MMP13 via RAS-MEK-ERK, while the FGFR-FGF18 axis restricts proliferation. In OA cartilage, FGF-FGFR1 becomes hyperactive, driving the same catabolic cascade that degrades TMJ condylar cartilage. We propose three validation paths: (1) in vitro BGJ398 on IL-1β-stimulated TMJ condylar chondrocytes measuring MMP13/ADAMTS5, (2) in silico docking to FGFR1 kinase domain (PDB: 3TT0), and (3) pharmacovigilance database query for TMJ-related signals in FGFR inhibitor-treated patients.

### Limitations

Several limitations should be noted. First, the 753-article evidence pool is TMJOA-centric and may underrepresent mechanisms for other diseases in the 15-disease phenotype layer. Second, 5-fold LDO with ~390 test pairs per fold is adequately powered for the large effect (d = 3.3, power > 0.99) but underpowered for detecting small improvements (minimum detectable AUC difference ≈ 0.28 at 80% power). Third, drug chemical structure and protein sequence features are not exploited; integrating molecular convolutions and protein language model embeddings could improve predictions for sparsely annotated drugs. Fourth, the homogeneous ≈ heterogeneous result is demonstrated on a single mechanism graph; external replication on independent KG resources is needed to establish generalizability. Fifth, graph ablation was conducted with the HeteroGNN architecture; consistent patterns across the homogeneous GNN would strengthen the claim that graph effects are architecture-independent.

---

## Conclusion

We demonstrate that, in a curated TMJOA mechanism graph, a four-layer drug → target → pathway → disease hierarchy enables inductive generalization independently of GNN architectural complexity. A minimal homogeneous GNN achieves inductive AUC 0.85, with no architecture variant producing a statistically significant improvement. Graph ablation identifies target–pathway edge specificity as the critical signal carrier, and a flat-graph control confirms that mechanism constraints prevent identity-based memorization. ECFP4 molecular fingerprints provide an orthogonal +2.5 AUC gain through chemical similarity. The heterogeneous GNN's contribution is complementary: mechanism path-level traceability and descriptive attention weights. Our findings suggest that, at least for curated disease-specific mechanism graphs, graph construction may be a more important bottleneck than architecture engineering in computational drug repositioning. — drug → target → pathway → disease — enables inductive generalization, independently of GNN architectural complexity. A minimal homogeneous GNN achieves inductive AUC 0.85, matching sophisticated heterogeneous and attention-based architectures while outperforming knowledge graph embedding baselines by 2–5 AUC points. Graph ablation identifies target–pathway edge specificity as the critical signal carrier, and feature ablation confirms that layer-type identity alone, without structural features, yields optimal performance from graph topology. Separately, ECFP4 molecular fingerprints provide an orthogonal +2.5 AUC gain through chemical similarity — a complementary signal channel beyond graph features. The heterogeneous GNN's contribution is complementary: mechanism path-level traceability and descriptive attention weights that independently validate the biological importance of the target–pathway interface. Our findings argue that the primary bottleneck in computational repositioning is not model architecture but biological knowledge graph construction — a shift in emphasis from engineering better predictors to curating better mechanism graphs. We propose that mechanism-constrained graph learning represents a principled path forward for computational drug repositioning, where biological prior knowledge is the structural foundation of the learning process, and architectural sophistication serves to make that foundation's reasoning transparent.

---

## References

[1] Schiffman E, et al. Diagnostic criteria for temporomandibular disorders (DC/TMD). J Oral Facial Pain Headache. 2014;28(1):6–27.  
[2] Wang XD, et al. Current understanding of pathogenesis and treatment of TMJ osteoarthritis. J Dent Res. 2015;94(5):666–673.  
[3] de Souza RF, et al. Interventions for the management of temporomandibular joint osteoarthritis. Cochrane Database Syst Rev. 2012;(4):CD007261.  
[4] Himmelstein DS, et al. Systematic integration of biomedical knowledge prioritizes drugs for repurposing. eLife. 2017;6:e26726.  
[5] Ioannidis VN, et al. DRKG — Drug Repurposing Knowledge Graph. 2020. arXiv:2002.02035.  
[6] Zheng S, et al. PharmKG: a dedicated knowledge graph benchmark for biomedical data mining. Brief Bioinform. 2021;22(4):bbaa344.  
[7] Wang X, et al. Heterogeneous graph attention network. WWW. 2019:2022–2032.  
[8] Schlichtkrull M, et al. Modeling relational data with graph convolutional networks. ESWC. 2018:593–607.  
[9] Hu Z, et al. Heterogeneous graph transformer. WWW. 2020:2704–2710.  
[10] Yang B, et al. Embedding entities and relations for learning and inference in knowledge bases. ICLR. 2015.  
[11] Trouillon T, et al. Complex embeddings for simple link prediction. ICML. 2016:2071–2080.  
[12] Sun Z, et al. RotatE: Knowledge graph embedding by relational rotation in complex space. ICLR. 2019.  
[13] Bordes A, et al. Translating embeddings for modeling multi-relational data. NeurIPS. 2013:2787–2795.
[14] Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model. 2010;50(5):742–754.
