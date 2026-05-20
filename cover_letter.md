# Cover Letter

**Date**: May 20, 2026

**To**: Editor-in-Chief, Briefings in Bioinformatics

**Manuscript Title**: Mechanism-Constrained Graph Learning for Inductive Drug Repositioning: The Graph Is the Message

---

Dear Editor,

We submit "Mechanism-Constrained Graph Learning for Inductive Drug Repositioning: The Graph Is the Message" for consideration in Briefings in Bioinformatics.

**What we did.** We constructed a four-layer heterogeneous graph — drug → target protein → signaling pathway → disease phenotype — to reformulate drug repositioning as a mechanism-constrained hierarchical prediction problem. We then made an unexpected empirical finding: a minimal homogeneous GNN on this graph achieves inductive leave-drug-out AUC of 0.85, indistinguishable from a full heterogeneous attention GNN. Architecture does not matter; the graph topology does.

**Why it matters.** Most published GNN repositioning models focus on architectural innovation while evaluating under transductive protocols that allow structural shortcuts. Our work demonstrates that (a) mechanism graph construction, not model architecture, is the primary bottleneck in computational repositioning, and (b) widely used transductive baselines (notably Node2Vec) overestimate performance by 21 AUC points. We provide an extensively validated framework with architecture ablation (5 models), graph ablation (5 variants), knowledge graph embedding baselines (4 methods), molecular feature enhancement (ECFP4), and multi-source literature validation across three independent databases (PubMed, OpenAlex, Scopus; 18/20 confirmed by all three).

**Why Briefings in Bioinformatics.** The paper's core contribution — that graph schema design is a first-order modeling decision that should precede architecture selection — is a methodological insight directly relevant to the bioinformatics community. The findings on transductive evaluation leakage and inductive discipline apply broadly beyond our TMJOA use case.

**Data availability.** All code, graph files, training data, and supplementary tables will be deposited in a public GitHub repository upon acceptance. A one-click reproduction script (`reproduce.sh`) is included.

We confirm that this manuscript has not been published elsewhere and is not under consideration by another journal. All authors have approved the manuscript and agree with its submission to Briefings in Bioinformatics.

Sincerely,

Li Haosen (Corresponding Author)
Department of Stomatology, Tongji Hospital, Tongji Medical College
Huazhong University of Science and Technology
Wuhan, Hubei, 430030, China
Email: chongchong@tjh.tjmu.edu.cn
