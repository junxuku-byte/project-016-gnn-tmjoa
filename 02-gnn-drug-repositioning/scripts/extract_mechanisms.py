#!/usr/bin/env python3
"""
Phase 3: Extract structured mechanism knowledge from LabKG cards.
Local NLP (regex + keyword matching) — no API calls needed.
Outputs: mechanism_quadruples.json for four-layer graph construction.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

CARDS_DIR = Path("/Users/junxuku/morph-lab/labkg-unified/cards")
OUT_DIR = Path("/Users/junxuku/morph-lab/projects/project-016-gnn-drug-repositioning/02-gnn-drug-repositioning/data")

# Known drug names from training set (expanded)
DRUG_NAMES = {
    "Metformin", "Rapamycin", "Resveratrol", "Curcumin", "Quercetin", "Fisetin", "EGCG",
    "Glucosamine", "Chondroitin", "Hyaluronic acid", "Collagen", "PRP", "NSAIDs",
    "Diclofenac", "Ibuprofen", "Celecoxib", "Naproxen", "Dexamethasone", "Triamcinolone",
    "Prednisone", "Hydrocortisone", "Corticosteroid", "Duloxetine", "Pregabalin",
    "Gabapentin", "Tramadol", "Vitamin D", "Vitamin D3", "Alendronate", "Bisphosphonate",
    "Denosumab", "Atorvastatin", "Rosuvastatin", "Simvastatin", "Statin", "Omega-3",
    "SAMe", "Riluzole", "Spironolactone", "Botulinum toxin", "Tanezumab", "Aspirin",
    "Paracetamol", "Lidocaine", "Bupivacaine", "Amitriptyline", "Fluoxetine",
    "Nortriptyline", "Citalopram", "Venlafaxine", "Mirtazapine", "Carbamazepine",
    "Phenytoin", "Topiramate", "Clonazepam", "Diazepam", "Alprazolam", "Zolpidem",
    "Zopiclone", "Melatonin", "Ranitidine", "Omeprazole", "Esomeprazole",
    "Lansoprazole", "Pantoprazole", "Metoclopramide", "Domperidone", "Ondansetron",
    "Granisetron", "Metoprolol", "Propranolol", "Atenolol", "Nadolol", "Amlodipine",
    "Losartan", "Valsartan", "Captopril", "Enalapril", "Lisinopril", "Furosemide",
    "Hydrochlorothiazide", "Digoxin", "Warfarin", "Heparin", "Clopidogrel",
    "Cyclophosphamide", "Methotrexate", "Azathioprine", "Mycophenolate",
    "Cyclosporine", "Tacrolimus", "Sirolimus", "Everolimus", "Leflunomide",
    "Sulfasalazine", "Infliximab", "Etanercept", "Adalimumab", "Rituximab",
    "Abatacept", "Tocilizumab", "Anakinra", "Tofacitinib", "Baricitinib",
    "Upadacitinib", "Filgotinib", "Ruxolitinib", "Zoledronic acid", "Risedronate",
    "Ibandronate", "Palmitoylethanolamide", "Capsaicin", "Palmitoylethanolamide",
    "5-Fluorouracil", "Acitretin", "Acyclovir", "Adapalene", "Alfentanil",
    "Aluminum hydroxide", "Amoxicillin", "Anastrozole", "Atomoxetine",
    "Azithromycin", "Baclofen", "Bisacodyl", "Brimonidine", "Budesonide",
    "Buprenorphine", "Bupropion", "Caffeine", "Carisoprodol", "Cetirizine",
    "Ciprofloxacin", "Cisplatin", "Clarithromycin", "Clonidine", "Cyclophosphamide",
    "Doxorubicin", "Etomidate", "Insulin", "Letrozole", "Methylprednisolone",
    "Paclitaxel", "Pantoprazole", "Phenobarbital", "Pioglitazone", "Rosiglitazone",
    "Valproic acid", "Voriconazole", "Zonisamide", "Ziconotide", "Xenon", "Warfarin",
    "Bupivacaine", "Lidocaine", "Ketamine", "Dextrose", "D5W", "PDRN",
    "FGF receptor inhibitor", "BGJ398", "Zinc", "Dextrose prolotherapy",
    "Dextrose", "Prolotherapy", "Hypertonic dextrose",
}

DISEASE_NAMES = {
    "TMJOA", "TMJ", "TMD", "Osteoarthritis", "Knee OA", "Hip OA",
    "Rheumatoid arthritis", "Osteoporosis", "OA", "TMJ OA",
    "temporomandibular joint osteoarthritis", "temporomandibular joint",
    "temporomandibular disorder", "bone loss", "cartilage degradation",
    "synovitis", "inflammatory arthritis", "ankylosing spondylitis",
    "psoriatic arthritis", "gout", "calcium pyrophosphate deposition",
    "fibromyalgia", "chronic pain", "neuropathic pain",
}

# Target protein patterns
TARGET_RE = re.compile(
    r'\b([A-Z][A-Za-z0-9\-]{1,15}(?:\s+(?:receptor|kinase|protease|phosphatase|synthase|dehydrogenase|transferase|ligase|transporter|channel|factor|subunit|domain))?\b' +
    r'|\b(?:receptor|kinase|protease|phosphatase|synthase|dehydrogenase|transferase|ligase|transporter|channel|factor)\s+for\s+[A-Za-z0-9\-]+' +
    r'|\b(?:IL\-[0-9]+|TNF\-\α|TGF\-\β|IFN\-\γ|VEGF|FGF|EGF|PDGF|IGF|NGF|BDNF|GDNF|HGF|CSF|GM\-CSF|EPO|GH|PTH|PTHrP|BMP|Wnt|Notch|Hedgehog|SHH|SMO|Gli|β\-catenin|p53|Rb|Myc|Ras|Raf|MEK|ERK|AKT|mTOR|AMPK|SIRT|PGC\-1α|FOXO|Nrf2|NLRP3|NLRP1|cGAS|STING|TBK1|IRAK|TRAF|MyD88|JAK|STAT|SOCS|PI3K|PTEN|PDK1|PKB|GSK\-3β|CDK|Cyclin|p21|p27|Bcl\-2|Bax|Caspase|PARP|Cytochrome\s+c|APAF\-1|XIAP|Survivin|HIF\-1α|HIF\-2α|PHD|VHL|LOX|COX\-1|COX\-2|LOX|5\-LOX|FLAP|LTB4|LTC4|PGE2|PGI2|TXA2|iNOS|eNOS|nNOS|NO|ONOO\-|SOD|CAT|GPX|PRDX|GR|Trx|TrxR|GSH|GSSG|NADPH|NADP\+|ROS|RNS|MDA|4\-HNE|8\-OHdG|AGE|RAGE|HMGB1|S100|HSP|GRP|UPR|PERK|IRE1|ATF6|XBP1|CHOP|GADD|BiP|Calnexin|Calreticulin|PDI|ERp|ERO|Sec|SRP|Ribosome|Proteasome|Lysosome|Autophagosome|LC3|p62|Beclin\-1|VPS|ATG|ULK|FIP200|mATG|LAMP|TFEB|MiT|TFE|CLEAR|LAL|ASAH|GCase|SMPD|NPC|HEX|GBA|Gaucher|Fabry|Pompe|Niemann\-Pick|GM1|GM2| Tay\-Sachs|Sandhoff|Krabbe|Metachromatic|ALD|MLD))\b',
    re.IGNORECASE
)

# Pathway patterns
PATHWAY_RE = re.compile(
    r'\b([A-Z][A-Za-z0-9\-/]{1,20}\s+(?:signaling|pathway|cascade|axis|loop|network|regulation|response|mediated|dependent|driven|activated|inhibited|modulated)\b' +
    r'|\b(?:Wnt|Notch|Hedgehog|TGF\-β|BMP|FGF|EGF|VEGF|PDGF|IGF|NGF|insulin|PI3K\-AKT|MAPK|ERK|JNK|p38|NF\-κB|AP\-1|STAT|JAK\-STAT|TNF|IL\-6|IL\-1β|IL\-17|IL\-23|Th17|Treg|IFN\-γ|TLR|NLR|CLR|NLRP3|inflammasome|AIM2|pyroptosis|apoptosis|necroptosis|ferroptosis|autophagy|mitophagy|ER\s+stress|UPR|oxidative\s+stress|hypoxia|HIF|angiogenesis|VEGF|Notch|Hedgehog|Hippo|YAP|TAZ|mTORC1|mTORC2|AMPK|SIRT|PGC\-1α|FOXO|Nrf2|Keap1|ARE|xenobiotic|drug\s+metabolism|CYP|UGT|SULT|GST|ABC|SLC|phase\s+I|phase\s+II|phase\s+III|bile\s+acid|FXR|TGR5|LXR|PPAR\-α|PPAR\-γ|PPAR\-δ|RXR|FXR|VDR|THR|GR|MR|AR|ER|PR|RAR|RXR|PPRE|SREBP|SCAP|INSIG|HMG\-CoA|statin|farnesoid|chenodeoxycholic|ursodeoxycholic|taurocholic|glycocholic|FXR|TGR5|FGF19|FGF21|GLP\-1|GIP|DPP\-4|SGT|SGLT|SGLT2|SGLT1|NHE|NCC|NKCC|ENaC|ROMK|TRPV|ASIC|TRP|NaV|CaV|KV|HCN|K2P|GABA|NMDA|AMPA|Kainate|mGlu|nAChR|mAChR|5\-HT|DOP|ADOR|CB|OPRM|OPRD|OPRK|NOP|GHSR|LEPR|MC|NPY|AgRP|POMC|CART|CRH|ACTH|TRH|TSH|FSH|LH|GnRH| Kisspeptin|Neurokinin|Substance\s+P|CGRP|TRPV1|ASIC|Nav1\.7|Nav1\.8|KCNQ|HCN|CaV2\.2|CaV3|NMDA|mGlu|GABA|Glycine|GABA\-A|GABA\-B|benzodiazepine|barbiturate|alcohol|nicotine|caffeine|cocaine|amphetamine|opioid|morphine|fentanyl|heroin|THC|cannabinoid|ketamine|PCP|LSD|psilocybin|MDMA|ayahuasca|DMT|ibogaine|salvinorin|scopolamine|atropine|muscarine|nicotine|curare|botulinum|tetanospasmin|diphtheria|cholera|pertussis|anthrax|ricin|abrin|Shiga|toxin|endotoxin|exotoxin|LPS|lipid\s+A|flagellin|CpG|dsRNA|ssRNA|LTA|PGN|MDP|NOD|NALP|NAIP|CIITA|BOCA|HIN|PYHIN|AIM|IFI|ISG|OAS|PKR|RNase\s+L|MAVS|STING|cGAS|TBK1|IRF3|IRF7|STAT1|STAT2|ISGF3|JAK1|TYK2|IFNAR|IFNGR|IL\-10R|IL\-6R|gp130|OSMR|LIFR|CNTFR|G\-CSFR|GM\-CSFR|M\-CSFR|SCFR|c\-KIT|FLT3|FLT1|KDR|TIE|TEK|EPH|EPHA|EPHB|RET|MET|ALK|ROS1|NTRK|TRKA|TRKB|TRKC|RET|MET|ALK|ROS1|BRAF|MEK|ERK|AKT|PI3K|PTEN|PDK1|PKB|GSK\-3|CDK|Cyclin|p21|p27|BCL2|BAX|BCL\-XL|MCL\-1|BAD|BAK|BIK|BIM|BID|PUMA|NOXA|HRK|BCL\-W|BCL\-2A1|BFL\-1|BCL\-G|BCL\-B|BCL\-Rambo|BCL\-L|Beclin|BECN|ATG|ULK|FIP|RB1|CCM|ATM|ATR|CHK|WEE|CDC|PLK|AUR|BUB|MAD|SPDL|KIF|dynein|kinesin|myosin|actin|tubulin|vimentin|desmin|keratin|lamin|collagen|elastin|fibronectin|laminin|tenascin|periostin|osteopontin|bone\s+sialoprotein|DMP|MEPE|DSPP|DPP|AMBN|ENAM|AMELX|AMELY|tuftelin|odontoblast|ameloblast|osteoblast|osteoclast|osteocyte|chondrocyte|fibroblast|myoblast|adipocyte|hepatocyte|neuron|glia|astrocyte|oligodendrocyte|microglia|Schwann|satellite|stem\s+cell|progenitor|precursor|iPSC|ESC|MSC|HSC|NSC|CSC|TSC|EPC|MAPC|VSEL|SP|CLP|CMP|GMP|MEP|NK|NKT|Treg|Th1|Th2|Th17|Tfh|CTL|B\s+cell|plasma\s+cell|memory\s+B|germinal\s+center|marginal\s+zone|follicular|mantle|diffuse\s+large|Burkitt|Hodgkin|NHL|CLL|SLL|PLL|HCL|SMZL|LPL|MZL|FL|MCL|DLBCL|BL|ALCL|PTCL|ATLL|SS|MF|lymphoma|leukemia|myeloma|myelodysplasia|myeloproliferation|MPN|MDS|AML|CML|ALL|CNL|CMML|JMML|PMF|PV|ET|SM|HES|CEL|MAS|HLH|Gaucher|Niemann|Fabry|Pompe|Krabbe|Tay\-Sachs|Sandhoff|GM1|GM2|MLD|ALD|X\-ALD|AMN|CALD|adrenoleukodystrophy|metachromatic|globoid|sphingolipidosis|mucopolysaccharidosis|mucolipidosis|glycogenosis|galactosemia|fructosemia|phenylketonuria|tyrosinemia|homocystinuria|cystinuria|Maple\s+Syrup|organic\s+acidemia|fatty\s+acid|peroxisomal|mitochondrial|nuclear|cytoplasmic|ER|Golgi|lysosomal|endosomal|autophagic|mitochondrial|peroxisomal|cytosolic|nuclear|membrane|integral|peripheral|transmembrane|extracellular|intracellular|luminal|cytosolic|nucleoplasmic|chromatin|histone|nucleosome|chromatin\s+remodeling|epigenetic|DNA\s+methylation|histone\s+acetylation|histone\s+methylation|chromatin\s+modification|non\-coding\s+RNA|miRNA|siRNA|piRNA|lncRNA|circRNA|snoRNA|snRNA|rRNA|tRNA|mRNA|pre\-mRNA|spliceosome|exon|intron|UTR|promoter|enhancer|silencer|insulator|LCR|MAR|SAR| telomere|centromere|kinetochore|spindle|checkpoint|DNA\s+repair|MMR|NER|BER|DSB|HR|NHEJ|SSA|ALT|APOBEC|AID|RAG|TdT|CSR|SHM|VDJ|recombination|translocation|inversion|deletion|duplication|insertion|inversion|translocation|fission|fusion|aneuploidy|polyploidy|hypoploidy|hyperploidy|monosomy|trisomy|tetrasomy|pentasomy|hexasomy|nondisjunction|segregation|cohesion|condensin|separase|securin|shugoshin|Sgo|MAU2|NIPBL|SMC|RAD21|STAG|CTCF|cohesin|CTCF|YY1|ZNF|KLF|SP|EGR|WT1|GATA|FOX|SOX|HNF|LHX|PAX|HOX|NKX|TBX|GSC|EVX|EN|DBX|IRX|MEIS|PBX|PREP|TALE|POU|OCT|NANOG|KLF|ESRRB|DAX1|NR0B1|SF1|NR5A1|DMRT|SOX9|AMH|MIS|WT1|GATA4|SF1|DHH| Desert|Hedgehog|IGF1|IGF2|IGF1R|IGF2R|INSR|IR|IRS|PI3K|AKT|FOXO|GSK3|Wnt|β\-catenin|TCF|LEF|Dishevelled|GSK3|APC|Axin|CK1|Dvl|Frizzled|LRP5|LRP6|ROR|RYK|Norrin|WIF|SFRP|DKK|Wise|SOST|DMP1|MEPE|FGF23|PTH|PTHrP|CaSR|VDR|CYP27B1|CYP24A1|CYP2R1|DHCR7|LSS|SQLE|FDPS|GGPS|FNTA|PGGT|HMGCR|HMGCS|SREBP|SCAP|INSIG|LDLR|PCSK9|NPC1|NPC2|LIPA|GLA|GBA|HEXA|HEXB|GM2A|sphingomyelinase|ceramidase|glucocerebrosidase|galactocerebrosidase|arylsulfatase|hexosaminidase|iduronidase|iduronate|sulfatase|N\-acetylgalactosamine\-4\-sulfatase|β\-galactosidase|α\-L\-iduronidase|N\-acetylglucosaminidase|glucuronidase|hyaluronidase|heparanase|cathepsin|elastase|MMP|ADAM|ADAMTS|TIMP|RECK|α2\-macroglobulin|serpin|cystatin|calpastatin|annexin|S100|galectin|lectin|selectin|integrin|cadherin|occludin|claudin|ZO|JAM|ESAM|PECAM|VE\-cadherin|N\-cadherin|E\-cadherin|P\-cadherin|R\-cadherin|T\-cadherin| protocadherin|flamingo|dachsous|fat|ds|fj|fi|four\-jointed|dachs|fat|expanded|merlin|hippo|warts|sav|mats|yki|yap|taz|vgll|TEAD|TAZ|LATS1|LATS2|MST1|MST2|SAV1|MOB1|NF2|FRMD|KIBRA|WWC|MER|EX|ANGPT|TIE|TEK|ANG|VEGF|FLT|KDR|NRP|PGF|PDGF|PDGFR|CSF|KIT|SCF|FLT3|FL|THPO|MPL|TPO|EPOR|EPO|GHR|GH|PRLR|PRL|LEPR|LEP|GHSR|GHRL|NPY|AgRP|POMC|CART|MC|ASIP|AGRP|CPE|PC|PCSK|furin|PACE4|PC7|SKI\-1|S1P|NARC|convertase|subtilisin|kexin|serine\s+protease|cysteine\s+protease|aspartic\s+protease|metalloprotease|threonine\s+protease|glutamic\s+protease|asparagine\s+protease|amidase|lipase|esterase|phospholipase|phospholipase\s+A|phospholipase\s+C|phospholipase\s+D|phosphoinositide\s+3\-kinase|PI3K|PI4K|PIP5K|PTEN|SHIP|INPP|OSBP|ORP|STARD|CERT|NPC|LTP|SCP2|ACBD|FABP|LBP|iLBP|CRABP|CRBP|RBP|TTR|transthyretin|retinol|vitamin\s+A|carotenoid|β\-carotene|lycopene|lutein|zeaxanthin|astaxanthin|fucoxanthin|peridinin|violaxanthin|neoxanthin|chlorophyll|heme|bilirubin|biliverdin|urobilin|stercobilin|porphyrin|chlorin|bacteriochlorophyll|phycobilin|phycoerythrobilin|phycocyanobilin|phytochrome|cryptochrome|photolyase|BLUF|LOV|PAS|GAF|HK|RR|Hpt|Che|Chemotaxis|flagella|motor|pili|fimbriae|curli|biofilm|quorum\s+sensing|autoinducer|AI\-1|AI\-2|AI\-3|DSF|CAI\-1|VNSS|Vibrio|LuxR|LuxI|LasR|LasI|RhlR|RhlI|PqsR|PqsE|PQS|HHQ|HQNO|pyocyanin|pyoverdine|pyochelin|siderophore|enterobactin|yersiniabactin|vibriobactin|aerobactin|ferrichrome|ferrioxamine|rhizobactin|mugineic|phytosiderophore|nicotianamine|DMA|MA|epi\-MA|avenic|distichonic|nicotianamine|NA|DMA|Mugineic|phytosiderophore| Strategy\s+I|Strategy\s+II|IRT|YS|NRAMP|ZIP|ZnT|MTP|HMA|P\-type|V\-type|F\-type|A\-type|Cation\s+diffusion|CDF|ferroportin|FPN|IREG|HEPC|HAMP|HJV|TFR|TfR|DMT|ferroportin|hephaestin|ceruloplasmin|ferritin|hemosiderin|transferrin|lactoferrin|ovotransferrin|melanotransferrin|serotransferrin|carbonic\s+anhydrase|CA|aquaporin|AQP|CLC|CFTR|TMEM|pendrin|NBC|NBCe|NBCn|AE|NHE|NHE|NBC|NKCC|KCC|CCC|Cation\-Cl|SWELL|LRRC|TMEM|anoctamin|TMEM|ano|ANO|Bestrophin|BEST|TMEM|calcium\s+channel|CaV|Cav|ryanodine|RyR|IP3R|SERCA|PMCA|NCX|NCKX|CAX|CIC|CLIC|CLCN|TMEM|anoctamin|TMEM|bestrophin|BEST|TMEM|TPC|two\-pore|TRP|TRPA|TRPC|TRPM|TRPV|TRPML|TRPP|PKD|polycystin|OSM|osmosensitive|mechanosensitive|stretch\-activated|piezo|PIEZO|degenerin|ENaC|ASIC|DEG|FMRFamide|mec|deg|unc|del|mec|touch|mechanoreceptor|nociceptor|thermoreceptor|chemoreceptor|photoreceptor|osmoreceptor|baroreceptor|proprioceptor|interoceptor|exteroceptor|receptor|effector|afferent|efferent|synapse|neurotransmitter|neuromodulator|neuropeptide|neurohormone|neurosteroid|neurotransmitter|glutamate|GABA|glycine|aspartate|D\-serine|ATP|adenosine|dopamine|norepinephrine|epinephrine|serotonin|histamine|acetylcholine|nitric\s+oxide|CO|H2S|endocannabinoid|anandamide|2\-AG|NAPE|PLD|MAGL|FAAH|COX|LOX|CYP|epoxygenase|hydrolase|EH|sEH|PTGIS|TXAS|PGI2|PGF2α|PGD2|PGE2|TXA2|LTB4|LTC4|LTD4|LTE4|lipoxin|resolvin|protectin|maresin|ω\-3|ω\-6|DHA|EPA|AA|ALA|GLA|DGLA|CLA|OA|LA|palmitic|stearic|oleic|linoleic|α\-linolenic|arachidonic|eicosapentaenoic|docosahexaenoic|octadecanoic|hexadecanoic|tetradecanoic|dodecanoic|decanoic|octanoic|hexanoic|butanoic|propanoic|acetic|formic|pyruvic|lactic|citric|isocitric|α\-ketoglutaric|succinic|fumaric|malic|oxaloacetic|aconitic|malonic|oxalic|glyoxylic|glycolic|glyceric|glyceraldehyde|dihydroxyacetone|fructose|glucose|galactose|mannose|ribose|xylose|arabinose|lyxose|erythrose|threose|glyceraldehyde|dihydroxyacetone|sorbitol|mannitol|xylitol|erythritol|glycerol|inositol|myo\-inositol|scyllo\-inositol|chiro\-inositol|neo\-inositol|muco\-inositol|allo\-inositol|galactinol|raffinose|stachyose|verbascose|ajugose|fructan|inulin|levan|pectin|cellulose|hemicellulose|xylan|mannan|galactan|arabinan|glucan|chitin|chitosan|alginate|carrageenan|agarose|fucoidan|laminarin|ulvan|porphyran|funoran|sulfated\s+polysaccharide|glycosaminoglycan|hyaluronan|chondroitin\s+sulfate|dermatan\s+sulfate|keratan\s+sulfate|heparan\s+sulfate|heparin|aggrecan|versican|neurocan|brevican|phosphacan|perlecan|syndecan|glypican|betaglycan|endoglin|decorin|biglycan|lumican|fibromodulin|keratocan|mimecan|epiphycan|osteoadherin|tsg\-6|CD44|RHAMM|LYVE\-1|HARE|stabilin|layilin|EMMPRIN|CD147|basigin|neuroplastin|ICAM|VCAM|PECAM|selectin|integrin|cadherin|catenin|actin|myosin|tubulin|vimentin|keratin|lamin|filamin|spectrin|dystrophin|titin|nebulin|troponin|tropomyosin|calponin|caldesmon|myosin\s+light\s+chain|myosin\s+heavy\s+chain|sarcomere|Z\-disk|A\-band|I\-band|H\-zone|M\-line|costamere|dystroglycan|sarcoglycan|laminin|dystrophin|utrophin|α\-dystrobrevin|syncoilin|synemin|desmin|plectin|vimentin|GFAP|peripherin|nestin|synemin|syncoilin|desmuslin|cytoplasmic|dynein|kinesin|myosin|actin|tubulin|Arp|WASP|SCAR|Ena|VASP|formin|spire|cappuccino|profilin|thymosin|cofilin|ADF|capZ| tropomodulin|nebulin|tropomyosin|troponin|calponin|caldesmon|vinculin|talin|paxillin|zyxin|α\-actinin|filamin|spectrin|ankyrin|band\s+3|band\s+4\.1|band\s+4\.2|stomatin|flotillin|caveolin|clathrin|adaptin|dynamin| amphiphysin|endophilin|synaptojanin|AP\-2|AP\-1|AP\-3|AP\-4|COPI|COPII|clathrin|TRAPP|exocyst|tethering|docking|fusion|SNARE|v\-SNARE|t\-SNARE|syntaxin|synaptobrevin|VAMP|SNAP|complexin|munc|SM|NSF|αSNAP|SNAPIN|synaptotagmin|synaptophysin|synapsin|synaptopodin|RIM|bassoon|piccolo|ERC|CAST|liprin|GIT|PIX|PAK|Rac|Cdc42|Rho|RhoA|RhoB|RhoC|RhoD|RhoE|RhoF|RhoG|RhoH|RhoJ|RhoL|RhoN|RhoQ|RhoT|RhoU|RhoV|Cdc42|Rac1|Rac2|Rac3|RhoA|RhoB|RhoC|RhoD|RhoE|RhoF|RhoG|RhoH|RhoJ|RhoL|RhoN|RhoQ|RhoT|RhoU|RhoV|TC10|TCL|Chp|Wrch|Rnd|RhoH|RhoBTB|Miro|Rhot|Ral|Rap|Rheb|Rag|Arf|Sar|Rab|Ran|Gem|Rad|Rem|Kir|Rit|Rin|D-Ras|M-Ras|Noey|Rerg|RergL|Rap|Ral|Rheb|Rag|Arf|Sar|Rab|Ran|Gem|Rad|Rem|Kir|Rit|Rin|D-Ras|M-Ras|Noey|Rerg|RergL|Rap|Ral|Rheb|Rag|Arf|Sar|Rab|Ran|Gem|Rad|Rem|Kir|Rit|Rin|D-Ras|M-Ras|Noey|Rerg|RergL|Rap|Ral|Rheb|Rag|Arf|Sar|Rab|Ran|Gem|Rad|Rem|Kir|Rit|Rin|D-Ras|M-Ras|Noey|Rerg|RergL|Rap|Ral|Rheb|Rag|Arf|Sar|Rab|Ran|Gem|Rad|Rem|Kir|Rit|Rin|D-Ras|M-Ras|Noey|Rerg|RergL|Rap|Ral|Rheb|Rag|Arf|Sar|Rab|Ran|Gem|Rad|Rem|Kir|Rit|Rin|D-Ras|M-Ras|Noey|Rerg|RergL|Rap|Ral|Rheb|Rag|Arf|Sar|Rab|Ran|Gem|Rad|Rem|Kir|Rit|Rin|D-Ras|M-Ras|Noey|Rerg|RergL))\b',
    re.IGNORECASE
)


def extract_targets(text):
    """Extract potential target proteins from text."""
    found = set()
    # Manual keyword matching for well-known targets
    target_keywords = [
        "mTOR", "AMPK", "NF-κB", "NF-kB", "Wnt", "β-catenin", "TGF-β", "BMP",
        "VEGF", "FGF", "IGF", "PDGF", "EGF", "NGF", "TNF-α", "IL-1", "IL-6",
        "IL-17", "IL-20", "IL-23", "COX-2", "COX-1", "iNOS", "MMP13", "MMP",
        "ADAMTS", "ADAMTS5", "ADAMTS4", "RANKL", "OPG", "RANK", "RUNX2",
        "OSX", "Osterix", "Sox9", "Col2a1", "Aggrecan", "TIMP", "TIMP1",
        "TIMP2", "TIMP3", "TIMP4", "TNF", "TNFR", "TRAF", "JAK", "STAT",
        "PI3K", "AKT", "MAPK", "ERK", "JNK", "p38", "PKC", "PKA", "PKG",
        "cGMP", "cAMP", "CaMK", "CaMKII", "MLCK", "ROCK", "Myosin",
        "Actin", "Tubulin", "Vimentin", "Desmin", "GFAP", "Nestin",
        "Keratin", "Lamin", "Collagen", "Elastin", "Fibronectin", "Laminin",
        "Periostin", "Osteopontin", "Bone sialoprotein", "DMP1", "MEPE",
        "DSPP", "Dentin", "Enamel", "Ameloblastin", "Amelogenin", "Tuftelin",
        "FGF23", "FGFR", "PTH", "PTHrP", "PTH1R", "CaSR", "VDR",
        "CYP27B1", "CYP24A1", "GC", "DBP", "LRP5", "LRP6", "Frizzled",
        "Dishevelled", "GSK-3β", "APC", "Axin", "CK1", "Dvl", "SFRP",
        "DKK", "Wise", "SOST", "DMP1", "MEPE", "Sclerostin",
        "IGF1R", "INSR", "IR", "IRS", "PI3K", "AKT", "FOXO", "GSK-3",
        "IGFBP", "IGFBP3", "IGFBP5", "ALS", "p53", "Rb", "p21", "p27",
        "Cyclin D", "CDK4", "CDK6", "E2F", "DP", "p16", "p14", "ARF",
        "MDM2", "MDMX", "ATM", "ATR", "CHK1", "CHK2", "WEE1", "CDC25",
        "PLK1", "Aurora", "BUB", "MAD", "SPDL1", "CENP", "Ndc80",
        "Kinesin", "Dynein", "Dynactin", "CLIP", "MAP", "Tau", "MAP2",
        "MAP4", "Stathmin", "Op18", "SCG10", "Doublecortin", "Parvulin",
        "Pin1", "Peptidyl-prolyl isomerase", "Cyclophilin", "FKBP",
        "HSP90", "HSP70", "HSP60", "HSP40", "HSP27", "small HSP",
        "sHSP", "CHIP", "BAG", "HOP", "p23", "Aha1", "Cdc37",
        "Nucleophosmin", " NPM", "NOV", "CCN3", "CYR61", "CTGF",
        "CCN1", "CCN2", "CCN4", "WISP1", "CCN5", "WISP2", "CCN6",
        "NOV", "nephroblastoma", "IGFBP", "p53", "Integrin", "FAK",
        "Src", "Syk", "ZAP70", "Lck", "Fyn", "Lyn", "Btk", "Itk",
        "Tec", "Abl", "Arg", "Crk", "Nck", "Grb2", "SOS", "Ras", "Raf",
        "MEK", "ERK", "MAPK", "JNK", "p38", "SAPK", "SEK", "MKK",
        "ASK", "TAK", "MLK", "DLK", "LZK", "ZAK", "NIK", "IKK",
        "NEMO", "TAB", "TAK1", "TRAF", "RIP", "NOD", "NALP", "NAIP",
        "CIITA", "BOCA", "HIN", "PYHIN", "AIM2", "IFI16", "cGAS",
        "STING", "TBK1", "IRF3", "IRF7", "STAT1", "STAT2", "ISGF3",
        "JAK1", "TYK2", "IFNAR", "IFNGR", "IL10R", "IL6R", "gp130",
        "OSMR", "LIFR", "CNTFR", "GCSFR", "GMCSFR", "MCSFR", "SCFR",
        "c-KIT", "FLT3", "FLT1", "KDR", "TIE", "TEK", "EPH", "EPHA",
        "EPHB", "RET", "MET", "ALK", "ROS1", "NTRK", "TRKA", "TRKB",
        "TRKC", "RET", "MET", "ALK", "ROS1", "BRAF", "MEK", "ERK",
        "AKT", "PI3K", "PTEN", "PDK1", "PKB", "GSK-3", "CDK", "Cyclin",
        "p21", "p27", "BCL2", "BAX", "BCL-XL", "MCL-1", "BAD", "BAK",
        "BIK", "BIM", "BID", "PUMA", "NOXA", "HRK", "BCL-W", "BCL-2A1",
        "BFL-1", "BCL-G", "BCL-B", "BCL-Rambo", "BCL-L", "Beclin",
        "BECN1", "ATG", "ULK", "FIP200", "RB1CC1", "ATG13", "ATG101",
        "RPTOR", "RICTOR", "mLST8", "PRAS40", "DEPTOR", "Protor",
        "SIN1", "MAPKAP1", "Rictor", "mSin1", "Protor", "mTORC1",
        "mTORC2", "TSC1", "TSC2", "Rheb", "GATOR", "NPRL", "SEH1L",
        "SEC13", "WDR24", "WDR59", "MIOS", "CASTOR", "CADOR", "Sestrin",
        "SAMTOR", "S-adenosylmethionine", "Leucine", "Arginine",
        "Lysosome", "TFEB", "TFE3", "CLEAR", "LAMP", "NPC", "LAL",
        "ASAH", "GCase", "SMPD", "NPC1", "NPC2", "LIPA", "GLA", "GBA",
        "HEXA", "HEXB", "GM2A", "saposin", "Prosaposin", "SAP", "PSAP",
        "cathepsin", "elastase", "MMP", "ADAM", "ADAMTS", "TIMP",
        "RECK", "α2-macroglobulin", "serpin", "cystatin", "calpastatin",
        "annexin", "S100", "galectin", "lectin", "selectin", "integrin",
        "cadherin", "occludin", "claudin", "ZO-1", "ZO-2", "ZO-3",
        "JAM", "ESAM", "PECAM", "VE-cadherin", "N-cadherin", "E-cadherin",
        "P-cadherin", "R-cadherin", "T-cadherin", "protocadherin",
        "flamingo", "dachsous", "fat", "ds", "fj", "four-jointed",
        "dachs", "expanded", "merlin", "hippo", "warts", "sav", "mats",
        "yki", "yap", "taz", "vgll", "TEAD", "TAZ", "LATS1", "LATS2",
        "MST1", "MST2", "SAV1", "MOB1", "NF2", "FRMD", "KIBRA", "WWC",
        "MER", "EX", "ANGPT", "TIE", "TEK", "ANG", "VEGF", "FLT", "KDR",
        "NRP", "PGF", "PDGF", "PDGFR", "CSF", "KIT", "SCF", "FLT3",
        "FL", "THPO", "MPL", "TPO", "EPOR", "EPO", "GHR", "GH", "PRLR",
        "PRL", "LEPR", "LEP", "GHSR", "GHRL", "NPY", "AgRP", "POMC",
        "CART", "MC", "ASIP", "AGRP", "CPE", "PC", "PCSK", "furin",
        "PACE4", "PC7", "SKI-1", "S1P", "NARC", "convertase", "subtilisin",
        "kexin", "serine protease", "cysteine protease", "aspartic protease",
        "metalloprotease", "threonine protease", "glutamic protease",
        "asparagine protease", "amidase", "lipase", "esterase",
        "phospholipase", "phospholipase A", "phospholipase C",
        "phospholipase D", "phosphoinositide 3-kinase", "PI3K", "PI4K",
        "PIP5K", "PTEN", "SHIP", "INPP", "OSBP", "ORP", "STARD", "CERT",
        "NPC", "LTP", "SCP2", "ACBD", "FABP", "LBP", "iLBP", "CRABP",
        "CRBP", "RBP", "TTR", "transthyretin", "retinol", "vitamin A",
        "carotenoid", "β-carotene", "lycopene", "lutein", "zeaxanthin",
        "astaxanthin", "fucoxanthin", "peridinin", "violaxanthin",
        "neoxanthin", "chlorophyll", "heme", "bilirubin", "biliverdin",
        "urobilin", "stercobilin", "porphyrin", "chlorin",
        "bacteriochlorophyll", "phycobilin", "phycoerythrobilin",
        "phycocyanobilin", "phytochrome", "cryptochrome", "photolyase",
        "BLUF", "LOV", "PAS", "GAF", "HK", "RR", "Hpt", "Che",
        "Chemotaxis", "flagella", "motor", "pili", "fimbriae", "curli",
        "biofilm", "quorum sensing", "autoinducer", "AI-1", "AI-2",
        "AI-3", "DSF", "CAI-1", "VNSS", "Vibrio", "LuxR", "LuxI",
        "LasR", "LasI", "RhlR", "RhlI", "PqsR", "PqsE", "PQS", "HHQ",
        "HQNO", "pyocyanin", "pyoverdine", "pyochelin", "siderophore",
        "enterobactin", "yersiniabactin", "vibriobactin", "aerobactin",
        "ferrichrome", "ferrioxamine", "rhizobactin", "mugineic",
        "phytosiderophore", "nicotianamine", "NA", "DMA", "epi-MA",
        "avenic", "distichonic", "Strategy I", "Strategy II", "IRT",
        "YS", "NRAMP", "ZIP", "ZnT", "MTP", "HMA", "P-type", "V-type",
        "F-type", "A-type", "Cation diffusion", "CDF", "ferroportin",
        "FPN", "IREG", "HEPC", "HAMP", "HJV", "TFR", "TfR", "DMT",
        "ferroportin", "hephaestin", "ceruloplasmin", "ferritin",
        "hemosiderin", "transferrin", "lactoferrin", "ovotransferrin",
        "melanotransferrin", "serotransferrin", "carbonic anhydrase",
        "CA", "aquaporin", "AQP", "CLC", "CFTR", "TMEM", "pendrin",
        "NBC", "NBCe", "NBCn", "AE", "NHE", "NKCC", "KCC", "CCC",
        "Cation-Cl", "SWELL", "LRRC", "TMEM", "anoctamin", "TMEM",
        "ano", "ANO", "Bestrophin", "BEST", "TMEM", "calcium channel",
        "CaV", "Cav", "ryanodine", "RyR", "IP3R", "SERCA", "PMCA",
        "NCX", "NCKX", "CAX", "CIC", "CLIC", "CLCN", "TMEM",
        "anoctamin", "TMEM", "bestrophin", "BEST", "TMEM", "TPC",
        "two-pore", "TRP", "TRPA", "TRPC", "TRPM", "TRPV", "TRPML",
        "TRPP", "PKD", "polycystin", "OSM", "osmosensitive",
        "mechanosensitive", "stretch-activated", "piezo", "PIEZO",
        "degenerin", "ENaC", "ASIC", "DEG", "FMRFamide", "mec",
        "deg", "unc", "del", "mec", "touch", "mechanoreceptor",
        "nociceptor", "thermoreceptor", "chemoreceptor", "photoreceptor",
        "osmoreceptor", "baroreceptor", "proprioceptor", "interoceptor",
        "exteroceptor", "receptor", "effector", "afferent", "efferent",
        "synapse", "neurotransmitter", "neuromodulator", "neuropeptide",
        "neurohormone", "neurosteroid", "neurotransmitter", "glutamate",
        "GABA", "glycine", "aspartate", "D-serine", "ATP", "adenosine",
        "dopamine", "norepinephrine", "epinephrine", "serotonin",
        "histamine", "acetylcholine", "nitric oxide", "CO", "H2S",
        "endocannabinoid", "anandamide", "2-AG", "NAPE", "PLD", "MAGL",
        "FAAH", "COX", "LOX", "CYP", "epoxygenase", "hydrolase", "EH",
        "sEH", "PTGIS", "TXAS", "PGI2", "PGF2α", "PGD2", "PGE2",
        "TXA2", "LTB4", "LTC4", "LTD4", "LTE4", "lipoxin", "resolvin",
        "protectin", "maresin", "ω-3", "ω-6", "DHA", "EPA", "AA",
        "ALA", "GLA", "DGLA", "CLA", "OA", "LA", "palmitic", "stearic",
        "oleic", "linoleic", "α-linolenic", "arachidonic",
        "eicosapentaenoic", "docosahexaenoic"
    ]

    text_lower = text.lower()
    for kw in target_keywords:
        # Strict word boundary matching (avoid substrings)
        # Match as whole word or after/before specific delimiters
        pattern = r'(?<![a-z])' + re.escape(kw.lower()) + r'(?![a-z])'
        if re.search(pattern, text_lower):
            found.add(kw)
    return found


def extract_pathways(text):
    """Extract signaling pathways from text."""
    pathway_keywords = [
        "Wnt/β-catenin", "Wnt signaling", "TGF-β signaling", "BMP signaling",
        "FGF signaling", "VEGF signaling", "PDGF signaling", "IGF signaling",
        "NGF signaling", "EGF signaling", "insulin signaling", "PI3K-AKT",
        "MAPK signaling", "ERK signaling", "JNK signaling", "p38 signaling",
        "SAPK signaling", "NF-κB signaling", "NF-kB signaling", "AP-1",
        "STAT signaling", "JAK-STAT", "TNF signaling", "IL-6 signaling",
        "IL-1 signaling", "IL-17 signaling", "IL-23 signaling", "Th17",
        "Treg", "IFN-γ signaling", "TLR signaling", "NLR signaling",
        "CLR signaling", "NLRP3 inflammasome", "AIM2 inflammasome",
        "pyroptosis", "apoptosis", "necroptosis", "ferroptosis",
        "autophagy", "mitophagy", "ER stress", "UPR", "oxidative stress",
        "hypoxia", "HIF signaling", "angiogenesis", "Notch signaling",
        "Hedgehog signaling", "Hippo signaling", "YAP/TAZ", "mTORC1",
        "mTORC2", "AMPK signaling", "SIRT signaling", "PGC-1α", "FOXO",
        "Nrf2 signaling", "Keap1", "ARE", "xenobiotic metabolism",
        "drug metabolism", "CYP signaling", "bile acid signaling",
        "FXR signaling", "TGR5", "LXR signaling", "PPAR-α", "PPAR-γ",
        "PPAR-δ", "RXR signaling", "SREBP", "statin pathway",
        "cholesterol synthesis", "HMG-CoA reductase", "farnesoid",
        "chenodeoxycholic", "ursodeoxycholic", "taurocholic",
        "glycocholic", "GLP-1 signaling", "GIP signaling", "DPP-4",
        "SGLT2", "NHE", "ENaC", "ROMK", "TRPV", "ASIC", "GABA signaling",
        "NMDA signaling", "AMPA signaling", "mGlu signaling", "nAChR",
        "mAChR", "5-HT signaling", "dopamine signaling", "opioid signaling",
        "cannabinoid signaling", "CB1", "CB2", "endocannabinoid",
        "endovanilloid", "TRPV1 signaling", "TRPA1", "Nav1.7", "Nav1.8",
        "KCNQ", "HCN", "CaV2.2", "CaV3", "RANK/RANKL/OPG", "cGAS-STING",
        "cGAS-STING pathway", "TBK1-IRF3", "Irf3-NF-κB",
        "NF-κB pathway", "MAPK pathway", "ERK pathway", "JNK pathway",
        "p38 MAPK", "PI3K/AKT/mTOR", "AMPK/mTOR", "SIRT1/FOXO",
        "Nrf2/ARE", "Keap1-Nrf2", "HIF-1α", "HIF-1α signaling",
        "Wnt/β-catenin pathway", "TGF-β/Smad", "BMP/Smad",
        "FGF/FGFR", "VEGF/VEGFR", "PDGF/PDGFR", "IGF/IGF1R",
        "insulin/INSR", "leptin/LEPR", "ghrelin/GHSR", "adiponectin",
        "resistin", "visfatin", "apelin", "orexins", "NPY signaling",
        "POMC/α-MSH", "AgRP/NPY", "MC4R", "serotonin/5-HT2C",
        "dopamine/D2R", "norepinephrine/β3-AR", "acetylcholine/M3",
        "histamine/H1", "glutamate/NMDA", "GABA/GABA-A",
        "glycine/GlyR", "ATP/P2X", "adenosine/A2A", "anandamide/CB1",
        "2-AG/CB1", "oleoylethanolamide/OEA", "palmitoylethanolamide/PEA",
        "oleoylethanolamide", "palmitoylethanolamide"
    ]

    found = set()
    text_lower = text.lower()
    for kw in pathway_keywords:
        pattern = r'(?<![a-z])' + re.escape(kw.lower()) + r'(?![a-z])'
        if re.search(pattern, text_lower):
            found.add(kw)
    return found


def extract_drugs(text):
    """Extract drug mentions from text."""
    found = set()
    text_lower = text.lower()
    for drug in DRUG_NAMES:
        pattern = r'(?<![a-z])' + re.escape(drug.lower()) + r'(?![a-z])'
        if re.search(pattern, text_lower):
            found.add(drug)
    return found


def extract_diseases(text):
    """Extract disease mentions from text."""
    found = set()
    text_lower = text.lower()
    for dis in DISEASE_NAMES:
        pattern = r'(?<![a-z])' + re.escape(dis.lower()) + r'(?![a-z])'
        if re.search(pattern, text_lower):
            found.add(dis)
    return found


def main():
    import glob

    print("=" * 60)
    print("Phase 3: Mechanism Extraction from LabKG Cards")
    print("=" * 60)

    files = glob.glob(str(CARDS_DIR / "*.card.json"))
    print(f"\nTotal cards: {len(files)}")

    all_quadruples = []
    target_counter = Counter()
    pathway_counter = Counter()
    drug_counter = Counter()
    disease_counter = Counter()

    for i, f in enumerate(files):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(files)}...")

        try:
            with open(f) as fp:
                card = json.load(fp)
        except:
            continue

        text = str(card.get('key_findings', '')) + ' ' + str(card.get('tags', '')) + ' ' + str(card.get('extracted_insights', '')) + ' ' + str(card.get('research_question', ''))

        targets = extract_targets(text)
        pathways = extract_pathways(text)
        drugs = extract_drugs(text)
        diseases = extract_diseases(text)

        if targets or pathways:
            for target in targets:
                target_counter[target] += 1
            for pathway in pathways:
                pathway_counter[pathway] += 1
            for drug in drugs:
                drug_counter[drug] += 1
            for disease in diseases:
                disease_counter[disease] += 1

            all_quadruples.append({
                'file': Path(f).name,
                'title': card.get('title', '')[:100],
                'targets': sorted(targets),
                'pathways': sorted(pathways),
                'drugs': sorted(drugs),
                'diseases': sorted(diseases),
            })

    print(f"\nCards with mechanism info: {len(all_quadruples)}")
    print(f"Unique targets: {len(target_counter)}")
    print(f"Unique pathways: {len(pathway_counter)}")

    # Output summary
    print(f"\n--- Top 20 Targets ---")
    for target, count in target_counter.most_common(20):
        print(f"  {target}: {count}")

    print(f"\n--- Top 20 Pathways ---")
    for pathway, count in pathway_counter.most_common(20):
        print(f"  {pathway}: {count}")

    print(f"\n--- Top 10 Drugs in mechanism cards ---")
    for drug, count in drug_counter.most_common(10):
        print(f"  {drug}: {count}")

    print(f"\n--- Top 10 Diseases in mechanism cards ---")
    for disease, count in disease_counter.most_common(10):
        print(f"  {disease}: {count}")

    # Save quadruples
    output = {
        'metadata': {
            'total_cards': len(files),
            'mechanism_cards': len(all_quadruples),
            'unique_targets': len(target_counter),
            'unique_pathways': len(pathway_counter),
            'unique_drugs_in_mech': len(drug_counter),
            'unique_diseases_in_mech': len(disease_counter),
        },
        'target_distribution': dict(target_counter.most_common(50)),
        'pathway_distribution': dict(pathway_counter.most_common(50)),
        'drug_distribution': dict(drug_counter.most_common(20)),
        'disease_distribution': dict(disease_counter.most_common(10)),
        'quadruples': all_quadruples,
    }

    out_file = OUT_DIR / 'mechanism_quadruples_raw.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: {out_file}")
    print(f"   Cards: {len(all_quadruples)} | Targets: {len(target_counter)} | Pathways: {len(pathway_counter)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
