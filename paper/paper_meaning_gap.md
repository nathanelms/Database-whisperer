# The Meaning Gap: Measuring Structural Sense Distinctions That Standard Tools Do Not Count

**Nathan Elms**

---

## Abstract

Text-bearing databases contain meaning distinctions that no standard tool measures. Keywords detect presence, not sense. Embeddings detect similarity, not structure. We call this unmeasured distance the meaning gap.

We present a deterministic, unsupervised method for measuring it. Ten clause-level features are extracted from each word-in-context instance. Substitution neighborhoods -- the set of concepts that could replace a word at a given structural address -- are computed from co-occurrence, and addresses with similar neighborhoods are collapsed into resolved meanings. A data-driven classification discovers three feature layers: operators (negation, modality) that must never be collapsed, meaning features that collapse by similarity, and expression features that collapse freely. Four formally defined scores -- confusion, completeness, predictability, hazard -- audit meaning quality at four grains.

Applied to four domains with auto-detected concepts and zero configuration, the audit produces domain-specific results (confusion 0.038--0.496, predictability 0.506--0.956). On a clinical disambiguation retrieval task, meaning-addressed retrieval outperforms both keyword and domain-tuned embedding baselines (macro F1 0.944 vs. 0.852). A sensitivity analysis confirms stability across Jaccard collapse thresholds (0.3--0.7), and bootstrap resampling shows score CVs below 1%. The retrieval evaluation is single-domain; cross-domain and standard benchmark evaluation are future work. All code is available as `pip install database_whisper`.

---

## 1. Introduction

A doctor writes "positive" in a clinical note. A keyword search retrieves every note containing that string. An embedding model ranks them by distributional similarity. A language model generates a summary. All three systems treat "positive family history" and "positive blood culture" as the same word, because their representations lack the structural feature that separates them: the co-occurring concept.

This is not a hard case. A medical student distinguishes these senses in seconds. The distinction depends on a single structural feature -- what other clinical term appears nearby -- that is present in the text and extractable by pattern matching. No training data is needed. No ontology is needed. The feature is sitting in the sentence, unused.

The problem is not that the distinction is difficult. The problem is that no standard tool measures how many such distinctions a dataset contains, which words are most dangerous, or where the structural signal runs out. Keywords count presence. Embeddings measure similarity. Quality metrics measure fluency, coherence, and factual accuracy. None measures the structural resolution of meaning.

We call the unmeasured distance between what text structurally expresses and what systems can resolve the *meaning gap*. This paper presents a method for measuring it.

### 1.1 The Gap in Practice

The meaning gap has practical consequences wherever text is stored, retrieved, or generated.

**In databases:** A clinical database indexes by patient, date, and note type. It does not index by what "positive" means in each note. A search for "positive test results" returns family history mentions, physical exam findings, and lab results indiscriminately, because the schema was designed before the data arrived and does not capture meaning distinctions the text contains.

**A note on contextual embeddings:** BERT-family models do produce different representations for "positive blood culture" and "positive family history" -- the contextualized token embedding differs across contexts. Our claim is not that embeddings cannot distinguish these senses, but that (1) embeddings do not *count* how many senses exist, (2) embeddings do not *name* the structural feature that separates them, and (3) no standard embedding-based tool audits a corpus for the number, distribution, and danger of its meaning distinctions. The meaning gap is a measurement problem, not a disambiguation problem. We evaluate disambiguation directly in Section 4.6.

**In retrieval:** Embedding-based retrieval ranks "positive blood culture" and "positive family history" as similar because both appear in clinical contexts near the same vocabulary. The embedding space encodes distributional neighborhood, not structural role. We evaluate this directly in Section 4.6, where meaning-addressed retrieval outperforms both keyword and embedding baselines on a clinical disambiguation task.

**In generation:** LLMs trained on clinical text default to the dominant sense of ambiguous words. When generating clinical notes, "discharge" overwhelmingly means "patient discharged" rather than wound discharge, ear discharge, or medication discharge. The training data contains all senses; the model collapses to one. A Structural Quality Index comparison between real and synthetic clinical text reveals that synthetic "discharge" has hazard 0.688 vs. 0.007 in real data -- a 98x difference invisible to lexical, entity, and embedding quality metrics.

**In testing:** Standardized exams contain ambiguous terms. When "right" appears in a law exam, does it mean legal entitlement or directional? The student must resolve the ambiguity before answering the question. Exams that require more resolution are harder for reasons unrelated to the tested knowledge. No fairness metric measures this.

Each of these is a different manifestation of the same gap: the text contains structural meaning distinctions that the downstream system cannot resolve. The gap is invisible because no tool measures it.

### 1.2 Contribution

Our contributions are:

1. **Substitution neighborhoods as operational meaning proxy.** The set of concepts that could replace a word at a given structural address, ranked by PMI surprise, serves as a computable proxy for word sense. We show that these neighborhoods recover known lexical relationships (WordNet antonym pairs) while discovering domain-specific senses that general-purpose inventories miss.

2. **Address collapse separates meaning from noise.** Greedy merging of addresses with similar substitution neighborhoods reduces 15,052 raw addresses to 2,701 resolved meanings. The collapse ratio is stable across a range of Jaccard thresholds (Section 4.7).

3. **Three-layer feature architecture, discovered from data.** Features self-classify into operators (never collapse), meaning (collapse by similarity), and expression (collapse freely). This classification is data-driven, not imposed. Expression features account for 69% of addresses; removing them improves resolution by 22%.

4. **Four-score meaning audit with formal definitions.** Confusion, completeness, predictability, and hazard (Section 3.6) measure meaning quality at document, dataset, sentence, and word levels respectively.

5. **Extrinsic evaluation on retrieval.** Meaning-addressed retrieval outperforms keyword and embedding baselines on a clinical disambiguation task (Section 4.6), demonstrating that the audit scores predict downstream utility.

6. **Cross-domain generalization.** The same tool, with no domain knowledge and no hand-picked concepts, produces meaningful audits on clinical, legal, news, and scientific text.

---

## 2. Background

### 2.1 Word Sense Disambiguation and Induction

Traditional WSD relies on sense inventories (WordNet, OntoNotes) and supervised classifiers trained on annotated corpora (Navigli, 2009). Word Sense Induction (WSI) discovers senses from data without predefined inventories, using clustering (Schutze, 1998), Bayesian models (Lau et al., 2012), or contextual embedding probes (Amrami & Goldberg, 2018).

Recent work uses contextual embeddings for WSD (Bevilacqua & Navigli, 2020; Pilehvar & Camacho-Collados, 2019) and WSI (Amrami & Goldberg, 2018), achieving strong results on standard benchmarks (SemEval). These systems learn representations from data and can capture sense distinctions that our pattern-matching features miss. However, they produce opaque clusters or similarity scores rather than interpretable feature-based addresses, and they require training data or at minimum a pretrained model.

Our method differs from both traditions. Unlike WSD, it requires no sense inventory. Unlike WSI, it produces interpretable addresses defined by named features rather than opaque clusters. Unlike contextual embedding approaches, it requires no pretrained model. Most critically, it does not ask "how many senses does this word have?" -- it asks "how many structural meaning distinctions does this dataset contain, and where does the structure run out?"

### 2.1.1 Data Profiling

The discriminator ladder is related to data profiling (Abedjan et al., 2015; Naumann, 2014): given a table with attribute columns, find which attributes best separate rows with the same key. Our algorithm applies this idea to text-derived features rather than database columns. The key difference is that our "attributes" are extracted from unstructured text by pattern matching, and the objective (collision reduction across all identity values simultaneously) enforces a universal ordering rather than per-key optimization.

### 2.2 Structural and Frame Semantics

The featurizer draws on established linguistic theory. Frame semantics (Fillmore, 1982) holds that a word evokes a situation schema with defined participant roles. Dependency grammar (Tesniere, 1959) treats the verb as the structural center of a clause. Information structure (Halliday, 1967; Lambrecht, 1994) distinguishes topic from focus. Speech act theory (Austin, 1962; Searle, 1969) classifies utterances by illocutionary force.

These frameworks motivate the ten extracted features. They are the minimal structural skeleton that linguistic theory identifies as carrying clause-level semantic load.

### 2.3 Information-Theoretic Foundations

The pair-reduction algorithm counts colliding pairs -- records sharing the same identity value AND the same value on a candidate field. This collision count is the quantity minimized by Renyi entropy of order 2 (H_2 = -log(sum(p_i^2))), the same quantity as Gini impurity in CART decision trees (Breiman et al., 1984).

The critical difference from CART: our method enforces a universal field ordering -- the same ladder for all identity neighborhoods, applied identically to every concept. CART selects different splits per node, optimizing per-leaf accuracy. The universal constraint trades per-concept expressiveness for cross-concept generalizability: the same coordinate system serves all concepts.

This is analogous to an information bottleneck (Tishby et al., 2000) -- compression that preserves task-relevant information -- but we do not claim formal equivalence. The ladder is a greedy approximation: it finds the minimal feature subset that preserves discrimination, not the optimal one.

**Relation to prior work:** The pair-reduction algorithm is a form of sequential forward selection (SFS; Whitney, 1971) with a collision-count objective. The key difference from standard feature selection is the universal ordering constraint: the same feature ranking serves all identity values simultaneously. This constraint is motivated by the data auditing use case, where a single coordinate system must be interpretable across all concepts in a corpus.

### 2.4 Distributional Semantics and Substitutability

The distributional hypothesis -- "a word is characterized by the company it keeps" (Firth, 1957) -- underlies embedding models. Our substitution neighborhoods are distributional, but addressed: they measure co-occurrence not of words in text, but of concepts at structural addresses. "Positive" near "blood" at a copular address has different neighbors than "positive" near "history" at a possessive address. The address localizes the distribution. This recovers the insight of Cruse (1986) that substitutability -- what could replace a word in context -- is the operational test of meaning identity.

---

## 3. Method

Figure 1 previews the end result: a single ambiguous word is routed through a sequence of structural features to distinct resolved meanings, each characterized by its substitution neighborhood. The subsections below describe each stage of the pipeline that produces this structure.

> **Figure 1.** Semantic routing tree for "positive" in clinical text. The discriminator ladder routes 1,847 instances through meaning-layer and operator-layer features to five resolved meanings. Each leaf lists its substitution neighborhood. *(See fig_routing_tree.pdf)*

### 3.1 Text Featurizer

Given a corpus of text records, the featurizer extracts concept-in-context instances. For each occurrence of each target concept in a record, it produces a structured record with ten features:

| Feature | Theoretical Basis |
|---|---|
| verb_class | Frame semantics: the verb determines the situation schema |
| syntactic_role | Dependency grammar: concept's relation to the verb frame |
| paired_concept | Compositional semantics: co-occurring argument constrains sense |
| contrast | Discourse pragmatics: contrastive structure separates uses |
| equation | Predicate logic: copular definitions mark identity relations |
| modality | Speech act theory: asserting vs. commanding vs. hedging |
| voice | Information structure: passive/active shifts topic/focus |
| negation | Discourse pragmatics: affirmed vs. negated inverts truth |
| clause_position | Information structure: early = topic, late = focus |
| transitivity | Predicate-argument: valency fingerprints sense |

All extraction uses deterministic pattern matching within a 5-word context window on each side of the target concept. This provides reproducibility and interpretability at the cost of coverage (see Limitations, Section 7). Multi-instance extraction: every occurrence of each concept in a document generates a separate instance with its own local context features.

Target concepts may be hand-selected for a domain or auto-detected: `auto_detect_concepts()` selects high-frequency content words (minimum 20 occurrences, minimum 4 characters, English stopwords excluded, ranked by frequency). Multi-instance extraction: if "positive" appears three times in one record, three separate instances are generated, each with features computed from its own local context window (5 words on each side of the concept). Verb class maps each detected verb to one of 25 universal classes (being, having, doing, giving, making, etc.) via a lookup table of approximately 200 forms.

### 3.2 Discriminator Ladder

The ladder algorithm takes featurized instances and discovers which features best separate different uses of the same concept.

1. Count colliding pairs: instances sharing the same concept AND the same value on every candidate field.
2. For each candidate field, count how many colliding pairs it would eliminate.
3. Select the field that eliminates the most. Record its reduction.
4. Remove that field from candidates. Repeat.
5. Stop when no candidate reduces colliding pairs by more than 5% of the current count.

The 5% threshold is a diminishing-returns cutoff: once a feature eliminates fewer than 1 in 20 remaining collisions, it is adding address complexity without meaningful discrimination. On MTSamples, this produces ladders of 4--5 rungs. At 1%, ladders extend to 7--8 rungs with marginal gains in resolved meanings but increased address sparsity. At 10%, ladders truncate to 2--3 rungs, losing discrimination on hard concepts. The retrieval experiment (Section 4.6) shows that going from 2 to 4 rungs improves macro F1 from 0.893 to 0.944, confirming that early stopping at 2--3 rungs would sacrifice useful signal. The 5% threshold is a practical default, not a tuned hyperparameter.

The output is an ordered list of fields -- the ladder -- defining a coordinate system. Each instance is assigned an address by concatenating its values along the ladder fields. Figure 2 shows the ladder discovered on MTSamples clinical text: each rung's collision reduction and the cumulative effect.

> **Figure 2.** Discriminator ladder on MTSamples (30 auto-detected concepts). *Left:* each rung's collision reduction, color-coded by feature layer. The 5% stop threshold terminates the ladder at five rungs. *Right:* cumulative remaining collisions drop from 100% to 3.0%, collapsing 15,052 raw addresses to 2,701 resolved meanings. *(See fig_ladder.pdf)*

### 3.3 Substitution Neighborhoods

Once concepts have addresses, they become addressed instances. "Positive" is no longer one word -- it is many instances at many addresses, each representing a specific structural meaning.

Substitution neighborhoods measure what other addressed instances co-occur with a given instance. For each concept at each address, we compute PMI surprise against all other addressed instances in the same documents:

    surprise(a, b) = log2( P(a,b) / (P(a) * P(b)) )

High surprise neighbors are meaningful: "negative" near "positive" at a lab-result address is surprising and informative. Low surprise neighbors are background: "patient" near everything is expected and uninformative.

The substitution neighborhood -- the set of surprising co-occurring addressed instances -- is our operational meaning representation. It answers the question: "What could replace this word at this structural address?" If structuralist linguistics is correct that substitutability defines sense (Cruse, 1986), then the neighborhood is the computationally recoverable portion of that sense. We do not claim it captures full semantic content -- only the structural distinctions that clause-level features can express.

### 3.4 Address Collapse

Raw addresses contain syntactic noise. "Positive" in active voice and "positive" in passive voice may have identical substitution neighborhoods -- the voice difference does not change what could replace the word. These should be the same resolved meaning.

Address collapse merges addresses with similar neighborhoods:

1. For each concept, compute the top-10 neighbor concept set (the neighborhood signature) for each address.
2. Sort addresses by instance count (most populated first).
3. Greedily merge: if a new address has Jaccard similarity >= 0.5 with an existing group's signature, merge it into that group and update the group signature to the union. Otherwise, start a new group. The 0.5 threshold was chosen as a moderate similarity requirement; we did not tune it.
4. Each group is a resolved meaning.

**Tie-breaking and stability:** In the ladder algorithm, when two candidate fields reduce the same number of colliding pairs, the field with fewer distinct values is preferred. If still tied, alphabetical order breaks the tie. After each rung is selected, colliding pairs are recomputed globally. The collapse algorithm is deterministic given a fixed processing order (largest address first); we do not claim optimality, only reproducibility. The sensitivity analysis (Section 4.7) shows that the resolved meaning count varies smoothly with the threshold, suggesting the collapse is not fragile to small parameter changes. The full pip-installable implementation is available as `database_whisper`.

On clinical text (30 auto-detected concepts, 5-feature ladder): 15,052 raw addresses collapse to 2,701 resolved meanings. 82% of the original address space was surface variation. (Sensitivity to the collapse threshold is analyzed in Section 4.7.)

### 3.5 Three-Layer Feature Classification

Not all features are the same kind of thing. We classify each feature by measuring whether changing its value changes the substitution neighborhood.

**Method:** For each feature, group addresses by that feature's value. Compute the average Jaccard similarity of substitution neighborhoods within groups (same feature value) and across groups (different feature values). If within >> across (ratio > 1.2), changing this feature changes meaning -- it is a MEANING feature. If within ~ across (ratio < 1.1), changing this feature does not change meaning -- it is an EXPRESSION feature.

**Operators** are a third category. Negation and modality score low on the ratio test (1.10 and 1.15) -- not because they don't matter, but because co-occurrence is blind to them. Negated and affirmed instances live in the same documents near the same concepts. A doctor who writes "no fever" and a doctor who writes "fever" are discussing the same topic with the same vocabulary. Co-occurrence can't see the negation. But collapsing across negation would merge "positive" with "not positive" -- a catastrophic error. Operators are detected by name from a known set and protected from collapse regardless of their ratio score.

**Results on clinical text (MTSamples, all 10 features):**

| Layer | Features | Ratio | Collapse Rule |
|---|---|---|---|
| OPERATOR | negation (1.10), modality (1.15) | Low (blind to co-occurrence) | Never collapse |
| MEANING | equation (1.62), contrast (1.34), paired_concept (1.22) | High (neighborhoods change) | Collapse by similarity |
| EXPRESSION | verb_class (1.09), voice (1.09), syntactic_role (1.15), clause_position (1.06), transitivity (1.06) | Low (neighborhoods stable) | Collapse freely |

This classification is data-driven. The features were not assigned to layers by linguistic theory -- though the assignment is consistent with theory. Expression features are exactly the ones that describe HOW something is written (active vs. passive, early vs. late, transitive vs. intransitive) rather than WHAT is being discussed.

### 3.6 Four Meaning Scores

After collapse, four scores audit meaning quality at four grains. Let $c$ denote a concept, $D_c$ the set of documents containing $c$, and $m_d$ the number of resolved meanings of $c$ in document $d$.

**Confusion** (document level): Does meaning drift within documents?

$$\text{confusion}(c) = 1 - \frac{1}{\bar{m}_c}, \quad \bar{m}_c = \frac{1}{|D_c|} \sum_{d \in D_c} m_d$$

When $c$ resolves to exactly one meaning per document, $\bar{m}_c = 1$ and confusion is 0 (safe). When $c$ averages five meanings per document, confusion is 0.8 (dangerous).

**Completeness** (dataset level): Are all meanings represented? Let $R_c = \{r_1, \ldots, r_k\}$ be the resolved meanings of $c$, with $n_i$ instances in meaning $r_i$ and $N_c = \sum_i n_i$.

$$\text{completeness}(c) = \frac{H(R_c)}{H_{\max}(R_c)}, \quad H(R_c) = -\sum_{i=1}^{k} \frac{n_i}{N_c} \log_2 \frac{n_i}{N_c}, \quad H_{\max} = \log_2 k$$

Shannon entropy normalized by maximum entropy. Completeness 1.0 means instances are evenly distributed across meanings. Completeness near 0 means one sense dominates.

**Predictability** (sentence level): Does context disambiguate?

$$\text{predictability}(c) = \frac{|\{d \in D_c : m_d = 1\}|}{|D_c|}$$

Fraction of documents where $c$ resolves to exactly one meaning (macro-averaged across concepts for the dataset score).

**Hazard** (word level): Which words will cause trouble?

$$\text{hazard}(c) = \text{confusion}(c) \times (1 - \text{predictability}(c)) \times \left(1 + \frac{\log_2(N_c + 1)}{20}\right)$$

Product of confusion and unpredictability, weighted by log-frequency. High-frequency words with high confusion and low predictability are the most dangerous.

**Dataset aggregation:** Each per-concept score is aggregated to a dataset score by instance-weighted average: $S = \sum_c S(c) \cdot N_c / \sum_c N_c$. Hazard is reported as a ranked list rather than averaged, since the most dangerous word matters more than the mean.

The audit is one function call: `meaning_audit(addresses)`.

---

## 4. Experiments

We validate claims with six experiments. All code and data are publicly available as the pip-installable package `database_whisper`.

**Note on experimental configurations:** Different experiments use different concept sets and feature subsets. Section 4.1 (cross-domain) uses 30 auto-detected concepts per domain with the 5-feature ladder discovered by the profiler. Sections 4.2--4.3 use 25 hand-selected clinical concepts. Sections 4.4--4.5 use all 10 features. This causes different raw address counts (e.g., 15,052 with 5 ladder features vs. 15,763 with all 10 features) and slightly different scores for the same dataset. Each table specifies its configuration.

### 4.1 Cross-Domain Audit (4 domains, auto-detected concepts)

**Claim:** The meaning audit produces meaningful, domain-specific results on any English text dataset with auto-detected concepts and zero configuration.

**Method:** We loaded four datasets from HuggingFace: MTSamples clinical notes (2,000 records), AG News articles (2,000), ArXiv scientific abstracts (2,000), and MMLU Professional Law questions (1,534). For each, `auto_detect_concepts()` selected the top 30 high-frequency content words. The full pipeline -- extraction, profiling, addressing, neighborhoods, collapse, and audit -- ran with no domain-specific configuration.

**Results:**

| Domain | Concepts | Instances | Raw Addrs | Resolved | Confusion | Complete | Predict |
|---|---|---|---|---|---|---|---|
| Clinical | 30 | 103,968 | 15,052 | 2,701 | 0.470 | 0.800 | 0.506 |
| Legal | 30 | 12,450 | 3,891 | 1,204 | 0.265 | 0.734 | 0.726 |
| News | 30 | 45,672 | 8,234 | 1,102 | 0.038 | 0.812 | 0.956 |
| Scientific | 30 | 28,891 | 6,448 | 1,856 | 0.496 | 0.789 | 0.567 |

The scores vary meaningfully across domains. News text has nearly zero confusion (0.038) and near-perfect predictability (0.956) -- words mean the same thing throughout a news article. Clinical and scientific text have high confusion (~0.5) and low predictability (~0.5) -- the same word shifts sense within documents, and context often doesn't disambiguate. Legal text falls between.

**Bootstrap stability (clinical domain).** Five bootstrap resamples of 2,000 records from MTSamples produce: confusion 0.465 ± 0.004, completeness 0.811 ± 0.005, predictability 0.520 ± 0.004 (coefficient of variation < 1% for all three scores). The ladder field ordering was identical in 4/5 resamples; the fifth swapped two fields of similar discriminatory power. The cross-domain numbers in Table 4.1 are not sampling artifacts.

**Hazard words are completely different per domain** (0/5 overlap in top-5 hazard words). Clinical hazards: patient, right, left, procedure, pain. Legal hazards: court, party, state, action, right. News hazards: (near zero -- almost no hazardous words). Scientific hazards: model, significant, expression, effect, control.

**Auto-detected concepts are completely different per domain** (0.5/15 average pairwise overlap). The tool discovers domain-specific vocabulary without being told the domain.

### 4.2 Scaffold Removal: Meaning Replaces Syntax

**Claim:** Substitution neighborhoods are more informative than the syntactic features used to discover them. Meaning features replace scaffold features.

**Method:** Four rounds of iterative refinement on MTSamples with 25 clinical concepts.

- Round 0: Standard pipeline -- syntactic features produce addresses, addresses produce neighborhoods.
- Round 1: Enrich instances with neighborhood features (neighborhood size, resolved meaning ID). Let the profiler choose between syntactic and meaning features.
- Round 2: Iterate on Round 1's neighborhoods.
- Round 3: Force meaning-only features (drop all syntactic features from candidates).

**Results:**

| Round | Features Used | Resolution |
|---|---|---|
| R0 (scaffold) | paired_concept, verb_class, clause_position, equation, voice | 0.5407 |
| R1 (profiler chooses) | neighborhood_size, resolved_meaning, paired_concept, ... | 0.5614 |
| R2 (iterate) | neighborhood_size, resolved_meaning, ... | 0.5782 |
| R3 (meaning only) | neighborhood_size, resolved_meaning | 0.5957 |

The profiler independently drops ALL scaffold fields in Round 1. It prefers meaning features (neighborhood size, resolved meaning) over every syntactic feature available to it. This is not a designed outcome -- the profiler is the same greedy pair-reduction algorithm, and it finds that meaning features carry more discriminatory information.

Resolution improves 10% from scaffold-only (0.5407) to meaning-only (0.5957). 22 of 25 concepts improve, 0 degrade.

**On circularity:** The meaning features (neighborhood size, resolved meaning) are derived from the syntactic features, so one might expect derived features to be preferred -- aggregation often subsumes its components. The non-trivial finding is the *completeness* of the replacement: ALL syntactic features are dropped, not just redundant ones. The profiler finds zero syntactic features worth keeping once meaning features are available. If the neighborhoods merely summarized the syntax, at least some syntactic features would carry residual information. They do not.

The scaffold metaphor: syntactic features are scaffolding. They are needed to discover substitution neighborhoods, but once the neighborhoods stand alone, the scaffolding can be removed.

### 4.3 WordNet Sanity Check

**Purpose:** Verify that substitution neighborhoods recover known lexical relationships. This is a sanity check, not a validation -- antonyms are expected to co-occur in clinical text (the same lab test is discussed as positive or negative), so PMI-based neighborhoods should surface them. The question is whether our structural addressing disrupts this expected behavior.

**Method:** For each concept pair where WordNet lists an antonym relationship among our 25 target concepts, check whether the concepts appear as top-10 neighbors in each other's substitution neighborhoods.

**Results:** All 16 antonym pairs among our concepts are recovered (positive/negative, left/right, acute/chronic, etc.). This confirms the neighborhoods behave as expected -- structural addressing preserves distributional relationships rather than disrupting them. We do not claim this as a validation of sense discrimination; antonym recovery validates co-occurrence statistics, which is a weaker claim (cf. Mohammad et al., 2013).

**Domain-specific senses not in WordNet:** More informatively, the neighborhoods discover senses that general-purpose inventories miss:
- "positive + culture" (19 instances): pathogen detected in a microbiological assay.
- "clear + auscultation" (346 instances): no abnormal sounds on listening.
- "discharge + wound" (87 instances): distinguished from "discharge + patient" by structural address.

These illustrate the value of corpus-specific sense discovery over fixed inventories.

### 4.4 Meaning vs. Expression Separation

**Claim:** Expression features account for the majority of raw addresses. Removing them improves audit accuracy.

**Method:** Classify all 10 features using the within-group vs. across-group Jaccard ratio method (Section 3.5). Run the meaning audit three ways: all features, meaning-only, expression-collapsed. (Configuration: MTSamples, 25 hand-selected clinical concepts, 5-feature ladder, Jaccard threshold 0.5.)

**Results:**

| Method | Raw Addrs | Resolved | Confusion | Complete | Predict |
|---|---|---|---|---|---|
| All features | 15,052 | 2,701 | 0.478 | 0.794 | 0.487 |
| Expression-collapsed | 4,590 | 1,450 | 0.382 | 0.597 | 0.572 |

69% of all raw addresses were expression variants, not meaning differences. Dropping expression features: resolution +22%, confusion -20%, predictability +17%.

Completeness drops from 0.794 to 0.597. This is honest: some real meaning distinctions correlate with expression features. But the reduction in false confusion (20%) and improvement in predictability (17%) indicate that the majority of what expression adds is noise, not signal.

**Feature redundancy discovered:** verb_class and transitivity are 96.7% correlated. equation and contrast are 96.0% correlated. The 10-feature set contains approximately 6 independent features.

### 4.5 Three-Layer Collapse

**Claim:** The three-layer collapse (operators protect, expression merges, meaning resolves) produces more honest results than flat collapse.

**Method:** Run the meaning audit with flat collapse (original method) and layered collapse (three-layer architecture) on MTSamples with all 10 features. (Configuration: 25 hand-selected clinical concepts, all 10 features as candidates, Jaccard threshold 0.5. Raw address count differs from Section 4.1 because all 10 features are used as candidates rather than the 5-feature ladder.)

**Results:**

| Method | Raw | Resolved | Expression Collapsed | Confusion | Complete | Predict |
|---|---|---|---|---|---|---|
| Flat | 15,763 | 2,440 | -- | 0.452 | 0.833 | 0.515 |
| Layered | 15,763 | 4,724 | 10,071 | 0.507 | 0.820 | 0.471 |

Layered collapse produces MORE resolved meanings (4,724 vs 2,440) because operator protection prevents merging across negation and modality boundaries. Confusion is higher and predictability is lower.

This is the correct behavior. The flat collapse was hiding real hazards by merging "positive" with "not positive." When operators are protected, the true complexity of the data becomes visible. Every concept's hazard score increases -- because the flat collapse was dishonest about negation.

**Operator protection verified:** Zero violations across all concepts. No resolved meaning contains instances with mixed negation values. "Positive" with negation=affirmed never merges with "positive" with negation=negated.

10,071 expression variants were auto-collapsed -- 64% of raw addresses were just writing-style differences.

### 4.6 Extrinsic Evaluation: Retrieval with Baselines

**Claim:** Meaning addresses improve retrieval on ambiguous clinical terms compared to keyword and embedding baselines.

**Method:** We construct a sense-disambiguation retrieval task on MTSamples. For the polysemous word "positive," we define ground-truth labels: instances where "positive" means "laboratory test positive" (identified by co-occurring clinical terms: culture, test, screen, specimen, gram, bacteria, etc.) vs. all other senses (positive family history, positive physical findings, positive outlook, etc.). We compare three retrieval methods:

1. **Keyword baseline:** Retrieve all instances containing "positive." This achieves 100% recall with precision equal to the base rate.
2. **Embedding baseline (all-MiniLM-L6-v2):** Encode all instance contexts as sentence embeddings. Query = "positive laboratory test result culture specimen." Rank by cosine similarity, report best F1 across all thresholds.
3. **DW meaning-address:** Build structural addresses using the discovered ladder. Identify which addresses correspond to the lab-positive sense. Retrieve all instances at those addresses.

We repeat for "discharge" (hospital release vs. fluid emission, base rate 64.3%) and "right" (anatomical direction vs. correct/entitlement, base rate 97.1%).

**Results:**

| Concept | Method | Precision | Recall | F1 |
|---|---|---|---|---|
| positive | Keyword | 0.556 | 1.000 | 0.715 |
| positive | Embedding (oracle) | 0.628 | 0.838 | 0.718 |
| positive | **DW address** | **0.944** | **0.875** | **0.908** |
| discharge | Keyword | 0.643 | 1.000 | 0.783 |
| discharge | Embedding (oracle) | 0.757 | 0.944 | 0.840 |
| discharge | **DW address** | **0.933** | **0.935** | **0.934** |
| right | Keyword | 0.971 | 1.000 | 0.985 |
| right | Embedding (oracle) | 0.971 | 1.000 | 0.985 |
| right | **DW address** | **0.979** | **0.998** | **0.988** |

**Macro-averaged across all three tasks:**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Keyword | 0.723 | 1.000 | 0.828 |
| MiniLM-L6-v2 (general, oracle) | 0.785 | 0.927 | 0.848 |
| S-PubMedBERT (domain-tuned, oracle) | 0.774 | 0.940 | 0.852 |
| DW meaning-address | **0.952** | **0.936** | **0.944** |

Both embedding baselines receive oracle thresholds (the threshold that maximizes F1 across all possible cutoffs) -- an advantage they would not have in practice. Despite this, DW outperforms the domain-tuned biomedical model (S-PubMedBERT, a PubMedBERT fine-tuned on MS MARCO for sentence similarity) by +0.092 macro F1.

Domain tuning barely improves the embedding baseline: S-PubMedBERT gains only +0.004 macro F1 over the general-purpose MiniLM. The bottleneck is the representational approach, not the training domain. Continuous similarity scores conflate senses that share distributional context; structural addresses separate them.

DW's main advantage is precision (+0.178 macro over S-PubMedBERT). On hard disambiguation tasks (base rate 55--65%), the margin is largest (+0.189 F1 on "positive," +0.083 on "discharge"). On easy tasks (97% base rate for "right"), all methods converge near ceiling.

**Ladder depth ablation:** Increasing from 2 to 3 to 4 ladder rungs improves macro F1 monotonically (0.893 → 0.929 → 0.944), validating the hierarchical feature selection.

### 4.7 Threshold Sensitivity Analysis

**Claim:** The collapse threshold (Jaccard >= 0.5) is a moderate choice; results are stable across a reasonable range.

**Method:** We run the full pipeline on MTSamples (2,000 records, 30 auto-detected concepts, 62,167 instances, 18,547 raw addresses) at five Jaccard thresholds: 0.3, 0.4, 0.5, 0.6, 0.7. All other parameters are held constant.

**Results:**

| Threshold | Resolved | Confusion | Completeness | Predictability |
|---|---|---|---|---|
| 0.3 | 1,155 | 0.418 | 0.661 | 0.544 |
| 0.4 | 2,531 | 0.456 | 0.753 | 0.515 |
| **0.5** | **4,442** | **0.470** | **0.798** | **0.506** |
| 0.6 | 7,817 | 0.477 | 0.834 | 0.501 |
| 0.7 | 10,328 | 0.483 | 0.854 | 0.499 |

Resolved meanings increase monotonically with threshold (stricter merging = more distinct addresses), as expected. The three audit scores are stable across the range:

- **Confusion:** 0.066 range (0.418--0.483). Robust to threshold choice.
- **Completeness:** 0.193 range (0.661--0.854). Most sensitive, because aggressive merging reduces the number of meanings the system distinguishes.
- **Predictability:** 0.045 range (0.499--0.544). Nearly invariant.

The 0.5 threshold sits at the elbow where completeness is reasonably high (~0.80) without over-collapsing. Lower thresholds (0.3) aggressively merge to only 1,155 resolved meanings from 18,547 raw, sacrificing completeness. Higher thresholds (0.7) preserve more distinctions but scores plateau. The qualitative findings of the paper -- which domains are confused, which words are hazardous -- do not change across this range.

**Note on ontology vs. scores:** While the audit scores are stable, the number of resolved meanings varies nearly 10x across this range (1,155 to 10,328). The method is robust in its *assessment* of the data but not in the *ontology* it discovers. Users who need a specific sense inventory should treat the threshold as a granularity control; users who need a data quality audit can use the default 0.5 with confidence that the scores are representative.

---

## 5. The Constitution as Illustration

We include one non-experimental example. The US Constitution (62 sections, 25 concepts, 216 instances) produces a ladder where section alone resolves 96.7% of ambiguity. The Constitution was designed to minimize the meaning gap -- each section addresses a specific structural concern, and the section label is nearly sufficient to determine the usage context of any term within it.

This contrasts with clinical notes, where specialty resolves 90% but three more features are needed. The meaning gap is small in text that was crafted for precision and large in text that was written for clinical efficiency. The measurement is a property of the text, not the tool.

---

## 6. Discussion

### 6.1 What the Meaning Gap Measures

The meaning gap is not a claim about what text "really means." It is a measurement of how many structural distinctions a given feature set can express in a given corpus. Below the resolution limit, uses are structurally separable. Above it, they are aliased -- structurally indistinguishable under the extracted features.

This measurement characterizes our feature set, not all possible representations. A system with richer features -- longer context windows, coreference resolution, external knowledge -- could raise the limit. But for any system operating on the same ten clause-level features, the gap we measure is the gap it faces. The measurement is a property of the (feature set, corpus) pair, not a universal ceiling on meaning discrimination.

### 6.2 Substitution Neighborhoods as Meaning Proxies

Our working hypothesis, following structuralist linguistics (de Saussure, 1916; Cruse, 1986), is that substitutability -- what could replace a word in context -- is a useful operational test of sense identity. The substitution neighborhood operationalizes this computationally, within the limits of the extracted features. We do not claim it captures full semantic content, only the structural distinctions that clause-level features can express.

The scaffold removal experiment (Section 4.2) provides empirical support for the neighborhoods' informativeness. The profiler -- which knows nothing about linguistics -- independently prefers neighborhood-derived features over the syntactic features that generated them. This is not circular: the syntactic features define the address space, but the neighborhoods aggregate distributional information across that space. That the aggregated representation is more discriminative than its components is an empirical finding, consistent with the hypothesis that substitutability captures something meaningful about sense.

### 6.3 The Negation Blind Spot

Negation is invisible to co-occurrence. A doctor who writes "no fever" and a doctor who writes "fever" are discussing the same topic in the same documents with the same vocabulary. Co-occurrence measures topic proximity; negation modifies truth value without changing topic.

This is the same blind spot that clinical NLP has confronted for twenty years (Chapman et al., 2001, NegEx). Our contribution is not solving negation detection but discovering that the blind spot falls out of the data-driven classification: negation scores low on the neighborhood similarity ratio because it is an operator, not because it is unimportant. The three-layer architecture addresses this by protecting operators from collapse regardless of their ratio score.

### 6.4 Applications

The meaning gap measurement enables several applications, each requiring the same tool pointed at a different target:

**Training data danger scores.** Point the audit at a training corpus. The hazard index identifies which words may cause problems in downstream models. On synthetic clinical data (Asclepius), "discharge" has hazard 0.688 vs. 0.007 in real data (MTSamples) -- a 98x difference for this single word. We present this as an illustrative example, not a systematic comparison; a full evaluation of the hazard score's predictive validity across all concepts and quality metrics is future work.

**Exam fairness.** Point the audit at exam questions. Predictability measures whether the test tests knowledge or language resolution. Across five medical and legal exams, predictability ranges from 0.835 (USMLE) to 0.995 (MedMCQA). Low predictability means candidates are failing language, not the tested subject.

**Hallucination detection without ground truth.** Compare a generated summary against its source using meaning addresses. Faithful summaries preserve the source's meaning-address structure. Hallucinated summaries don't. No reference summaries needed.

**Self-improving synthetic data.** Run the audit on generated data. Identify hazardous words. Regenerate those passages with disambiguation constraints. Re-audit. Iterate.

Each of these is a separate paper-length investigation. We mention them here as evidence that the meaning gap measurement is not a single-purpose tool but a reusable primitive.

---

## 7. Limitations

**English only.** The current featurizer targets English syntax. The algorithm and architecture are language-independent; extension requires language-specific pattern sets.

**Text only.** The method operates on written text. Spoken language, images, and multimodal content are out of scope.

**No machine learning -- a tradeoff, not a virtue.** The deterministic, training-free pipeline provides interpretability and reproducibility: every address is a named feature combination, every score is a closed-form formula, and the same input always produces the same output. The cost is coverage and adaptability. A lookup table of 200 verb forms cannot match a pretrained model's vocabulary. A 5-word context window cannot capture discourse-level dependencies. The pipeline is designed so the featurizer can be replaced with a richer one (including learned features) without changing the downstream algorithms. We view the current featurizer as a proof of concept for the measurement framework, not a claim that pattern matching is superior to learning.

**Auto-detect parameters.** The `auto_detect_concepts()` function uses fixed thresholds (minimum 20 occurrences, minimum 4 characters, top 30 by frequency) that may require adjustment for corpora with different characteristics (short texts, technical notation, non-English content).

**Negation blind spot.** Co-occurrence cannot detect operators. We mitigate this with the three-layer architecture (known operators are protected), but the method cannot discover new operator-like features from co-occurrence alone.

**Minimum corpus size.** The method requires approximately 1,000 records for stable neighborhoods. On smaller corpora (the Constitution's 216 instances), the ladder is robust but address-level statistics are sparse.

**Runtime.** The full pipeline (extraction, profiling, addressing, neighborhoods, collapse, audit) runs in approximately 45 seconds on MTSamples (2,000 records, 103,968 instances, 30 concepts) on a single CPU core. The pair-reduction step is O(n·k) where n is instances and k is candidate fields. Neighborhood computation is O(n·c) where c is concepts. Collapse is O(a²·t) where a is addresses per concept and t is the top-k neighborhood size. The method is single-threaded and requires no GPU.

**No discourse structure.** The featurizer captures clause-level relationships but misses coreference chains, paragraph-level argumentation, and document-level narrative structure. These are above the structural ceiling of the current feature set.

**Evaluation circularity.** The retrieval ground truth in Section 4.6 is defined by co-occurring clinical terms (culture, specimen, etc.), and the method's primary disambiguation feature is paired_concept -- a co-occurring term. The method is tested on labels it is structurally predisposed to recover. This makes the retrieval experiment a demonstration that the method works as designed, not an independent validation against human sense judgments. Evaluation against standard WSD benchmarks (SemEval, MSH-WSD) or human-annotated sense labels remains future work.

**Retrieval evaluation scope.** Section 4.6 covers three polysemous words on one clinical corpus (MTSamples) with two embedding baselines (general-purpose MiniLM and domain-tuned S-PubMedBERT). Evaluation on established WSD benchmarks (SemEval, MSH-WSD) and cross-domain retrieval evaluation are future work.

**Cross-domain sample stability.** Bootstrap resampling on the clinical domain (Section 4.1) confirms CVs below 1% for all three scores. We report bootstrap results for one domain; the remaining three domains use single runs on the available data.

**Feature coverage and error rates.** Ten features capture clause-level structure using pattern matching within a 5-word window. This is deliberately shallow -- no dependency parser, no coreference, no pretrained model. The featurizer defaults to "none" when no pattern matches. On MTSamples, default rates per feature range from 8% (negation) to 35% (verb_class) -- meaning the featurizer assigns a non-default value to 65-92% of instances per feature. Complex noun phrases, relative clauses, and sentences where the nearest verb is outside the 5-word window are common failure modes. Richer features (longer context, external knowledge, coreference resolution) would raise the resolution limit. The current limit is a property of these features, not a claim about text in general. The pipeline is designed so the featurizer can be upgraded independently -- richer features produce more addresses, but the ladder, collapse, and audit algorithms are feature-agnostic.

---

## 8. Conclusion

The meaning gap -- the distance between what text structurally expresses and what systems can resolve -- is present in every text-bearing dataset we tested. Keywords do not measure it. Embeddings do not measure it. Standard quality metrics do not measure it.

We have shown that it is measurable. A domain-agnostic tool discovers substitution neighborhoods from structural co-occurrence, collapses syntactic noise into resolved meanings, classifies features into three functional layers, and audits meaning quality with four scores at four grains. The tool requires no training data, no annotation, and no domain expertise. It discovers structure rather than imposing it.

The three-layer architecture -- operators that protect truth values, meaning features that define topic, expression features that describe style -- falls out of the data. It is not imposed by the method. The scaffolding metaphor holds: syntactic features are needed to discover meaning, but once meaning stands alone, the scaffolding can be removed. The profiler independently confirms this.

Applied to four domains with auto-detected concepts, the audit produces meaningfully different results per domain. Different domains have different confusion levels, different hazard words, and different concept vocabularies. The tool measures what is there, not what we expect.

Whether richer models (longer context, coreference, learned representations) can close the meaning gap is an empirical question we do not answer here. What we show is that the gap is measurable, that it varies across domains, and that a simple deterministic tool suffices to characterize it. The measurement is a starting point, not a final answer.

---

## References

Amrami, A., & Goldberg, Y. (2018). Word sense induction with neural biLM and symmetric patterns. *Proceedings of EMNLP*.

Austin, J. L. (1962). *How to Do Things with Words*. Oxford University Press.

Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.

Chapman, W. W., Bridewell, W., Hanbury, P., Cooper, G. F., & Buchanan, B. G. (2001). A simple algorithm for identifying negated findings and diseases in discharge summaries. *Journal of Biomedical Informatics*, 34(5), 301-310.

Cruse, D. A. (1986). *Lexical Semantics*. Cambridge University Press.

de Saussure, F. (1916). *Course in General Linguistics*. (C. Bally & A. Sechehaye, Eds.).

Fillmore, C. J. (1982). Frame semantics. *Linguistics in the Morning Calm*, 111-137.

Firth, J. R. (1957). A synopsis of linguistic theory 1930-1955. *Studies in Linguistic Analysis*, 1-32.

Halliday, M. A. K. (1967). Notes on transitivity and theme in English: Part 2. *Journal of Linguistics*, 3(2), 199-244.

Lambrecht, K. (1994). *Information Structure and Sentence Form*. Cambridge University Press.

Lau, J. H., Cook, P., McCarthy, D., Newman, D., & Baldwin, T. (2012). Word sense induction for novel sense detection. *Proceedings of EACL*.

Navigli, R. (2009). Word sense disambiguation: A survey. *ACM Computing Surveys*, 41(2), 1-69.

Schutze, H. (1998). Automatic word sense discrimination. *Computational Linguistics*, 24(1), 97-123.

Tesniere, L. (1959). *Elements de syntaxe structurale*. Klincksieck.

Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. *Proceedings of the 37th Annual Allerton Conference*.

Whitney, A. W. (1971). A direct method of nonparametric measurement selection. *IEEE Transactions on Computers*, C-20(9), 1100-1103.

Abedjan, Z., Golab, L., & Naumann, F. (2015). Profiling relational data: A survey. *The VLDB Journal*, 24(4), 557-581.

Naumann, F. (2014). Data profiling revisited. *ACM SIGMOD Record*, 42(4), 40-49.

Bevilacqua, M., & Navigli, R. (2020). Breaking through the 80% glass ceiling: Raising the state of the art in word sense disambiguation by incorporating knowledge graph information. *Proceedings of ACL*.

Pilehvar, M. T., & Camacho-Collados, J. (2019). WiC: The word-in-context dataset for evaluating context-sensitive meaning representations. *Proceedings of NAACL*.

Mohammad, S. M., Dorr, B. J., Hirst, G., & Turney, P. D. (2013). Computing lexical contrast. *Computational Linguistics*, 39(3), 555-590.

---

## Appendix A: Package

All code is available as the pip-installable package `database_whisper`. Core modules:

- `text.py` -- featurizer (10 features, multi-instance extraction, auto-detect concepts)
- `substitution.py` -- neighborhoods, collapse, three-layer classification, four-score audit, diagnostic tools
- `retrieve.py` -- meaning-addressed retrieval
- `compare.py` -- Structural Quality Index for dataset comparison

One-liner audit:

```python
import database_whisper as dw

instances = dw.extract_concept_instances(records, text_field="text", concepts=dw.auto_detect_concepts(records, text_field="text"))
profile = dw.profile_records(instances, identity_fields=["concept"])
addresses = dw.meaning_addresses(instances, ladder_fields=profile.ladder_fields)
audit = dw.meaning_audit(addresses)
print(audit)
```
