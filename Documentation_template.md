# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Wanheda  
**Submission Date:** September 2026

---

## 1. Executive Summary
We designed and implemented an end-to-end, ultra-optimized Business Entity Resolution pipeline capable of scaling seamlessly to over 12 million business records across three heterogeneous data sources. The solution combines multi-key country-partitioned inverted index blocking ($\ge 99.63\%$ recall ceiling) with 32-bit compact integer arrays, ultra-fast C++ pairwise string and structural feature extraction via RapidFuzz, a GPU-accelerated XGBoost decision tree classifier running natively on CUDA, and a bipartite max-confidence global assignment algorithm to maximize the precision-heavy macro-averaged $F_{0.5}$ metric with singleton credit. The pipeline operates under a strict $< 1.5\text{ GB}$ RAM ceiling with direct disk streaming, guaranteeing zero memory paging and zero disk thrashing.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 2,206,821 Source-1 training entities and ~10.3M Source-2/3 records revealed several critical noise patterns:
1. **Source Discrepancies:**
   - **Source 2:** Frequently has missing addresses (~3.4% fully null; many partial), requiring matching to rely primarily on business name and country.
   - **Source 3:** Frequently contains domain-formatted business names (`maurewilliamscolombier.com`), OCR/leetspeak corruption (`#5enaum Lsb` for `#Senaum Labs`), and occasionally corrupted business names (`???? ????`) where the true entity link must be resolved exclusively via address tokens (`1-1A - 201 Shilpitha Splendour`).
2. **Country Consistency:** Empirical analysis over all 7,638,365 ground truth pairs revealed 0.0000% cross-country mismatches. Partitioning candidate generation by country strictly preserves 100% recall while cutting the search space by orders of magnitude.
3. **Singleton Impact:** ~5.58% of Source 1 entities in training are singletons (0 matches). Because false merges on singletons yield an instant $1.0 \to 0.0$ macro penalty under macro $F_{0.5}$, the system incorporates singleton-protective confidence margins and a precision-favoring decision threshold ($0.80$).
4. **Generalization Requirement:** The test set introduces `France`, which never appears in training data. The pipeline dynamically groups by country without any hardcoded `{US, India}` checks, normalizing international legal forms (`SA`, `SAS`, `SARL`).

### 2.2 Solution Strategy
**Approach Type:** Multi-Stage Inverted Index Blocking + Pairwise GPU GBDT Classifier + Global Bipartite Conflict Assignment.

**Core Innovations:**
- **Compact 32-Bit Integer Array Blocker (`compact_blocking.py`):** Replaces heavy Python object dictionaries with compact 32-bit unsigned integer arrays (`array('I')`) and zero-copy Arrow representations, slashing index RAM from 8 GB to <150 MB.
- **Direct TSV Disk Streaming (`predict_compact.py`):** Candidate pairs and matching results are buffered and stream-written directly to disk, completely bypassing in-memory storage of 34+ million string records.
- **CUDA GPU Acceleration:** Inference runs via native XGBoost CUDA DMatrix on the NVIDIA GeForce RTX 3050 Laptop GPU, achieving >1.6 million pair evaluations per second.
- **Bipartite Global Conflict Resolution:** Resolves multi-claims where multiple Source 1 entities compete for the same Source 2 or Source 3 record by greedy highest-confidence assignment, eliminating duplicate target claims that severely degrade $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 1.72 \times 10^{13}$ pairwise comparison space, we constructed multi-key country-partitioned inverted indices:

- **Blocking Keys:**
  1. `nm_<country>_<clean_name>`: Exact normalized name.
  2. `cp_<country>_<compressed_name>`: Concatenated alphanumeric name.
  3. `cr_<country>_<core_compressed>`: Core name omitting legal suffixes and stop-words.
  4. `sc_<country>_<sorted_tokens>`: Sorted non-legal tokens (resolves word transpositions).
  5. `f2_<country>_<first_two_tokens>`: First two business name tokens.
  6. `f1_<country>_<first_token>`: Discriminative first token ($\ge 3$ characters, non-stopword).
  7. `pin_<country>_<postal_code>_<token_prefix>`: Postal/PIN code + first 3 characters of name.
  8. `st_addr_<country>_<street_num>_<rare_token>`: Street/Plot number + rare address token.
  9. `st_rd_<country>_<street_num>_<road_token>`: Street number + road token.
  10. `st_nm_<country>_<street_num>_<name_prefix>`: Street number + 3 characters of business name.
  11. `tok_<country>_<rare_token>`: Secondary inverted index on informative name tokens.

- **Posting List Capping:** Keys with $> 250$ candidate postings are pruned to prevent common words ("hotel", "enterprises") from producing uninformative candidate hubs.
- **Recall Ceiling:** Tested on the held-out stratified validation set:
  - **Blocking Recall:** **99.63%** (17,365 / 17,429 true match pairs retrieved).
  - **Reduction Ratio:** **$> 99.999\%$** (candidates capped to top-30 per entity out of 10M records).

---

## 4. Matching Model

### 4.1 Feature Engineering (`features.py` & `enhanced_features.py`)
For every candidate pair, an enriched 42-dimensional feature vector is computed:
- **String & Token Similarity (C++ RapidFuzz):**
  - Jaro-Winkler similarity (`name_jw`) & normalized Levenshtein ratio (`name_lev_ratio`)
  - Token-Sort and Token-Set ratios (`name_token_sort`, `name_token_set`)
  - Token Jaccard coefficient (`name_token_jaccard`) & containment ratio (`name_token_containment`)
  - Prefix-4 match indicator (`name_prefix4_match`) & sorted-tokens Jaro-Winkler (`name_sorted_jw`)
  - Character 3-gram Jaccard coefficient (`name_char3_jaccard`)
  - Exact match and substring match indicators (`name_exact_match`, `name_comp_exact`, `name_comp_substr`)
  - Core Acronym match flag (`name_acronym_match`)
  - String length difference and length ratio (`name_len_diff`, `name_len_ratio`)
- **Information Retrieval & Term Weighting (TF-IDF & BM25):**
  - Per-country 30,000-term TF-IDF cosine similarity for business names (`name_tfidf_cosine`)
  - Per-country 30,000-term TF-IDF cosine similarity for addresses (`addr_tfidf_cosine`) — #1 feature by gain (53.9%)
  - BM25 relevance score for names (`name_bm25`) & addresses (`addr_bm25`)
  - Maximum shared IDF value between matching tokens (`name_tfidf_max_shared_idf`, `addr_tfidf_max_shared_idf`)
- **Phonetic Encoding (Pure-Python Metaphone & Soundex):**
  - Double Metaphone exact phonetic match flag (`name_metaphone_match`)
  - Soundex code match flag (`name_soundex_match`)
  - Phonetic token match ratio across multi-token business names (`name_phonetic_token_ratio`)
- **Address Structure & Agreement:**
  - Address presence/missingness flags (`addr_s1_missing`, `addr_cand_missing`, `addr_both_present`)
  - Address Jaro-Winkler, Token-Sort, Token-Set, and Character 3-gram Jaccard
  - Token containment ratio (`addr_token_containment`)
  - Postal/PIN code agreement flag (`pin_match`: +1 if match, -1 if mismatch, 0 if missing)
  - PIN both present / PIN disagree flags (`pin_both_present`, `pin_disagree`)
  - Street number agreement flag (`st_num_match`: +1 if match, -1 if mismatch, 0 if missing)
  - Street number both present / disagree flags (`st_num_both_present`, `st_num_disagree`)
- **Meta / Structural:**
  - Candidate source indicator (`cand_source_is_s2`)
  - Retrieval rank in candidate blocking list (`cand_rank`)
  - Candidate blocking overlap score (`cand_blocking_score`)

### 4.2 Model Architecture & Training (`train_v5_fast.py`)
- **Base Models:**
  1. **GPU XGBoost:** `device='cuda'`, `tree_method='hist'`, `n_estimators=450`, `max_depth=7`, `learning_rate=0.06`, `scale_pos_weight=2.22`.
  2. **LightGBM:** `n_estimators=350`, `num_leaves=63`, `learning_rate=0.06`.
- **Meta-Learner:** Out-of-fold Logistic Regression stacking (`XGB_weight=10.89`, `LGB_weight=1.33`).
- **License & Size:** Apache-2.0 / MIT, total parameters <500,000 (well within the $\le 8\text{B}$ parameter limit).
- **Validation Split:** 10,000–50,000 Source-1 entities held out, stratified on `(country, is_singleton)`. Zero data leakage.
- **Top Feature Importances (Gain):**
  1. `addr_tfidf_cosine` (53.94% gain)
  2. `addr_cand_missing` (14.08% gain)
  3. `addr_both_present` (11.83% gain)
  4. `addr_token_containment` (8.78% gain)
  5. `addr_bm25` (7.23% gain)

### 4.3 Assignment & Threshold Optimization (`assign.py`)
- Precision-weighted decision thresholds tuned per country.
- **Global Assignment:** Greedy confidence-ordered bipartite assignment ensures no candidate record in S2 or S3 is assigned to more than one S1 entity, eliminating precision penalties from duplicate claims.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
Evaluated using the exact competition metric via `eval_metric.py`:

| Metric | Score |
| :--- | :--- |
| **US Macro $F_{0.5}$** | **0.9595 (95.95%)** |
| **Overall Macro $F_{0.5}$** | **0.9152–0.9203** |
| **Macro Precision** | **0.9547 (95.47%)** |
| **Macro Recall** | **0.8498 (84.98%)** |
| **Singleton Accuracy** | **97.90%** |
| **Blocking Recall Ceiling** | **99.63%** |

### 5.2 Error Analysis
- **Transliteration Recovery:** Over 638,000 non-ASCII Indian records (written in Hindi, Gujarati, Tamil, Telugu, Odia) were recovered using Unicode transliteration, boosting name-matching recall significantly.
- **False Positives:** Occur primarily when distinct businesses share identical franchise names at neighboring street addresses within the same commercial complex where unit/floor numbers are omitted.
- **False Negatives:** Traced to records where both the business name underwent an extreme rebranding and the address had a severe typo in both street number and road token simultaneously.

---

## 6. Conclusion
The developed pipeline achieves state-of-the-art entity resolution performance (US $F_{0.5} = 0.9595$, Overall $F_{0.5} = 0.9152$) through a principled multi-stage design: ultra-high recall blocking (99.63%), 42-dimensional TF-IDF/BM25/phonetic/RapidFuzz feature engineering, GPU-accelerated gradient boosting ensemble, and precision-optimized bipartite global assignment. The solution operates under a $<1.5$ GB RAM ceiling, has zero external network dependencies, is fully reproducible, and scales efficiently across 12+ million records.

---

## Appendix

### A. Code Artefacts
The complete runnable codebase is located in `code/business_entity_resolution/`:
```
code/business_entity_resolution/
  ├── src/
  │   ├── normalize.py            # Text normalization, unidecode transliteration, Indic/French legal forms
  │   ├── blocking.py             # Multi-key inverted index candidate generation
  │   ├── compact_blocking.py     # 32-bit compact integer array blocker (<150 MB RAM)
  │   ├── features.py             # 33-dim pairwise RapidFuzz and structured similarity features
  │   ├── enhanced_features.py    # 42-dim TF-IDF, BM25, and pure-Python Metaphone/Soundex features
  │   ├── train_v5_fast.py        # Fast GPU XGBoost + LightGBM ensemble training (<15 min)
  │   ├── assign.py               # Global bipartite assignment and threshold sweep
  │   ├── eval_metric.py          # Official Macro F_0.5 metric implementation
  │   ├── split_data.py           # Stratified train/val split generator
  │   ├── predict_enhanced.py     # Streaming test set inference and output generator
  │   ├── package_submission.py   # Bundles <team_name>_submission.zip & runs validation
  │   └── test_*.py               # Comprehensive pytest test suite (16 unit tests)
  ├── requirements.txt            # Pinned dependencies
  └── README.md                   # Step-by-step reproduction guide
```

**Reproduction Command:**
```bash
python code/business_entity_resolution/src/predict_enhanced.py --test-dir student_resource/dataset/test --output-dir output
```
Output files `output/matching_results.tsv` and `output/candidate_pairs.tsv` are generated and automatically verified via `student_resource/utils/validate_submission.py`.
