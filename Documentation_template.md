# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Wanheda  
**Submission Date:** September 2026

---

## 1. Executive Summary
We designed and implemented an end-to-end, high-performance Business Entity Resolution pipeline capable of scaling to over 12 million business records across three heterogeneous data sources. The solution combines multi-key country-partitioned inverted index blocking ($\ge 99.6\%$ recall ceiling), ultra-fast C++ pairwise string and structural feature extraction via RapidFuzz, a LightGBM gradient-boosted decision tree classifier, and a bipartite max-confidence assignment algorithm to maximize the precision-heavy macro-averaged $F_{0.5}$ metric while strictly adhering to the 8B parameter ceiling and MIT/Apache-2.0 license constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 2,206,821 Source-1 training entities and ~10.3M Source-2/3 records revealed several critical noise patterns:
1. **Source Discrepancies:**
   - **Source 2:** Frequently has missing addresses (~3.4% fully null; many partial), requiring matching to rely primarily on business name and country.
   - **Source 3:** Frequently contains domain-formatted business names (`maurewilliamscolombier.com`), OCR/leetspeak corruption (`#5enaum Lsb` for `#Senaum Labs`), and occasionally corrupted business names (`???? ????`) where the true entity link must be resolved exclusively via address tokens (`1-1A - 201 Shilpitha Splendour`).
2. **Country Consistency:** Empirical analysis over 172,731 ground truth pairs revealed 0.000% cross-country mismatches. Partitioning candidate generation by normalized country strictly preserves recall while drastically reducing the candidate search space.
3. **Singleton Impact:** ~5.58% of Source 1 entities in training are singletons (0 matches). Because false merges on singletons yield an instant $1.0 \to 0.0$ macro penalty, the system incorporates singleton-aware confidence margins.
4. **Generalization Requirement:** The test set introduces `France`, which never appears in training data. The pipeline dynamically groups by country without any hardcoded `{US, India}` checks.

### 2.2 Solution Strategy
**Approach Type:** Multi-Stage Inverted Index Blocking + Pairwise GBDT Feature Classifier + Global Bipartite Conflict Assignment.

**Core Innovations:**
- **Core-Compressed Normalization:** Strips domain extensions (`.com`, `.c0m`, `.org`), website prefixes (`www.`), OCR artifacts, and legal suffixes (`pvt`, `ltd`, `corp`, `inc`) to produce a compressed core token representation that collapses permutations, unspaced URLs, and legal variations into exact index matches.
- **Bipartite Global Conflict Resolution:** Resolves multi-claims where multiple Source 1 entities compete for the same Source 2 or Source 3 record by greedy highest-confidence assignment, eliminating false merges that severely degrade $F_{0.5}$.
- **Lightweight, High-Speed Execution:** Built on Polars, RapidFuzz, and LightGBM (all MIT-licensed) enabling multi-million row processing within memory and time constraints without needing massive deep learning infrastructure.

---

## 3. Candidate Generation (Blocking)

To reduce the $2.2\text{M} \times 10.3\text{M} \approx 2.2 \times 10^{13}$ pairwise comparison space, we constructed multi-key country-partitioned inverted indices:

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

- **Posting List Capping:** Keys with $> 300$ candidate postings are pruned to prevent common words ("hotel", "enterprises") from producing uninformative candidate hubs.
- **Recall Ceiling:** Tested on the held-out stratified validation set:
  - **Blocking Recall:** **99.63%** (17,365 / 17,429 true match pairs retrieved).
  - **Reduction Ratio:** **$> 99.999\%$** (candidates capped to top-35 per entity out of 10M records).

---

## 4. Matching Model

### 4.1 Feature Engineering (`features.py`)
For every surviving candidate pair, a 27-dimensional feature vector is computed:
- **Name Metrics:**
  - Jaro-Winkler similarity (`name_jw`)
  - Normalized Levenshtein ratio (`name_lev_ratio`)
  - Token-Sort and Token-Set ratios (`name_token_sort`, `name_token_set`)
  - Token Jaccard coefficient (`name_token_jaccard`)
  - Character 3-gram Jaccard coefficient (`name_char3_jaccard`)
  - Exact match and substring match indicators (`name_comp_exact`, `name_comp_substr`)
  - Core Acronym match flag (`name_acronym_match`, e.g., "IBM" $\leftrightarrow$ "International Business Machines")
  - String length difference and length ratio
- **Address Metrics:**
  - Address presence/missingness flags (`addr_s1_missing`, `addr_cand_missing`, `addr_both_present`)
  - Address Jaro-Winkler, Token-Sort, Token-Set, and Character 3-gram Jaccard
  - Postal/PIN code agreement flag (`pin_match`: +1 if match, -1 if mismatch, 0 if missing)
  - Street number agreement flag (`st_num_match`: +1 if match, -1 if mismatch, 0 if missing)
- **Meta / Structural:**
  - Candidate source indicator (`cand_source_is_s2`)
  - Retrieval rank in candidate blocking list (`cand_rank`)
  - Candidate blocking overlap score (`cand_blocking_score`)

### 4.2 Model Architecture & Training (`train.py`)
- **Model:** LightGBM Gradient Boosted Decision Trees (250 estimators, learning rate 0.08, num_leaves 31).
- **License & Size:** MIT License, <500,000 parameters (well within the $\le 8\text{B}$ constraint).
- **Validation Split:** 50,000 Source-1 entities held out prior to modeling, stratified on `(country, is_singleton)`. Zero data leakage into transforms.
- **Top Feature Importances:**
  1. `name_char3_jaccard` (696 splits)
  2. `addr_token_jaccard` (608 splits)
  3. `cand_rank` (540 splits)
  4. `addr_token_set` (520 splits)
  5. `name_lev_ratio` (518 splits)
  6. `addr_token_sort` (508 splits)
  7. `name_jw` (491 splits)

### 4.3 Assignment & Threshold Optimization (`assign.py`)
- Decision thresholds swept in $[0.40, 0.90]$ with step 0.05 on the validation split.
- **Optimal Threshold:** $0.75$ yielded the highest macro $F_{0.5}$. The precision-heavy metric favors a higher threshold to penalize false merges.
- **Global Assignment:** Conflict resolution prevents any candidate record in S2 or S3 from being assigned to more than one S1 entity.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
Evaluated using the exact competition metric via `eval_metric.py`:

| Metric | Score |
| :--- | :--- |
| **Macro-averaged $F_{0.5}$** | **0.9047** |
| **Macro Precision** | **0.9417 (94.17%)** |
| **Macro Recall** | **0.8447 (84.47%)** |
| **Singleton Accuracy** | **88.89%** |
| **Blocking Recall** | **99.63%** |

### 5.2 Error Analysis
- **False Positives:** Occur primarily when two distinct businesses share identical franchise names at neighboring street addresses within the same commercial complex where unit/floor numbers are omitted.
- **False Negatives:** Traced to records where both the business name underwent an extreme rebranding and the address had a severe typo in both street number and road token simultaneously.

---

## 6. Conclusion
The developed pipeline achieves a high Macro $F_{0.5}$ score of **0.9047** on held-out data through a principled multi-stage design: ultra-high recall blocking (99.63%), fine-grained string and structured address feature engineering, and precision-optimized global assignment. The solution operates with zero external dependencies or internet lookups, is fully reproducible, and scales efficiently across millions of records.

---

## Appendix

### A. Code Artefacts
The complete runnable codebase is located in `code/business_entity_resolution/`:
```
code/business_entity_resolution/
  ├── src/
  │   ├── normalize.py            # Text normalization, legal forms, domain & address parsing
  │   ├── blocking.py             # Multi-key inverted index candidate generation
  │   ├── features.py             # 27-dim pairwise RapidFuzz and structured similarity features
  │   ├── train.py                # Pairwise feature training and LightGBM model fitting
  │   ├── assign.py               # Global bipartite assignment and threshold sweep
  │   ├── eval_metric.py          # Official Macro F_0.5 metric implementation
  │   ├── split_data.py           # Stratified train/val split generator
  │   ├── predict.py              # Test set inference and output writer
  │   ├── pipeline.py             # End-to-end unified driver
  │   └── test_*.py               # Comprehensive pytest test suite (15 unit tests)
  ├── requirements.txt            # Pinned dependencies
  └── README.md                   # Step-by-step reproduction guide
```

**Reproduction Command:**
```bash
python code/business_entity_resolution/src/pipeline.py --mode all
```
Output files `output/matching_results.tsv` and `output/candidate_pairs.tsv` are generated and automatically verified via `student_resource/utils/validate_submission.py`.
