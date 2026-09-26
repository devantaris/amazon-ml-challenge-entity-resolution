# Business Entity Resolution Pipeline

High-performance, scalable business entity resolution pipeline designed for the **Amazon ML Challenge 2026**.

## 1. Overview & Architecture

Given multi-source, noisy business identity fragments across Source 1 (reference source), Source 2, and Source 3, the pipeline resolves matching business records while maximizing the precision-heavy **macro-averaged $F_{0.5}$** metric with singleton credit.

The solution operates through five core stages:

```
[Raw TSV Files (S1, S2, S3)]
            │
            ▼
┌──────────────────────────────────────────────┐
│  Phase 1: Normalization (normalize.py)       │
│  - Unicode NFKD & ASCII flattening           │
│  - Bidirectional legal suffix normalization   │
│  - Domain name & OCR noise removal           │
│  - Address components & PIN extraction       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Compact Blocking                   │
│  (compact_blocking.py / blocking.py)         │
│  - Partitioning by country (US, India, FR)   │
│  - 32-bit compact integer array posting lists│
│  - Zero Python object overhead (<150 MB RAM) │
│  - Name, core compressed & token prefix keys │
│  - Address street & PIN keys                 │
│  - Posting list frequency capping            │
│  -> Produces candidate_pairs.tsv             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 3: Pairwise Features                  │
│  (features.py / enhanced_features.py)        │
│  - 42-dimensional enriched pairwise features │
│  - Jaro-Winkler & Levenshtein similarity     │
│  - Token-sort & token-set ratios             │
│  - Token & char 3-gram Jaccard coefficients  │
│  - Token containment & acronym matching flags│
│  - TF-IDF Cosine Similarity (Name & Address) │
│  - BM25 Relevance Scoring (Name & Address)   │
│  - Max Shared IDF & Phonetic Encodings       │
│  - Postal code & street number agreement     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 4: Ensemble Stacking Classifier       │
│  (train_v5_fast.py)                          │
│  - GPU-accelerated XGBoost (CUDA hist tree)  │
│  - CPU LightGBM (63 leaves)                  │
│  - Logistic Regression Meta-Learner (Stacking│
│  - Validated on held-out stratified split    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 5: Global Assignment (assign.py)      │
│  - Bipartite conflict resolution             │
│  - Threshold optimization for macro F_0.5    │
│  - Singleton protection margin               │
│  -> Produces matching_results.tsv            │
└──────────────────────────────────────────────┘
```

---

## 2. Validation Performance

Evaluated against the held-out stratified validation set (50,000 entities stratified on `(country, is_singleton)`):

| Metric | Score |
| :--- | :--- |
| **Validation Macro $F_{0.5}$** | **0.9203** |
| **Validation Macro Precision** | **0.9584 (95.84%)** |
| **Validation Macro Recall** | **0.8528 (85.28%)** |
| **Singleton Accuracy** | **97.17%** |
| **Blocking Recall Ceiling** | **99.63%** |
| **Candidate Reduction Ratio** | **> 99.999%** |

---

## 3. Strict Compliance & License Verification

- **No External Data:** Runs strictly on provided train and test TSV files without any internet lookups, external geocoding, or remote APIs.
- **Model License:** XGBoost (Apache-2.0 License), LightGBM (MIT License), scikit-learn (BSD 3-Clause), RapidFuzz (MIT License), Polars (MIT License).
- **Parameter Ceiling:** The model has ~300 decision trees (< 500,000 parameters), well under the 8 Billion parameter ceiling.
- **Unseen Country Generalization:** Handles `France` (and any novel country label) dynamically through string-partitioned indexing without hardcoded country gates.
- **Memory Efficiency:** Strictly caps heap allocations to $< 1.5\text{ GB}$ using direct disk streaming and 32-bit integer posting lists to ensure zero OS disk paging.

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.11 (or 3.10+)
- Virtual environment
- NVIDIA GPU (optional, for CUDA acceleration)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### Running Tests
All 16 unit tests verify the metric, normalization, candidate blocking, pairwise features, global assignment, and France generalization:

```bash
pytest src/ -v
```

---

## 5. End-to-End Execution Pipeline

To run inference on the test set:

```bash
# Run ultra-compact GPU inference streaming directly to disk:
python src/predict_compact.py \
    --test-dir student_resource/dataset/test \
    --output-dir output

# Validate submission files against official challenge checker:
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test

# Package official submission zip:
python src/package_submission.py --team-name wanheda
```

The outputs are written directly to:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`
- `wanheda_submission.zip`
