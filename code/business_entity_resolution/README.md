# Business Entity Resolution Pipeline

High-performance, scalable business entity resolution pipeline designed for the **Amazon ML Challenge 2026**.

## 1. Overview & Architecture

Given multi-source, noisy business identity fragments across Source 1 (reference source), Source 2, and Source 3, the pipeline resolves matching business records while maximizing the precision-heavy **macro-averaged $F_{0.5}$** metric.

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
│  Phase 2: Candidate Blocking (blocking.py)   │
│  - Partitioning by country (US, India, FR)   │
│  - Multi-key inverted indexing (10M records) │
│  - Name, core compressed & token prefix keys │
│  - Address street & PIN keys                 │
│  - Posting list frequency capping            │
│  -> Produces candidate_pairs.tsv             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 3: Pairwise Features (features.py)    │
│  - Jaro-Winkler & Levenshtein similarity     │
│  - Token-sort & token-set ratios             │
│  - Token & char 3-gram Jaccard coefficients  │
│  - Acronym & substring matching flags        │
│  - Postal code & street number agreement     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 4: Pairwise Classifier (train.py)     │
│  - LightGBM GBDT (MIT License, <500K params) │
│  - Fast C++ inference via RapidFuzz          │
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
| **Validation Macro $F_{0.5}$** | **0.9047** |
| **Validation Macro Precision** | **0.9417 (94.17%)** |
| **Validation Macro Recall** | **0.8447 (84.47%)** |
| **Singleton Accuracy** | **88.89%** |
| **Blocking Recall Ceiling** | **99.63%** |
| **Candidate Reduction Ratio** | **> 99.999%** |

---

## 3. Strict Compliance & License Verification

- **No External Data:** Runs strictly on provided train and test TSV files without any internet lookups, external geocoding, or remote APIs.
- **Model License:** LightGBM (MIT License), scikit-learn (BSD 3-Clause), RapidFuzz (MIT License), Polars (MIT License).
- **Parameter Ceiling:** The model has ~250 decision trees (< 500,000 parameters), well under the 8 Billion parameter ceiling.
- **Unseen Country Generalization:** Handles `France` (and any novel country label) dynamically through string-partitioned indexing without hardcoded country gates.

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.11 (or 3.10+)
- Virtual environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### Running Tests
All 15 unit tests verify the metric, normalization, candidate blocking, pairwise features, global assignment, and France generalization:

```bash
pytest src/ -v
```

---

## 5. End-to-End Execution Pipeline

To run the entire pipeline end-to-end (or individual stages):

```bash
# Full pipeline: split -> train -> predict -> validate
python src/pipeline.py --mode all

# Or run inference only on test set:
python src/pipeline.py --mode predict \
    --test-dir student_resource/dataset/test \
    --output-dir output
```

The outputs are written directly to:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

And validated against `student_resource/utils/validate_submission.py`.
