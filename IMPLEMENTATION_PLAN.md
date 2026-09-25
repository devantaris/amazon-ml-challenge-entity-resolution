# Implementation Plan — Business Entity Resolution Pipeline

**Audience:** AI coding agent (e.g. Claude Code) building this end-to-end.
**Goal:** Maximize macro-averaged F₀.₅ on `matching_results.tsv`, produce a valid `candidate_pairs.tsv`, and package a reproducible submission.

**Hard constraints (do not violate):**
- No external data lookups, APIs, geocoding services, or internet augmentation of any kind. Everything must run on the provided train/test TSVs only.
- Final matching model must be MIT or Apache-2.0 licensed and ≤8B parameters.
- Every `test_source1.tsv` entity must appear exactly once in `matching_results.tsv` and in `candidate_pairs.tsv`.
- No duplicate `source1_entity_id` rows; no duplicate IDs within an ID list; only S2-/S3- IDs that exist in the test set.
- `matching_results.tsv` matches must be a subset of `candidate_pairs.tsv` candidates for the same entity.
- Run `utils/validate_submission.py` before every leaderboard upload.

---

## Phase 0 — Project setup

1. Create the repo skeleton matching the required final submission structure from the start:
   ```
   code/business_entity_resolution/
     src/
       normalize.py
       blocking.py
       features.py
       train.py
       predict.py
       assign.py
       pipeline.py
       eval_metric.py
     README.md
     requirements.txt
   output/
     matching_results.tsv
     candidate_pairs.tsv
   Documentation_template.md   (fill in at the end)
   ```
2. Pin dependencies in `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `lightgbm` (or `xgboost`/`catboost`), `rapidfuzz`, `unidecode`, `scipy`. Add `sentence-transformers` + a specific small MIT/Apache model only if Phase 3 embeddings are used — verify the model's license and parameter count explicitly before adding it, and record the check in the README.
3. Load `train_source1/2/3.tsv` and `train_ground_truth.tsv` with `sep="\t"`. Sanity-check row counts, null rates per column, ID prefix consistency, and duplicate IDs.
4. **Build the validation split first, before any modeling.** Hold out a stratified slice of Source-1 entities from `train` (stratify on has-match-vs-singleton, and on country) to simulate the test set. Never let any held-out entity's records leak into feature-fitting (e.g. TF-IDF vocabulary, embedding fine-tuning) — fit those transforms on the training portion only.
5. Implement `eval_metric.py`: per-entity precision/recall/F₀.₅ and the macro-average, exactly matching the spec's formula. This is the single source of truth used everywhere below — every phase reports against it.

**Exit criteria:** clean train/val split, metric implementation unit-tested against the worked example in the problem statement (precision=2/3, recall=1.0 → F₀.₅≈0.714).

---

## Phase 1 — Normalization (`normalize.py`)

1. Build a shared text-normalization function applied identically to `business_name` and `business_address` fields across all three sources:
   - Lowercase, unicode-normalize (NFKD via `unidecode`) to flatten transliteration artifacts.
   - Strip punctuation except meaningful separators; collapse whitespace.
   - Expand a legal-suffix dictionary both directions (Corp/Corporation, Ltd/Limited, Pvt/Private, Inc/Incorporated, LLC, Co, & vs and, etc.) — mine this dictionary empirically from the actual training vocabulary, don't hand-guess it blind.
   - Address-specific: expand Rd/Road, St/Street, Ave/Avenue, Blvd, Apt, Fl, etc.; strip landmark phrases ("near X", "opposite X", "behind X") into a separate `landmark` field rather than deleting them outright (they may still carry weak signal); extract a postal/PIN code into its own field when present; extract a leading street number into its own field when present.
2. Produce two representations per record: (a) a normalized full string for name and address, (b) a token set (for Jaccard/set-based features later).
3. Output a normalized parquet/CSV cache per source so downstream stages don't re-normalize repeatedly.

**Exit criteria:** spot-check ~30 random records per source pre/post normalization; confirm known noise patterns from the problem statement (Corp↔Corporation, Rd↔Road, & vs and, landmark refs) are handled.

---

## Phase 2 — Blocking / candidate generation (`blocking.py`)

Goal: maximize recall — a match missed here can never be recovered later. Union multiple blocking strategies rather than relying on one.

1. **Token-overlap blocking:** inverted index on normalized name tokens (and bigrams for short names); retrieve S2/S3 records sharing at least one rare token with each S1 entity.
2. **Character n-gram TF-IDF + cosine:** vectorize normalized name (+ address) with char n-grams (e.g. 3–5 grams), retrieve top-k nearest neighbors per S1 entity via sparse cosine similarity or an ANN index (FAISS) for speed at scale.
3. **Phonetic blocking:** Double Metaphone (or Soundex) on the first name token(s); block on phonetic code match — catches transliteration/spelling variants token methods miss.
4. **Embedding-based blocking (optional, if license/size checked):** encode `normalized_name + normalized_address` with a small sentence-embedding model, ANN top-k retrieval. This is usually what catches the hardest cases (word-order transpositions, heavy transliteration).
5. **Country pre-filter as a soft signal, never a hard filter:** don't drop cross-country candidates outright unless country strings clearly and confidently disagree after normalization — country fields may be noisy, and France (unseen in training) must not break the pipeline. Test this explicitly with a held-out-country simulation (see Phase 6).
6. Union all candidate sets per S1 entity into one deduplicated list. This union is `candidate_pairs.tsv`'s content.
7. **Measure blocking recall on the validation split**: for each val entity, is every true positive from ground truth present in the candidate set? Target ≥99.5%. If below that, add another blocking pass or widen top-k before proceeding — do not move to Phase 3 until this holds.
8. Log the **reduction ratio** (avg candidates per S1 entity vs. total S2+S3 pool) for the methodology writeup.

**Exit criteria:** blocking recall ≥99.5% on validation split; reduction ratio reported; `candidate_pairs.tsv` format validated.

---

## Phase 3 — Pairwise feature engineering (`features.py`)

For every (S1 entity, candidate) pair surviving blocking, compute a feature vector:

- **Name similarity:** Levenshtein ratio, Jaro-Winkler, token-sort ratio, token-set ratio (via `rapidfuzz`), Jaccard on token sets, Jaccard on char n-grams, TF-IDF cosine, acronym-match flag (e.g. "IBM" vs "International Business Machines" — check if one side's initials match the other's).
- **Address similarity:** same string/set metrics on normalized address; exact-match flags for extracted postal code and street number when both sides have them (and a separate "both missing" vs "one missing" flag, since missingness itself is informative but shouldn't force a False result); locality token overlap.
- **Embedding similarity** (if used in blocking): cosine similarity between S1 and candidate embeddings, reused as a feature rather than recomputed.
- **Country signal:** exact match flag on normalized country string, but never make this a hard gate in the model — let the classifier learn its weight, since it must generalize to the unseen France label.
- **Structural:** string length difference (name, address), source (S2 vs S3) as a categorical/one-hot feature, count of candidates competing for this S1 entity (weak prior signal).

Output one row per (S1, candidate) pair with all features plus the ground-truth label (for training rows only).

**Exit criteria:** feature matrix built for train and val candidate pairs; no leakage (features computed without referencing ground truth); null-handling verified.

---

## Phase 4 — Pairwise classifier (`train.py`)

1. Train a gradient-boosted tree model (LightGBM/XGBoost/CatBoost — all MIT/Apache/BSD-licensed, well under the 8B param ceiling) on the engineered features, label = is-true-match.
2. Use **stratified k-fold CV**, stratified by (has-match vs singleton) and by country, to get robust out-of-fold probability estimates and to check variance across folds.
3. **Calibrate** output probabilities (isotonic regression or Platt scaling) so a chosen threshold behaves consistently at inference time.
4. Watch for class imbalance (most candidate pairs are non-matches after blocking) — use `scale_pos_weight` / class weighting or focal-style reweighting as needed, but validate against the real metric, not just AUC/logloss.
5. Run feature importance / SHAP to sanity-check the model isn't leaning on something spurious (e.g. a data-artifact correlated with source file order).

**Exit criteria:** out-of-fold predictions available for the full validation split; calibrated probabilities.

---

## Phase 5 — Threshold tuning + global assignment (`assign.py`)

1. **Threshold:** sweep decision thresholds on the validation split, computing macro F₀.₅ (not accuracy/AUC) at each, and pick the threshold that maximizes it. Expect the optimal threshold to sit above 0.5 given the precision-heavy metric.
2. **Global consistency pass:** after thresholding, resolve conflicts where one S2/S3 record would be claimed by multiple S1 entities. Implement this as a max-weight bipartite assignment (e.g. `scipy.optimize.linear_sum_assignment` on the relevant sparse submatrix, or a greedy highest-confidence-first assignment with a "claimed" set) so each S2/S3 ID is used by at most one S1 entity — many-to-one violations directly hurt precision.
3. **Singleton protection:** for S1 entities whose best-scoring candidate is close to the threshold, bias toward predicting no match — a false merge on a true singleton costs a full point (1.0 → 0.0) under this metric, more than a missed weak match costs under recall. Consider a slightly stricter effective threshold for borderline single-candidate cases, tuned on validation.
4. Re-measure macro F₀.₅ on the validation split after both threshold + assignment steps — this is your best estimate of leaderboard score.

**Exit criteria:** validation macro F₀.₅ measured and logged; assignment step demonstrably improves precision without materially hurting recall (compare before/after).

---

## Phase 6 — Generalization check (France / unseen country)

1. Before touching the real test set, simulate the unseen-country scenario: hold out one training country entirely from model fitting (e.g. train on US only, validate matching performance on India as if it were "unseen"), confirming the pipeline doesn't implicitly depend on country-specific normalization rules or hard filters.
2. Confirm no code path assumes `country ∈ {US, India}` (no hardcoded lists, no country-specific branching that would silently no-op or error on "France").
3. Re-run the full pipeline end-to-end on a synthetic run with an injected unfamiliar country string to confirm no crashes and reasonable (if lower-confidence) behavior.

**Exit criteria:** pipeline runs cleanly and produces sane output when a country label outside {US, India} is present.

---

## Phase 7 — Inference on test set + output generation (`predict.py`, `pipeline.py`)

1. Run the full pipeline (normalize → block → feature → predict → threshold → assign) on `test_source1/2/3.tsv`.
2. Write `output/candidate_pairs.tsv` (post-blocking, pre-model-narrowing — the exact set fed to the classifier) and `output/matching_results.tsv` (post-threshold, post-assignment final matches), matching the exact column names and tab-separated, no-quoting format from the spec.
3. Ensure every `test_source1` entity has exactly one row in both files, empty string for no-match rows.
4. Run `utils/validate_submission.py` against both files — fix everything it flags before considering this phase done.

**Exit criteria:** `validate_submission.py` returns PASS (exit 0).

---

## Phase 8 — Error analysis loop (iterate until score plateaus)

1. On the validation split, pull false positives and false negatives, bucket them by cause: missed-by-blocking, abbreviation not in dictionary, transliteration variant, landmark-address noise, near-duplicate-but-distinct-business, genuinely ambiguous.
2. Patch the specific root cause per bucket (extend the normalization dictionary, add a targeted feature, adjust blocking top-k) rather than broad re-tuning.
3. Re-run Phases 3–5 after each patch and re-measure. Stop when marginal gains flatten or time budget is spent.

---

## Phase 9 — Packaging

1. Write `code/business_entity_resolution/README.md` with exact reproduction steps: environment setup, command to regenerate both output files from raw train/test TSVs, expected runtime.
2. Finalize `requirements.txt` with pinned versions.
3. Fill in `Documentation_template.md`: methodology, blocking strategy + measured recall/reduction ratio, model architecture + feature list, validation strategy, threshold selection rationale, France-generalization notes, and final validation-split F₀.₅.
4. Zip per the required structure (`<team_name>_submission.zip`) and confirm it matches the spec exactly before final submission.

---

## Suggested execution order for the agent (condensed)

`Phase 0 → 1 → 2 (iterate until recall ≥99.5%) → 3 → 4 → 5 → 6 → 7 → 8 (loop) → 9`

Do not proceed past Phase 2 until blocking recall is verified on the validation split — it silently caps the maximum achievable score for every later phase.
