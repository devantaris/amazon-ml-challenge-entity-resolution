# Recovery Plan — from 0.647 to 0.95+ (stretch 0.97)

**Status: DRAFT FOR USER REVIEW — do not implement until approved.**

## 1. Why we scored 0.647 (root causes, verified from code + data)

The leaderboard score is real; the 0.92 "validation" was measured in a fantasy world. Four compounding causes:

### C1 — The submission used the wrong (weakest) model
`predict_compact.py` loads `cache/models/xgb_gpu_matcher.pkl`: a **single XGBoost trained on 12,000 of the 2,206,821 S1 entities** (0.5%), 33 base features, threshold 0.80 tuned on 2,000 entities. The v5 ensemble (XGBoost + LightGBM + meta-learner, 42 features, per-country thresholds, val 0.9127 on 10K entities) was never wired into inference — `predict_enhanced.py` exists but the submission never used it.

### C2 — Validation never simulated test conditions
Training and validation both ran against a **curated target pool of ~90K records** (true targets force-added + first 25–50K rows of each source) = **0.9% of the real 10.3M-record universe**. At test time the blocker indexes the full pool and its **30-candidate cap saturates** (mean 29.8/30 in the submitted `candidate_pairs.tsv`). The model never saw realistic distractor density, and the val "blocking recall 99.63%" was measured where true targets were pre-seeded into the pool.

### C3 — France (15% of the test set) has zero coverage
Test = India 810K / US 663K / **France 259K S1 entities**. No French training data, no per-country threshold for France, no TF-IDF vocabulary for France. If France scores ~0.3–0.45, it alone drags the overall score down by 0.07–0.10.

### C4 — Blocking strategy breaks at full-pool scale
- 25% of S1 entities share an **exact normalized name** with another S1 entity ("Eye Group" ×46, "Primary Care Group" ×47, in different cities) → name-only keys flood the 30-candidate cap with same-brand-different-city junk, and the true match (heavily perturbed) falls out of the top-30.
- No IDF weighting, no address anchoring, no re-ranking: candidate ordering is a raw key-overlap count.
- India is underpredicted (2.66 links/entity vs 3.66 for US/France in the submission; truth avg ≈ 3.46) → transliteration noise handling is incomplete.

### C5 — Smaller verified bugs (to fix along the way)
- `cand_rank` computed over a Python `set` at training time (arbitrary order) — the feature is noise during training but meaningful at inference.
- `blocking_score` is a constant 1.0 — dead feature.
- Meta-learner KFold splits **pairs**, not S1 entities → same entity's pairs in train+val folds of the OOF stack (leakage).
- PIN regex eats phone-number fragments as postal codes ("Ph. 989, 9487203" → PIN 487203).
- Street-number regex is start-anchored — fails on reversed French addresses ("Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac").
- Verified correct and kept: greedy one-to-one assignment (ground truth is strictly one-to-one — checked all 7.64M train links, zero targets shared across S1 entities).

**What 0.97 requires numerically:** with 94.4% multi-match entities (avg 3.67 links), macro-F0.5 ≈ 0.97 needs per-entity precision ≈0.99 and recall ≈0.95. One wrong link on a 4-link entity costs 0.21 of that entity's score. So the whole plan is oriented around: (a) never losing true candidates (blocking), (b) near-perfect discrimination (features+model+data), (c) smart global decisions (assignment).

---

## 2. The plan

### Phase 0 — Honest validation harness (build FIRST, ~½ day)
Everything else is measured through this; no more curated-pool numbers.

1. Fixed validation split: 100K S1 train entities stratified by (country × singleton × match-count bucket). Model training never sees them.
2. Index the **full** per-country train pools once (US 6.19M / India 4.13M records, ~5–10 min each) and cache to disk for reuse.
3. Harness reports: blocking recall@K per country and per bucket, end-to-end macro-F0.5 / precision / recall / singleton accuracy per country, threshold sweep curves, error bucket counts.
4. **M0 milestone:** run the *current submitted pipeline* through the harness. It should land near 0.65 — that validates the harness against the leaderboard (our one free calibration point).

### Phase 1 — Blocking 2.0: recall ceiling ≥ 99.3% at realistic K (~1 day)
1. **IDF-weighted candidate scoring** — replace raw overlap counts; posting-list length gives IDF; name-key hits get an **address-agreement multiplier** (shared PIN/street/city boosts; disagreeing PIN dampens).
2. **Address-anchored gating for hub names** — when a name key's posting list exceeds τ (≈200), only admit candidates that also share a postal code / street number / rare address token. This kills the "Eye Group ×46" junk flood.
3. **Overgenerate → re-rank → truncate**: keep ~200–400 raw candidates, re-rank by TF-IDF cosine computed on the fly (S1 vector vs its own candidates only — cheap, no global ANN), keep top-K. Tune K per country for recall ≥ 99.3% (expect K ≈ 50–100). The Phase-0 harness quantifies recall@30/60/100 of the old blocker as the baseline. (Note: a first attempt at this measurement, `scratch/diagnose_blocking_recall.py`, showed the current full-pool index build is too slow/memory-heavy at 6.2M records — Phase 1 includes a streaming index build that fixes this.)
4. France-safe: all keys/IDF computed from whatever pool is indexed — no country-specific logic.

### Phase 2 — Features 2.0 (~½–1 day, overlaps with Phase 3 data gen)
1. Fix C5 bugs (deterministic `cand_rank`, real retrieval score, per-country PIN validation, street-number extraction anywhere in string, meta-learner grouped by S1 entity).
2. Strengthen the **address block** (the disambiguator): PIN agreement, street-number match, street-name token Jaccard, city/region/state overlap, unit/apt flags — as its own feature group.
3. Strengthen the **name block**: data-mined legal-suffix/variant dictionary (replace hand list), ST↔SAINT disambiguation, acronym containment both directions, digit-token features, phonetic codes (already implemented).
4. Keep the 42-feature TF-IDF/BM25 block, but fit TF-IDF **transductively at inference time on the test pool itself** (provided data only — allowed; this gives France a vocabulary).

### Phase 3 — Train at full scale (~1 day, mostly compute)
1. Pair generation over the **full pool** with the new blocker: positives = 7.64M GT links; negatives = rank-stratified hard negatives (~8 per entity: top-4 hardest + 4 random) → ~25M pairs.
2. Memory-fit: on Kaggle (30 GB RAM + 16 GB-VRAM GPU) the full ~25M pairs train natively; on the laptop, subsample negatives to ~12–15M pairs (≈2.5–3 GB) or use quantized DMatrix. LightGBM member on CPU.
3. GroupKFold-by-entity OOF → LogisticRegression meta-learner. Keep the XGB+LGB stack.
4. **M2 milestone:** harness F0.5 ≥ 0.90 expected here.

### Phase 4 — Decision layer + France/India (~½ day)
1. Sweep per-country threshold × retention cutoff × singleton margin on the harness (coordinate descent on macro-F0.5). Global fallback threshold for unseen countries.
2. **France de-risk:** "mask-US" simulation — train with US entirely hidden, evaluate on US as a pseudo-unseen country; validates that transductive TF-IDF + fallback threshold + country-blind features degrade gracefully. 
3. **India boost:** mine transliteration variant pairs from confident matches (token-level alignment, edit distance ≤ 2) and auto-extend `LEGAL_MAPPINGS`.

### Phase 5 — Error-analysis loop + collective post-processing (~1–2 days, the last mile)
1. Bucket every harness error: blocking-miss / below-threshold / assignment-conflict / singleton-false-merge / partial-set. Patch per bucket, re-measure.
2. **Twin bootstrap (recall + precision):** after round-1 assignment, exploit transitivity — if S2-a and S3-b are near-duplicates and S2-a is assigned to S1-x, boost S3-b for S1-x; re-run the one-to-one assignment. Reconstructs missing siblings, fixes contested targets.
3. Optional: ultra-confident self-training (P>0.99 + exact-key corroboration) as extra training signal, validated on the harness first.

### Phase 6 — Submission cadence
- Submission #2 after Phase 4 (blocking 2.0 + full retrain + honest thresholds): expected 0.85–0.92.
- Submission #3 after Phase 5: expected 0.92–0.96 (stretch 0.97).
- Each submission calibrates harness-vs-leaderboard delta; `candidate_pairs.tsv` = exact post-re-rank set fed to the model (subset rule holds automatically).

---

## 3. Compute strategy — laptop guardrails + cloud options

### 3a. Non-negotiable local guardrails (applies to every script I run)
The 2026-09-26 diagnostic incident (RAM 15.0/15.2 GB, SSD 100% active = page thrashing) happened because the index build retained the Polars frame, full Python string lists, and a growing dict index simultaneously. Rules going forward:

1. **Memory watermark, enforced in code**: every heavy script polls RSS via `psutil` and hard-stops / checkpoints at 10 GB (of 15.2 GB), so the OS never pages. No step is allowed to rely on swap.
2. **Streaming everything**: chunked Polars scans (~250K rows), per-chunk index parts merged at the end; target text accessed via row-index lookups, never materialized as full Python lists.
3. **Disk**: only `cache/` and `output/` writes (a few GB total); nothing else touches the SSD; heavy steps never run silently in the background.
4. **GPU for what it's good at**: XGBoost `device="cuda"` hist training and batched inference scoring. **NPU note (honest)**: NPUs accelerate ONNX/int8 transformer inference (DirectML/OpenVINO) — they cannot run XGBoost/LightGBM training, so for this workload the NPU stays idle; the GPU is our accelerator.
5. Progress prints every chunk so you can always see what's running and stop it.

### 3b. Cloud alternatives (faster, zero strain on your laptop)
The binding constraint locally is **16 GB RAM** for full-pool indexing and ~25M-pair training. Options, best first (updated 2026-09-26: user has **Colab Pro** with ~100 compute units/month):

1. **Colab Pro L4 runtime (primary workhorse — already owned)**: L4 GPU (22–24 GB VRAM) with **~43–53 GB system RAM** at ~2.4 CU/hr → ~40 hrs/month. Full 10.3M-record indexing and 25M-pair training fit natively; one full iteration ≈ 2–4 hrs. Cheaper T4 (~1.43 CU/hr, ~25 GB RAM) for light runs; A100 (~8.2 CU/hr, ~83 GB RAM) unnecessary. Artifacts persist via Google Drive.
2. **Kaggle Notebooks (free backup)**: CPU notebooks 4 cores / **30 GB RAM unlimited**, GPU P100/T4×2 (30 GPU-hrs/week, 12-h sessions). Same pipeline runs there unchanged when Colab units run low.
3. **AWS credits (user has a student-verified AWS account — sprint reserve, not backbone)**: credits are a finite pool, so use them for deadline sprints, not routine runs. Best fits: spot `g6.2xlarge` (L4, ~32 GB RAM, ~$0.30–0.40/hr spot) or spot `g5.2xlarge` (A10G, 32 GB, ~$0.35–0.50/hr spot) for fast full retrains, and a spot big-RAM CPU instance (`m7i.4xlarge`, 64 GB, ~$0.28/hr spot) if ever needed for the index build. **Mandatory:** set a Billing budget alarm at 50% of credits and always stop/terminate instances; check the credits' expiry date.
4. **Hourly rentals (alternative sprint option)**: Vast.ai / RunPod RTX 4090 or A10, 32–64 GB RAM, ~$0.20–0.70/hr.
5. **GCP/AWS free credits or university allocations**, if additional ones become available.

**Recommended split**: laptop = development, small-sample tests, unit tests, final packaging (all memory-capped per 3a). Kaggle = full-pool blocking builds, full-scale training, full test inference (~30–60 min on a T4). Everything orchestrated through the same git-tracked `src/` so local and cloud runs are identical code; the challenge's fair-play rule bans external *data* lookups, not compute — training on a cloud VM with only the provided TSVs is fully compliant.

---

## 4. Honest expectations
- 0.90–0.93 is the realistic strong outcome of this plan. 0.95+ requires Phases 0–5 all landing well; 0.97 is a stretch that additionally needs the twin bootstrap and transliteration mining to pay off. The harness will tell us after each milestone where the ceiling actually is — no more inflated numbers.
- Heavy jobs run on the user's **Colab Pro L4 runtime** (~43–53 GB RAM, 22 GB GPU, ~2.4 CU/hr — see §3b); the laptop only does capped development runs and final packaging. Kaggle's free notebooks are the backup when compute units run low.

**Suggested first work session:** Phase 0 + Phase 1 (harness + blocking 2.0), then decide.
