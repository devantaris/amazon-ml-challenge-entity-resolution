"""
Enhanced Training Pipeline v2 — All Phases Combined.

Phase 1: Hard-negative mining from blocking candidates.
Phase 2: TF-IDF cosine + BM25 features per country.
Phase 3: Phonetic encoding features.
Phase 4: Ensemble stacking (XGBoost + LightGBM + meta-learner).
Phase 5: Per-country threshold tuning.

Memory budget: ≤ 2.0 GB Python heap.
GPU: NVIDIA RTX 3050 (6 GB VRAM) via XGBoost CUDA.
"""

import os
import gc
import sys
import time
import pickle
import argparse
import collections
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import psutil

from normalize import normalize_country
from blocking import CandidateBlocker, extract_blocking_keys
from features import FEATURE_NAMES, RecordRepresentation, compute_pair_features
from enhanced_features import (
    ENHANCED_FEATURE_NAMES,
    CompactTFIDF,
    compute_enhanced_features,
)
from eval_metric import compute_macro_f05
from assign import global_bipartite_assignment


def log_ram(label=""):
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"  [RAM] {label}: {rss:.0f} MB", flush=True)


def load_splits(splits_dir, n_train, n_val):
    """Load train/val S1 entity IDs."""
    with open(os.path.join(splits_dir, "train_s1_ids.txt")) as f:
        train_ids = [line.strip() for line in f if line.strip()][:n_train]
    with open(os.path.join(splits_dir, "val_s1_ids.txt")) as f:
        val_ids = [line.strip() for line in f if line.strip()][:n_val]
    return set(train_ids), set(val_ids)


def load_ground_truth(data_dir, needed_s1_ids):
    """Load ground truth for needed S1 IDs. Returns gt_map and needed_target_ids."""
    gt_df = pl.scan_csv(
        os.path.join(data_dir, "train_ground_truth.tsv"), separator="\t"
    ).filter(pl.col("source1_entity_id").is_in(needed_s1_ids)).collect()

    gt_map = {}
    needed_target_ids = set()
    for row in gt_df.to_dicts():
        s1_id = row["source1_entity_id"]
        m_str = (row.get("matched_entity_ids") or "").strip()
        if m_str:
            targets = {x.strip() for x in m_str.split(",") if x.strip()}
            gt_map[s1_id] = targets
            needed_target_ids.update(targets)
        else:
            gt_map[s1_id] = set()
    return gt_map, needed_target_ids


def load_s1_reps(data_dir, s1_ids):
    """Load and pre-process S1 RecordRepresentations."""
    s1_df = pl.scan_csv(
        os.path.join(data_dir, "train_source1.tsv"), separator="\t"
    ).filter(pl.col("entity_id").is_in(s1_ids)).collect()

    reps = {}
    for r in s1_df.to_dicts():
        reps[r["entity_id"]] = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )
    return reps


def build_target_pool(data_dir, needed_target_ids, sample_per_source=40000):
    """
    Build target pool: all ground-truth targets + random sample from S2/S3.
    Returns target_reps dict and per-country token lists for TF-IDF.
    """
    target_reps = {}
    country_name_tokens = collections.defaultdict(list)
    country_addr_tokens = collections.defaultdict(list)

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(data_dir, src_file)
        df_targets = pl.scan_csv(path, separator="\t").filter(
            pl.col("entity_id").is_in(needed_target_ids)
        ).collect()
        df_sample = pl.scan_csv(path, separator="\t").head(sample_per_source).collect()
        df_combined = pl.concat([df_targets, df_sample]).unique(subset=["entity_id"])

        for r in df_combined.to_dicts():
            eid = r["entity_id"]
            rep = RecordRepresentation(
                eid, r["business_name"] or "", r["business_address"] or "", r["country"] or ""
            )
            target_reps[eid] = rep
            c = rep.country
            country_name_tokens[c].append(rep.name_tokens)
            country_addr_tokens[c].append(rep.addr_tokens)

        del df_targets, df_sample, df_combined
        gc.collect()

    return target_reps, country_name_tokens, country_addr_tokens


def build_tfidf_models(country_name_tokens, country_addr_tokens):
    """Fit per-country TF-IDF vocabularies."""
    name_tfidfs = {}
    addr_tfidfs = {}
    for c in country_name_tokens:
        nt = CompactTFIDF(max_features=30000)
        nt.fit(country_name_tokens[c])
        name_tfidfs[c] = nt

        at = CompactTFIDF(max_features=30000)
        at.fit(country_addr_tokens[c])
        addr_tfidfs[c] = at

        print(f"  TF-IDF fitted for {c}: name vocab={len(nt.vocab)}, addr vocab={len(at.vocab)}", flush=True)

    return name_tfidfs, addr_tfidfs


def generate_pairs_with_hard_negatives(
    s1_ids, s1_reps, target_reps, gt_map, blocker,
    name_tfidfs, addr_tfidfs,
    max_hard_negs=8, max_cands=30,
):
    """
    Generate training pairs with hard-negative mining.
    For each S1 entity:
      - All positive matches (ground truth)
      - Top-K hardest negatives from blocking candidates
    Returns X (n_pairs x 42), y (n_pairs,)
    """
    n_features = len(ENHANCED_FEATURE_NAMES)
    max_pairs = len(s1_ids) * (max_hard_negs + 10)
    X = np.zeros((max_pairs, n_features), dtype=np.float32)
    y = np.zeros(max_pairs, dtype=np.int32)
    idx = 0

    for count, s1_id in enumerate(s1_ids):
        s1_rep = s1_reps.get(s1_id)
        if not s1_rep:
            continue

        true_targets = gt_map.get(s1_id, set())
        cands = blocker.retrieve_candidates_for_entity(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )

        # Ensure all true targets appear
        all_cands = list(dict.fromkeys(cands + [t for t in true_targets if t in target_reps]))

        c = s1_rep.country
        name_tfidf = name_tfidfs.get(c)
        addr_tfidf = addr_tfidfs.get(c)

        pos_feats = []
        neg_feats = []

        for rank, cid in enumerate(all_cands):
            cand_rep = target_reps.get(cid)
            if not cand_rep:
                continue

            base_feats = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            enhanced = compute_enhanced_features(
                base_feats,
                s1_rep.name_tokens, s1_rep.addr_tokens,
                cand_rep.name_tokens, cand_rep.addr_tokens,
                s1_rep.name_clean, cand_rep.name_clean,
                name_tfidf, addr_tfidf,
            )

            is_match = 1 if cid in true_targets else 0
            if is_match:
                pos_feats.append(enhanced)
            else:
                neg_feats.append(enhanced)

        # Keep all positives
        for feat in pos_feats:
            if idx < max_pairs:
                X[idx] = feat
                y[idx] = 1
                idx += 1

        # Keep top-K hardest negatives (highest name similarity = hardest)
        if neg_feats:
            # Sort by name_jw (index 0) descending = hardest negatives first
            neg_feats.sort(key=lambda f: -f[0])
            for feat in neg_feats[:max_hard_negs]:
                if idx < max_pairs:
                    X[idx] = feat
                    y[idx] = 0
                    idx += 1

        if (count + 1) % 5000 == 0:
            print(f"  Generated pairs for {count+1:,} / {len(s1_ids):,} entities (idx={idx:,})", flush=True)

    X = X[:idx]
    y = y[:idx]
    return X, y


def train_ensemble(X_train, y_train, random_state=42):
    """
    Train XGBoost (CUDA) + LightGBM + Logistic Regression meta-learner.
    Uses 5-fold CV for OOF predictions.
    """
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    print(f"\n  Training ensemble on {n_samples:,} pairs x {n_features} features", flush=True)
    print(f"  Positive rate: {np.mean(y_train):.2%}", flush=True)

    # === Model 1: XGBoost on CUDA GPU ===
    print("\n  --- Training XGBoost (CUDA GPU) ---", flush=True)
    xgb_model = xgb.XGBClassifier(
        device="cuda",
        tree_method="hist",
        n_estimators=400,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=max(1.0, np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)),
        random_state=random_state,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)
    log_ram("After XGBoost fit")

    # === Model 2: LightGBM (CPU) ===
    print("\n  --- Training LightGBM (CPU) ---", flush=True)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=350,
        num_leaves=63,
        max_depth=-1,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        scale_pos_weight=max(1.0, np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)),
        random_state=random_state,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    log_ram("After LightGBM fit")

    # === 5-Fold OOF Predictions for Meta-Learner ===
    print("\n  --- Generating 5-fold OOF predictions ---", flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)
    oof_lgb = np.zeros(n_samples, dtype=np.float32)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]

        xgb_fold = xgb.XGBClassifier(
            device="cuda", tree_method="hist",
            n_estimators=300, max_depth=7, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=5, gamma=0.1,
            scale_pos_weight=max(1.0, np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1)),
            random_state=random_state, eval_metric="logloss",
        )
        xgb_fold.fit(X_tr, y_tr)
        oof_xgb[va_idx] = xgb_fold.predict_proba(X_va)[:, 1]

        lgb_fold = lgb.LGBMClassifier(
            n_estimators=250, num_leaves=63,
            learning_rate=0.06, subsample=0.85, colsample_bytree=0.85,
            min_child_samples=20,
            scale_pos_weight=max(1.0, np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1)),
            random_state=random_state, verbose=-1,
        )
        lgb_fold.fit(X_tr, y_tr)
        oof_lgb[va_idx] = lgb_fold.predict_proba(X_va)[:, 1]

        del xgb_fold, lgb_fold, X_tr, X_va, y_tr
        gc.collect()
        print(f"    Fold {fold_idx+1}/5 complete", flush=True)

    # === Meta-Learner: Logistic Regression ===
    print("\n  --- Training Meta-Learner (Logistic Regression) ---", flush=True)
    meta_X = np.column_stack([oof_xgb, oof_lgb])
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    meta_model.fit(meta_X, y_train)
    print(f"  Meta-learner weights: XGB={meta_model.coef_[0][0]:.4f}, LGB={meta_model.coef_[0][1]:.4f}", flush=True)

    # Print feature importances from the full XGBoost model
    print("\n  --- Top 15 Feature Importances (XGBoost Gain) ---", flush=True)
    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(-importances)
    for i, idx in enumerate(sorted_idx[:15]):
        fname = ENHANCED_FEATURE_NAMES[idx] if idx < len(ENHANCED_FEATURE_NAMES) else f"f{idx}"
        print(f"    {i+1:2d}. {fname:30s}: {importances[idx]:.4f}", flush=True)

    return xgb_model, lgb_model, meta_model


def evaluate_ensemble(
    val_s1_ids, s1_reps, target_reps, gt_map, blocker,
    xgb_model, lgb_model, meta_model,
    name_tfidfs, addr_tfidfs,
    max_cands=30,
):
    """Evaluate ensemble on validation set. Returns per-country and overall metrics."""
    val_candidate_probs = {}
    val_true_gt = {}

    for count, s1_id in enumerate(val_s1_ids):
        s1_rep = s1_reps.get(s1_id)
        if not s1_rep:
            continue
        val_true_gt[s1_id] = gt_map.get(s1_id, set())

        cands = blocker.retrieve_candidates_for_entity(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )
        if not cands:
            val_candidate_probs[s1_id] = []
            continue

        c = s1_rep.country
        name_tfidf = name_tfidfs.get(c)
        addr_tfidf = addr_tfidfs.get(c)

        feat_batch = []
        valid_cids = []
        for rank, cid in enumerate(cands):
            cand_rep = target_reps.get(cid)
            if not cand_rep:
                continue
            base_feats = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            enhanced = compute_enhanced_features(
                base_feats,
                s1_rep.name_tokens, s1_rep.addr_tokens,
                cand_rep.name_tokens, cand_rep.addr_tokens,
                s1_rep.name_clean, cand_rep.name_clean,
                name_tfidf, addr_tfidf,
            )
            feat_batch.append(enhanced)
            valid_cids.append(cid)

        if feat_batch:
            X_val = np.array(feat_batch, dtype=np.float32)
            p_xgb = xgb_model.predict_proba(X_val)[:, 1]
            p_lgb = lgb_model.predict_proba(X_val)[:, 1]
            meta_X = np.column_stack([p_xgb, p_lgb])
            p_meta = meta_model.predict_proba(meta_X)[:, 1]
            cand_pairs = list(zip(valid_cids, [float(p) for p in p_meta]))
        else:
            cand_pairs = []

        val_candidate_probs[s1_id] = cand_pairs

        if (count + 1) % 2000 == 0:
            print(f"  Evaluated {count+1:,} / {len(val_s1_ids):,} val entities", flush=True)

    # Phase 5: Per-country threshold tuning
    print("\n--- Phase 5: Per-Country Threshold Tuning ---", flush=True)
    countries_in_val = set()
    for s1_id in val_s1_ids:
        rep = s1_reps.get(s1_id)
        if rep:
            countries_in_val.add(rep.country)

    country_thresholds = {}
    thresholds_to_try = [round(x, 2) for x in np.arange(0.50, 0.92, 0.02)]

    for c in sorted(countries_in_val):
        c_s1_ids = [s1_id for s1_id in val_s1_ids if s1_reps.get(s1_id) and s1_reps[s1_id].country == c]
        c_probs = {s1_id: val_candidate_probs.get(s1_id, []) for s1_id in c_s1_ids}
        c_gt = {s1_id: val_true_gt.get(s1_id, set()) for s1_id in c_s1_ids}

        best_t, best_f05 = 0.70, 0.0
        for t in thresholds_to_try:
            preds = global_bipartite_assignment(c_probs, threshold=t)
            res = compute_macro_f05(c_gt, preds)
            if res["macro_f05"] > best_f05:
                best_f05 = res["macro_f05"]
                best_t = t

        country_thresholds[c] = best_t
        print(f"  {c}: best threshold = {best_t:.2f}, F0.5 = {best_f05:.4f}", flush=True)

    # Global evaluation with per-country thresholds
    print("\n--- Global Evaluation with Per-Country Thresholds ---", flush=True)
    all_matches = {}
    for c in countries_in_val:
        c_s1_ids = [s1_id for s1_id in val_s1_ids if s1_reps.get(s1_id) and s1_reps[s1_id].country == c]
        c_probs = {s1_id: val_candidate_probs.get(s1_id, []) for s1_id in c_s1_ids}
        c_matches = global_bipartite_assignment(c_probs, threshold=country_thresholds[c])
        all_matches.update(c_matches)

    final_metrics = compute_macro_f05(val_true_gt, all_matches)
    print(f"\n  *** FINAL ENSEMBLE MACRO F0.5: {final_metrics['macro_f05']:.4f} ***", flush=True)
    print(f"  Macro Precision:   {final_metrics['macro_precision']:.4f}", flush=True)
    print(f"  Macro Recall:      {final_metrics['macro_recall']:.4f}", flush=True)
    print(f"  Singleton Accuracy:{final_metrics.get('singleton_accuracy', 0):.4f}", flush=True)
    print(f"  Non-Singleton F05: {final_metrics.get('non_singleton_f05', 0):.4f}", flush=True)

    # Also evaluate with global uniform threshold for comparison
    print("\n--- Comparison: Global Uniform Threshold Sweep ---", flush=True)
    best_global_t, best_global_f05 = 0.70, 0.0
    for t in thresholds_to_try:
        preds = global_bipartite_assignment(val_candidate_probs, threshold=t)
        res = compute_macro_f05(val_true_gt, preds)
        if res["macro_f05"] > best_global_f05:
            best_global_f05 = res["macro_f05"]
            best_global_t = t
    print(f"  Best global threshold: {best_global_t:.2f}, F0.5: {best_global_f05:.4f}", flush=True)

    return final_metrics, country_thresholds, best_global_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset/train")
    parser.add_argument("--splits-dir", default="cache/splits")
    parser.add_argument("--models-dir", default="cache/models")
    parser.add_argument("--n-train", type=int, default=50000,
                        help="Number of S1 train entities (more = better but slower)")
    parser.add_argument("--n-val", type=int, default=10000,
                        help="Number of S1 val entities for evaluation")
    parser.add_argument("--max-hard-negs", type=int, default=8)
    parser.add_argument("--target-sample", type=int, default=50000,
                        help="Random target records to sample per source beyond ground truth")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 60, flush=True)
    print(" ENHANCED TRAINING PIPELINE v2", flush=True)
    print(" Phases 1-5: Hard Negatives + TF-IDF + Phonetic + Ensemble", flush=True)
    print("=" * 60, flush=True)
    log_ram("Start")

    # ── Load Splits ──────────────────────────────────────────
    print("\n--- Loading Splits ---", flush=True)
    train_s1_ids, val_s1_ids = load_splits(args.splits_dir, args.n_train, args.n_val)
    all_s1_ids = train_s1_ids | val_s1_ids
    print(f"Train: {len(train_s1_ids):,}, Val: {len(val_s1_ids):,}", flush=True)

    # ── Load Ground Truth ────────────────────────────────────
    print("\n--- Loading Ground Truth ---", flush=True)
    gt_map, needed_target_ids = load_ground_truth(args.data_dir, all_s1_ids)
    print(f"Ground truth entries: {len(gt_map):,}, needed targets: {len(needed_target_ids):,}", flush=True)
    log_ram("After GT")

    # ── Load S1 Representations ──────────────────────────────
    print("\n--- Loading S1 Representations ---", flush=True)
    s1_reps = load_s1_reps(args.data_dir, all_s1_ids)
    print(f"Loaded {len(s1_reps):,} S1 reps", flush=True)
    log_ram("After S1 reps")

    # ── Build Target Pool ────────────────────────────────────
    print("\n--- Building Target Pool ---", flush=True)
    target_reps, country_name_tokens, country_addr_tokens = build_target_pool(
        args.data_dir, needed_target_ids, sample_per_source=args.target_sample
    )
    print(f"Total target reps: {len(target_reps):,}", flush=True)
    log_ram("After target pool")

    # ── Phase 2: Build TF-IDF Models ─────────────────────────
    print("\n--- Phase 2: Building Per-Country TF-IDF Models ---", flush=True)
    name_tfidfs, addr_tfidfs = build_tfidf_models(country_name_tokens, country_addr_tokens)
    del country_name_tokens, country_addr_tokens
    gc.collect()
    log_ram("After TF-IDF")

    # ── Build Candidate Blocker ──────────────────────────────
    print("\n--- Building Candidate Blocker Index ---", flush=True)
    blocker = CandidateBlocker(max_candidates_per_key=250, max_candidates_per_entity=30)
    for eid, rep in target_reps.items():
        keys, rare_toks = extract_blocking_keys(
            rep.name_clean, rep.addr_clean, rep.country
        )
        for k in keys:
            blocker.key_index[k].append(eid)
        c_norm = normalize_country(rep.country)
        for t in rare_toks:
            blocker.token_index[f"tok_{c_norm}_{t}"].append(eid)
        blocker.total_indexed_records += 1
    print(f"Indexed {blocker.total_indexed_records:,} target records", flush=True)
    log_ram("After blocker index")

    # ── Phase 1: Generate Training Pairs with Hard Negatives ─
    print("\n--- Phase 1: Hard-Negative Mining & Pair Generation ---", flush=True)
    X_train, y_train = generate_pairs_with_hard_negatives(
        list(train_s1_ids), s1_reps, target_reps, gt_map, blocker,
        name_tfidfs, addr_tfidfs,
        max_hard_negs=args.max_hard_negs,
    )
    print(f"Training set: {X_train.shape[0]:,} pairs, {X_train.shape[1]} features", flush=True)
    print(f"Positive rate: {np.mean(y_train):.2%}", flush=True)
    log_ram("After pair generation")

    # ── Phase 4: Train Ensemble ──────────────────────────────
    print("\n--- Phase 4: Training Ensemble (XGBoost + LightGBM + Meta) ---", flush=True)
    xgb_model, lgb_model, meta_model = train_ensemble(X_train, y_train)
    log_ram("After ensemble training")

    # Free training data
    del X_train, y_train
    gc.collect()

    # ── Phase 5: Evaluate & Tune Thresholds ──────────────────
    print("\n--- Phase 5: Evaluating on Validation Set ---", flush=True)
    final_metrics, country_thresholds, best_global_threshold = evaluate_ensemble(
        list(val_s1_ids), s1_reps, target_reps, gt_map, blocker,
        xgb_model, lgb_model, meta_model,
        name_tfidfs, addr_tfidfs,
    )
    log_ram("After evaluation")

    # ── Save Models ──────────────────────────────────────────
    print("\n--- Saving Ensemble Models ---", flush=True)
    save_path = os.path.join(args.models_dir, "ensemble_v2.pkl")
    tfidf_path = os.path.join(args.models_dir, "tfidf_models.pkl")

    with open(save_path, "wb") as f:
        pickle.dump({
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "meta_model": meta_model,
            "country_thresholds": country_thresholds,
            "global_threshold": best_global_threshold,
            "metrics": final_metrics,
            "feature_names": ENHANCED_FEATURE_NAMES,
        }, f)

    with open(tfidf_path, "wb") as f:
        pickle.dump({
            "name_tfidfs": name_tfidfs,
            "addr_tfidfs": addr_tfidfs,
        }, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f" TRAINING COMPLETE in {elapsed/60:.1f} minutes", flush=True)
    print(f" Ensemble model saved to: {save_path}", flush=True)
    print(f" TF-IDF models saved to: {tfidf_path}", flush=True)
    print(f" Final Macro F0.5: {final_metrics['macro_f05']:.4f}", flush=True)
    print(f" Country thresholds: {country_thresholds}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
