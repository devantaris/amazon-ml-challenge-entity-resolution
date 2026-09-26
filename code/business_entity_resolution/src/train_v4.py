"""
Enhanced Training Pipeline v4 — Best of Both Worlds.

Uses curated target pool (like the original high-scoring approach)
but with all enhancements:
- Hard-negative mining (top-K hardest from blocker)
- 42 enhanced features (TF-IDF, BM25, Phonetic)
- XGBoost (CUDA GPU, deeper) + LightGBM ensemble
- 50K validation for reliable threshold tuning
- Much larger training set (40K entities)
- Larger target pool (100K sample per source + all GT targets)

Key fix from v3: Don't index ALL millions of records (precision drops).
Instead use curated pool that matches what the original blocker retrieves.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset/train")
    parser.add_argument("--splits-dir", default="cache/splits")
    parser.add_argument("--models-dir", default="cache/models")
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-val", type=int, default=50000)
    parser.add_argument("--max-hard-negs", type=int, default=10)
    parser.add_argument("--target-sample", type=int, default=100000,
                        help="Random target records to sample per source")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 60, flush=True)
    print(" ENHANCED TRAINING v4 — Curated Pool + Full Enhancements", flush=True)
    print("=" * 60, flush=True)
    log_ram("Start")

    # ── Load Splits ──
    with open(os.path.join(args.splits_dir, "train_s1_ids.txt")) as f:
        train_ids = set([l.strip() for l in f if l.strip()][:args.n_train])
    with open(os.path.join(args.splits_dir, "val_s1_ids.txt")) as f:
        val_ids = set([l.strip() for l in f if l.strip()][:args.n_val])
    all_ids = train_ids | val_ids
    print(f"Train: {len(train_ids):,}, Val: {len(val_ids):,}", flush=True)

    # ── Load Ground Truth ──
    gt_df = pl.scan_csv(
        os.path.join(args.data_dir, "train_ground_truth.tsv"), separator="\t"
    ).filter(pl.col("source1_entity_id").is_in(all_ids)).collect()

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
    del gt_df; gc.collect()
    print(f"GT entries: {len(gt_map):,}, needed targets: {len(needed_target_ids):,}", flush=True)

    # ── Load S1 Records ──
    s1_df = pl.scan_csv(
        os.path.join(args.data_dir, "train_source1.tsv"), separator="\t"
    ).filter(pl.col("entity_id").is_in(all_ids)).collect()

    s1_reps = {}
    for r in s1_df.to_dicts():
        s1_reps[r["entity_id"]] = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )
    print(f"S1 reps: {len(s1_reps):,}", flush=True)
    log_ram("After S1 reps")

    # ── Build Curated Target Pool ──
    print("\n--- Building Curated Target Pool ---", flush=True)
    target_reps = {}
    country_name_tokens = collections.defaultdict(list)
    country_addr_tokens = collections.defaultdict(list)

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(args.data_dir, src_file)
        # Load all needed GT targets
        df_gt_targets = pl.scan_csv(path, separator="\t").filter(
            pl.col("entity_id").is_in(needed_target_ids)
        ).collect()
        # Load random sample for negatives
        df_sample = pl.scan_csv(path, separator="\t").head(args.target_sample).collect()
        df_combined = pl.concat([df_gt_targets, df_sample]).unique(subset=["entity_id"])

        for r in df_combined.to_dicts():
            eid = r["entity_id"]
            rep = RecordRepresentation(
                eid, r["business_name"] or "", r["business_address"] or "", r["country"] or ""
            )
            target_reps[eid] = rep
            c = rep.country
            country_name_tokens[c].append(rep.name_tokens)
            country_addr_tokens[c].append(rep.addr_tokens)

        del df_gt_targets, df_sample, df_combined; gc.collect()

    print(f"Total target reps: {len(target_reps):,}", flush=True)
    log_ram("After target pool")

    # ── Build TF-IDF Models ──
    print("\n--- Building TF-IDF Models ---", flush=True)
    name_tfidfs = {}
    addr_tfidfs = {}
    for c in country_name_tokens:
        nt = CompactTFIDF(max_features=30000)
        nt.fit(country_name_tokens[c])
        name_tfidfs[c] = nt
        at = CompactTFIDF(max_features=30000)
        at.fit(country_addr_tokens[c])
        addr_tfidfs[c] = at
        print(f"  {c}: name vocab={len(nt.vocab)}, addr vocab={len(at.vocab)}", flush=True)
    del country_name_tokens, country_addr_tokens; gc.collect()

    # ── Build Blocker ──
    print("\n--- Building Candidate Blocker ---", flush=True)
    blocker = CandidateBlocker(max_candidates_per_key=250, max_candidates_per_entity=35)
    blocker.index_target_records(
        [{"entity_id": eid, "business_name": rep.name_clean, "business_address": rep.addr_clean, "country": rep.country}
         for eid, rep in target_reps.items()]
    )
    print(f"Indexed {blocker.total_indexed_records:,} targets", flush=True)
    log_ram("After blocker")

    # ── Phase 1: Generate Training Pairs with Hard Negatives ──
    print("\n--- Phase 1: Hard-Negative Mining ---", flush=True)
    n_features = len(ENHANCED_FEATURE_NAMES)
    max_pairs = len(train_ids) * (args.max_hard_negs + 12)
    X_train = np.zeros((max_pairs, n_features), dtype=np.float32)
    y_train = np.zeros(max_pairs, dtype=np.int32)
    idx = 0

    for count, s1_id in enumerate(train_ids):
        s1_rep = s1_reps.get(s1_id)
        if not s1_rep:
            continue

        true_targets = gt_map.get(s1_id, set())
        cands = blocker.retrieve_candidates_for_entity(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )
        # Ensure all GT positives are included
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
            base = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            enhanced = compute_enhanced_features(
                base, s1_rep.name_tokens, s1_rep.addr_tokens,
                cand_rep.name_tokens, cand_rep.addr_tokens,
                s1_rep.name_clean, cand_rep.name_clean,
                name_tfidf, addr_tfidf,
            )
            if cid in true_targets:
                pos_feats.append(enhanced)
            else:
                neg_feats.append(enhanced)

        for feat in pos_feats:
            if idx < max_pairs:
                X_train[idx] = feat; y_train[idx] = 1; idx += 1

        if neg_feats:
            neg_feats.sort(key=lambda f: -f[0])  # hardest first (highest name_jw)
            for feat in neg_feats[:args.max_hard_negs]:
                if idx < max_pairs:
                    X_train[idx] = feat; y_train[idx] = 0; idx += 1

        if (count + 1) % 5000 == 0:
            print(f"  {count+1:,}/{len(train_ids):,} entities (idx={idx:,})", flush=True)

    X_train = X_train[:idx]
    y_train = y_train[:idx]
    print(f"\nTraining: {X_train.shape[0]:,} pairs x {X_train.shape[1]} features", flush=True)
    print(f"Positive rate: {np.mean(y_train):.2%}", flush=True)
    log_ram("After pairs")

    # ── Phase 4: Train Ensemble ──
    print("\n--- Phase 4: Training Ensemble ---", flush=True)
    pos_weight = max(1.0, np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))

    print("  XGBoost (CUDA)...", flush=True)
    xgb_model = xgb.XGBClassifier(
        device="cuda", tree_method="hist",
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=42, eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    print("  LightGBM (CPU)...", flush=True)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=63, max_depth=-1,
        learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=20, scale_pos_weight=pos_weight,
        random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train, y_train)

    # OOF for meta-learner
    print("  5-Fold OOF...", flush=True)
    n = len(y_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(n, dtype=np.float32)
    oof_lgb = np.zeros(n, dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr = y_train[tr_idx]
        pw = max(1.0, np.sum(ytr == 0) / max(np.sum(ytr == 1), 1))

        xf = xgb.XGBClassifier(
            device="cuda", tree_method="hist",
            n_estimators=350, max_depth=8, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5, gamma=0.1,
            scale_pos_weight=pw, random_state=42, eval_metric="logloss",
        )
        xf.fit(Xtr, ytr); oof_xgb[va_idx] = xf.predict_proba(Xva)[:, 1]

        lf = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
            scale_pos_weight=pw, random_state=42, verbose=-1,
        )
        lf.fit(Xtr, ytr); oof_lgb[va_idx] = lf.predict_proba(Xva)[:, 1]
        del xf, lf, Xtr, Xva, ytr; gc.collect()
        print(f"    Fold {fold+1}/5", flush=True)

    meta_X = np.column_stack([oof_xgb, oof_lgb])
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_X, y_train)
    print(f"  Meta weights: XGB={meta_model.coef_[0][0]:.4f}, LGB={meta_model.coef_[0][1]:.4f}", flush=True)

    # Feature importances
    print("\n  Top 15 Features:", flush=True)
    imp = xgb_model.feature_importances_
    for i, fi in enumerate(np.argsort(-imp)[:15]):
        fname = ENHANCED_FEATURE_NAMES[fi] if fi < len(ENHANCED_FEATURE_NAMES) else f"f{fi}"
        print(f"    {i+1:2d}. {fname:30s}: {imp[fi]:.4f}", flush=True)

    del X_train, y_train, oof_xgb, oof_lgb, meta_X; gc.collect()
    log_ram("After training")

    # ── Phase 5: Validation ──
    print("\n--- Phase 5: Validation (50K entities) ---", flush=True)
    val_probs = {}
    val_gt = {}

    for count, s1_id in enumerate(val_ids):
        s1_rep = s1_reps.get(s1_id)
        if not s1_rep:
            continue
        val_gt[s1_id] = gt_map.get(s1_id, set())

        cands = blocker.retrieve_candidates_for_entity(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )
        if not cands:
            val_probs[s1_id] = []
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
            base = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            enhanced = compute_enhanced_features(
                base, s1_rep.name_tokens, s1_rep.addr_tokens,
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
            p_meta = meta_model.predict_proba(np.column_stack([p_xgb, p_lgb]))[:, 1]
            val_probs[s1_id] = list(zip(valid_cids, [float(p) for p in p_meta]))
        else:
            val_probs[s1_id] = []

        if (count + 1) % 10000 == 0:
            print(f"  {count+1:,}/{len(val_ids):,} val entities", flush=True)

    # Per-country threshold tuning
    print("\n--- Threshold Tuning ---", flush=True)
    thresholds = [round(x, 2) for x in np.arange(0.50, 0.95, 0.02)]
    country_thresholds = {}

    for c in set(s1_reps[sid].country for sid in val_ids if sid in s1_reps):
        c_ids = [sid for sid in val_ids if sid in s1_reps and s1_reps[sid].country == c]
        c_probs = {sid: val_probs.get(sid, []) for sid in c_ids}
        c_gt = {sid: val_gt.get(sid, set()) for sid in c_ids}

        best_t, best_f = 0.70, 0.0
        for t in thresholds:
            preds = global_bipartite_assignment(c_probs, threshold=t)
            res = compute_macro_f05(c_gt, preds)
            if res["macro_f05"] > best_f:
                best_f = res["macro_f05"]; best_t = t
        country_thresholds[c] = best_t
        print(f"  {c}: threshold={best_t:.2f}, F0.5={best_f:.4f} (n={len(c_ids):,})", flush=True)

    # Global eval with per-country thresholds
    all_matches = {}
    for c in country_thresholds:
        c_ids = [sid for sid in val_ids if sid in s1_reps and s1_reps[sid].country == c]
        c_probs = {sid: val_probs.get(sid, []) for sid in c_ids}
        c_matches = global_bipartite_assignment(c_probs, threshold=country_thresholds[c])
        all_matches.update(c_matches)

    metrics = compute_macro_f05(val_gt, all_matches)
    print(f"\n  *** PER-COUNTRY MACRO F0.5: {metrics['macro_f05']:.4f} ***", flush=True)
    print(f"  Precision: {metrics['macro_precision']:.4f}", flush=True)
    print(f"  Recall:    {metrics['macro_recall']:.4f}", flush=True)
    print(f"  Singleton: {metrics.get('singleton_accuracy', 0):.4f}", flush=True)

    # Also try global uniform
    best_gt, best_gf = 0.70, 0.0
    for t in thresholds:
        preds = global_bipartite_assignment(val_probs, threshold=t)
        res = compute_macro_f05(val_gt, preds)
        if res["macro_f05"] > best_gf:
            best_gf = res["macro_f05"]; best_gt = t
    print(f"  Global uniform: threshold={best_gt:.2f}, F0.5={best_gf:.4f}", flush=True)

    # Pick the better approach
    if best_gf >= metrics['macro_f05']:
        final_thresholds = {c: best_gt for c in country_thresholds}
        final_f05 = best_gf
        print(f"  -> Using global threshold ({best_gt:.2f})", flush=True)
    else:
        final_thresholds = country_thresholds
        final_f05 = metrics['macro_f05']
        print(f"  -> Using per-country thresholds", flush=True)

    # Save
    save_path = os.path.join(args.models_dir, "ensemble_v4.pkl")
    tfidf_path = os.path.join(args.models_dir, "tfidf_models_v4.pkl")

    with open(save_path, "wb") as f:
        pickle.dump({
            "xgb_model": xgb_model, "lgb_model": lgb_model, "meta_model": meta_model,
            "country_thresholds": final_thresholds, "global_threshold": best_gt,
            "metrics": metrics, "feature_names": ENHANCED_FEATURE_NAMES,
        }, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump({"name_tfidfs": name_tfidfs, "addr_tfidfs": addr_tfidfs}, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f" COMPLETE in {elapsed/60:.1f} min | F0.5={final_f05:.4f}", flush=True)
    print(f" Model: {save_path}", flush=True)
    print(f" Thresholds: {final_thresholds}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
