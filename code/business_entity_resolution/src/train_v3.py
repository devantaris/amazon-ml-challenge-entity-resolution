"""
Enhanced Training Pipeline v3 — Full-Coverage Country-Streaming.

Mirrors the test inference pipeline exactly:
- Indexes ALL S2/S3 training records per country via CompactCandidateBlocker
- Creates RecordRepresentations on-the-fly for retrieved candidates only
- Hard-negative mining from full blocking pool (not a tiny subsample)
- 42 enhanced features (base + TF-IDF + BM25 + Phonetic)
- Ensemble: XGBoost (CUDA) + LightGBM + LogReg meta-learner
- Per-country threshold tuning on full 50K validation set

Memory: streams country-by-country, peak < 2 GB per country pass.
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
from compact_blocking import CompactCandidateBlocker
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


def generate_country_pairs(
    s1_ids, s1_df, targets_df, gt_map,
    name_tfidf, addr_tfidf,
    max_hard_negs=10, max_cands=35,
):
    """
    Generate training/val pairs for one country using full target pool.
    Indexes ALL targets with CompactCandidateBlocker, retrieves candidates,
    creates RecordRepresentations on-the-fly.
    """
    n_features = len(ENHANCED_FEATURE_NAMES)

    # Build compact blocker from full targets
    blocker = CompactCandidateBlocker(
        max_candidates_per_key=250, max_candidates_per_entity=max_cands
    )
    blocker.index_dataframe(targets_df)

    # Prepare target lookup arrays
    tgt_ids = targets_df["entity_id"].to_list()
    tgt_names = targets_df["business_name"].fill_null("").to_list()
    tgt_addrs = targets_df["business_address"].fill_null("").to_list()
    tgt_countries = targets_df["country"].fill_null("").to_list()

    # Build target ID -> row index map for ground truth injection
    tgt_id_to_idx = {eid: i for i, eid in enumerate(tgt_ids)}

    # Filter S1 records for these IDs
    s1_c = s1_df.filter(pl.col("entity_id").is_in(s1_ids))
    s1_id_list = s1_c["entity_id"].to_list()
    s1_names = s1_c["business_name"].fill_null("").to_list()
    s1_addrs = s1_c["business_address"].fill_null("").to_list()
    s1_countries_list = s1_c["country"].fill_null("").to_list()

    max_pairs = len(s1_id_list) * (max_hard_negs + 12)
    X = np.zeros((max_pairs, n_features), dtype=np.float32)
    y = np.zeros(max_pairs, dtype=np.int32)
    idx = 0

    for count, (s1_id, s1_name, s1_addr, s1_country) in enumerate(
        zip(s1_id_list, s1_names, s1_addrs, s1_countries_list)
    ):
        s1_rep = RecordRepresentation(s1_id, s1_name, s1_addr, s1_country)
        true_targets = gt_map.get(s1_id, set())

        # Retrieve candidates from full blocker
        cand_indices = blocker.retrieve_candidate_indices(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )

        # Also inject ground truth targets that are in this country's pool
        gt_indices = []
        for t_id in true_targets:
            t_idx = tgt_id_to_idx.get(t_id)
            if t_idx is not None and t_idx not in cand_indices:
                gt_indices.append(t_idx)

        all_indices = list(dict.fromkeys(list(cand_indices) + gt_indices))

        pos_feats = []
        neg_feats = []

        for rank, ci in enumerate(all_indices):
            cand_rep = RecordRepresentation(
                tgt_ids[ci], tgt_names[ci], tgt_addrs[ci], tgt_countries[ci]
            )
            base_feats = compute_pair_features(
                s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0
            )
            enhanced = compute_enhanced_features(
                base_feats,
                s1_rep.name_tokens, s1_rep.addr_tokens,
                cand_rep.name_tokens, cand_rep.addr_tokens,
                s1_rep.name_clean, cand_rep.name_clean,
                name_tfidf, addr_tfidf,
            )

            is_match = 1 if tgt_ids[ci] in true_targets else 0
            if is_match:
                pos_feats.append(enhanced)
            else:
                neg_feats.append(enhanced)

        # Keep ALL positives
        for feat in pos_feats:
            if idx < max_pairs:
                X[idx] = feat
                y[idx] = 1
                idx += 1

        # Keep top-K hardest negatives (highest name_jw = most confusing)
        if neg_feats:
            neg_feats.sort(key=lambda f: -f[0])
            for feat in neg_feats[:max_hard_negs]:
                if idx < max_pairs:
                    X[idx] = feat
                    y[idx] = 0
                    idx += 1

        if (count + 1) % 5000 == 0:
            print(f"    Pairs for {count+1:,}/{len(s1_id_list):,} entities (idx={idx:,})", flush=True)

    del blocker, tgt_ids, tgt_names, tgt_addrs, tgt_countries, tgt_id_to_idx
    gc.collect()

    return X[:idx], y[:idx]


def generate_country_val_probs(
    val_ids, s1_df, targets_df,
    xgb_model, lgb_model, meta_model,
    name_tfidf, addr_tfidf,
    max_cands=35,
):
    """
    Generate validation probabilities for one country using full target pool.
    Returns val_candidate_probs dict.
    """
    blocker = CompactCandidateBlocker(
        max_candidates_per_key=250, max_candidates_per_entity=max_cands
    )
    blocker.index_dataframe(targets_df)

    tgt_ids = targets_df["entity_id"].to_list()
    tgt_names = targets_df["business_name"].fill_null("").to_list()
    tgt_addrs = targets_df["business_address"].fill_null("").to_list()
    tgt_countries = targets_df["country"].fill_null("").to_list()

    s1_c = s1_df.filter(pl.col("entity_id").is_in(val_ids))
    s1_id_list = s1_c["entity_id"].to_list()
    s1_names = s1_c["business_name"].fill_null("").to_list()
    s1_addrs = s1_c["business_address"].fill_null("").to_list()
    s1_countries_list = s1_c["country"].fill_null("").to_list()

    # Get XGBoost booster for CUDA prediction
    booster = xgb_model.get_booster()
    booster.set_param({"device": "cuda:0"})

    val_probs = {}
    batch_size = 10000

    for start in range(0, len(s1_id_list), batch_size):
        end = min(start + batch_size, len(s1_id_list))

        feat_batch = []
        pair_meta = []
        batch_s1_ids = []

        for i in range(start, end):
            s1_id = s1_id_list[i]
            s1_rep = RecordRepresentation(s1_id, s1_names[i], s1_addrs[i], s1_countries_list[i])
            batch_s1_ids.append(s1_id)

            cand_indices = blocker.retrieve_candidate_indices(
                s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
            )

            if not cand_indices:
                val_probs[s1_id] = []
                continue

            for rank, ci in enumerate(cand_indices):
                cand_rep = RecordRepresentation(
                    tgt_ids[ci], tgt_names[ci], tgt_addrs[ci], tgt_countries[ci]
                )
                base_feats = compute_pair_features(
                    s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0
                )
                enhanced = compute_enhanced_features(
                    base_feats,
                    s1_rep.name_tokens, s1_rep.addr_tokens,
                    cand_rep.name_tokens, cand_rep.addr_tokens,
                    s1_rep.name_clean, cand_rep.name_clean,
                    name_tfidf, addr_tfidf,
                )
                feat_batch.append(enhanced)
                pair_meta.append((s1_id, tgt_ids[ci]))

        if feat_batch:
            X_val = np.array(feat_batch, dtype=np.float32)
            dmat = xgb.DMatrix(X_val)
            p_xgb = booster.predict(dmat)
            del dmat
            p_lgb = lgb_model.predict_proba(X_val)[:, 1]
            meta_X = np.column_stack([p_xgb, p_lgb])
            p_meta = meta_model.predict_proba(meta_X)[:, 1]
            del X_val

            for (s1_id, cid), prob in zip(pair_meta, p_meta):
                if s1_id not in val_probs:
                    val_probs[s1_id] = []
                val_probs[s1_id].append((cid, float(prob)))

        # Ensure all S1 IDs have entries
        for s1_id in batch_s1_ids:
            if s1_id not in val_probs:
                val_probs[s1_id] = []

        if end % 10000 == 0 or end >= len(s1_id_list):
            print(f"    Val scored {end:,}/{len(s1_id_list):,} entities", flush=True)

    del blocker, tgt_ids, tgt_names, tgt_addrs, tgt_countries
    gc.collect()
    return val_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset/train")
    parser.add_argument("--splits-dir", default="cache/splits")
    parser.add_argument("--models-dir", default="cache/models")
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-val", type=int, default=50000)
    parser.add_argument("--max-hard-negs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 60, flush=True)
    print(" ENHANCED TRAINING v3 — Full-Coverage Country-Streaming", flush=True)
    print("=" * 60, flush=True)
    log_ram("Start")

    # Load splits
    with open(os.path.join(args.splits_dir, "train_s1_ids.txt")) as f:
        train_ids = [l.strip() for l in f if l.strip()][:args.n_train]
    with open(os.path.join(args.splits_dir, "val_s1_ids.txt")) as f:
        val_ids = [l.strip() for l in f if l.strip()][:args.n_val]
    all_ids = set(train_ids) | set(val_ids)
    print(f"Train: {len(train_ids):,}, Val: {len(val_ids):,}", flush=True)

    # Load ground truth
    gt_df = pl.scan_csv(
        os.path.join(args.data_dir, "train_ground_truth.tsv"), separator="\t"
    ).filter(pl.col("source1_entity_id").is_in(all_ids)).collect()

    gt_map = {}
    for row in gt_df.to_dicts():
        s1_id = row["source1_entity_id"]
        m_str = (row.get("matched_entity_ids") or "").strip()
        gt_map[s1_id] = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()
    del gt_df
    gc.collect()
    print(f"Ground truth loaded: {len(gt_map):,} entries", flush=True)

    # Load full S1 dataframe (only needed IDs)
    s1_df = pl.scan_csv(
        os.path.join(args.data_dir, "train_source1.tsv"), separator="\t"
    ).filter(pl.col("entity_id").is_in(all_ids)).collect()

    # Determine countries
    countries = s1_df["country"].unique().to_list()
    print(f"Countries: {countries}", flush=True)

    # Country -> set of train/val S1 IDs
    s1_country_map = {}
    for r in s1_df.select(["entity_id", "country"]).to_dicts():
        s1_country_map[r["entity_id"]] = r["country"]

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    # ── PHASE 1-3: Generate training pairs country-by-country ──
    print("\n--- Generating Training Pairs (Country-by-Country Streaming) ---", flush=True)
    all_X_parts = []
    all_y_parts = []
    name_tfidfs = {}
    addr_tfidfs = {}

    for c in countries:
        c_norm = normalize_country(c)
        print(f"\n  Country: {c} ({c_norm})", flush=True)

        # Load ALL S2+S3 targets for this country
        s2_c = pl.scan_csv(
            os.path.join(args.data_dir, "train_source2.tsv"), separator="\t"
        ).filter(pl.col("country") == c).collect()
        s3_c = pl.scan_csv(
            os.path.join(args.data_dir, "train_source3.tsv"), separator="\t"
        ).filter(pl.col("country") == c).collect()
        targets_c = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        gc.collect()
        print(f"  Target records: {len(targets_c):,}", flush=True)
        log_ram(f"After loading targets for {c}")

        # Build TF-IDF for this country (sample first 100K records)
        print(f"  Fitting TF-IDF...", flush=True)
        sample_n = min(100000, len(targets_c))
        name_tok_lists = []
        addr_tok_lists = []
        t_names = targets_c["business_name"].fill_null("").to_list()[:sample_n]
        t_addrs = targets_c["business_address"].fill_null("").to_list()[:sample_n]
        for n, a in zip(t_names, t_addrs):
            tmp = RecordRepresentation("tmp", n, a, c)
            name_tok_lists.append(tmp.name_tokens)
            addr_tok_lists.append(tmp.addr_tokens)
        del t_names, t_addrs

        nt = CompactTFIDF(max_features=30000)
        nt.fit(name_tok_lists)
        at = CompactTFIDF(max_features=30000)
        at.fit(addr_tok_lists)
        name_tfidfs[c_norm] = nt
        addr_tfidfs[c_norm] = at
        del name_tok_lists, addr_tok_lists
        gc.collect()
        print(f"  TF-IDF: name vocab={len(nt.vocab)}, addr vocab={len(at.vocab)}", flush=True)

        # Get train S1 IDs for this country
        c_train_ids = {sid for sid in train_ids if s1_country_map.get(sid) == c}
        print(f"  Train S1 entities in {c}: {len(c_train_ids):,}", flush=True)

        if c_train_ids:
            X_c, y_c = generate_country_pairs(
                c_train_ids, s1_df, targets_c, gt_map,
                nt, at,
                max_hard_negs=args.max_hard_negs,
            )
            all_X_parts.append(X_c)
            all_y_parts.append(y_c)
            print(f"  Pairs: {len(y_c):,}, Pos rate: {np.mean(y_c):.2%}", flush=True)

        del targets_c
        gc.collect()
        log_ram(f"After pairs for {c}")

    # Combine all country pairs
    X_train = np.concatenate(all_X_parts, axis=0)
    y_train = np.concatenate(all_y_parts, axis=0)
    del all_X_parts, all_y_parts
    gc.collect()
    print(f"\nTotal training: {X_train.shape[0]:,} pairs x {X_train.shape[1]} features", flush=True)
    print(f"Overall positive rate: {np.mean(y_train):.2%}", flush=True)
    log_ram("After all pairs combined")

    # ── PHASE 4: Train Ensemble ──
    print("\n--- Phase 4: Training Ensemble ---", flush=True)

    pos_weight = max(1.0, np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    print(f"scale_pos_weight: {pos_weight:.2f}", flush=True)

    print("  Training XGBoost (CUDA)...", flush=True)
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
    log_ram("After XGBoost")

    print("  Training LightGBM (CPU)...", flush=True)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=63, max_depth=-1,
        learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=20, scale_pos_weight=pos_weight,
        random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    log_ram("After LightGBM")

    # 5-Fold OOF for meta-learner
    print("  5-Fold OOF predictions...", flush=True)
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
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=5, gamma=0.1,
            scale_pos_weight=pw, random_state=42, eval_metric="logloss",
        )
        xf.fit(Xtr, ytr)
        oof_xgb[va_idx] = xf.predict_proba(Xva)[:, 1]

        lf = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63,
            learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
            min_child_samples=20, scale_pos_weight=pw,
            random_state=42, verbose=-1,
        )
        lf.fit(Xtr, ytr)
        oof_lgb[va_idx] = lf.predict_proba(Xva)[:, 1]

        del xf, lf, Xtr, Xva, ytr
        gc.collect()
        print(f"    Fold {fold+1}/5 done", flush=True)

    print("  Training meta-learner...", flush=True)
    meta_X = np.column_stack([oof_xgb, oof_lgb])
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_X, y_train)
    print(f"  Meta weights: XGB={meta_model.coef_[0][0]:.4f}, LGB={meta_model.coef_[0][1]:.4f}", flush=True)

    # Print feature importances
    print("\n  Top 15 Feature Importances (XGBoost Gain):", flush=True)
    imp = xgb_model.feature_importances_
    for i, fi in enumerate(np.argsort(-imp)[:15]):
        fname = ENHANCED_FEATURE_NAMES[fi] if fi < len(ENHANCED_FEATURE_NAMES) else f"f{fi}"
        print(f"    {i+1:2d}. {fname:30s}: {imp[fi]:.4f}", flush=True)

    # Free training data
    del X_train, y_train, oof_xgb, oof_lgb, meta_X
    gc.collect()
    log_ram("After training freed")

    # ── PHASE 5: Evaluate on full validation set, country-by-country ──
    print("\n--- Phase 5: Validation Evaluation (Country-Streaming) ---", flush=True)
    all_val_probs = {}

    for c in countries:
        c_norm = normalize_country(c)
        c_val_ids = {sid for sid in val_ids if s1_country_map.get(sid) == c}
        if not c_val_ids:
            continue
        print(f"\n  Val for {c}: {len(c_val_ids):,} entities", flush=True)

        s2_c = pl.scan_csv(
            os.path.join(args.data_dir, "train_source2.tsv"), separator="\t"
        ).filter(pl.col("country") == c).collect()
        s3_c = pl.scan_csv(
            os.path.join(args.data_dir, "train_source3.tsv"), separator="\t"
        ).filter(pl.col("country") == c).collect()
        targets_c = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        gc.collect()
        print(f"  Targets: {len(targets_c):,}", flush=True)

        nt = name_tfidfs.get(c_norm)
        at = addr_tfidfs.get(c_norm)

        c_probs = generate_country_val_probs(
            c_val_ids, s1_df, targets_c,
            xgb_model, lgb_model, meta_model,
            nt, at,
        )
        all_val_probs.update(c_probs)

        del targets_c
        gc.collect()
        log_ram(f"After val for {c}")

    # Per-country threshold tuning
    print("\n--- Per-Country Threshold Tuning ---", flush=True)
    val_gt = {sid: gt_map.get(sid, set()) for sid in val_ids}
    thresholds_to_try = [round(x, 2) for x in np.arange(0.50, 0.95, 0.02)]
    country_thresholds = {}

    for c in countries:
        c_norm = normalize_country(c)
        c_val_ids = [sid for sid in val_ids if s1_country_map.get(sid) == c]
        if not c_val_ids:
            continue
        c_probs = {sid: all_val_probs.get(sid, []) for sid in c_val_ids}
        c_gt = {sid: val_gt.get(sid, set()) for sid in c_val_ids}

        best_t, best_f05 = 0.70, 0.0
        for t in thresholds_to_try:
            preds = global_bipartite_assignment(c_probs, threshold=t)
            res = compute_macro_f05(c_gt, preds)
            if res["macro_f05"] > best_f05:
                best_f05 = res["macro_f05"]
                best_t = t
        country_thresholds[c_norm] = best_t
        print(f"  {c} ({c_norm}): threshold={best_t:.2f}, F0.5={best_f05:.4f} (n={len(c_val_ids):,})", flush=True)

    # Global evaluation
    print("\n--- Global Results ---", flush=True)
    all_matches = {}
    for c in countries:
        c_norm = normalize_country(c)
        c_val_ids = [sid for sid in val_ids if s1_country_map.get(sid) == c]
        c_probs = {sid: all_val_probs.get(sid, []) for sid in c_val_ids}
        c_matches = global_bipartite_assignment(c_probs, threshold=country_thresholds[c_norm])
        all_matches.update(c_matches)

    metrics = compute_macro_f05(val_gt, all_matches)
    print(f"\n  *** FINAL ENSEMBLE MACRO F0.5: {metrics['macro_f05']:.4f} ***", flush=True)
    print(f"  Macro Precision:    {metrics['macro_precision']:.4f}", flush=True)
    print(f"  Macro Recall:       {metrics['macro_recall']:.4f}", flush=True)
    print(f"  Singleton Accuracy: {metrics.get('singleton_accuracy', 0):.4f}", flush=True)
    print(f"  Non-Singleton F05:  {metrics.get('non_singleton_f05', 0):.4f}", flush=True)

    # Also try global uniform threshold
    best_gt, best_gf = 0.70, 0.0
    for t in thresholds_to_try:
        preds = global_bipartite_assignment(all_val_probs, threshold=t)
        res = compute_macro_f05(val_gt, preds)
        if res["macro_f05"] > best_gf:
            best_gf = res["macro_f05"]
            best_gt = t
    print(f"  Global uniform: threshold={best_gt:.2f}, F0.5={best_gf:.4f}", flush=True)

    # Use whichever is better
    if best_gf > metrics['macro_f05']:
        print(f"  -> Using global uniform threshold ({best_gt:.2f}) as it's better", flush=True)
        final_thresholds = {c_norm: best_gt for c_norm in [normalize_country(c) for c in countries]}
        final_f05 = best_gf
    else:
        print(f"  -> Using per-country thresholds as they're better", flush=True)
        final_thresholds = country_thresholds
        final_f05 = metrics['macro_f05']

    # Save
    save_path = os.path.join(args.models_dir, "ensemble_v3.pkl")
    tfidf_path = os.path.join(args.models_dir, "tfidf_models_v3.pkl")

    with open(save_path, "wb") as f:
        pickle.dump({
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "meta_model": meta_model,
            "country_thresholds": final_thresholds,
            "global_threshold": best_gt,
            "metrics": metrics,
            "feature_names": ENHANCED_FEATURE_NAMES,
        }, f)

    with open(tfidf_path, "wb") as f:
        pickle.dump({"name_tfidfs": name_tfidfs, "addr_tfidfs": addr_tfidfs}, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f" TRAINING COMPLETE in {elapsed/60:.1f} minutes", flush=True)
    print(f" Model: {save_path}", flush=True)
    print(f" TF-IDF: {tfidf_path}", flush=True)
    print(f" Final F0.5: {final_f05:.4f}", flush=True)
    print(f" Thresholds: {final_thresholds}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
