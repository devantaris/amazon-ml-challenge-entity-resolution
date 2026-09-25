"""
Model Training Module for Business Entity Resolution.

Trains a LightGBM gradient-boosted tree on pairwise candidate features:
- Pairs S1 entities with candidates from blocking
- Computes pairwise similarity features
- Fits binary classification model
- Validates on held-out stratified validation set
- Sweeps decision threshold to maximize macro F_0.5
"""

import os
import argparse
import pickle
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from normalize import normalize_country
from blocking import CandidateBlocker
from features import FEATURE_NAMES, RecordRepresentation, compute_pair_features
from eval_metric import compute_macro_f05
from assign import sweep_optimal_threshold, global_bipartite_assignment


def train_matching_model(
    data_dir: str = "student_resource/dataset/train",
    splits_dir: str = "cache/splits",
    models_dir: str = "cache/models",
    n_train_entities: int = 15000,
    n_val_entities: int = 3000,
    max_cands_per_entity: int = 35,
    random_state: int = 42,
) -> None:
    os.makedirs(models_dir, exist_ok=True)

    print("--- 1. Loading Training and Validation Splits ---", flush=True)
    with open(os.path.join(splits_dir, "train_s1_ids.txt"), "r") as f:
        train_s1_ids = set([line.strip() for line in f if line.strip()][:n_train_entities])

    with open(os.path.join(splits_dir, "val_s1_ids.txt"), "r") as f:
        val_s1_ids = set([line.strip() for line in f if line.strip()][:n_val_entities])

    all_eval_s1_ids = train_s1_ids | val_s1_ids
    print(f"Loaded {len(train_s1_ids):,} train entities, {len(val_s1_ids):,} val entities.", flush=True)

    print("\n--- 2. Loading Ground Truth ---", flush=True)
    gt_df = pl.scan_csv(os.path.join(data_dir, "train_ground_truth.tsv"), separator="\t")
    gt_rows = gt_df.filter(pl.col("source1_entity_id").is_in(all_eval_s1_ids)).collect()

    gt_map = {}
    needed_target_ids = set()
    for row in gt_rows.to_dicts():
        s1_id = row["source1_entity_id"]
        m_str = (row.get("matched_entity_ids") or "").strip()
        if m_str:
            targets = set([x.strip() for x in m_str.split(",") if x.strip()])
            gt_map[s1_id] = targets
            needed_target_ids.update(targets)
        else:
            gt_map[s1_id] = set()

    print(f"Total ground truth true matches needed for evaluation: {len(needed_target_ids):,}", flush=True)

    print("\n--- 3. Loading S1 Records ---", flush=True)
    s1_df = pl.scan_csv(os.path.join(data_dir, "train_source1.tsv"), separator="\t")
    s1_rows = s1_df.filter(pl.col("entity_id").is_in(all_eval_s1_ids)).collect()

    s1_rep_map = {}
    for r in s1_rows.to_dicts():
        s1_rep_map[r["entity_id"]] = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )

    print("\n--- 4. Building Candidate Blocker Index ---")
    blocker = CandidateBlocker(max_candidates_per_key=300, max_candidates_per_entity=max_cands_per_entity)

    # For fast and memory-efficient training, load relevant portions or scan S2/S3
    print("Indexing S2 records...", flush=True)
    s2_df = pl.scan_csv(os.path.join(data_dir, "train_source2.tsv"), separator="\t")
    s2_targets = s2_df.filter(pl.col("entity_id").is_in(needed_target_ids)).collect()
    s2_sample = s2_df.head(100000).collect()
    s2_slice = pl.concat([s2_targets, s2_sample]).unique(subset=["entity_id"])
    blocker.index_target_records(s2_slice.to_dicts())

    print("Indexing S3 records...", flush=True)
    s3_df = pl.scan_csv(os.path.join(data_dir, "train_source3.tsv"), separator="\t")
    s3_targets = s3_df.filter(pl.col("entity_id").is_in(needed_target_ids)).collect()
    s3_sample = s3_df.head(100000).collect()
    s3_slice = pl.concat([s3_targets, s3_sample]).unique(subset=["entity_id"])
    blocker.index_target_records(s3_slice.to_dicts())

    # Build target representations dictionary for fast feature lookup
    target_reps = {}
    for r in s2_slice.to_dicts():
        target_reps[r["entity_id"]] = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )
    for r in s3_slice.to_dicts():
        target_reps[r["entity_id"]] = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )
    print(f"Total target records indexed: {len(target_reps):,}", flush=True)

    print("\n--- 5. Generating Training Pairs & Features ---")
    X_train_list = []
    y_train_list = []

    for s1_id in train_s1_ids:
        s1_rep = s1_rep_map.get(s1_id)
        if not s1_rep:
            continue
        cands = blocker.retrieve_candidates_for_entity(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )
        true_targets = gt_map.get(s1_id, set())

        # Always include true targets in training pairs to learn true positive signals
        all_train_cands = set(cands) | (true_targets & target_reps.keys())

        for rank, cid in enumerate(all_train_cands):
            cand_rep = target_reps.get(cid)
            if not cand_rep:
                continue
            is_match = 1 if cid in true_targets else 0
            feats = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            X_train_list.append(feats)
            y_train_list.append(is_match)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"X_train shape: {X_train.shape}, Positive rate: {np.mean(y_train):.2%}")

    print("\n--- 6. Fitting LightGBM Classifier ---")
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.08,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    sorted_idx = np.argsort(-importances)
    for idx in sorted_idx[:15]:
        print(f"  {FEATURE_NAMES[idx]:22s}: {importances[idx]}")

    print("\n--- 7. Validating and Tuning Threshold on Validation Set ---")
    val_candidate_probs = {}
    val_true_gt = {}

    for s1_id in val_s1_ids:
        s1_rep = s1_rep_map.get(s1_id)
        if not s1_rep:
            continue
        val_true_gt[s1_id] = gt_map.get(s1_id, set())
        cands = blocker.retrieve_candidates_for_entity(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )
        if not cands:
            val_candidate_probs[s1_id] = []
            continue

        cand_pairs = []
        feat_batch = []
        valid_cids = []
        for rank, cid in enumerate(cands):
            cand_rep = target_reps.get(cid)
            if not cand_rep:
                continue
            feats = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            feat_batch.append(feats)
            valid_cids.append(cid)

        if feat_batch:
            X_val_batch = np.array(feat_batch, dtype=np.float32)
            probs = model.predict_proba(X_val_batch)[:, 1]
            cand_pairs = list(zip(valid_cids, probs))
        
        val_candidate_probs[s1_id] = cand_pairs

    best_thresh, best_f05, best_metrics = sweep_optimal_threshold(
        val_candidate_probs, val_true_gt
    )

    # Save model and config
    model_save_path = os.path.join(models_dir, "lgb_matcher.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": model, "threshold": best_thresh, "metrics": best_metrics}, f)

    print(f"\nModel and configuration saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=15000)
    parser.add_argument("--val-size", type=int, default=3000)
    args = parser.parse_args()

    train_matching_model(
        n_train_entities=args.train_size,
        n_val_entities=args.val_size,
    )
