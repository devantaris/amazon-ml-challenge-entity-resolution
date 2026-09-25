"""
GPU-Accelerated Model Training Module for Business Entity Resolution.

Memory-safe, high-speed training on NVIDIA GeForce RTX GPU:
- Pre-allocated numpy arrays (zero Python list fragmentation)
- Compact target pool (< 600 MB RAM total)
- NVIDIA CUDA execution via XGBoost GPU hist
- Calibrated optimal threshold for macro F_0.5
"""

import os
import gc
import psutil
import argparse
import pickle
import numpy as np
import polars as pl
import xgboost as xgb

from normalize import normalize_country
from blocking import CandidateBlocker
from features import FEATURE_NAMES, RecordRepresentation, compute_pair_features
from eval_metric import compute_macro_f05
from assign import sweep_optimal_threshold


def train_gpu_matching_model(
    data_dir: str = "student_resource/dataset/train",
    splits_dir: str = "cache/splits",
    models_dir: str = "cache/models",
    n_train_entities: int = 12000,
    n_val_entities: int = 2000,
    max_cands_per_entity: int = 25,
    random_state: int = 42,
) -> None:
    os.makedirs(models_dir, exist_ok=True)
    process = psutil.Process(os.getpid())

    print("========================================", flush=True)
    print(" GPU Accelerated Model Training (CUDA)  ", flush=True)
    print("========================================", flush=True)
    print(f"Initial RAM: {process.memory_info().rss / (1024*1024):.1f} MB", flush=True)

    print("\n--- 1. Loading Training and Validation Splits ---", flush=True)
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

    print(f"Ground truth true matches needed: {len(needed_target_ids):,}", flush=True)

    print("\n--- 3. Loading S1 Records ---", flush=True)
    s1_df = pl.scan_csv(os.path.join(data_dir, "train_source1.tsv"), separator="\t")
    s1_rows = s1_df.filter(pl.col("entity_id").is_in(all_eval_s1_ids)).collect()

    s1_rep_map = {}
    for r in s1_rows.to_dicts():
        s1_rep_map[r["entity_id"]] = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )

    print("\n--- 4. Building Candidate Blocker Index ---", flush=True)
    blocker = CandidateBlocker(max_candidates_per_key=250, max_candidates_per_entity=max_cands_per_entity)

    print("Indexing S2 records...", flush=True)
    s2_df = pl.scan_csv(os.path.join(data_dir, "train_source2.tsv"), separator="\t")
    s2_targets = s2_df.filter(pl.col("entity_id").is_in(needed_target_ids)).collect()
    s2_sample = s2_df.head(25000).collect()
    s2_slice = pl.concat([s2_targets, s2_sample]).unique(subset=["entity_id"])
    blocker.index_target_records(s2_slice.to_dicts())

    print("Indexing S3 records...", flush=True)
    s3_df = pl.scan_csv(os.path.join(data_dir, "train_source3.tsv"), separator="\t")
    s3_targets = s3_df.filter(pl.col("entity_id").is_in(needed_target_ids)).collect()
    s3_sample = s3_df.head(25000).collect()
    s3_slice = pl.concat([s3_targets, s3_sample]).unique(subset=["entity_id"])
    blocker.index_target_records(s3_slice.to_dicts())

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
    print(f"RAM after indexing: {process.memory_info().rss / (1024*1024):.1f} MB", flush=True)

    print("\n--- 5. Generating Training Pairs & 33 Features ---", flush=True)
    # Estimate capacity and pre-allocate
    max_estimated_pairs = len(train_s1_ids) * (max_cands_per_entity + 3)
    X_train = np.zeros((max_estimated_pairs, len(FEATURE_NAMES)), dtype=np.float32)
    y_train = np.zeros(max_estimated_pairs, dtype=np.int32)
    num_pairs = 0

    for s1_id in train_s1_ids:
        s1_rep = s1_rep_map.get(s1_id)
        if not s1_rep:
            continue
        cands = blocker.retrieve_candidates_for_entity(
            s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
        )
        true_targets = gt_map.get(s1_id, set())
        all_train_cands = set(cands) | (true_targets & target_reps.keys())

        for rank, cid in enumerate(all_train_cands):
            cand_rep = target_reps.get(cid)
            if not cand_rep:
                continue
            is_match = 1 if cid in true_targets else 0
            feats = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            X_train[num_pairs] = feats
            y_train[num_pairs] = is_match
            num_pairs += 1

    # Slice to actual size
    X_train = X_train[:num_pairs]
    y_train = y_train[:num_pairs]
    print(f"X_train shape: {X_train.shape}, Positive rate: {np.mean(y_train):.2%}", flush=True)
    print(f"RAM before GPU fitting: {process.memory_info().rss / (1024*1024):.1f} MB", flush=True)

    print("\n--- 6. Fitting XGBoost Classifier on NVIDIA RTX GPU (CUDA) ---", flush=True)
    model = xgb.XGBClassifier(
        device="cuda",
        tree_method="hist",
        n_estimators=400,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    print("\n--- Feature Importances ---", flush=True)
    importances = model.feature_importances_
    sorted_idx = np.argsort(-importances)
    for idx in sorted_idx[:15]:
        print(f"  {FEATURE_NAMES[idx]:22s}: {importances[idx]:.4f}", flush=True)

    print("\n--- 7. Validating and Tuning Threshold on Validation Set ---", flush=True)
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
        else:
            cand_pairs = []

        val_candidate_probs[s1_id] = cand_pairs

    best_thresh, best_f05, best_metrics = sweep_optimal_threshold(
        val_candidate_probs, val_true_gt
    )

    # Save GPU model and threshold
    model_save_path = os.path.join(models_dir, "xgb_gpu_matcher.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": model, "threshold": best_thresh, "metrics": best_metrics}, f)

    print(f"\nGPU Model saved to {model_save_path}", flush=True)
    print(f"Final Peak RAM: {process.memory_info().rss / (1024*1024):.1f} MB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=2000)
    args = parser.parse_args()

    train_gpu_matching_model(
        n_train_entities=args.train_size,
        n_val_entities=args.val_size,
    )
