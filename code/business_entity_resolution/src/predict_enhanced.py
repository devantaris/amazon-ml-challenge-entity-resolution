"""
Enhanced Streaming Inference with Ensemble Model.

Uses the ensemble (XGBoost + LightGBM + Meta-Learner) with 42 enhanced features.
Streams country-by-country with direct disk output.
Peak RAM: < 1.5 GB.
"""

import os
import gc
import sys
import argparse
import pickle
import time
from typing import Dict, List, Optional, Set
import numpy as np
import polars as pl
import xgboost as xgb

from normalize import normalize_country
from compact_blocking import CompactCandidateBlocker
from features import RecordRepresentation, compute_pair_features
from enhanced_features import (
    ENHANCED_FEATURE_NAMES,
    CompactTFIDF,
    compute_enhanced_features,
)
from assign import global_bipartite_assignment


def run_enhanced_inference(
    test_dir: str = "student_resource/dataset/test",
    model_path: str = "cache/models/ensemble_v5.pkl",
    tfidf_path: str = "cache/models/tfidf_models_v5.pkl",
    output_dir: str = "output",
    max_cands_per_entity: int = 30,
    batch_size: int = 15000,
    sample_size: Optional[int] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    print("Loading ensemble model...", flush=True)
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    xgb_model = saved["xgb_model"]
    lgb_model = saved["lgb_model"]
    meta_model = saved["meta_model"]
    country_thresholds = saved.get("country_thresholds", {})
    global_threshold = saved.get("global_threshold", 0.75)

    # Set XGBoost booster to CUDA for native GPU inference
    booster = xgb_model.get_booster()
    booster.set_param({"device": "cuda:0"})

    print(f"Country thresholds: {country_thresholds}", flush=True)
    print(f"Global fallback threshold: {global_threshold:.2f}", flush=True)

    # Load TF-IDF models
    print("Loading TF-IDF models...", flush=True)
    name_tfidfs = {}
    addr_tfidfs = {}
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            tfidf_saved = pickle.load(f)
        name_tfidfs = tfidf_saved.get("name_tfidfs", {})
        addr_tfidfs = tfidf_saved.get("addr_tfidfs", {})
        print(f"Loaded TF-IDF for countries: {list(name_tfidfs.keys())}", flush=True)
    else:
        print(f"WARNING: {tfidf_path} not found, TF-IDF features will be zero", flush=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"Loading S1 test records...", flush=True)
    s1_df = pl.read_csv(s1_path, separator="\t")
    if sample_size is not None and sample_size > 0:
        s1_df = s1_df.head(sample_size)
    total_s1 = len(s1_df)
    print(f"Total S1 test entities: {total_s1:,}", flush=True)

    detected_countries = s1_df["country"].unique().to_list()
    ordered_countries = [c for c in ["France", "US", "India"] if c in detected_countries]
    for c in detected_countries:
        if c not in ordered_countries:
            ordered_countries.append(c)

    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    cand_f = open(cand_path, "w", encoding="utf-8", buffering=2 * 1024 * 1024)
    match_f = open(match_path, "w", encoding="utf-8", buffering=2 * 1024 * 1024)
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    match_f.write("source1_entity_id\tmatched_entity_ids\n")

    total_processed = 0
    num_matched = 0
    total_links = 0

    for c in ordered_countries:
        c_norm = normalize_country(c)
        print(f"\n{'='*50}", flush=True)
        print(f" Processing Country: {c} ({c_norm})", flush=True)
        print(f"{'='*50}", flush=True)

        s1_c_df = s1_df.filter(pl.col("country") == c)
        n_s1_c = len(s1_c_df)
        print(f"S1 entities: {n_s1_c:,}", flush=True)

        # Load targets for this country
        s2_c = pl.scan_csv(s2_path, separator="\t").filter(pl.col("country") == c).collect()
        s3_c = pl.scan_csv(s3_path, separator="\t").filter(pl.col("country") == c).collect()
        targets_df = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        gc.collect()

        n_targets = len(targets_df)
        print(f"Target records (S2+S3): {n_targets:,}", flush=True)

        # Build TF-IDF for this country's test targets if not already fitted
        name_tfidf = name_tfidfs.get(c_norm)
        addr_tfidf = addr_tfidfs.get(c_norm)

        if name_tfidf is None:
            print(f"  Fitting test TF-IDF for {c_norm}...", flush=True)
            name_tok_lists = []
            addr_tok_lists = []
            names_list = targets_df["business_name"].fill_null("").to_list()
            addrs_list = targets_df["business_address"].fill_null("").to_list()
            # Sample for TF-IDF fitting (first 100K records)
            sample_n = min(100000, len(names_list))
            for i in range(sample_n):
                rep_tmp = RecordRepresentation("tmp", names_list[i], addrs_list[i], c)
                name_tok_lists.append(rep_tmp.name_tokens)
                addr_tok_lists.append(rep_tmp.addr_tokens)
            name_tfidf = CompactTFIDF(max_features=30000)
            name_tfidf.fit(name_tok_lists)
            addr_tfidf = CompactTFIDF(max_features=30000)
            addr_tfidf.fit(addr_tok_lists)
            del name_tok_lists, addr_tok_lists
            gc.collect()
            print(f"  TF-IDF fitted: name vocab={len(name_tfidf.vocab)}, addr vocab={len(addr_tfidf.vocab)}", flush=True)

        # Build compact blocker
        print(f"Indexing with compact 32-bit integer array...", flush=True)
        blocker = CompactCandidateBlocker(max_candidates_per_key=250, max_candidates_per_entity=max_cands_per_entity)
        blocker.index_dataframe(targets_df)

        tgt_ids = targets_df["entity_id"].to_list()
        tgt_names = targets_df["business_name"].fill_null("").to_list()
        tgt_addrs = targets_df["business_address"].fill_null("").to_list()
        tgt_countries = targets_df["country"].fill_null("").to_list()

        del targets_df
        gc.collect()

        s1_ids_c = s1_c_df["entity_id"].to_list()
        s1_names_c = s1_c_df["business_name"].fill_null("").to_list()
        s1_addrs_c = s1_c_df["business_address"].fill_null("").to_list()
        s1_countries_c = s1_c_df["country"].fill_null("").to_list()

        threshold = country_thresholds.get(c_norm, global_threshold)
        min_prob_keep = max(0.20, threshold - 0.15)
        print(f"Using threshold: {threshold:.2f} (retention cutoff: {min_prob_keep:.2f})", flush=True)

        s1_candidate_probs: Dict[str, List[tuple]] = {s1_id: [] for s1_id in s1_ids_c}

        print(f"Scoring S1 entities...", flush=True)
        for idx in range(0, n_s1_c, batch_size):
            end_idx = min(idx + batch_size, n_s1_c)

            batch_s1_reps = []
            for i in range(idx, end_idx):
                s1_rep = RecordRepresentation(
                    s1_ids_c[i], s1_names_c[i], s1_addrs_c[i], s1_countries_c[i]
                )
                batch_s1_reps.append(s1_rep)

            feat_batch = []
            pair_meta = []

            for s1_rep in batch_s1_reps:
                cand_indices = blocker.retrieve_candidate_indices(
                    s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
                )
                cands_eids = [tgt_ids[ci] for ci in cand_indices]
                cand_f.write(f"{s1_rep.entity_id}\t{','.join(cands_eids)}\n")

                if not cand_indices:
                    continue

                for rank, ci in enumerate(cand_indices):
                    cand_rep = RecordRepresentation(
                        tgt_ids[ci], tgt_names[ci], tgt_addrs[ci], tgt_countries[ci]
                    )
                    base_feats = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
                    enhanced = compute_enhanced_features(
                        base_feats,
                        s1_rep.name_tokens, s1_rep.addr_tokens,
                        cand_rep.name_tokens, cand_rep.addr_tokens,
                        s1_rep.name_clean, cand_rep.name_clean,
                        name_tfidf, addr_tfidf,
                    )
                    feat_batch.append(enhanced)
                    pair_meta.append((s1_rep.entity_id, tgt_ids[ci]))

            if feat_batch:
                X_batch = np.array(feat_batch, dtype=np.float32)

                # XGBoost prediction via native CUDA DMatrix
                dmat = xgb.DMatrix(X_batch)
                p_xgb = booster.predict(dmat)
                del dmat

                # LightGBM prediction (CPU)
                p_lgb = lgb_model.predict_proba(X_batch)[:, 1]

                # Meta-learner ensemble
                meta_X = np.column_stack([p_xgb, p_lgb])
                p_meta = meta_model.predict_proba(meta_X)[:, 1]

                del X_batch

                for (s1_id, cid), prob in zip(pair_meta, p_meta):
                    if prob >= min_prob_keep:
                        s1_candidate_probs[s1_id].append((cid, float(prob)))

            if end_idx % 50000 == 0 or end_idx >= n_s1_c:
                print(f"  Scored {end_idx:,} / {n_s1_c:,} entities in {c}...", flush=True)

        # Assignment
        c_matches = global_bipartite_assignment(s1_candidate_probs, threshold=threshold)

        for s1_id in s1_ids_c:
            m = c_matches.get(s1_id, set())
            match_f.write(f"{s1_id}\t{','.join(sorted(m))}\n")
            if m:
                num_matched += 1
                total_links += len(m)
            total_processed += 1

        cand_f.flush()
        match_f.flush()

        elapsed = time.time() - t_start
        print(f"Completed {c} in {elapsed:.0f}s total", flush=True)

        del blocker, tgt_ids, tgt_names, tgt_addrs, tgt_countries
        del s1_c_df, s1_ids_c, s1_names_c, s1_addrs_c, s1_countries_c, s1_candidate_probs, c_matches
        gc.collect()

    cand_f.close()
    match_f.close()

    total_time = time.time() - t_start
    print(f"\n--- Output Files Finalized ({total_time/60:.1f} min) ---", flush=True)
    print(f"  Total S1 entities: {total_processed:,}", flush=True)
    print(f"  Entities with matches: {num_matched:,} ({num_matched/max(total_processed,1):.2%})", flush=True)
    print(f"  Singletons: {total_processed - num_matched:,}", flush=True)
    print(f"  Total links: {total_links:,}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--model-path", default="cache/models/ensemble_v5.pkl")
    parser.add_argument("--tfidf-path", default="cache/models/tfidf_models_v5.pkl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    run_enhanced_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        tfidf_path=args.tfidf_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    )
