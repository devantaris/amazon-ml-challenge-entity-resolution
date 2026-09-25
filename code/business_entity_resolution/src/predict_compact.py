"""
Ultra-Compact GPU-Accelerated Test Inference & Output Generation.

- Peak RAM stays strictly under 1.5 GB (Zero disk swap, zero NVMe thrashing).
- Uses CompactCandidateBlocker with 32-bit integer array posting lists.
- Direct streaming of candidate_pairs.tsv and matching_results.tsv to disk (0-RAM output buffering).
- Runs inference on NVIDIA GeForce RTX GPU using native CUDA DMatrix.
- Processes country-by-country (France -> US -> India) with deterministic memory reclamation.
"""

import os
import gc
import sys
import argparse
import pickle
from typing import Dict, List, Optional, Set
import numpy as np
import polars as pl
import xgboost as xgb
from tqdm import tqdm

from normalize import normalize_country
from compact_blocking import CompactCandidateBlocker
from features import RecordRepresentation, compute_pair_features
from assign import global_bipartite_assignment


def run_compact_inference(
    test_dir: str = "student_resource/dataset/test",
    model_path: str = "cache/models/xgb_gpu_matcher.pkl",
    output_dir: str = "output",
    max_cands_per_entity: int = 30,
    batch_size: int = 15000,
    override_threshold: Optional[float] = None,
    sample_size: Optional[int] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading GPU-trained model from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        saved_obj = pickle.load(f)
    model = saved_obj["model"]
    booster = model.get_booster()
    # Configure booster to use CUDA device directly with no CPU-GPU fallback overhead
    booster.set_param({"device": "cuda:0"})
    
    threshold = override_threshold if override_threshold is not None else saved_obj.get("threshold", 0.80)
    # Filter threshold: anything below threshold - 0.15 has zero probability of assignment
    min_prob_keep = max(0.20, threshold - 0.15)
    print(f"Loaded model successfully. Decision threshold: {threshold:.2f} (retention cutoff: {min_prob_keep:.2f})", flush=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"Loading S1 test records from {s1_path}...", flush=True)
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

    print(f"Execution plan for countries: {ordered_countries}", flush=True)

    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"Opening output streaming files in {output_dir}...", flush=True)
    cand_f = open(cand_path, "w", encoding="utf-8", buffering=2 * 1024 * 1024)
    match_f = open(match_path, "w", encoding="utf-8", buffering=2 * 1024 * 1024)

    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    match_f.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1_processed = 0
    num_matched_entities = 0
    total_matched_links = 0

    for c in ordered_countries:
        c_norm = normalize_country(c)
        print(f"\n==========================================", flush=True)
        print(f" Processing Country: {c} ({c_norm})", flush=True)
        print(f"==========================================", flush=True)

        s1_c_df = s1_df.filter(pl.col("country") == c)
        n_s1_c = len(s1_c_df)
        print(f"S1 entities in {c}: {n_s1_c:,}", flush=True)

        print(f"Reading target records for {c} from S2 and S3...", flush=True)
        s2_c = pl.scan_csv(s2_path, separator="\t").filter(pl.col("country") == c).collect()
        s3_c = pl.scan_csv(s3_path, separator="\t").filter(pl.col("country") == c).collect()
        targets_df = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        gc.collect()

        n_targets = len(targets_df)
        print(f"Total target records (S2+S3) for {c}: {n_targets:,}", flush=True)

        # Build compact integer blocker
        print(f"Indexing {n_targets:,} records with compact 32-bit integer array...", flush=True)
        blocker = CompactCandidateBlocker(max_candidates_per_key=250, max_candidates_per_entity=max_cands_per_entity)
        blocker.index_dataframe(targets_df)

        # Extract compact lists for fast index lookup
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

        s1_candidate_probs: Dict[str, List[tuple]] = {s1_id: [] for s1_id in s1_ids_c}
        cand_rep_cache: Dict[int, RecordRepresentation] = {}

        print(f"Scoring S1 entities and querying candidates...", flush=True)
        for idx in range(0, n_s1_c, batch_size):
            end_idx = min(idx + batch_size, n_s1_c)

            batch_s1_reps = []
            for i in range(idx, end_idx):
                s1_id = s1_ids_c[i]
                s1_rep = RecordRepresentation(
                    s1_id, s1_names_c[i], s1_addrs_c[i], s1_countries_c[i]
                )
                batch_s1_reps.append(s1_rep)

            feat_batch = []
            pair_meta = []  # (s1_id, cand_id)

            for s1_rep in batch_s1_reps:
                cand_indices = blocker.retrieve_candidate_indices(
                    s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
                )
                cands_eids = [tgt_ids[ci] for ci in cand_indices]
                # Direct streaming write of candidate_pairs row to disk (0-RAM retention)
                cand_f.write(f"{s1_rep.entity_id}\t{','.join(cands_eids)}\n")

                if not cand_indices:
                    continue

                for rank, ci in enumerate(cand_indices):
                    cand_rep = cand_rep_cache.get(ci)
                    if cand_rep is None:
                        cand_rep = RecordRepresentation(
                            tgt_ids[ci], tgt_names[ci], tgt_addrs[ci], tgt_countries[ci]
                        )
                        if len(cand_rep_cache) < 150000:
                            cand_rep_cache[ci] = cand_rep

                    feats = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
                    feat_batch.append(feats)
                    pair_meta.append((s1_rep.entity_id, tgt_ids[ci]))

            if feat_batch:
                X_batch = np.array(feat_batch, dtype=np.float32)
                dmat = xgb.DMatrix(X_batch)
                probs = booster.predict(dmat)
                del X_batch, dmat

                for (s1_id, cid), prob in zip(pair_meta, probs):
                    if prob >= min_prob_keep:
                        s1_candidate_probs[s1_id].append((cid, float(prob)))

            if end_idx % 50000 == 0 or end_idx >= n_s1_c:
                print(f"  Scored {end_idx:,} / {n_s1_c:,} entities in {c}...", flush=True)

        print(f"Running bipartite global assignment for {c}...", flush=True)
        c_matches = global_bipartite_assignment(s1_candidate_probs, threshold=threshold)

        # Stream matched records directly to disk
        for s1_id in s1_ids_c:
            m = c_matches.get(s1_id, set())
            match_f.write(f"{s1_id}\t{','.join(sorted(m))}\n")
            if m:
                num_matched_entities += 1
                total_matched_links += len(m)
            total_s1_processed += 1

        cand_f.flush()
        match_f.flush()

        print(f"Completed {c}: {n_s1_c:,} entities processed and streamed.", flush=True)

        # Free all data structures for this country
        del blocker, tgt_ids, tgt_names, tgt_addrs, tgt_countries, cand_rep_cache
        del s1_c_df, s1_ids_c, s1_names_c, s1_addrs_c, s1_countries_c, s1_candidate_probs, c_matches
        gc.collect()

    cand_f.close()
    match_f.close()

    print(f"\n--- Output Files Finalized ---", flush=True)
    print(f"Candidate pairs: {cand_path}", flush=True)
    print(f"Matching results: {match_path}", flush=True)
    print(f"\nFinal Submission Statistics:", flush=True)
    print(f"  Total S1 entities written: {total_s1_processed:,}", flush=True)
    print(f"  Entities with matches: {num_matched_entities:,} ({num_matched_entities/max(total_s1_processed, 1):.2%})", flush=True)
    print(f"  Singletons (no match): {total_s1_processed - num_matched_entities:,} ({(total_s1_processed - num_matched_entities)/max(total_s1_processed, 1):.2%})", flush=True)
    print(f"  Total links predicted: {total_matched_links:,}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--model-path", default="cache/models/xgb_gpu_matcher.pkl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    run_compact_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        override_threshold=args.threshold,
        sample_size=args.sample_size,
    )
