"""
Inference & Output Generation Module.

Generates:
1. output/candidate_pairs.tsv (blocking candidates fed to model)
2. output/matching_results.tsv (final thresholded and globally assigned matches)
3. Country-partitioned execution (France, US, India) to maintain optimal memory and speed.
"""

import os
import gc
import sys
import argparse
import pickle
from typing import Dict, List, Optional, Set
import numpy as np
import polars as pl
from tqdm import tqdm

from normalize import normalize_country
from blocking import CandidateBlocker
from features import RecordRepresentation, compute_pair_features
from assign import global_bipartite_assignment


def run_inference(
    test_dir: str = "student_resource/dataset/test",
    model_path: str = "cache/models/lgb_matcher.pkl",
    output_dir: str = "output",
    max_cands_per_entity: int = 35,
    batch_size: int = 10000,
    override_threshold: float = None,
    sample_size: Optional[int] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading trained model from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        saved_obj = pickle.load(f)
    model = saved_obj["model"]
    threshold = override_threshold if override_threshold is not None else saved_obj.get("threshold", 0.72)
    print(f"Loaded model successfully. Decision threshold: {threshold:.2f}", flush=True)

    # File paths
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"Loading S1 test records from {s1_path}...", flush=True)
    s1_df = pl.read_csv(s1_path, separator="\t")
    if sample_size is not None and sample_size > 0:
        s1_df = s1_df.head(sample_size)
    total_s1 = len(s1_df)
    print(f"Total S1 test entities: {total_s1:,}", flush=True)

    all_s1_ids = s1_df["entity_id"].to_list()
    countries = s1_df["country"].unique().to_list()
    print(f"Countries detected in S1: {countries}", flush=True)

    candidate_pairs_map: Dict[str, List[str]] = {}
    matching_results_map: Dict[str, Set[str]] = {}

    for c in countries:
        c_norm = normalize_country(c)
        print(f"\n==========================================", flush=True)
        print(f" Processing Country: {c} (normalized: {c_norm})", flush=True)
        print(f"==========================================", flush=True)

        s1_c_df = s1_df.filter(pl.col("country") == c)
        print(f"S1 {c} entities: {len(s1_c_df):,}", flush=True)

        print(f"Filtering S2 records for {c}...", flush=True)
        s2_c = pl.scan_csv(s2_path, separator="\t").filter(pl.col("country") == c).collect()
        print(f"  S2 {c} records: {len(s2_c):,}", flush=True)

        print(f"Filtering S3 records for {c}...", flush=True)
        s3_c = pl.scan_csv(s3_path, separator="\t").filter(pl.col("country") == c).collect()
        print(f"  S3 {c} records: {len(s3_c):,}", flush=True)

        blocker = CandidateBlocker(max_candidates_per_key=300, max_candidates_per_entity=max_cands_per_entity)
        target_reps: Dict[str, RecordRepresentation] = {}

        print(f"Indexing S2 & S3 targets for {c}...", flush=True)
        for r in s2_c.to_dicts():
            rep = RecordRepresentation(
                r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
            )
            target_reps[r["entity_id"]] = rep
            blocker.index_target_records([r])

        for r in s3_c.to_dicts():
            rep = RecordRepresentation(
                r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
            )
            target_reps[r["entity_id"]] = rep
            blocker.index_target_records([r])

        print(f"Indexed {len(target_reps):,} targets for {c}.", flush=True)

        s1_c_rows = s1_c_df.to_dicts()
        s1_candidate_probs: Dict[str, List[tuple]] = {}

        for idx in range(0, len(s1_c_rows), batch_size):
            batch = s1_c_rows[idx : idx + batch_size]
            for r in batch:
                s1_id = r["entity_id"]
                s1_rep = RecordRepresentation(
                    s1_id, r["business_name"] or "", r["business_address"] or "", r["country"] or ""
                )
                cands = blocker.retrieve_candidates_for_entity(
                    s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country
                )
                candidate_pairs_map[s1_id] = cands

                if not cands:
                    s1_candidate_probs[s1_id] = []
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
                    X_batch = np.array(feat_batch, dtype=np.float32)
                    probs = model.predict_proba(X_batch)[:, 1]
                    s1_candidate_probs[s1_id] = list(zip(valid_cids, probs))
                else:
                    s1_candidate_probs[s1_id] = []

            if (idx + batch_size) % 50000 == 0 or (idx + batch_size) >= len(s1_c_rows):
                print(f"  Processed {min(idx + batch_size, len(s1_c_rows)):,} / {len(s1_c_rows):,} S1 entities in {c}", flush=True)

        print(f"Applying bipartite global assignment for {c}...", flush=True)
        c_matches = global_bipartite_assignment(s1_candidate_probs, threshold=threshold)
        matching_results_map.update(c_matches)

        # Free memory
        del blocker, target_reps, s2_c, s3_c, s1_c_df, s1_c_rows, s1_candidate_probs
        gc.collect()

    print(f"\n--- Writing Final Output Files ---", flush=True)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"Writing {cand_path}...", flush=True)
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in all_s1_ids:
            cands = candidate_pairs_map.get(eid, [])
            cand_str = ",".join(cands)
            f.write(f"{eid}\t{cand_str}\n")

    print(f"Writing {match_path}...", flush=True)
    num_matched_entities = 0
    total_matched_links = 0
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in all_s1_ids:
            matches = matching_results_map.get(eid, set())
            match_str = ",".join(sorted(matches))
            f.write(f"{eid}\t{match_str}\n")
            if matches:
                num_matched_entities += 1
                total_matched_links += len(matches)

    print(f"\nFinal Test Matching Summary:", flush=True)
    print(f"  Total S1 entities written: {total_s1:,}", flush=True)
    print(f"  Entities with matches: {num_matched_entities:,} ({num_matched_entities/total_s1:.2%})", flush=True)
    print(f"  Singletons (no matches): {total_s1 - num_matched_entities:,} ({(total_s1 - num_matched_entities)/total_s1:.2%})", flush=True)
    print(f"  Total matched links: {total_matched_links:,}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--model-path", default="cache/models/lgb_matcher.pkl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    run_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        override_threshold=args.threshold,
        sample_size=args.sample_size,
    )
