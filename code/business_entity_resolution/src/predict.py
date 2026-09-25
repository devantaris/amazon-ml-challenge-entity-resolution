"""
Inference & Output Generation Module.

Generates:
1. output/candidate_pairs.tsv (blocking candidates fed to model)
2. output/matching_results.tsv (final thresholded and globally assigned matches)
3. Validates outputs with student_resource/utils/validate_submission.py
"""

import os
import argparse
import pickle
import numpy as np
import polars as pl
from tqdm import tqdm

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
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        saved_obj = pickle.load(f)
    model = saved_obj["model"]
    threshold = override_threshold if override_threshold is not None else saved_obj.get("threshold", 0.65)
    print(f"Loaded model successfully. Decision threshold: {threshold:.2f}")

    # File paths
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n--- 1. Indexing Target Records (S2 & S3) ---")
    blocker = CandidateBlocker(max_candidates_per_key=300, max_candidates_per_entity=max_cands_per_entity)
    target_reps = {}

    print(f"Reading S2 records from {s2_path}...")
    s2_df = pl.read_csv(s2_path, separator="\t")
    print(f"S2 rows: {len(s2_df):,}")
    for r in tqdm(s2_df.iter_rows(named=True), total=len(s2_df), desc="Indexing S2"):
        rep = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )
        target_reps[r["entity_id"]] = rep
        blocker.index_target_records([r])

    print(f"Reading S3 records from {s3_path}...")
    s3_df = pl.read_csv(s3_path, separator="\t")
    print(f"S3 rows: {len(s3_df):,}")
    for r in tqdm(s3_df.iter_rows(named=True), total=len(s3_df), desc="Indexing S3"):
        rep = RecordRepresentation(
            r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or ""
        )
        target_reps[r["entity_id"]] = rep
        blocker.index_target_records([r])

    print(f"Total target records indexed: {len(target_reps):,}")

    print(f"\n--- 2. Processing Test Source 1 Entities ---")
    s1_df = pl.read_csv(s1_path, separator="\t")
    total_s1 = len(s1_df)
    print(f"Total S1 test entities: {total_s1:,}")

    candidate_pairs_map = {}
    s1_candidate_probs = {}

    s1_rows = s1_df.to_dicts()

    for idx in tqdm(range(0, total_s1, batch_size), desc="Scoring S1 Batches"):
        batch_slice = s1_rows[idx : idx + batch_size]

        for r in batch_slice:
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

    print(f"\n--- 3. Running Global Bipartite Assignment ---")
    matching_results_map = global_bipartite_assignment(
        s1_candidate_probs, threshold=threshold
    )

    print(f"\n--- 4. Writing Output Files ---")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"Writing {cand_path}...")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in s1_rows:
            eid = r["entity_id"]
            cands = candidate_pairs_map.get(eid, [])
            cand_str = ",".join(cands)
            f.write(f"{eid}\t{cand_str}\n")

    print(f"Writing {match_path}...")
    num_matched_entities = 0
    total_matched_links = 0
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for r in s1_rows:
            eid = r["entity_id"]
            matches = matching_results_map.get(eid, set())
            match_str = ",".join(sorted(matches))
            f.write(f"{eid}\t{match_str}\n")
            if matches:
                num_matched_entities += 1
                total_matched_links += len(matches)

    print(f"\nMatching Results Summary:")
    print(f"  Total S1 entities: {total_s1:,}")
    print(f"  Entities with matches: {num_matched_entities:,} ({num_matched_entities/total_s1:.2%})")
    print(f"  Singletons (no match): {total_s1 - num_matched_entities:,} ({(total_s1 - num_matched_entities)/total_s1:.2%})")
    print(f"  Total matched links: {total_matched_links:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--model-path", default="cache/models/lgb_matcher.pkl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    run_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        override_threshold=args.threshold,
    )
