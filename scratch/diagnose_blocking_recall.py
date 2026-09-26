"""
Diagnostic: measure TRUE blocking recall of CompactCandidateBlocker on a FULL
per-country target pool (all train S2+S3 records for that country), using the
same parameters as predict_compact.py used for the submitted run.

Also measures end-to-end F0.5 of the submitted xgb_gpu_matcher model on the
same sample, per country, to decompose the leaderboard 0.647.
"""

import os
import sys
import time
import pickle
import random
import collections
import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))

from compact_blocking import CompactCandidateBlocker
from features import RecordRepresentation, compute_pair_features
from eval_metric import compute_macro_f05
import xgboost as xgb

DATA = "student_resource/dataset/train"
N_SAMPLE = 2000
CAPS = [30, 60, 100]


def load_gt():
    gt = pl.read_csv(os.path.join(DATA, "train_ground_truth.tsv"), separator="\t")
    out = {}
    for row in gt.to_dicts():
        m = (row["matched_entity_ids"] or "").strip()
        out[row["source1_entity_id"]] = {x.strip() for x in m.split(",") if x.strip()} if m else set()
    return out


def main():
    random.seed(42)
    print("Loading GT...", flush=True)
    gt_map = load_gt()

    print("Loading S1...", flush=True)
    s1 = pl.read_csv(os.path.join(DATA, "train_source1.tsv"), separator="\t")

    # stratified sample: per country, half singleton / half matched
    s1 = s1.with_columns(
        pl.col("entity_id").map_elements(lambda e: 1 if gt_map.get(e) else 0, return_dtype=pl.Int32).alias("has_match")
    )
    sample_ids = []
    for c in ["US", "INDIA"]:
        for hm in [0, 1]:
            pool = s1.filter((pl.col("country") == c) & (pl.col("has_match") == hm))
            k = min(N_SAMPLE // 4, len(pool))
            ids = pool.sample(n=k, seed=42)["entity_id"].to_list()
            sample_ids.extend(ids)
    print(f"Sampled {len(sample_ids)} S1 entities", flush=True)

    model_path = "cache/models/xgb_gpu_matcher.pkl"
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    booster = model.get_booster()
    threshold = saved.get("threshold", 0.80)
    print(f"Submitted model threshold: {threshold}", flush=True)

    for country in ["US", "INDIA"]:
        t0 = time.time()
        print(f"\n===== COUNTRY {country} =====", flush=True)
        s2 = pl.scan_csv(os.path.join(DATA, "train_source2.tsv"), separator="\t").filter(pl.col("country") == country).collect()
        s3 = pl.scan_csv(os.path.join(DATA, "train_source3.tsv"), separator="\t").filter(pl.col("country") == country).collect()
        targets = pl.concat([s2, s3])
        del s2, s3
        n_tgt = len(targets)
        print(f"Target pool: {n_tgt:,} records ({time.time()-t0:.0f}s)", flush=True)

        blocker = CompactCandidateBlocker(max_candidates_per_key=250, max_candidates_per_entity=max(CAPS))
        blocker.index_dataframe(targets)
        tgt_ids = targets["entity_id"].to_list()
        tgt_names = targets["business_name"].fill_null("").to_list()
        tgt_addrs = targets["business_address"].fill_null("").to_list()
        tgt_countries = targets["country"].fill_null("").to_list()
        del targets
        print(f"Indexed in {time.time()-t0:.0f}s", flush=True)

        s1c = s1.filter(pl.col("country") == country)
        s1c = s1c.filter(pl.col("entity_id").is_in(sample_ids))
        recs = s1c.to_dicts()

        recall_hits = {cap: 0 for cap in CAPS}
        total_true = 0
        n_cands_list = []
        probs_map = {}
        gt_sub = {}

        t1 = time.time()
        for r in recs:
            s1_rep = RecordRepresentation(r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or "")
            true_t = gt_map.get(r["entity_id"], set())
            gt_sub[r["entity_id"]] = true_t
            total_true += len(true_t)

            idxs = blocker.retrieve_candidate_indices(s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country)
            n_cands_list.append(len(idxs))
            got = {tgt_ids[ci] for ci in idxs}
            for cap in CAPS:
                recall_hits[cap] += len(true_t & got) if cap == max(CAPS) else 0
            # recall at intermediate caps: recompute from first cap entries
            for cap in CAPS[:-1]:
                recall_hits[cap] += len(true_t & {tgt_ids[ci] for ci in idxs[:cap]})

            feats = []
            cids = []
            for rank, ci in enumerate(idxs):
                cand_rep = RecordRepresentation(tgt_ids[ci], tgt_names[ci], tgt_addrs[ci], tgt_countries[ci])
                feats.append(compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0))
                cids.append(tgt_ids[ci])
            if feats:
                X = np.array(feats, dtype=np.float32)
                p = booster.predict(xgb.DMatrix(X))
                probs_map[r["entity_id"]] = [(cid, float(pr)) for cid, pr in zip(cids, p) if pr >= threshold - 0.15]
            else:
                probs_map[r["entity_id"]] = []

        print(f"Scored in {time.time()-t1:.0f}s", flush=True)

        print(f"\n--- {country} BLOCKING RECALL (full pool, n={len(recs)} entities, {total_true} true links) ---")
        for cap in CAPS:
            print(f"  cap={cap:3d}: recall={recall_hits[cap]/max(total_true,1):.4f}")
        import statistics
        print(f"  mean candidates/entity: {statistics.mean(n_cands_list):.1f}")
        print(f"  entities with 0 candidates: {sum(1 for x in n_cands_list if x == 0)}")

        # end-to-end with submitted model + greedy assignment
        from assign import global_bipartite_assignment
        preds = global_bipartite_assignment(probs_map, threshold=threshold)
        m = compute_macro_f05(gt_sub, preds)
        print(f"\n--- {country} END-TO-END (submitted model, thr={threshold}) ---")
        print(f"  macro_f05={m['macro_f05']:.4f} P={m['macro_precision']:.4f} R={m['macro_recall']:.4f} singleton_acc={m['singleton_accuracy']:.4f}")

        del blocker, tgt_ids, tgt_names, tgt_addrs, tgt_countries, probs_map, preds
        import gc
        gc.collect()


if __name__ == "__main__":
    main()
