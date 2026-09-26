"""
Phase 0 — Honest Validation Harness.

Measures, under REALISTIC test conditions (full per-country target pools,
no pre-seeded true matches):
  1. Streaming full-pool index build (memory-capped: chunked CSV reads,
     psutil watermark abort — never relies on swap).
  2. Blocking recall@K of the current CompactCandidateBlocker (K = 30/60/100).
  3. End-to-end macro-F0.5 of the submitted model/threshold on top-30 and
     top-100 candidate sets (when a model pickle is available).

Run:  python src/phase0_harness.py --countries US India
Results are printed and saved as JSON to --output-dir.
"""

import os
import gc
import json
import time
import pickle
import argparse
import psutil

import numpy as np
import polars as pl

from compact_blocking import CompactCandidateBlocker
from blocking_v2 import StreamingBlockerV2
from features import RecordRepresentation, compute_pair_features
from eval_metric import compute_macro_f05

WATERMARK_GB = 10.0  # hard abort threshold — leave headroom, never touch swap
CHUNK = 250_000

blocker = None              # set per-country in run_country
sampled_ids_by_country = {} # set in main
xgb = None                  # xgboost module, bound in main


def check_rss(label):
    gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"    [rss {gb:.2f} GB] {label}", flush=True)
    if gb > WATERMARK_GB:
        raise MemoryError(f"RSS {gb:.2f} GB exceeded {WATERMARK_GB} GB watermark — aborting before swap.")
    return gb


def sample_val_ids(data_dir, per_country, seed=42):
    """Stratified S1 sample per country: 5/6 matched + 1/6 singleton."""
    s1 = pl.scan_csv(os.path.join(data_dir, "train_source1.tsv"), separator="\t").select(
        ["entity_id", "country"]
    ).collect()
    gt = pl.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), separator="\t")
    gt_flag = gt.select(
        pl.col("source1_entity_id"),
        (pl.col("matched_entity_ids").fill_null("").str.strip_chars() != "").alias("has_match"),
    )
    df = s1.join(gt_flag, left_on="entity_id", right_on="source1_entity_id", how="inner")
    sampled = []
    for c in ["US", "India"]:
        n_matched = int(per_country * 5 / 6)
        n_single = per_country - n_matched
        for hm, n in ((True, n_matched), (False, n_single)):
            pool = df.filter((pl.col("country") == c) & (pl.col("has_match") == hm))
            k = min(n, len(pool))
            sampled.extend(pool.sample(n=k, seed=seed)["entity_id"].to_list())
    return set(sampled)


def build_gt_map(data_dir, s1_ids):
    """Ground truth for the sampled S1 ids only."""
    gt = pl.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), separator="\t")
    gt = gt.filter(pl.col("source1_entity_id").is_in(list(s1_ids)))
    gt_map = {}
    for row in gt.to_dicts():
        m = (row["matched_entity_ids"] or "").strip()
        gt_map[row["source1_entity_id"]] = {x.strip() for x in m.split(",") if x.strip()} if m else set()
    return gt_map


def stream_index_pool(path, country, tgt_ids, tgt_names, tgt_addrs, tgt_countries):
    """Chunked CSV streaming: index + append compact text arrays, never retain frames."""
    reader = pl.read_csv_batched(path, separator="\t", batch_size=CHUNK)
    index_fn = blocker.index_chunk if hasattr(blocker, "index_chunk") else blocker.index_dataframe
    n_indexed = 0
    while True:
        batches = reader.next_batches(2)
        if not batches:
            break
        for df in batches:
            df = df.filter(pl.col("country") == country)
            if df.is_empty():
                continue
            index_fn(df)
            tgt_ids.extend(df["entity_id"].to_list())
            tgt_names.extend(df["business_name"].fill_null("").to_list())
            tgt_addrs.extend(df["business_address"].fill_null("").to_list())
            tgt_countries.extend(df["country"].fill_null("").to_list())
            n_indexed += len(df)
            del df
        check_rss(f"indexed {n_indexed:,} rows from {os.path.basename(path)}")
    return n_indexed


def run_country(country, args, gt_map, model_art):
    global blocker
    print(f"\n=================== {country} ===================", flush=True)

    if not sampled_ids_by_country.get(country):
        print(f"  SKIPPED: no sampled S1 entities for {country!r} "
              f"(raw country labels in data: US / India / France)", flush=True)
        return {"country": country, "skipped": True}

    tgt_ids, tgt_names, tgt_addrs, tgt_countries = [], [], [], []
    if args.blocker == "v2":
        blocker = StreamingBlockerV2(
            overgenerate=args.overgenerate, topk=args.max_k
        )
    else:
        blocker = CompactCandidateBlocker(
            max_candidates_per_key=250, max_candidates_per_entity=args.max_k
        )

    t0 = time.time()
    n2 = stream_index_pool(os.path.join(args.data_dir, "train_source2.tsv"), country, tgt_ids, tgt_names, tgt_addrs, tgt_countries)
    n3 = stream_index_pool(os.path.join(args.data_dir, "train_source3.tsv"), country, tgt_ids, tgt_names, tgt_addrs, tgt_countries)
    if args.blocker == "v2":
        blocker.text_names = tgt_names
        blocker.text_addrs = tgt_addrs
    print(f"  Pool indexed: {n2 + n3:,} records in {time.time()-t0:.0f}s [{args.blocker}]", flush=True)
    check_rss("after pool index")

    s1 = pl.scan_csv(os.path.join(args.data_dir, "train_source1.tsv"), separator="\t").select(
        ["entity_id", "business_name", "business_address", "country"]
    ).filter(pl.col("country") == country).filter(
        pl.col("entity_id").is_in(sorted(sampled_ids_by_country[country]))
    ).collect()
    recs = s1.to_dicts()

    caps = (30, 60, 100)
    hits = {k: 0 for k in caps}
    overgen_hits = 0
    total_true = 0
    n_cands = []
    probs_map = {}
    gt_sub = {}
    t1 = time.time()
    for r in recs:
        s1_rep = RecordRepresentation(r["entity_id"], r["business_name"] or "", r["business_address"] or "", r["country"] or "")
        true_t = gt_map.get(r["entity_id"], set())
        gt_sub[r["entity_id"]] = true_t
        total_true += len(true_t)

        if args.blocker == "v2":
            idxs = blocker.retrieve(s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country, k=args.overgenerate)
        else:
            idxs = blocker.retrieve_candidate_indices(s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country)
        n_cands.append(len(idxs))
        for k in caps:
            hits[k] += len(true_t & {tgt_ids[ci] for ci in idxs[:k]})
        overgen_hits += len(true_t & {tgt_ids[ci] for ci in idxs})

        feats, cids = [], []
        for rank, ci in enumerate(idxs[:100]):
            cand_rep = RecordRepresentation(tgt_ids[ci], tgt_names[ci], tgt_addrs[ci], tgt_countries[ci])
            feats.append(compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0))
            cids.append(tgt_ids[ci])
        if feats and model_art is not None:
            X = np.array(feats, dtype=np.float32)
            dmat = xgb.DMatrix(X)
            p = model_art["booster"].predict(dmat)
            probs_map[r["entity_id"]] = [(cid, float(pr)) for cid, pr in zip(cids, p)]
        else:
            probs_map[r["entity_id"]] = []
    print(f"  Scored {len(recs):,} S1 entities in {time.time()-t1:.0f}s", flush=True)

    result = {
        "country": country,
        "blocker": args.blocker,
        "pool_size": n2 + n3,
        "n_entities": len(recs),
        "total_true_links": total_true,
        "blocking_recall": {str(k): round(hits[k] / max(total_true, 1), 4) for k in caps},
        "mean_cands_per_entity": round(float(np.mean(n_cands)), 3) if n_cands else 0.0,
    }
    if args.blocker == "v2":
        result["recall_at_overgen"] = round(overgen_hits / max(total_true, 1), 4)

    if model_art is not None and any(probs_map.values()):
        from assign import global_bipartite_assignment
        for cap in (30, 100):
            capped = {sid: lst[:cap] for sid, lst in probs_map.items()}
            preds = global_bipartite_assignment(capped, threshold=model_art["threshold"])
            m = compute_macro_f05(gt_sub, preds)
            result[f"end_to_end_cap{cap}"] = {
                "threshold": model_art["threshold"],
                "macro_f05": round(m["macro_f05"], 4),
                "precision": round(m["macro_precision"], 4),
                "recall": round(m["macro_recall"], 4),
                "singleton_acc": round(m["singleton_accuracy"], 4),
            }

    print(json.dumps(result, indent=2), flush=True)

    del blocker, tgt_ids, tgt_names, tgt_addrs, tgt_countries, probs_map, gt_sub
    gc.collect()
    return result


def main():
    global sampled_ids_by_country, xgb
    import xgboost as _xgb
    xgb = _xgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset/train")
    parser.add_argument("--model", default="cache/models/xgb_gpu_matcher.pkl")
    parser.add_argument("--countries", nargs="+", default=["US", "India"])
    parser.add_argument("--n-per-country", type=int, default=3000)
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--blocker", choices=["v1", "v2"], default="v2")
    parser.add_argument("--overgenerate", type=int, default=800)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Sampling validation S1 entities...", flush=True)
    sampled_ids = sample_val_ids(args.data_dir, args.n_per_country)
    print(f"Sampled {len(sampled_ids):,} S1 entities", flush=True)

    s1c = pl.scan_csv(os.path.join(args.data_dir, "train_source1.tsv"), separator="\t").select(
        ["entity_id", "country"]
    ).filter(pl.col("entity_id").is_in(sorted(sampled_ids))).collect()
    for row in s1c.to_dicts():
        sampled_ids_by_country.setdefault(row["country"], set()).add(row["entity_id"])
    print({c: len(v) for c, v in sampled_ids_by_country.items()}, flush=True)
    check_rss("after sampling")

    gt_map = build_gt_map(args.data_dir, sampled_ids)
    check_rss("after ground truth")

    model_art = None
    if os.path.exists(args.model):
        try:
            with open(args.model, "rb") as f:
                saved = pickle.load(f)
            booster = saved["model"].get_booster()
            model_art = {"booster": booster, "threshold": saved.get("threshold", 0.80)}
            print(f"Loaded model (threshold={model_art['threshold']})", flush=True)
        except Exception as e:
            print(f"WARNING: could not load model ({e}) — recall-only mode", flush=True)
    else:
        print(f"Model not found at {args.model} — recall-only mode", flush=True)

    all_results = {}
    for c in args.countries:
        all_results[c] = run_country(c, args, gt_map, model_art)

    out_path = os.path.join(args.output_dir, f"phase0_{args.blocker}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
