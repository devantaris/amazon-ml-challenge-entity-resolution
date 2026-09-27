"""
Fast full-test inference — cached per-row preprocessing + multiprocessing.

Designed for deadline runs: every S1 test entity gets a row in BOTH output
files no matter what (entities not yet scored when time runs out are written
with empty match lists — a valid submission).

Usage:
  python src/fast_inference.py \
      --test-dir <...>/dataset/test \
      --models-dir cache/models \
      --out-dir /kaggle/working/output_fast \
      --n-workers 4
"""

import os
import sys
import gc
import time
import json
import pickle
import argparse
import collections
import multiprocessing as mp

import numpy as np
import polars as pl

from normalize import normalize_country
from blocking_v2 import StreamingBlockerV2
from features import RecordRepresentation, compute_pair_features
from fast_features import compute_enhanced_features_fast, metaphone_m, soundex_m
from assign import global_bipartite_assignment

WATERMARK_GB = 22.0
CHUNK = 4000  # entities per worker task


def check_rss(label):
    try:
        import psutil
        gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        print(f"    [rss {gb:.2f} GB] {label}", flush=True)
        assert gb < WATERMARK_GB, "watermark exceeded"
    except AssertionError:
        raise
    except Exception:
        pass


# ── worker-side globals (inherited via fork) ─────────────────────────────────
G = {}


def _row_cache(ci):
    """RecordRepresentation + precomputed phonetics for a target row, cached."""
    c = G["row_cache"]
    ent = c.get(ci)
    if ent is None:
        rep = RecordRepresentation(G["tgt_ids"][ci], G["tgt_names"][ci],
                                   G["tgt_addrs"][ci], G["tgt_countries"][ci])
        ent = (rep, metaphone_m(rep.name_clean), soundex_m(rep.name_clean))
        if len(c) > 3_500_000:
            c.clear()
        c[ci] = ent
    return ent


def process_chunk(chunk):
    """chunk: list of (s1_id, name, addr, country). Returns (cand_rows, scored)."""
    blocker = G["blocker"]
    tgt_ids = G["tgt_ids"]
    K = G["k"]
    cand_rows = []   # (s1_id, "id,id,id")
    scored = []      # (s1_id, [(cand_id, prob), ...])

    buf = []          # feature rows
    pending = []      # (s1_id, cand_ids) aligned with buf rows
    nt = G["name_tfidf"].get(G["cur_country"])
    at = G["addr_tfidf"].get(G["cur_country"])

    def flush():
        if not buf:
            return
        import xgboost as xgb
        X = np.array(buf, dtype=np.float32)
        p1 = G["xgb"].predict(xgb.DMatrix(X))
        p2 = G["lgb"].predict_proba(X)[:, 1]
        p3 = G["meta"].predict_proba(np.column_stack([p1, p2]))[:, 1]
        i = 0
        for s1_id, cids in pending:
            scored.append((s1_id, list(zip(cids, p3[i:i + len(cids)].tolist()))))
            i += len(cids)
        buf.clear()
        pending.clear()

    for s1_id, name, addr, country in chunk:
        s1_rep = RecordRepresentation(s1_id, name, addr, country)
        s1_meta = metaphone_m(s1_rep.name_clean)
        s1_sdx = soundex_m(s1_rep.name_clean)
        idxs = blocker.retrieve(s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country, k=K)
        cids = [tgt_ids[i] for i in idxs]
        cand_rows.append((s1_id, ",".join(cids)))
        pending.append((s1_id, cids))
        for rank, ci in enumerate(idxs):
            cand_rep, c_meta, c_sdx = _row_cache(ci)
            base = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
            buf.append(compute_enhanced_features_fast(
                base, s1_rep.name_tokens, s1_rep.addr_tokens,
                cand_rep.name_tokens, cand_rep.addr_tokens,
                s1_rep.name_clean, cand_rep.name_clean, nt, at, s1_meta, s1_sdx))
        if len(buf) >= 6000:
            flush()

    flush()
    return cand_rows, scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--overgen", type=int, default=150)
    ap.add_argument("--tok-posting", type=int, default=3000)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--fallback-threshold", type=float, default=0.75)
    ap.add_argument("--countries", nargs="+", default=["US", "India", "France"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    use_fork = True
    try:
        mp.set_start_method("fork", force=True)
    except (ValueError, OSError):
        use_fork = False
        print("fork unavailable — single-process fallback", flush=True)

    print("Loading models...", flush=True)
    with open(os.path.join(args.models_dir, "ensemble_v5.pkl"), "rb") as f:
        ens = pickle.load(f)
    with open(os.path.join(args.models_dir, "tfidf_models_v5.pkl"), "rb") as f:
        tf = pickle.load(f)
    G["xgb"] = ens["xgb_model"].get_booster()
    G["lgb"] = ens["lgb_model"]
    G["meta"] = ens["meta_model"]
    G["name_tfidf"] = tf["name_tfidfs"]
    G["addr_tfidf"] = tf["addr_tfidfs"]
    thresholds = ens.get("country_thresholds", {})
    print(f"country thresholds: {thresholds} | fallback {args.fallback_threshold}", flush=True)

    s1_all = pl.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), separator="\t")
    all_ids = set(s1_all["entity_id"].to_list())
    print(f"Test S1 entities: {len(all_ids):,}", flush=True)

    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")
    cand_f = open(cand_path, "w", encoding="utf-8", buffering=1 << 20)
    match_f = open(match_path, "w", encoding="utf-8", buffering=1 << 20)
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    match_f.write("source1_entity_id\tmatched_entity_ids\n")

    written = set()

    for country in args.countries:
        s1c = s1_all.filter(pl.col("country") == country).sort("entity_id")
        n_s1 = len(s1c)
        print(f"\n=========== {country}: {n_s1:,} entities ===========", flush=True)
        if n_s1 == 0:
            continue
        G["cur_country"] = normalize_country(country)

        t0 = time.time()
        blocker = StreamingBlockerV2(tok_posting=args.tok_posting,
                                     overgenerate=args.overgen, topk=args.k)
        tgt_ids, tgt_names, tgt_addrs, tgt_countries = [], [], [], []
        for path in ("test_source2.tsv", "test_source3.tsv"):
            reader = pl.read_csv_batched(os.path.join(args.test_dir, path),
                                         separator="\t", batch_size=250_000)
            while True:
                batches = reader.next_batches(2)
                if not batches:
                    break
                for df in batches:
                    df = df.filter(pl.col("country") == country)
                    if df.is_empty():
                        continue
                    blocker.index_chunk(df)
                    tgt_ids.extend(df["entity_id"].to_list())
                    tgt_names.extend(df["business_name"].fill_null("").to_list())
                    tgt_addrs.extend(df["business_address"].fill_null("").to_list())
                    tgt_countries.extend(df["country"].fill_null("").to_list())
                    del df
                check_rss(f"indexing {path}")
        blocker.text_names = tgt_names
        blocker.text_addrs = tgt_addrs
        G["blocker"] = blocker
        G["tgt_ids"] = tgt_ids
        G["tgt_names"] = tgt_names
        G["tgt_addrs"] = tgt_addrs
        G["tgt_countries"] = tgt_countries
        G["k"] = args.k
        G["row_cache"] = {}
        print(f"Pool indexed: {len(tgt_ids):,} in {time.time()-t0:.0f}s", flush=True)
        check_rss("after index")

        # pool is created AFTER indexing so forked workers inherit the blocker
        pool_ctx = None
        if use_fork and args.n_workers > 1:
            pool_ctx = mp.Pool(args.n_workers)
            print(f"multiprocessing: fork x {args.n_workers}", flush=True)

        thr = thresholds.get(G["cur_country"], args.fallback_threshold)
        print(f"threshold for {country}: {thr}", flush=True)

        names = s1c["business_name"].fill_null("").to_list()
        addrs = s1c["business_address"].fill_null("").to_list()
        ids_c = s1c["entity_id"].to_list()
        countries_c = s1c["country"].to_list()
        chunks = [[(ids_c[i], names[i], addrs[i], countries_c[i])
                   for i in range(a, min(a + CHUNK, n_s1))]
                  for a in range(0, n_s1, CHUNK)]

        probs_map = {}
        t1 = time.time()
        done = 0
        if pool_ctx is not None:
            results_iter = pool_ctx.imap_unordered(process_chunk, chunks)
        else:
            results_iter = map(process_chunk, chunks)
        for cand_rows, scored in results_iter:
            for s1_id, cand_str in cand_rows:
                cand_f.write(f"{s1_id}\t{cand_str}\n")
                written.add(s1_id)
            for s1_id, pairs in scored:
                probs_map[s1_id] = pairs
            done += len(cand_rows)
            el = time.time() - t1
            rate = done / max(el, 1e-9)
            print(f"    [rss n/a] {done:,}/{n_s1:,} entities ({rate:.0f}/s, "
                  f"ETA {(n_s1 - done) / max(rate, 1e-9) / 60:.0f} min)", flush=True)
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()
        check_rss("after scoring")
        check_rss("after scoring")

        preds = global_bipartite_assignment(probs_map, threshold=thr)
        for s1_id in ids_c:
            m = preds.get(s1_id, set())
            match_f.write(f"{s1_id}\t{','.join(sorted(m))}\n")
        cand_f.flush(); match_f.flush()
        n_matched = sum(1 for v in preds.values() if v)
        print(f"{country} DONE: {n_matched:,} matched entities, "
              f"{time.time()-t1:.0f}s", flush=True)

        G["blocker"] = None
        G["row_cache"] = {}
        gc.collect()

    # safety: any entity not yet written (timeout mid-country) gets an empty row
    missing = all_ids - written
    for s1_id in sorted(missing):
        cand_f.write(f"{s1_id}\t\n")
        match_f.write(f"{s1_id}\t\n")
    if missing:
        print(f"WARNING: {len(missing):,} entities had no predictions (empty rows written)", flush=True)

    cand_f.close(); match_f.close()
    print(f"\nOutputs: {cand_path}\n         {match_path}", flush=True)


if __name__ == "__main__":
    main()
