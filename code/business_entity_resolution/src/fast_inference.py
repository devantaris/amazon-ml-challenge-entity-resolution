"""
Fast full-test inference — spawn-mode multiprocessing (deadlock-free).

Each worker builds its OWN blocker index and model stack via an initializer
(no fork state inheritance, no deadlocks), then scores chunks of entities.

Safety: every S1 entity gets a row in both output files no matter what —
entities not yet scored when time runs out are written with empty lists.

Usage:
  python src/fast_inference.py --test-dir <...>/test --models-dir cache/models \
      --out-dir /kaggle/working/output_fast --n-workers 3
"""

import os
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

CHUNK = 4000
_W = {}  # per-worker state, built by _init_worker


def _init_worker(test_dir, country, models_dir, map_dir, k, overgen, tok_posting):
    """Runs once per spawned worker: build blocker + load models."""
    os.chdir(map_dir)  # blocking_v2 loads the translit map from ./cache
    t0 = time.time()
    blocker = StreamingBlockerV2(tok_posting=tok_posting, overgenerate=overgen, topk=k)
    tgt_ids, tgt_names, tgt_addrs, tgt_countries = [], [], [], []
    for path in ("test_source2.tsv", "test_source3.tsv"):
        reader = pl.read_csv_batched(os.path.join(test_dir, path), separator="\t", batch_size=250_000)
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
    blocker.text_names = tgt_names
    blocker.text_addrs = tgt_addrs
    with open(os.path.join(models_dir, "ensemble_v5.pkl"), "rb") as f:
        ens = pickle.load(f)
    with open(os.path.join(models_dir, "tfidf_models_v5.pkl"), "rb") as f:
        tf = pickle.load(f)
    _W.update(
        blocker=blocker, tgt_ids=tgt_ids, tgt_names=tgt_names, tgt_addrs=tgt_addrs,
        tgt_countries=tgt_countries, k=k, row_cache={},
        cur_country=normalize_country(country),
        xgb=ens["xgb_model"].get_booster(), lgb=ens["lgb_model"], meta=ens["meta_model"],
        name_tfidf=tf["name_tfidfs"], addr_tfidf=tf["addr_tfidfs"],
    )
    print(f"  [worker {os.getpid()}] {country}: indexed {len(tgt_ids):,} in {time.time()-t0:.0f}s",
          flush=True)


def _row_cache(ci):
    c = _W["row_cache"]
    ent = c.get(ci)
    if ent is None:
        rep = RecordRepresentation(_W["tgt_ids"][ci], _W["tgt_names"][ci],
                                   _W["tgt_addrs"][ci], _W["tgt_countries"][ci])
        ent = (rep, metaphone_m(rep.name_clean), soundex_m(rep.name_clean))
        if len(c) > 3_000_000:
            c.clear()
        c[ci] = ent
    return ent


def process_chunk(chunk):
    import xgboost as xgb
    blocker = _W["blocker"]
    tgt_ids = _W["tgt_ids"]
    K = _W["k"]
    nt, at = _W["name_tfidf"].get(_W["cur_country"]), _W["addr_tfidf"].get(_W["cur_country"])

    cand_rows, scored = [], []
    buf, pending = [], []

    def flush():
        if not buf:
            return
        X = np.array(buf, dtype=np.float32)
        p1 = _W["xgb"].predict(xgb.DMatrix(X))
        p2 = _W["lgb"].predict_proba(X)[:, 1]
        p3 = _W["meta"].predict_proba(np.column_stack([p1, p2]))[:, 1]
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
    ap.add_argument("--map-dir", required=True,
                    help="dir whose ./cache holds translit_map.json (usually the code root)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--overgen", type=int, default=150)
    ap.add_argument("--tok-posting", type=int, default=3000)
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--fallback-threshold", type=float, default=0.75)
    ap.add_argument("--countries", nargs="+", default=["US", "India", "France"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ctx = mp.get_context("spawn")

    print("Loading models + mining check...", flush=True)
    with open(os.path.join(args.models_dir, "ensemble_v5.pkl"), "rb") as f:
        ens = pickle.load(f)
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

        thr = thresholds.get(normalize_country(country), args.fallback_threshold)
        print(f"threshold for {country}: {thr}", flush=True)

        names = s1c["business_name"].fill_null("").to_list()
        addrs = s1c["business_address"].fill_null("").to_list()
        ids_c = s1c["entity_id"].to_list()
        countries_c = s1c["country"].to_list()
        chunks = [[(ids_c[i], names[i], addrs[i], countries_c[i])
                   for i in range(a, min(a + CHUNK, n_s1))]
                  for a in range(0, n_s1, CHUNK)]

        t1 = time.time()
        done = 0
        probs_map = {}
        print(f"Spawning {args.n_workers} workers (each indexes the {country} pool, ~5-8 min)...", flush=True)
        with ctx.Pool(args.n_workers, initializer=_init_worker,
                      initargs=(args.test_dir, country, args.models_dir,
                                args.map_dir, args.k, args.overgen, args.tok_posting)) as pool:
            for cand_rows, scored in pool.imap_unordered(process_chunk, chunks):
                for s1_id, cand_str in cand_rows:
                    cand_f.write(f"{s1_id}\t{cand_str}\n")
                    written.add(s1_id)
                for s1_id, pairs in scored:
                    probs_map[s1_id] = pairs
                done += len(cand_rows)
                el = time.time() - t1
                rate = done / max(el, 1e-9)
                print(f"    {done:,}/{n_s1:,} entities ({rate:.0f}/s, "
                      f"ETA {(n_s1 - done) / max(rate, 1e-9) / 60:.0f} min)", flush=True)

        preds = global_bipartite_assignment(probs_map, threshold=thr)
        for s1_id in ids_c:
            m = preds.get(s1_id, set())
            match_f.write(f"{s1_id}\t{','.join(sorted(m))}\n")
        cand_f.flush(); match_f.flush()
        n_matched = sum(1 for v in preds.values() if v)
        print(f"{country} DONE: {n_matched:,} matched entities, {time.time()-t1:.0f}s", flush=True)
        gc.collect()

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
