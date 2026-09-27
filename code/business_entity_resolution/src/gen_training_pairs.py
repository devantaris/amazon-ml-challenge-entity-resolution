"""
Phase 3, step 1 — generate training pairs at scale.

For a large stratified sample of S1 entities, retrieves top-K candidates from
the FULL per-country pool with Blocking v2 (same blocker as inference), merges
ground-truth targets that blocking missed (force-added, appended rank), and
writes 42-feature labelled pair shards (parquet) for XGBoost/LightGBM training.

Design notes:
  - Excludes the harness validation entities (same seed) -> no train/val mix.
  - Forced positives get ranks appended after the retrieved list (positions the
    model would rarely see at inference); this is an approximation, NOT a GT
    leak: the rank feature never encodes the label.
  - Sharded parquet output + resume: safe against Colab session death.
  - Memory-safe: streaming CSV, per-country blocker teardown, RSS watermark.
"""

import os
import gc
import json
import time
import shutil
import random
import argparse

import numpy as np
import polars as pl

from phase0_harness import sample_val_ids, check_rss, build_gt_map
from blocking_v2 import StreamingBlockerV2
from features import RecordRepresentation, compute_pair_features
from enhanced_features import CompactTFIDF
from fast_features import (
    compute_enhanced_features_fast,
    metaphone_m,
    soundex_m,
)

CHUNK = 250_000


def tfidf_from_df(counter, n_docs, max_features):
    """Build CompactTFIDF directly from a token->DF counter (no extra pass)."""
    tf = CompactTFIDF(max_features=max_features)
    top = sorted(counter.items(), key=lambda kv: -kv[1])[: max_features]
    tf.vocab = {term: i for i, (term, _) in enumerate(top)}
    tf.idf = np.zeros(len(tf.vocab), dtype=np.float32)
    for term, i in tf.vocab.items():
        tf.idf[i] = np.log((1 + n_docs) / (1 + counter[term])) + 1.0
    tf.fitted = True
    return tf


def stream_s1_records(path, ids_sorted, chunk=CHUNK):
    """Yield dicts for the given S1 ids in the given order."""
    id_set = set(ids_sorted)
    reader = pl.read_csv_batched(path, separator="\t", batch_size=chunk)
    by_id = {}
    while True:
        batches = reader.next_batches(4)
        if not batches:
            break
        for df in batches:
            df = df.filter(pl.col("entity_id").is_in(id_set))
            if df.is_empty():
                continue
            for r in df.to_dicts():
                by_id[r["entity_id"]] = r
            check_rss(f"s1 buffer {len(by_id):,}")
    for eid in ids_sorted:
        r = by_id.get(eid)
        if r is not None:
            yield r
    del by_id
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="student_resource/dataset/train")
    ap.add_argument("--out-dir", default="cache/pairs")
    ap.add_argument("--backup-dir", default="")  # e.g. /content/drive/MyDrive/ER_Challenge/pairs
    ap.add_argument("--n-entities", type=int, default=500_000)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--overgen", type=int, default=120)
    ap.add_argument("--tok-posting", type=int, default=1500,
                    help="posting cap for token/composite keys — training pairs "
                         "only need top-K candidates, so this can be far below "
                         "the inference-time value")
    ap.add_argument("--shard-entities", type=int, default=50_000)
    ap.add_argument("--max-feats", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-exclude-val", action="store_true", help="smoke-test only")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    # 1. validation exclusion (must match phase0_harness sampling exactly)
    val_ids = set()
    if not args.no_exclude_val:
        print("Excluding harness validation entities...", flush=True)
        val_ids = sample_val_ids(args.data_dir, 3000)
        print(f"Excluding {len(val_ids):,} val entities", flush=True)

    # 2. sample train S1 (natural country + singleton proportions)
    s1_all = pl.read_csv(
        os.path.join(args.data_dir, "train_source1.tsv"), separator="\t"
    ).select(["entity_id", "country"])
    pool = s1_all.filter(~pl.col("entity_id").is_in(list(val_ids)))
    n = min(args.n_entities, len(pool))
    sampled = pool.sample(n=n, seed=args.seed).sort("entity_id")
    print(f"Sampled {n:,} train entities", flush=True)
    del s1_all, pool
    gc.collect()

    gt_map = build_gt_map(args.data_dir, set(sampled["entity_id"].to_list()))
    print(f"GT map: {len(gt_map):,} ({sum(1 for v in gt_map.values() if v):,} matched)", flush=True)
    check_rss("after GT")

    meta = {
        "args": vars(args),
        "entities": len(sampled),
        "started": time.strftime("%Y-%m-%d %H:%M"),
        "shards": 0,
        "pairs": 0,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for country in ["US", "India"]:
        ids_c = sampled.filter(pl.col("country") == country)["entity_id"].to_list()
        if not ids_c:
            continue
        print(f"\n=============== {country}: {len(ids_c):,} entities ===============", flush=True)

        # 3. index full pool for this country
        blocker = StreamingBlockerV2(tok_posting=args.tok_posting, overgenerate=args.overgen, topk=args.k)
        tgt_ids, tgt_names, tgt_addrs, tgt_countries = [], [], [], []
        t0 = time.time()
        for path in ("train_source2.tsv", "train_source3.tsv"):
            reader = pl.read_csv_batched(
                os.path.join(args.data_dir, path), separator="\t", batch_size=CHUNK
            )
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
                check_rss(f"indexing {os.path.basename(path)}")
        blocker.text_names = tgt_names
        blocker.text_addrs = tgt_addrs
        row_of = {tid: i for i, tid in enumerate(tgt_ids)}
        print(f"Pool indexed: {len(tgt_ids):,} in {time.time()-t0:.0f}s", flush=True)

        # 4. TF-IDF from blocker DF counters (full-pool IDF, no extra pass)
        name_tfidf = tfidf_from_df(blocker.name_df, blocker.total_indexed, args.max_feats)
        addr_tfidf = tfidf_from_df(blocker.addr_df, blocker.total_indexed, args.max_feats)
        print(f"TF-IDF vocabs: name={len(name_tfidf.vocab):,} addr={len(addr_tfidf.vocab):,}", flush=True)
        check_rss("after tfidf")

        # 5. resume support
        done_file = os.path.join(args.out_dir, f"done_{country}.txt")
        done = 0
        if os.path.exists(done_file):
            done = int(open(done_file).read().strip() or 0)
            print(f"Resuming after {done:,} entities", flush=True)
        ids_c = ids_c[done:]
        shard_idx = done // args.shard_entities

        # 6. process in shards
        shard_feats, shard_ids, shard_cids, shard_labels = [], [], [], []
        t1 = time.time()
        n_pos = n_pairs = 0

        def flush_shard():
            nonlocal shard_idx, n_pos, n_pairs
            if not shard_feats:
                return
            X = np.array(shard_feats, dtype=np.float32)
            df = pl.DataFrame({
                **{f"f{i}": X[:, i] for i in range(X.shape[1])},
                "s1_id": shard_ids,
                "cand_id": shard_cids,
                "label": shard_labels,
            })
            out = os.path.join(args.out_dir, f"pairs_{country}_{shard_idx:03d}.parquet")
            df.write_parquet(out, compression="zstd")
            if args.backup_dir:
                os.makedirs(args.backup_dir, exist_ok=True)
                shutil.copy(out, args.backup_dir)
            print(f"  shard {shard_idx:03d}: {X.shape[0]:,} pairs -> {out} "
                  f"({time.time()-t1:.0f}s elapsed)", flush=True)
            shard_idx += 1
            shard_feats.clear(); shard_ids.clear(); shard_cids.clear(); shard_labels.clear()
            check_rss("after shard flush")

        s1_by_id = {}
        id_set = set(ids_c)
        reader = pl.read_csv_batched(
            os.path.join(args.data_dir, "train_source1.tsv"), separator="\t", batch_size=CHUNK
        )
        processed = 0
        while True:
            batches = reader.next_batches(4)
            if not batches:
                break
            for df in batches:
                df = df.filter(pl.col("entity_id").is_in(id_set))
                if df.is_empty():
                    continue
                for r in df.to_dicts():
                    s1_by_id[r["entity_id"]] = r
                del df
                if len(s1_by_id) % 200_000 == 0 and len(s1_by_id) > 0:
                    check_rss(f"s1 buffer {len(s1_by_id):,}")

        for eid in ids_c:
            r = s1_by_id.get(eid)
            if r is None:
                continue
            s1_rep = RecordRepresentation(eid, r["business_name"] or "", r["business_address"] or "", r["country"] or "")
            true_t = gt_map.get(eid, set())
            s1_meta = metaphone_m(s1_rep.name_clean)
            s1_sdx = soundex_m(s1_rep.name_clean)

            rows = blocker.retrieve(s1_rep.name_clean, s1_rep.addr_clean, s1_rep.country, k=args.k)
            got = {tgt_ids[i] for i in rows}
            forced = [row_of[t] for t in true_t if t in row_of and t not in got]

            all_rows = list(rows) + forced  # forced appended after retrieved
            for rank, ci in enumerate(all_rows):
                cand_rep = RecordRepresentation(
                    tgt_ids[ci], tgt_names[ci], tgt_addrs[ci], tgt_countries[ci]
                )
                base = compute_pair_features(s1_rep, cand_rep, cand_rank=rank, blocking_score=1.0)
                feats = compute_enhanced_features_fast(
                    base, s1_rep.name_tokens, s1_rep.addr_tokens,
                    cand_rep.name_tokens, cand_rep.addr_tokens,
                    s1_rep.name_clean, cand_rep.name_clean,
                    name_tfidf, addr_tfidf, s1_meta, s1_sdx,
                )
                label = 1 if tgt_ids[ci] in true_t else 0
                shard_feats.append(feats)
                shard_ids.append(eid)
                shard_cids.append(tgt_ids[ci])
                shard_labels.append(label)
                n_pairs += 1
                n_pos += label

            processed += 1
            if processed % args.shard_entities == 0:
                flush_shard()
                done_now = done + processed
                open(done_file, "w").write(str(done_now))
            if processed % 1_000 == 0:
                el = time.time() - t1
                rate = processed / max(el, 1e-9)
                eta_min = (len(ids_c) - processed) / max(rate, 1e-9) / 60
                check_rss(f"{processed:,}/{len(ids_c):,} entities ({rate:.0f}/s, "
                          f"ETA {eta_min:.0f} min), {n_pairs:,} pairs ({n_pos:,} pos)")

        flush_shard()
        open(done_file, "w").write(str(done + processed))
        print(f"{country} DONE: {processed:,} entities, {n_pairs:,} pairs ({n_pos:,} pos, "
              f"{n_pos/max(n_pairs,1):.2%}), {time.time()-t1:.0f}s", flush=True)

        del blocker, tgt_ids, tgt_names, tgt_addrs, tgt_countries, row_of, s1_by_id
        gc.collect()
        check_rss("after country teardown")

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M")
    meta["elapsed_min"] = round((time.time() - t_start) / 60, 1)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nALL DONE in {meta['elapsed_min']} min", flush=True)


if __name__ == "__main__":
    main()
