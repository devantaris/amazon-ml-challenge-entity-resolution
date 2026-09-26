"""
Mine a transliteration/typo variant dictionary from training ground truth.

For matched (S1, target) name pairs, tokens that co-occur across the two sides
of a true match with small edit distance are variant spellings of the same
word (chaudhary/choudhary, ltd/limitted, ...). We count such co-occurrences,
pick the higher-dictionary-frequency form as canonical, and emit a
country-specific map variant -> canonical.

Output: cache/translit_map.json  {"India": {...}, "US": {...}}
Used by blocking_v2 (key collapsing) — no external data involved.
"""

import os
import gc
import json
import random
import argparse
import collections

import polars as pl
from rapidfuzz.distance import Levenshtein

from normalize import normalize_business_name, normalize_country
from blocking import LEGAL_SET
from enhanced_features import soundex

from phase0_harness import check_rss

STOP = {"and", "the", "of", "in", "for", "to", "co", "corp", "inc", "ltd",
        "pvt", "llc", "llp", "grp", "svc", "soln", "ent", "ind", "tech"}


def tokens_of(name: str):
    _, toks, _, _ = normalize_business_name(name or "")
    return tuple(t for t in toks if len(t) >= 3 and t not in LEGAL_SET and t not in STOP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="student_resource/dataset/train")
    ap.add_argument("--n-entities", type=int, default=400_000)
    ap.add_argument("--min-pair-count", type=int, default=15)
    ap.add_argument("--max-map", type=int, default=50_000)
    ap.add_argument("--out", default="cache/translit_map.json")
    args = ap.parse_args()

    random.seed(42)
    print("Loading GT and sampling matched entities...", flush=True)
    gt = pl.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), separator="\t")
    gt = gt.filter(pl.col("matched_entity_ids").fill_null("").str.strip_chars() != "")
    sample = gt.sample(n=min(args.n_entities, len(gt)), seed=42)
    s1_ids = set(sample["source1_entity_id"].to_list())
    links = []          # (s1_id, tgt_id)
    needed = set()
    for row in sample.to_dicts():
        for t in row["matched_entity_ids"].split(","):
            t = t.strip()
            if t:
                links.append((row["source1_entity_id"], t))
                needed.add(t)
    print(f"Sampled {len(s1_ids):,} entities, {len(links):,} links, {len(needed):,} targets", flush=True)
    check_rss("after GT")

    name_toks = {}      # entity_id -> (tokens, country)
    for path, ids in (("train_source1.tsv", s1_ids), ("train_source2.tsv", needed), ("train_source3.tsv", needed)):
        df = pl.scan_csv(os.path.join(args.data_dir, path), separator="\t").filter(
            pl.col("entity_id").is_in(list(ids))).collect()
        for r in df.to_dicts():
            name_toks[r["entity_id"]] = (tokens_of(r["business_name"]), normalize_country(r["country"] or ""))
        del df
        gc.collect()
        check_rss(f"after {path}")

    pair_count = collections.Counter()      # (country, sorted-pair) -> n
    tok_df = [collections.Counter(), collections.Counter()]  # per country idx
    cidx = {"US": 0, "INDIA": 1}
    n_links = 0
    for s1_id, t_id in links:
        a = name_toks.get(s1_id)
        b = name_toks.get(t_id)
        if not a or not b:
            continue
        n_links += 1
        c = a[1] if a[1] in cidx else "INDIA"
        ci = cidx[c]
        for t in set(a[0]):
            tok_df[ci][t] += 1
        for t in set(b[0]):
            tok_df[ci][t] += 1
        sa, sb = set(a[0]), set(b[0])
        a_only, b_only = sa - sb, sb - sa
        if not a_only or not b_only:
            continue
        for x in a_only:
            for y in b_only:
                if abs(len(x) - len(y)) > 2:
                    continue
                if soundex(x) != soundex(y):
                    continue
                if Levenshtein.distance(x, y) <= 2:
                    pair_count[(c, x, y) if x < y else (c, y, x)] += 1
        if n_links % 200_000 == 0:
            check_rss(f"{n_links:,} links, {len(pair_count):,} distinct pairs")

    # build maps: canonical = higher DF form
    maps = {"US": {}, "India": {}}
    for (c, x, y), n in pair_count.items():
        if n < args.min_pair_count:
            continue
        ci = cidx[c]
        dx, dy = tok_df[ci].get(x, 0), tok_df[ci].get(y, 0)
        variant, canonical = (y, x) if dx >= dy else (x, y)
        if len(variant) < 4 or len(canonical) < 4:
            continue
        raw_c = "US" if c == "US" else "India"
        prev = maps[raw_c].get(variant)
        if prev is None or n > prev[0]:
            maps[raw_c][variant] = (n, canonical)

    out = {}
    for c in maps:
        ranked = sorted(maps[c].items(), key=lambda kv: -kv[1][0])[: args.max_map]
        out[c] = {v: can for v, (n, can) in ranked}
        print(f"{c}: {len(out[c]):,} variant rules", flush=True)
        for v, can in list(ranked)[:12]:
            print(f"   {v} -> {can}  (n={maps[c][v][0]})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
