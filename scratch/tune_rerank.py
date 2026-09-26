"""Grid-search re-rank blend weights on in-pool mini links (fast, local)."""

import sys
import itertools
import polars as pl

sys.path.insert(0, "code/business_entity_resolution/src")
from blocking_v2 import StreamingBlockerV2

D = "scratch/mini_train2/"
OVERGEN = 800

s1 = pl.read_csv(D + "train_source1.tsv", separator="\t")
targets = pl.read_csv(D + "targets.tsv", separator="\t")
gt = pl.read_csv(D + "train_ground_truth.tsv", separator="\t")
tgt_ids = targets["entity_id"].to_list()
pool_ids = set(tgt_ids)

inpool = {}
for row in gt.to_dicts():
    m = (row["matched_entity_ids"] or "").strip()
    if not m:
        continue
    ts = {x.strip() for x in m.split(",") if x.strip() in pool_ids}
    if ts:
        inpool[row["source1_entity_id"]] = ts
total_links = sum(len(v) for v in inpool.values())
s1map = {r["entity_id"]: r for r in s1.to_dicts()}
print(f"in-pool links: {total_links}")

b = StreamingBlockerV2(topk=100, overgenerate=OVERGEN)
b.index_chunk(targets)
b.text_names = targets["business_name"].fill_null("").to_list()
b.text_addrs = targets["business_address"].fill_null("").to_list()

# collect components + labels per query
samples = []  # (true_set, rows, comps)
for s1id, trues in inpool.items():
    r = s1map[s1id]
    rows, comps = b.retrieve(
        r["business_name"], r["business_address"], r["country"],
        k=OVERGEN, return_components=True,
    )
    samples.append((trues, rows, comps))
print("components collected")

def recall(ws):
    w_cos, w_fn, w_fa, w_k = ws
    hits30 = hits100 = 0
    for trues, rows, comps in samples:
        scored = sorted(
            range(len(rows)),
            key=lambda i: -(w_cos * min(comps[i]["cos"] * 1.4, 1.0)
                            + w_fn * comps[i]["fz_n"]
                            + w_fa * comps[i]["fz_a"]
                            + w_k * comps[i]["kr"]),
        )
        top30 = {rows[i] for i in scored[:30]}
        top100 = {rows[i] for i in scored[:100]}
        got30 = {tgt_ids[i] for i in top30}
        got100 = {tgt_ids[i] for i in top100}
        hits30 += len(trues & got30)
        hits100 += len(trues & got100)
    return hits30 / total_links, hits100 / total_links

best = None
grid = []
for w_cos in [0.0, 0.2, 0.4, 0.6]:
    for w_fn in [0.0, 0.2, 0.4, 0.6]:
        for w_fa in [0.0, 0.1, 0.2, 0.3]:
            for w_k in [0.0, 0.1, 0.2, 0.4]:
                if abs(w_cos + w_fn + w_fa + w_k - 1.0) > 0.01:
                    continue
                grid.append((w_cos, w_fn, w_fa, w_k))

results = []
for ws in grid:
    r30, r100 = recall(ws)
    results.append((r30, r100, ws))
    if best is None or (r100, r30) > (best[1], best[0]):
        best = (r30, r100, ws)

results.sort(key=lambda x: -x[1])
print("\ntop 8 blends by recall@100 (r30, r100, weights cos/fz_n/fz_a/key):")
for r30, r100, ws in results[:8]:
    print(f"  @30={r30:.4f} @100={r100:.4f}  w={ws}")
print(f"\nBEST: weights={best[2]} recall@30={best[0]:.4f} recall@100={best[1]:.4f}")
