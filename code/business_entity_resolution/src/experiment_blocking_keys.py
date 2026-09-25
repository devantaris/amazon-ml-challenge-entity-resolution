"""
Blocking Key Recall Diagnostics.

Tests individual and combined blocking strategies on a sample of ground-truth matches
to measure recall and candidate set sizes before running at full scale.
"""

import collections
import re
import polars as pl
from normalize import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    COMMON_NAME_STOPWORDS,
)

def run_experiment(sample_size: int = 5000):
    print(f"Loading {sample_size} validation entities with ground truth...")
    
    # Read val ids
    with open("cache/splits/val_s1_ids.txt", "r") as f:
        val_ids = set([line.strip() for line in f if line.strip()][:sample_size])

    # Load S1
    s1_df = pl.scan_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
    s1_rows = s1_df.filter(pl.col("entity_id").is_in(val_ids)).collect()
    
    # Load GT
    gt_df = pl.scan_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")
    gt_rows = gt_df.filter(pl.col("source1_entity_id").is_in(val_ids)).collect()
    
    # Map S1 to true targets
    gt_map = {}
    all_target_ids = set()
    for row in gt_rows.to_dicts():
        s1_id = row["source1_entity_id"]
        m_str = (row.get("matched_entity_ids") or "").strip()
        if m_str:
            targets = set([x.strip() for x in m_str.split(",") if x.strip()])
            gt_map[s1_id] = targets
            all_target_ids.update(targets)
        else:
            gt_map[s1_id] = set()

    total_true_matches = sum(len(v) for v in gt_map.values())
    print(f"Sampled S1 entities: {len(s1_rows):,}")
    print(f"Singletons: {sum(1 for v in gt_map.values() if not v):,}")
    print(f"Entities with matches: {sum(1 for v in gt_map.values() if v):,}")
    print(f"Total true match pairs: {total_true_matches:,}")

    # Load true targets from S2 and S3 to see which keys match
    s2_targets = {x for x in all_target_ids if x.startswith("S2-")}
    s3_targets = {x for x in all_target_ids if x.startswith("S3-")}

    print(f"Retrieving {len(s2_targets):,} S2 records and {len(s3_targets):,} S3 records...")
    s2_rows = pl.scan_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(s2_targets)).collect()
    s3_rows = pl.scan_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(s3_targets)).collect()

    target_records = {}
    for r in s2_rows.to_dicts():
        target_records[r["entity_id"]] = r
    for r in s3_rows.to_dicts():
        target_records[r["entity_id"]] = r

    # Extended address stopwords
    ADDR_STOPWORDS = {
        "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
        "court", "ct", "boulevard", "blvd", "floor", "fl", "unit", "apt", "suite", "ste",
        "building", "bldg", "north", "south", "east", "west", "near", "opp", "opposite",
        "behind", "door", "shop", "plot", "sector", "sec", "block", "blk", "village",
        "city", "town", "state", "india", "usa", "us", "haryana", "maharashtra", "karnataka",
        "california", "texas", "delhi", "bengal", "york", "tamil", "nadu", "kerala"
    }

    LEGAL_SET = {
        "corp", "corporation", "inc", "incorporated", "ltd", "limited", "pvt", "private",
        "co", "company", "ent", "enterprises", "llc", "llp", "tech", "technologies",
        "svc", "services", "ind", "industries", "mfg", "pc", "dds", "md", "pllc"
    }

    def clean_domain_and_symbols(raw: str) -> str:
        s = raw.lower()
        # Clean leading noise characters
        s = re.sub(r"^[\*\#\@\-\_\s]+", "", s)
        # Clean domain prefixes/suffixes
        s = re.sub(r"^(?:https?:\/\/)?(?:www\.)?", "", s)
        s = re.sub(r"\.(?:c0m|com|org|net|in|co|io|biz|info|gov)(?:\.in)?\b", "", s)
        s = re.sub(r"(?:c0m|com|org|net|biz|info)$", "", s)
        return s

    def extract_keys(name, addr, country):
        c_norm = normalize_country(country)
        cleaned_raw_name = clean_domain_and_symbols(name or "")
        n_clean, n_tokens, n_acr, n_comp = normalize_business_name(cleaned_raw_name)
        a_clean, a_tokens, pin, st_num, lmark = normalize_business_address(addr or "")

        keys = set()
        # 1. Exact name
        if n_clean:
            keys.add(("name_clean", c_norm, n_clean))
        # 2. Compressed name
        if len(n_comp) >= 3:
            keys.add(("name_comp", c_norm, n_comp))

        # Core tokens (excluding legal forms and 'and')
        core_tokens = [t for t in n_tokens if t not in LEGAL_SET and t not in ("and", "&", "the", "of", "in") and len(t) >= 2]
        if core_tokens:
            core_comp = "".join(core_tokens)
            if len(core_comp) >= 3:
                keys.add(("name_core_comp", c_norm, core_comp))
            sorted_comp = "".join(sorted(core_tokens))
            if len(sorted_comp) >= 3:
                keys.add(("name_sorted_comp", c_norm, sorted_comp))

        # 3. First 2 tokens
        if len(n_tokens) >= 2:
            keys.add(("name_first2", c_norm, f"{n_tokens[0]} {n_tokens[1]}"))
        # 4. First token if length >= 3
        if n_tokens and len(n_tokens[0]) >= 3 and n_tokens[0] not in COMMON_NAME_STOPWORDS and n_tokens[0] not in LEGAL_SET:
            keys.add(("name_first1", c_norm, n_tokens[0]))

        # 5. PIN + first token (or first 3 chars)
        if pin and n_tokens:
            keys.add(("pin_token", c_norm, pin, n_tokens[0][:3]))

        # Address keys
        # Extract rare address tokens
        rare_addr_tokens = [t for t in a_tokens if len(t) >= 4 and t not in ADDR_STOPWORDS and not t.isdigit()]
        if st_num and rare_addr_tokens:
            keys.add(("st_rare_addr", c_norm, st_num, rare_addr_tokens[0]))
            if len(rare_addr_tokens) >= 2:
                keys.add(("st_rare_addr2", c_norm, st_num, rare_addr_tokens[1]))

        # Street num + road token
        if st_num and a_tokens:
            road_tokens = [t for t in a_tokens if t != st_num and len(t) >= 3]
            if road_tokens:
                keys.add(("st_road", c_norm, st_num, road_tokens[0][:4]))

        # Street num + name 3-gram
        if st_num and len(n_comp) >= 3:
            keys.add(("st_name", c_norm, st_num, n_comp[:3]))

        return keys, n_tokens, rare_addr_tokens

    # Pre-extract keys for targets
    target_keys = {}
    for tid, r in target_records.items():
        t_keys, t_toks, t_addr = extract_keys(r["business_name"] or "", r["business_address"] or "", r["country"] or "")
        target_keys[tid] = (t_keys, t_toks, set(t_addr))

    # Now evaluate recall across S1 entities
    strategy_hits = collections.defaultdict(int)
    union_hits = 0
    missed_examples = []

    for s1_r in s1_rows.to_dicts():
        s1_id = s1_r["entity_id"]
        true_set = gt_map[s1_id]
        if not true_set:
            continue

        s1_keys, s1_toks, s1_addr = extract_keys(s1_r["business_name"] or "", s1_r["business_address"] or "", s1_r["country"] or "")
        s1_rare_tokens = {t for t in s1_toks if len(t) >= 4 and t not in COMMON_NAME_STOPWORDS}
        s1_addr_set = set(s1_addr)

        for tid in true_set:
            if tid not in target_keys:
                continue
            t_keys, t_toks, t_addr_set = target_keys[tid]
            
            # Check individual keys
            shared_keys = s1_keys & t_keys
            for k in shared_keys:
                strategy_hits[k[0]] += 1
            
            # Check token overlap
            t_rare_tokens = {t for t in t_toks if len(t) >= 4 and t not in COMMON_NAME_STOPWORDS}
            token_overlap = bool(s1_rare_tokens & t_rare_tokens)
            if token_overlap:
                strategy_hits["rare_token_overlap"] += 1

            addr_overlap = bool(s1_addr_set & t_addr_set)
            if addr_overlap:
                strategy_hits["addr_rare_token_overlap"] += 1

            if shared_keys or token_overlap or addr_overlap:
                union_hits += 1
            else:
                if len(missed_examples) < 10:
                    missed_examples.append((s1_r, target_records[tid]))

    print("\n--- Individual Key Recall on True Matches ---")
    for strat, hits in sorted(strategy_hits.items(), key=lambda x: -x[1]):
        print(f"  {strat:20s}: {hits:,} / {total_true_matches:,} ({hits / total_true_matches:.2%})")

    print(f"\nUnion Recall: {union_hits:,} / {total_true_matches:,} ({union_hits / total_true_matches:.2%})")
    
    if missed_examples:
        print(f"\nShowing first {len(missed_examples)} missed pairs:")
        for s1_m, tgt_m in missed_examples:
            s1_info = f"S1:  {s1_m['entity_id']} | {s1_m['business_name']} | {s1_m['business_address']}"
            tgt_info = f"TGT: {tgt_m['entity_id']} | {tgt_m['business_name']} | {tgt_m['business_address']}"
            print(s1_info.encode("ascii", "replace").decode("ascii"))
            print(tgt_info.encode("ascii", "replace").decode("ascii"))
            print("-" * 60)

if __name__ == "__main__":
    run_experiment()
