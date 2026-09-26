"""
Blocking 2.0 — full-pool candidate generation with IDF weighting, address
anchoring, and TF-IDF re-ranking.

Why v1 fails at full-pool scale (Phase 0 measurements):
  - Posting lists were truncated to the FIRST 250 records per key, so hub keys
    (common names, common tokens) silently dropped most of their members.
  - Candidate ordering was a raw key-overlap count: on a 6.2M-record pool the
    30-candidate cap filled with same-brand-different-city junk
    (recall@30 = 0.77 US, 0.57 India).

What v2 does:
  1. Composite keys anchor name tokens on address facts (pin, rare address
     tokens, street number) — naturally rare, so posting caps rarely bind.
  2. Retrieval weights every key hit by IDF computed from the posting length
     at query time — hub keys contribute almost nothing, no DF pre-pass.
  3. Address fingerprints (pin / street number / rare addr tokens) are stored
     per row in compact int arrays; candidates agreeing with the query on
     address facts get a boost even when name keys are weak.
  4. Overgenerate (~300) -> re-rank with IDF-weighted cosine on name+address
     tokens -> keep top-K.

Memory-safe: chunked indexing, compact int arrays, single pass per file.
Set `blocker.text_names` / `blocker.text_addrs` (raw strings per row) after
indexing — the re-ranker reads candidate text through them.
"""

import re
import os
import json
import math
import array
import heapq
import collections
from typing import Dict, List, Optional, Tuple

import polars as pl
from rapidfuzz import fuzz

from normalize import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    COMMON_NAME_STOPWORDS,
)
from blocking import clean_domain_and_symbols, ADDR_STOPWORDS, LEGAL_SET
from enhanced_features import soundex

NON_WORD = re.compile(r"[^\w\s]")

W_EXACT = 4.0     # nm / cp / cr / sc keys
W_COMPOSITE = 3.0 # name anchor x address fact
W_TOKEN = 2.0     # rare single-field tokens
W_PHON = 1.5      # soundex-of-core-token keys
W_ADDR_ONLY = 2.0 # address-only keys (pin, addr-token pairs) — for garbled names
W_PIN = 2.0
W_STNUM = 1.5
W_ADDR_TOK = 0.5
STOP_EXTRA = ("and", "the", "of", "in", "for", "to")
MIN_TOK_LEN = 3   # v2.1: 3-char tokens matter for Indian names/localities

# re-rank blend weights (tuned offline on in-pool mini links; see scratch/tune_rerank.py)
W_COS = 0.40
W_FUZZ_NAME = 0.0
W_FUZZ_ADDR = 0.20
W_KEY = 0.40


def _fingerprint(text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """(pin, street_number, rare_addr_tokens) from a raw address string."""
    pin, st_num = None, None
    if text:
        _, _, pin, st_num, _ = normalize_business_address(text)
    toks = NON_WORD.sub(" ", (text or "").lower()).split()
    rare = [t for t in toks if len(t) >= MIN_TOK_LEN and not t.isdigit() and t not in ADDR_STOPWORDS]
    return pin, st_num, rare[:4]


# ── mined transliteration map (produced by mine_translit.py) ─────────────────
_TRANSLIT: Optional[Dict[str, Dict[str, str]]] = None


def _load_translit() -> Dict[str, Dict[str, str]]:
    """Keys are normalized country labels (US / INDIA / FRANCE)."""
    global _TRANSLIT
    if _TRANSLIT is None:
        path = os.path.join("cache", "translit_map.json")
        raw = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            print(f"[blocking_v2] loaded translit map: "
                  f"{ {k: len(v) for k, v in raw.items()} }", flush=True)
        _TRANSLIT = {
            "US": raw.get("US", {}),
            "INDIA": raw.get("India", {}),
            "FRANCE": raw.get("France", {}),
        }
    return _TRANSLIT


def _map_toks(toks, tmap: Dict[str, str]):
    if not tmap:
        return toks
    return [tmap.get(t, t) for t in toks]


class StreamingBlockerV2:
    """Chunk-streamed index. Call index_chunk(df) per chunk, then retrieve()."""

    def __init__(self, max_posting: int = 20000, tok_posting: int = 5000,
                 overgenerate: int = 800, topk: int = 100):
        self.max_posting = max_posting      # exact-name keys
        self.tok_posting = tok_posting      # token / composite keys
        self.overgenerate = overgenerate
        self.topk = topk
        self.total_indexed = 0

        self.key_postings: Dict[str, array.array] = {}
        self.row_pin: array.array = array.array("I")    # hash or 0
        self.row_stnum: array.array = array.array("I")  # hash or 0
        self.row_atoks: List[Tuple[int, ...]] = []      # up to 4 tok hashes
        self.name_df: Dict[str, int] = collections.Counter()
        self.addr_df: Dict[str, int] = collections.Counter()

        # raw text per row, set by the harness after indexing (for re-rank)
        self.text_names: List[str] = []
        self.text_addrs: List[str] = []

    # ── indexing ──────────────────────────────────────────────────────────
    def index_chunk(self, df: pl.DataFrame) -> None:
        names = df["business_name"].fill_null("").to_list()
        addrs = df["business_address"].fill_null("").to_list()
        countries = df["country"].fill_null("").to_list()

        for name, addr, country in zip(names, addrs, countries):
            i = self.total_indexed
            c = normalize_country(country)
            tmap = _load_translit().get(c, {})
            cleaned = clean_domain_and_symbols(name)
            n_clean, n_toks, _, n_comp = normalize_business_name(cleaned)
            n_toks = _map_toks(n_toks, tmap)
            pin, st_num, a_rare = _fingerprint(addr)
            a_rare = _map_toks(a_rare, tmap)
            _, a_toks, _, _, _ = normalize_business_address(addr or "")
            a_toks = _map_toks(a_toks, tmap)

            self.name_df.update(n_toks)
            self.addr_df.update(a_toks)

            self.row_pin.append(hash(pin) & 0xFFFFFFFF if pin else 0)
            self.row_stnum.append(hash(st_num) & 0xFFFFFFFF if st_num else 0)
            self.row_atoks.append(tuple(hash(t) & 0xFFFFFFFF for t in a_rare))

            core = [t for t in n_toks
                    if t not in LEGAL_SET and t not in COMMON_NAME_STOPWORDS
                    and t not in STOP_EXTRA and len(t) >= 2]
            rare_name = [t for t in n_toks if len(t) >= MIN_TOK_LEN
                         and t not in COMMON_NAME_STOPWORDS and t not in LEGAL_SET]
            phon = list(dict.fromkeys(soundex(t) for t in core if len(t) >= 3))[:3]

            def add(key: str, row: int, cap: int):
                postings = self.key_postings.get(key)
                if postings is None:
                    postings = array.array("I")
                    self.key_postings[key] = postings
                if len(postings) < cap:
                    postings.append(row)

            # exact-ish name keys
            if n_clean:
                add(f"nm_{c}|{n_clean}", i, self.max_posting)
            if len(n_comp) >= 3:
                add(f"cp_{c}|{n_comp}", i, self.max_posting)
            if core:
                cc = "".join(core)
                if len(cc) >= 3:
                    add(f"cr_{c}|{cc}", i, self.max_posting)
                    add(f"sc_{c}|{''.join(sorted(core))}", i, self.max_posting)

            # composite keys: name anchor x address fact (naturally rare)
            if core:
                first = core[0]
                if pin:
                    add(f"cmp_{c}|{first}|p{pin}", i, self.tok_posting)
                for t in a_rare[:2]:
                    add(f"cmp_{c}|{first}|a{t}", i, self.tok_posting)
                if st_num:
                    add(f"cmp_{c}|{first}|s{st_num}", i, self.tok_posting)
            if pin and n_toks:
                add(f"cmp_{c}|p{pin}|{n_toks[0][:3]}", i, self.tok_posting)
            if st_num:
                for t in a_rare[:2]:
                    add(f"cmp_{c}|s{st_num}|a{t}", i, self.tok_posting)

            # address-only keys (catch garbled-name records via address facts)
            if pin:
                add(f"pin_{c}|{pin}", i, self.tok_posting)
            if len(a_rare) >= 2:
                add(f"apair_{c}|{a_rare[0]}|{a_rare[1]}", i, self.tok_posting)

            # rare single-field token indices + phonetic keys
            for t in rare_name:
                add(f"tok_n_{c}|{t}", i, self.tok_posting)
            for t in a_rare:
                add(f"tok_a_{c}|{t}", i, self.tok_posting)
            for p in phon:
                add(f"ph_{c}|{p}", i, self.tok_posting)

            self.total_indexed += 1

    # ── retrieval ─────────────────────────────────────────────────────────
    def _query_prep(self, name: str, addr: str, country: str):
        c = normalize_country(country)
        tmap = _load_translit().get(c, {})
        cleaned = clean_domain_and_symbols(name)
        n_clean, n_toks, _, n_comp = normalize_business_name(cleaned)
        n_toks = _map_toks(n_toks, tmap)
        pin, st_num, a_rare = _fingerprint(addr)
        a_rare = _map_toks(a_rare, tmap)

        core = [t for t in n_toks
                if t not in LEGAL_SET and t not in COMMON_NAME_STOPWORDS
                and t not in STOP_EXTRA and len(t) >= 2]
        rare_name = [t for t in n_toks if len(t) >= MIN_TOK_LEN
                     and t not in COMMON_NAME_STOPWORDS and t not in LEGAL_SET]
        phon = list(dict.fromkeys(soundex(t) for t in core if len(t) >= 3))[:3]

        keys: List[Tuple[str, float]] = []
        if n_clean:
            keys.append((f"nm_{c}|{n_clean}", W_EXACT))
        if len(n_comp) >= 3:
            keys.append((f"cp_{c}|{n_comp}", W_EXACT))
        if core:
            cc = "".join(core)
            if len(cc) >= 3:
                keys.append((f"cr_{c}|{cc}", W_EXACT))
                keys.append((f"sc_{c}|{''.join(sorted(core))}", W_EXACT))
            first = core[0]
            if pin:
                keys.append((f"cmp_{c}|{first}|p{pin}", W_COMPOSITE))
            for t in a_rare[:2]:
                keys.append((f"cmp_{c}|{first}|a{t}", W_COMPOSITE))
            if st_num:
                keys.append((f"cmp_{c}|{first}|s{st_num}", W_COMPOSITE))
        if pin and n_toks:
            keys.append((f"cmp_{c}|p{pin}|{n_toks[0][:3]}", W_COMPOSITE))
        if st_num:
            for t in a_rare[:2]:
                keys.append((f"cmp_{c}|s{st_num}|a{t}", W_COMPOSITE))
        if pin:
            keys.append((f"pin_{c}|{pin}", W_ADDR_ONLY))
        if len(a_rare) >= 2:
            keys.append((f"apair_{c}|{a_rare[0]}|{a_rare[1]}", W_ADDR_ONLY))
        for t in rare_name:
            keys.append((f"tok_n_{c}|{t}", W_TOKEN))
        for t in a_rare:
            keys.append((f"tok_a_{c}|{t}", W_TOKEN))
        for p in phon:
            keys.append((f"ph_{c}|{p}", W_PHON))

        return n_toks, a_rare, pin, st_num, keys

    def retrieve(self, name: str, addr: str, country: str,
                 k: Optional[int] = None, return_components: bool = False):
        k = k or self.topk
        n = max(self.total_indexed, 1)
        _, a_rare_q, pin_q, stnum_q, keys = self._query_prep(name, addr, country)
        q_pin = hash(pin_q) & 0xFFFFFFFF if pin_q else 0
        q_st = hash(stnum_q) & 0xFFFFFFFF if stnum_q else 0
        q_atoks = frozenset(hash(t) & 0xFFFFFFFF for t in a_rare_q)

        scores: Dict[int, float] = {}
        max_score = 1e-9
        for key, base_w in keys:
            postings = self.key_postings.get(key)
            if not postings:
                continue
            w = base_w * math.log(1.0 + n / len(postings))
            for row in postings:
                if row in scores:
                    scores[row] += w
                else:
                    scores[row] = w
            if w > max_score:
                max_score = w

        if not scores:
            return []

        # address-agreement boost on accumulated candidates
        for row in list(scores):
            s = scores[row]
            if q_pin and self.row_pin[row] == q_pin:
                s += W_PIN
            if q_st and self.row_stnum[row] == q_st:
                s += W_STNUM
            if a_rare_q:
                shared = 0
                for h in self.row_atoks[row]:
                    if h in q_atoks:
                        shared += 1
                if shared:
                    s += W_ADDR_TOK * min(shared, 2)
            scores[row] = s

        n_over = min(self.overgenerate, len(scores))
        if len(scores) > n_over:
            top = heapq.nlargest(n_over, scores.items(), key=lambda kv: kv[1])
        else:
            top = list(scores.items())

        # TF-IDF cosine re-rank on name + address tokens
        total_n = max(sum(self.name_df.values()), 1)
        total_a = max(sum(self.addr_df.values()), 1)

        def vec(name_tokens, addr_tokens) -> Dict[str, float]:
            v: Dict[str, float] = {}
            for t in name_tokens:
                v[t] = v.get(t, 0.0) + math.log(1.0 + total_n / max(self.name_df.get(t, 1), 1))
            for t in addr_tokens:
                key = "a:" + t
                v[key] = v.get(key, 0.0) + math.log(1.0 + total_a / max(self.addr_df.get(t, 1), 1))
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return {t: x / norm for t, x in v.items()}

        tmap_r = _load_translit().get(normalize_country(country), {})
        q_clean = clean_domain_and_symbols(name)
        qn_clean, qn_toks, _, _ = normalize_business_name(q_clean)
        qa_clean, qa_toks, _, _, _ = normalize_business_address(addr or "")
        qv = vec(_map_toks(qn_toks, tmap_r), _map_toks(qa_toks, tmap_r))

        denom = max_score + W_PIN + W_STNUM + 2 * W_ADDR_TOK
        reranked: List[Tuple[float, int]] = []
        comps: List[Dict[str, float]] = []
        for row, s in top:
            cn_clean, cn_toks, _, _ = normalize_business_name(
                clean_domain_and_symbols(self.text_names[row]))
            ca_clean, ca_toks, _, _, _ = normalize_business_address(self.text_addrs[row])
            cv = vec(_map_toks(cn_toks, tmap_r), _map_toks(ca_toks, tmap_r))
            common = qv.keys() & cv.keys()
            cos = sum(qv[t] * cv[t] for t in common) if common else 0.0
            fz_n = fuzz.token_set_ratio(qn_clean, cn_clean) / 100.0
            fz_a = fuzz.token_set_ratio(qa_clean, ca_clean) / 100.0
            kr = min(s / denom, 1.0)
            final = W_COS * min(cos * 1.4, 1.0) + W_FUZZ_NAME * fz_n \
                + W_FUZZ_ADDR * fz_a + W_KEY * kr
            reranked.append((final, row))
            if return_components:
                comps.append({"cos": cos, "fz_n": fz_n, "fz_a": fz_a, "kr": kr})

        reranked.sort(key=lambda x: -x[0])
        rows = [row for _, row in reranked[:k]]
        if return_components:
            order = {row: i for i, row in enumerate(rows)}
            return rows, [c for c, (_, row) in zip(comps, reranked) if row in order]
        return rows

    # v1-compatible alias so the harness can call either blocker uniformly
    def retrieve_candidate_indices(self, name: str, addr: str, country: str) -> List[int]:
        return self.retrieve(name, addr, country)
