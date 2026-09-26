"""
Fast path for the 42-feature computation — produces values IDENTICAL to
compute_enhanced_features, but:
  - TF-IDF cosine via sparse dicts instead of dense 30K-dim vectors
  - Metaphone/Soundex memoized (tokens repeat massively across the pool)
  - S1-side phonetics computed once per entity, not once per pair

Use this everywhere features are computed in bulk (pair generation and
inference). Verified against enhanced_features.compute_enhanced_features
with np.allclose on sampled pairs (see scratch/check_fast_features.py).
"""

import math
import collections
from typing import Dict, List, Optional

import numpy as np

from enhanced_features import (
    ENHANCED_FEATURE_NAMES,
    metaphone as _metaphone_raw,
    soundex as _soundex_raw,
)
from features import RecordRepresentation  # noqa: F401  (type compat)

_memo_meta: Dict[str, str] = {}
_memo_sdx: Dict[str, str] = {}


def metaphone_m(t: str) -> str:
    v = _memo_meta.get(t)
    if v is None:
        v = _metaphone_raw(t)
        if len(_memo_meta) > 1_500_000:
            _memo_meta.clear()
        _memo_meta[t] = v
    return v


def soundex_m(t: str) -> str:
    v = _memo_sdx.get(t)
    if v is None:
        v = _soundex_raw(t)
        if len(_memo_sdx) > 1_500_000:
            _memo_sdx.clear()
        _memo_sdx[t] = v
    return v


def _weights(tfidf, tokens: List[str]) -> Dict[str, float]:
    """L2-normalized in-vocab TF-IDF weights — identical math to transform()."""
    w: Dict[str, float] = {}
    tf = collections.Counter(tokens)
    for t, c in tf.items():
        idx = tfidf.vocab.get(t)
        if idx is not None:
            w[t] = (1.0 + math.log(c)) * tfidf.idf[idx]
    norm = math.sqrt(sum(x * x for x in w.values()))
    if norm > 0:
        inv = 1.0 / norm
        for t in w:
            w[t] *= inv
    return w


def cosine_sparse(tfidf, tokens_a: List[str], tokens_b: List[str]) -> float:
    if not tfidf.fitted:
        return 0.0
    wa = _weights(tfidf, tokens_a)
    if not wa:
        return 0.0
    wb = _weights(tfidf, tokens_b)
    if len(wb) < len(wa):
        wa, wb = wb, wa
    return float(sum(v * wb.get(t, 0.0) for t, v in wa.items()))


def bm25_sparse(tfidf, query_tokens: List[str], doc_tokens: List[str],
                k1: float = 1.5, b: float = 0.75, avgdl: float = 5.0) -> float:
    """Same math as CompactTFIDF.bm25_score (already loop-based)."""
    if not tfidf.fitted or not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    doc_tf = collections.Counter(doc_tokens)
    score = 0.0
    for t in set(query_tokens):
        idx = tfidf.vocab.get(t)
        if idx is None:
            continue
        tf_val = doc_tf.get(t, 0)
        if tf_val == 0:
            continue
        numerator = tf_val * (k1 + 1)
        denominator = tf_val + k1 * (1 - b + b * dl / avgdl)
        score += tfidf.idf[idx] * numerator / denominator
    return score


def phonetic_token_match_ratio_fast(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Identical to phonetic_token_match_ratio, with memoized codes."""
    if not tokens_a or not tokens_b:
        return 0.0
    if len(tokens_a) > len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a
    codes_b = {metaphone_m(t) for t in tokens_b if t}
    hits = 0
    for t in tokens_a:
        if t and metaphone_m(t) in codes_b:
            hits += 1
    return hits / max(len(tokens_a), 1)


def compute_enhanced_features_fast(
    base_features: np.ndarray,
    s1_name_tokens: List[str],
    s1_addr_tokens: List[str],
    cand_name_tokens: List[str],
    cand_addr_tokens: List[str],
    s1_name_clean: str,
    cand_name_clean: str,
    name_tfidf=None,
    addr_tfidf=None,
    s1_meta: Optional[str] = None,
    s1_sdx: Optional[str] = None,
) -> np.ndarray:
    enhanced = np.zeros(len(ENHANCED_FEATURE_NAMES), dtype=np.float32)
    enhanced[:33] = base_features

    if name_tfidf is not None and name_tfidf.fitted:
        enhanced[33] = cosine_sparse(name_tfidf, s1_name_tokens, cand_name_tokens)
        enhanced[35] = bm25_sparse(name_tfidf, s1_name_tokens, cand_name_tokens)
        shared = set(s1_name_tokens) & set(cand_name_tokens)
        if shared:
            max_idf = 0.0
            for t in shared:
                idx = name_tfidf.vocab.get(t)
                if idx is not None:
                    max_idf = max(max_idf, name_tfidf.idf[idx])
            enhanced[37] = max_idf

    if addr_tfidf is not None and addr_tfidf.fitted:
        enhanced[34] = cosine_sparse(addr_tfidf, s1_addr_tokens, cand_addr_tokens)
        enhanced[36] = bm25_sparse(addr_tfidf, s1_addr_tokens, cand_addr_tokens)
        shared = set(s1_addr_tokens) & set(cand_addr_tokens)
        if shared:
            max_idf = 0.0
            for t in shared:
                idx = addr_tfidf.vocab.get(t)
                if idx is not None:
                    max_idf = max(max_idf, addr_tfidf.idf[idx])
            enhanced[38] = max_idf

    if s1_name_clean and cand_name_clean:
        if s1_meta is None:
            s1_meta = metaphone_m(s1_name_clean)
        cand_meta = metaphone_m(cand_name_clean)
        enhanced[39] = 1.0 if (s1_meta and s1_meta == cand_meta) else 0.0

        if s1_sdx is None:
            s1_sdx = soundex_m(s1_name_clean)
        cand_sdx = soundex_m(cand_name_clean)
        enhanced[40] = 1.0 if (s1_sdx and s1_sdx == cand_sdx) else 0.0

        enhanced[41] = phonetic_token_match_ratio_fast(s1_name_tokens, cand_name_tokens)

    return enhanced
