"""
Enhanced Feature Engineering for Business Entity Resolution v2.

Extends the base 33 features with:
- TF-IDF cosine similarity on name and address tokens (per-country vocabulary)
- BM25-inspired relevance score
- Max shared IDF (information value of overlapping tokens)
- Pure-Python phonetic encoding (Soundex + simplified Metaphone)

Total features: 33 (base) + 6 (TF-IDF/BM25) + 3 (phonetic) = 42 features.
Zero external dependencies beyond numpy.
"""

import math
import collections
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from normalize import normalize_country


# ─── Pure-Python Soundex ─────────────────────────────────────────────────────

_SOUNDEX_MAP = {
    'b': '1', 'f': '1', 'p': '1', 'v': '1',
    'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
    'd': '3', 't': '3',
    'l': '4',
    'm': '5', 'n': '5',
    'r': '6',
}


def soundex(word: str) -> str:
    """Pure-Python Soundex encoding."""
    if not word:
        return ""
    word = word.lower().strip()
    # Keep only alpha chars
    word = ''.join(c for c in word if c.isalpha())
    if not word:
        return ""

    result = [word[0].upper()]
    prev_code = _SOUNDEX_MAP.get(word[0], '0')

    for ch in word[1:]:
        code = _SOUNDEX_MAP.get(ch, '0')
        if code != '0' and code != prev_code:
            result.append(code)
            if len(result) == 4:
                break
        prev_code = code if code != '0' else prev_code

    return ''.join(result).ljust(4, '0')


# ─── Pure-Python Simplified Metaphone ────────────────────────────────────────

def metaphone(word: str) -> str:
    """
    Simplified Metaphone encoding (pure Python).
    Maps words to phonetic codes for fuzzy matching.
    """
    if not word:
        return ""
    word = word.lower().strip()
    word = ''.join(c for c in word if c.isalpha())
    if not word:
        return ""

    # Drop duplicate adjacent letters
    deduped = [word[0]]
    for c in word[1:]:
        if c != deduped[-1]:
            deduped.append(c)
    word = ''.join(deduped)

    # Drop initial silent letters
    if word[:2] in ('ae', 'gn', 'kn', 'pn', 'wr'):
        word = word[1:]
    if not word:
        return ""

    code = []
    i = 0
    while i < len(word) and len(code) < 6:
        c = word[i]
        remaining = word[i:]

        if c in 'aeiou':
            if i == 0:
                code.append(c.upper())
            i += 1
        elif c == 'b':
            code.append('B')
            i += 2 if i + 1 < len(word) and word[i + 1] == 'b' else 1
        elif c == 'c':
            if remaining.startswith(('ch',)):
                code.append('X')
                i += 2
            elif remaining.startswith(('ci', 'ce', 'cy')):
                code.append('S')
                i += 1
            else:
                code.append('K')
                i += 1
        elif c == 'd':
            if remaining.startswith(('dg',)):
                i += 1
            else:
                code.append('T')
                i += 1
        elif c == 'f':
            code.append('F')
            i += 2 if i + 1 < len(word) and word[i + 1] == 'f' else 1
        elif c == 'g':
            if i + 1 < len(word) and word[i + 1] == 'h':
                if i + 2 < len(word) and word[i + 2] not in 'aeiou':
                    i += 2
                else:
                    code.append('K')
                    i += 2
            elif remaining.startswith(('gn',)):
                i += 2
            else:
                code.append('K')
                i += 2 if i + 1 < len(word) and word[i + 1] == 'g' else 1
        elif c == 'h':
            if i == 0 or word[i - 1] not in 'aeiou':
                if i + 1 < len(word) and word[i + 1] in 'aeiou':
                    code.append('H')
            i += 1
        elif c == 'j':
            code.append('J')
            i += 1
        elif c == 'k':
            if i == 0 or word[i - 1] != 'c':
                code.append('K')
            i += 1
        elif c == 'l':
            code.append('L')
            i += 2 if i + 1 < len(word) and word[i + 1] == 'l' else 1
        elif c == 'm':
            code.append('M')
            i += 2 if i + 1 < len(word) and word[i + 1] == 'm' else 1
        elif c == 'n':
            code.append('N')
            i += 2 if i + 1 < len(word) and word[i + 1] == 'n' else 1
        elif c == 'p':
            if remaining.startswith('ph'):
                code.append('F')
                i += 2
            else:
                code.append('P')
                i += 2 if i + 1 < len(word) and word[i + 1] == 'p' else 1
        elif c == 'q':
            code.append('K')
            i += 1
        elif c == 'r':
            code.append('R')
            i += 2 if i + 1 < len(word) and word[i + 1] == 'r' else 1
        elif c == 's':
            if remaining.startswith(('sh', 'sio', 'sia')):
                code.append('X')
                i += 2
            else:
                code.append('S')
                i += 2 if i + 1 < len(word) and word[i + 1] == 's' else 1
        elif c == 't':
            if remaining.startswith(('th',)):
                code.append('0')  # theta
                i += 2
            elif remaining.startswith(('tio', 'tia')):
                code.append('X')
                i += 2
            else:
                code.append('T')
                i += 2 if i + 1 < len(word) and word[i + 1] == 't' else 1
        elif c == 'v':
            code.append('F')
            i += 1
        elif c == 'w':
            if i + 1 < len(word) and word[i + 1] in 'aeiou':
                code.append('W')
            i += 1
        elif c == 'x':
            code.append('KS')
            i += 1
        elif c == 'y':
            if i + 1 < len(word) and word[i + 1] in 'aeiou':
                code.append('Y')
            i += 1
        elif c == 'z':
            code.append('S')
            i += 1
        else:
            i += 1

    return ''.join(code)


# ─── TF-IDF Vocabulary Builder ───────────────────────────────────────────────

class CompactTFIDF:
    """
    Lightweight per-country TF-IDF vocabulary.
    Stores only IDF weights and vocabulary mapping. No scipy sparse matrices.
    """

    def __init__(self, max_features: int = 30000):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, token_lists: List[List[str]]) -> None:
        """Fit IDF weights from a list of token-lists (each = one document)."""
        n_docs = len(token_lists)
        if n_docs == 0:
            self.fitted = False
            return

        df_counts: Dict[str, int] = collections.Counter()
        for toks in token_lists:
            for t in set(toks):
                df_counts[t] += 1

        sorted_terms = sorted(df_counts.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocab = {term: idx for idx, (term, _) in enumerate(sorted_terms)}

        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + df_counts[term])) + 1.0

        self.fitted = True

    def transform(self, tokens: List[str]) -> np.ndarray:
        """Transform a single token list into a TF-IDF vector."""
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        if not self.fitted or not tokens:
            return vec

        tf = collections.Counter(tokens)
        for t, count in tf.items():
            idx = self.vocab.get(t)
            if idx is not None:
                vec[idx] = (1.0 + math.log(count)) * self.idf[idx]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def cosine_sim(self, tokens_a: List[str], tokens_b: List[str]) -> float:
        """Compute cosine similarity between two token lists."""
        if not self.fitted:
            return 0.0
        va = self.transform(tokens_a)
        vb = self.transform(tokens_b)
        return float(np.dot(va, vb))

    def bm25_score(self, query_tokens: List[str], doc_tokens: List[str],
                   k1: float = 1.5, b: float = 0.75, avgdl: float = 5.0) -> float:
        """Compute simplified BM25 score."""
        if not self.fitted or not query_tokens or not doc_tokens:
            return 0.0
        dl = len(doc_tokens)
        doc_tf = collections.Counter(doc_tokens)
        score = 0.0
        for t in set(query_tokens):
            idx = self.vocab.get(t)
            if idx is None:
                continue
            tf_val = doc_tf.get(t, 0)
            if tf_val == 0:
                continue
            idf_val = self.idf[idx]
            numerator = tf_val * (k1 + 1)
            denominator = tf_val + k1 * (1 - b + b * dl / avgdl)
            score += idf_val * numerator / denominator
        return score


# ─── Phonetic Token Matching ─────────────────────────────────────────────────

def phonetic_token_match_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Fraction of tokens in the shorter list that have a phonetic match
    (same Metaphone code) in the longer list.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    if len(tokens_a) > len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a

    codes_b = set()
    for t in tokens_b:
        c = metaphone(t)
        if c:
            codes_b.add(c)

    if not codes_b:
        return 0.0

    matches = 0
    for t in tokens_a:
        c = metaphone(t)
        if c and c in codes_b:
            matches += 1

    return matches / len(tokens_a)


# ─── Enhanced Feature Names ──────────────────────────────────────────────────

ENHANCED_FEATURE_NAMES: List[str] = [
    # === Original 33 base features (indices 0-32) ===
    "name_jw", "name_lev_ratio", "name_token_sort", "name_token_set",
    "name_token_jaccard", "name_token_containment", "name_prefix4_match",
    "name_sorted_jw", "name_char3_jaccard", "name_exact_match",
    "name_comp_exact", "name_comp_substr", "name_acronym_match",
    "name_len_diff", "name_len_ratio",
    "addr_s1_missing", "addr_cand_missing", "addr_both_present",
    "addr_jw", "addr_token_sort", "addr_token_set", "addr_token_jaccard",
    "addr_token_containment", "addr_char3_jaccard",
    "pin_match", "pin_both_present", "pin_disagree",
    "st_num_match", "st_num_both_present", "st_num_disagree",
    "cand_source_is_s2", "cand_rank", "cand_blocking_score",
    # === TF-IDF / BM25 features (indices 33-38) ===
    "name_tfidf_cosine", "addr_tfidf_cosine",
    "name_bm25", "addr_bm25",
    "name_tfidf_max_shared_idf", "addr_tfidf_max_shared_idf",
    # === Phonetic features (indices 39-41) ===
    "name_metaphone_match", "name_soundex_match", "name_phonetic_token_ratio",
]


def compute_enhanced_features(
    base_features: np.ndarray,
    s1_name_tokens: List[str],
    s1_addr_tokens: List[str],
    cand_name_tokens: List[str],
    cand_addr_tokens: List[str],
    s1_name_clean: str,
    cand_name_clean: str,
    name_tfidf: Optional[CompactTFIDF] = None,
    addr_tfidf: Optional[CompactTFIDF] = None,
) -> np.ndarray:
    """
    Extends a 33-dim base feature vector to 42 dimensions.
    """
    enhanced = np.zeros(len(ENHANCED_FEATURE_NAMES), dtype=np.float32)
    enhanced[:33] = base_features

    # TF-IDF cosine similarity
    if name_tfidf and name_tfidf.fitted:
        enhanced[33] = name_tfidf.cosine_sim(s1_name_tokens, cand_name_tokens)
        enhanced[35] = name_tfidf.bm25_score(s1_name_tokens, cand_name_tokens)
        shared_name = set(s1_name_tokens) & set(cand_name_tokens)
        if shared_name:
            max_idf = 0.0
            for t in shared_name:
                idx = name_tfidf.vocab.get(t)
                if idx is not None:
                    max_idf = max(max_idf, name_tfidf.idf[idx])
            enhanced[37] = max_idf

    if addr_tfidf and addr_tfidf.fitted:
        enhanced[34] = addr_tfidf.cosine_sim(s1_addr_tokens, cand_addr_tokens)
        enhanced[36] = addr_tfidf.bm25_score(s1_addr_tokens, cand_addr_tokens)
        shared_addr = set(s1_addr_tokens) & set(cand_addr_tokens)
        if shared_addr:
            max_idf = 0.0
            for t in shared_addr:
                idx = addr_tfidf.vocab.get(t)
                if idx is not None:
                    max_idf = max(max_idf, addr_tfidf.idf[idx])
            enhanced[38] = max_idf

    # Phonetic features
    if s1_name_clean and cand_name_clean:
        s1_meta = metaphone(s1_name_clean)
        cand_meta = metaphone(cand_name_clean)
        enhanced[39] = 1.0 if (s1_meta and s1_meta == cand_meta) else 0.0

        s1_sx = soundex(s1_name_clean)
        cand_sx = soundex(cand_name_clean)
        enhanced[40] = 1.0 if (s1_sx and s1_sx == cand_sx) else 0.0

        enhanced[41] = phonetic_token_match_ratio(s1_name_tokens, cand_name_tokens)

    return enhanced
