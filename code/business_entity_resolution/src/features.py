"""
Pairwise Feature Engineering for Business Entity Resolution.

Computes comprehensive similarity features between Source 1 records and candidates:
- Rapid string distance metrics (Levenshtein, Jaro-Winkler, Token-Sort, Token-Set)
- Jaccard token and character 3-gram overlaps
- Postal code and street number agreement
- Acronym and compressed name matching
- Address presence/missingness indicators
- Candidate retrieval rank and blocking score
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.fuzz as fuzz

from normalize import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    COMMON_NAME_STOPWORDS,
)

FEATURE_NAMES: List[str] = [
    # Name string metrics
    "name_jw",
    "name_lev_ratio",
    "name_token_sort",
    "name_token_set",
    "name_token_jaccard",
    "name_char3_jaccard",
    "name_exact_match",
    "name_comp_exact",
    "name_comp_substr",
    "name_acronym_match",
    "name_len_diff",
    "name_len_ratio",
    # Address string metrics
    "addr_s1_missing",
    "addr_cand_missing",
    "addr_both_present",
    "addr_jw",
    "addr_token_sort",
    "addr_token_set",
    "addr_token_jaccard",
    "addr_char3_jaccard",
    # Structured address flags
    "pin_match",
    "pin_both_present",
    "st_num_match",
    "st_num_both_present",
    # Meta and blocking rank features
    "cand_source_is_s2",
    "cand_rank",
    "cand_blocking_score",
]


def char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Generates set of character n-grams."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class RecordRepresentation:
    """
    Cached pre-normalized representation of an entity record for instant feature extraction.
    """

    __slots__ = (
        "entity_id",
        "country",
        "name_clean",
        "name_tokens",
        "name_token_set",
        "name_char3",
        "name_acr",
        "name_comp",
        "addr_clean",
        "addr_tokens",
        "addr_token_set",
        "addr_char3",
        "pin",
        "st_num",
        "is_s2",
    )

    def __init__(self, entity_id: str, name: str, address: str, country: str):
        self.entity_id = entity_id
        self.country = normalize_country(country)
        self.is_s2 = 1.0 if entity_id.startswith("S2-") else 0.0

        n_clean, n_toks, n_acr, n_comp = normalize_business_name(name)
        self.name_clean = n_clean
        self.name_tokens = n_toks
        self.name_token_set = set(n_toks)
        self.name_char3 = char_ngrams(n_comp, 3)

        # Core acronym (excluding legal suffixes and stopwords)
        core_toks = [
            t for t in n_toks
            if t not in COMMON_NAME_STOPWORDS and t not in (
                "corp", "corporation", "inc", "incorporated", "ltd", "limited",
                "pvt", "private", "co", "company", "llc", "llp", "tech"
            )
        ]
        self.name_acr = "".join(t[0] for t in core_toks if t) if core_toks else n_acr
        self.name_comp = n_comp

        a_clean, a_toks, pin, st_num, _ = normalize_business_address(address)
        self.addr_clean = a_clean
        self.addr_tokens = a_toks
        self.addr_token_set = set(a_toks)
        self.addr_char3 = char_ngrams(a_clean.replace(" ", ""), 3) if a_clean else set()
        self.pin = pin
        self.st_num = st_num


def compute_pair_features(
    s1: RecordRepresentation,
    cand: RecordRepresentation,
    cand_rank: int = 0,
    blocking_score: float = 0.0,
) -> np.ndarray:
    """
    Computes dense 1D float32 numpy array of features for an (S1, Candidate) pair.
    """
    feats = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # 1. Name features
    s1_nc = s1.name_clean
    cand_nc = cand.name_clean

    feats[0] = jw.similarity(s1_nc, cand_nc)
    feats[1] = lev.normalized_similarity(s1_nc, cand_nc)
    feats[2] = fuzz.token_sort_ratio(s1_nc, cand_nc) / 100.0
    feats[3] = fuzz.token_set_ratio(s1_nc, cand_nc) / 100.0
    feats[4] = jaccard(s1.name_token_set, cand.name_token_set)
    feats[5] = jaccard(s1.name_char3, cand.name_char3)
    feats[6] = 1.0 if s1_nc and s1_nc == cand_nc else 0.0

    s1_comp = s1.name_comp
    cand_comp = cand.name_comp
    feats[7] = 1.0 if s1_comp and s1_comp == cand_comp else 0.0
    feats[8] = 1.0 if (s1_comp and cand_comp and (s1_comp in cand_comp or cand_comp in s1_comp)) else 0.0

    # Acronym matching
    acr_match = 0.0
    if len(s1.name_acr) >= 2 and (s1.name_acr == cand_comp or s1.name_acr in cand.name_token_set):
        acr_match = 1.0
    elif len(cand.name_acr) >= 2 and (cand.name_acr == s1_comp or cand.name_acr in s1.name_token_set):
        acr_match = 1.0
    feats[9] = acr_match

    len1 = len(s1_nc)
    len2 = len(cand_nc)
    feats[10] = abs(len1 - len2)
    feats[11] = min(len1, len2) / max(len1, len2, 1)

    # 2. Address features
    s1_has_addr = bool(s1.addr_clean)
    cand_has_addr = bool(cand.addr_clean)
    feats[12] = 0.0 if s1_has_addr else 1.0
    feats[13] = 0.0 if cand_has_addr else 1.0
    feats[14] = 1.0 if (s1_has_addr and cand_has_addr) else 0.0

    if s1_has_addr and cand_has_addr:
        feats[15] = jw.similarity(s1.addr_clean, cand.addr_clean)
        feats[16] = fuzz.token_sort_ratio(s1.addr_clean, cand.addr_clean) / 100.0
        feats[17] = fuzz.token_set_ratio(s1.addr_clean, cand.addr_clean) / 100.0
        feats[18] = jaccard(s1.addr_token_set, cand.addr_token_set)
        feats[19] = jaccard(s1.addr_char3, cand.addr_char3)

    # Structured fields
    if s1.pin and cand.pin:
        feats[20] = 1.0 if s1.pin == cand.pin else -1.0
        feats[21] = 1.0
    else:
        feats[20] = 0.0
        feats[21] = 0.0

    if s1.st_num and cand.st_num:
        feats[22] = 1.0 if s1.st_num == cand.st_num else -1.0
        feats[23] = 1.0
    else:
        feats[22] = 0.0
        feats[23] = 0.0

    # Meta
    feats[24] = cand.is_s2
    feats[25] = float(cand_rank)
    feats[26] = float(blocking_score)

    return feats
