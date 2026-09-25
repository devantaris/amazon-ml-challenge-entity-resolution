"""
Candidate Blocking Module for Business Entity Resolution.

High-recall, memory-efficient candidate generator:
- Partitions candidates by country.
- Multi-key inverted indexing:
    1. Exact normalized name
    2. Core compressed name (excluding legal suffixes)
    3. Sorted core tokens
    4. First two name tokens
    5. Informative first name token
    6. Postal code + name prefix
    7. Street / Plot number + road / rare address token
    8. Street number + name prefix
- Posting list frequency capping to avoid uninformative hub keys.
- Produces candidate_pairs.tsv matching the competition specification.
"""

import collections
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import polars as pl
from normalize import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    COMMON_NAME_STOPWORDS,
)

# Generic address tokens to ignore when building address keys
ADDR_STOPWORDS: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "floor", "fl", "unit", "apt", "suite", "ste",
    "building", "bldg", "north", "south", "east", "west", "near", "opp", "opposite",
    "behind", "door", "shop", "plot", "sector", "sec", "block", "blk", "village",
    "city", "town", "state", "india", "usa", "us", "haryana", "maharashtra", "karnataka",
    "california", "texas", "delhi", "bengal", "york", "tamil", "nadu", "kerala",
    "rajasthan", "gujarat", "punjab", "pradesh", "uttar", "madhya", "andhra", "telangana",
    "no", "nr", "null", "none"
}

LEGAL_SET: Set[str] = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited", "pvt", "private",
    "co", "company", "ent", "enterprises", "llc", "llp", "tech", "technologies",
    "svc", "services", "ind", "industries", "mfg", "pc", "dds", "md", "pllc"
}

RE_LEADING_NOISE = re.compile(r"^[\*\#\@\-\_\s]+")
RE_DOMAIN_PREFIX = re.compile(r"^(?:https?:\/\/)?(?:www\.)?")
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:c0m|com|org|net|in|co|io|biz|info|gov)(?:\.in)?\b", re.IGNORECASE)
RE_DOMAIN_UNSPACED = re.compile(r"(?:c0m|com|org|net|biz|info)$", re.IGNORECASE)
RE_PLOT_NUM = re.compile(r"\b(?:plot|khewat|kh|h\.?no|door|shop|no\.?)\s*[-:]?\s*([a-z0-9\-\/]+)\b", re.IGNORECASE)


def clean_domain_and_symbols(raw: str) -> str:
    """Strip website prefixes, noisy punctuation and domain extensions."""
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.lower().strip()
    s = RE_LEADING_NOISE.sub("", s)
    s = RE_DOMAIN_PREFIX.sub("", s)
    s = RE_DOMAIN_SUFFIX.sub("", s)
    s = RE_DOMAIN_UNSPACED.sub("", s)
    return s


def extract_blocking_keys(
    name: str,
    address: str,
    country: str,
) -> Tuple[List[str], List[str]]:
    """
    Extracts blocking keys and rare tokens from a record.

    Returns
    -------
    Tuple[List[str], List[str]]
        (keys_list, rare_tokens_list)
    """
    c_norm = normalize_country(country)
    cleaned_raw_name = clean_domain_and_symbols(name)
    n_clean, n_tokens, n_acr, n_comp = normalize_business_name(cleaned_raw_name)
    a_clean, a_tokens, pin, st_num, lmark = normalize_business_address(address)

    keys: List[str] = []

    # 1. Exact normalized name
    if n_clean:
        keys.append(f"nm_{c_norm}_{n_clean}")

    # 2. Compressed name
    if len(n_comp) >= 3:
        keys.append(f"cp_{c_norm}_{n_comp}")

    # 3. Core tokens (excluding legal suffix words and stopwords)
    core_tokens = [
        t for t in n_tokens
        if t not in LEGAL_SET and t not in ("and", "&", "the", "of", "in", "for", "to") and len(t) >= 2
    ]
    if core_tokens:
        core_comp = "".join(core_tokens)
        if len(core_comp) >= 3:
            keys.append(f"cr_{c_norm}_{core_comp}")
        sorted_comp = "".join(sorted(core_tokens))
        if len(sorted_comp) >= 3:
            keys.append(f"sc_{c_norm}_{sorted_comp}")

    # 4. First 2 tokens
    if len(n_tokens) >= 2:
        keys.append(f"f2_{c_norm}_{n_tokens[0]}_{n_tokens[1]}")

    # 5. First token (if informative, len >= 3)
    if n_tokens and len(n_tokens[0]) >= 3 and n_tokens[0] not in COMMON_NAME_STOPWORDS and n_tokens[0] not in LEGAL_SET:
        keys.append(f"f1_{c_norm}_{n_tokens[0]}")

    # 6. Postal code + first token prefix
    if pin and n_tokens and len(n_tokens[0]) >= 2:
        keys.append(f"pin_{c_norm}_{pin}_{n_tokens[0][:3]}")

    # 7. Street / Plot number + road / address token
    rare_addr_tokens = [t for t in a_tokens if len(t) >= 4 and t not in ADDR_STOPWORDS and not t.isdigit()]

    if st_num and rare_addr_tokens:
        keys.append(f"st_addr_{c_norm}_{st_num}_{rare_addr_tokens[0]}")
        if len(rare_addr_tokens) >= 2:
            keys.append(f"st_addr_{c_norm}_{st_num}_{rare_addr_tokens[1]}")

    if st_num and a_tokens:
        road_tokens = [t for t in a_tokens if t != st_num and len(t) >= 3 and t not in ADDR_STOPWORDS]
        if road_tokens:
            keys.append(f"st_rd_{c_norm}_{st_num}_{road_tokens[0][:4]}")

    if st_num and len(n_comp) >= 3:
        keys.append(f"st_nm_{c_norm}_{st_num}_{n_comp[:3]}")

    # Check for plot / property number
    plot_match = RE_PLOT_NUM.search(address or "")
    if plot_match:
        plot_val = plot_match.group(1).lower().replace(" ", "")
        if rare_addr_tokens:
            keys.append(f"plt_{c_norm}_{plot_val}_{rare_addr_tokens[0]}")

    # Extract rare name tokens for secondary inverted index lookup
    rare_name_tokens = [
        t for t in n_tokens
        if len(t) >= 4 and t not in COMMON_NAME_STOPWORDS and t not in LEGAL_SET
    ]

    return keys, rare_name_tokens


class CandidateBlocker:
    """
    Multi-stage candidate indexer and generator.
    """

    def __init__(self, max_candidates_per_key: int = 500, max_candidates_per_entity: int = 40):
        self.max_candidates_per_key = max_candidates_per_key
        self.max_candidates_per_entity = max_candidates_per_entity
        # key -> list of candidate entity IDs
        self.key_index: Dict[str, List[str]] = collections.defaultdict(list)
        # token -> list of candidate entity IDs
        self.token_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.total_indexed_records = 0

    def index_target_records(self, records_iter: Iterable[Dict[str, str]]) -> None:
        """
        Indexes Source 2 and Source 3 records into inverted posting lists.
        """
        for r in records_iter:
            eid = r["entity_id"]
            name = r.get("business_name") or ""
            addr = r.get("business_address") or ""
            country = r.get("country") or ""

            keys, rare_toks = extract_blocking_keys(name, addr, country)
            for k in keys:
                self.key_index[k].append(eid)

            c_norm = normalize_country(country)
            for t in rare_toks:
                self.token_index[f"tok_{c_norm}_{t}"].append(eid)

            self.total_indexed_records += 1

    def retrieve_candidates_for_entity(
        self,
        name: str,
        address: str,
        country: str,
    ) -> List[str]:
        """
        Retrieves top candidate IDs for a single Source 1 entity.
        """
        keys, rare_toks = extract_blocking_keys(name, address, country)
        c_norm = normalize_country(country)

        candidate_scores: Dict[str, int] = collections.defaultdict(int)

        # 1. Primary keys
        for k in keys:
            postings = self.key_index.get(k)
            if postings and len(postings) <= self.max_candidates_per_key:
                weight = 3 if k.startswith(("nm_", "cp_", "cr_", "sc_")) else 2
                for cid in postings:
                    candidate_scores[cid] += weight

        # 2. Rare token overlap
        for t in rare_toks:
            tok_key = f"tok_{c_norm}_{t}"
            postings = self.token_index.get(tok_key)
            if postings and len(postings) <= self.max_candidates_per_key:
                for cid in postings:
                    candidate_scores[cid] += 1

        if not candidate_scores:
            return []

        # Sort by candidate score (highest overlap first) and truncate to top-k
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return [cid for cid, _ in sorted_cands[: self.max_candidates_per_entity]]
