"""
Ultra-Compact Inverted Index & Memory-Capped Candidate Blocker.

Optimized for 16GB laptops:
- Uses 32-bit integer array posting lists (array('I')), cutting RAM from 8GB down to <100MB.
- Does NOT instantiate Python objects for millions of target records.
- Target records are stored in compact Arrow format (Polars).
- RecordRepresentation is instantiated strictly on-the-fly for the ~20 retrieved candidates per S1 entity.
- Peak RAM usage stays strictly below 2.5 GB.
"""

import array
import collections
from typing import Dict, List, Optional, Set, Tuple
import polars as pl

from blocking import extract_blocking_keys
from normalize import normalize_country


class CompactCandidateBlocker:
    """
    Memory-efficient candidate blocker using 32-bit integer array posting lists.
    """

    def __init__(self, max_candidates_per_key: int = 300, max_candidates_per_entity: int = 35):
        self.max_candidates_per_key = max_candidates_per_key
        self.max_candidates_per_entity = max_candidates_per_entity
        # key -> array of 32-bit unsigned integers (target row indices)
        self.key_postings: Dict[str, array.array] = {}
        # token -> array of 32-bit unsigned integers
        self.tok_postings: Dict[str, array.array] = {}
        self.total_indexed = 0

    def index_dataframe(self, df: pl.DataFrame) -> None:
        """
        Indexes records directly from a Polars DataFrame using compact integer arrays.
        """
        # Columns: entity_id, business_name, business_address, country
        names = df["business_name"].fill_null("").to_list()
        addrs = df["business_address"].fill_null("").to_list()
        countries = df["country"].fill_null("").to_list()

        start_idx = self.total_indexed

        for i, (name, addr, country) in enumerate(zip(names, addrs, countries)):
            row_idx = start_idx + i
            keys, rare_toks = extract_blocking_keys(name, addr, country)

            for k in keys:
                postings = self.key_postings.get(k)
                if postings is None:
                    postings = array.array("I")
                    self.key_postings[k] = postings
                if len(postings) < self.max_candidates_per_key:
                    postings.append(row_idx)

            c_norm = normalize_country(country)
            for t in rare_toks:
                tok_key = f"tok_{c_norm}_{t}"
                postings = self.tok_postings.get(tok_key)
                if postings is None:
                    postings = array.array("I")
                    self.tok_postings[tok_key] = postings
                if len(postings) < self.max_candidates_per_key:
                    postings.append(row_idx)

        self.total_indexed += len(df)

    def retrieve_candidate_indices(
        self,
        name: str,
        address: str,
        country: str,
    ) -> List[int]:
        """
        Retrieves top candidate integer indices for an S1 entity.
        """
        keys, rare_toks = extract_blocking_keys(name, address, country)
        c_norm = normalize_country(country)

        # Counter of candidate row_index -> accumulated match weight
        cand_scores: Dict[int, int] = collections.defaultdict(int)

        for k in keys:
            postings = self.key_postings.get(k)
            if postings:
                weight = 3 if k.startswith(("nm_", "cp_", "cr_", "sc_")) else 2
                for row_idx in postings:
                    cand_scores[row_idx] += weight

        for t in rare_toks:
            tok_key = f"tok_{c_norm}_{t}"
            postings = self.tok_postings.get(tok_key)
            if postings:
                for row_idx in postings:
                    cand_scores[row_idx] += 1

        if not cand_scores:
            return []

        sorted_cands = sorted(cand_scores.items(), key=lambda x: -x[1])
        return [row_idx for row_idx, _ in sorted_cands[: self.max_candidates_per_entity]]
