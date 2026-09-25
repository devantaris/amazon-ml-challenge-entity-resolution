"""
Generalization Tests for Unseen Countries (e.g., France).

Verifies Phase 6 requirements:
- Pipeline does not hardcode country ∈ {US, India}.
- Unseen country labels (like "France", "FR", "Germany") are preserved and normalized gracefully.
- Candidate blocking and pairwise feature extraction function properly for France records.
"""

from normalize import normalize_country
from blocking import CandidateBlocker, extract_blocking_keys
from features import RecordRepresentation, compute_pair_features


def test_france_country_normalization():
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("FR") == "FRANCE"
    assert normalize_country("FRA") == "FRANCE"
    assert normalize_country("france") == "FRANCE"
    assert normalize_country("Spain") == "SPAIN"


def test_france_blocking_and_matching():
    # French entity: L'Oreal Paris
    s1_name = "L'Oreal Produits De Beaute France SAS"
    s1_addr = "14 Rue Royale, 75008 Paris"
    s1_country = "France"

    keys, rare_toks = extract_blocking_keys(s1_name, s1_addr, s1_country)
    assert any("FRANCE" in k for k in keys)

    # Setup blocker
    blocker = CandidateBlocker(max_candidates_per_key=100, max_candidates_per_entity=10)
    s2_record = {
        "entity_id": "S2-FR01",
        "business_name": "L'Oreal Beaute France",
        "business_address": "14 Rue Royale, Paris",
        "country": "France",
    }
    s3_record = {
        "entity_id": "S3-FR02",
        "business_name": "lorealfrance.fr",
        "business_address": "14 Rue Royale, 75008 Paris",
        "country": "FR",
    }
    blocker.index_target_records([s2_record, s3_record])

    cands = blocker.retrieve_candidates_for_entity(s1_name, s1_addr, s1_country)
    assert "S2-FR01" in cands or "S3-FR02" in cands

    # Feature extraction
    s1_rep = RecordRepresentation("S1-FR01", s1_name, s1_addr, s1_country)
    cand_rep = RecordRepresentation("S2-FR01", s2_record["business_name"], s2_record["business_address"], s2_record["country"])

    feats = compute_pair_features(s1_rep, cand_rep)
    assert feats is not None
    assert len(feats) > 0
    # Jaro-Winkler should be high
    assert feats[0] > 0.70
