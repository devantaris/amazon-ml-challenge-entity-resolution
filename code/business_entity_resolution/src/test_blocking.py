"""
Unit tests for blocking.py.
"""

from blocking import CandidateBlocker, extract_blocking_keys


def test_extract_blocking_keys():
    keys, rare_toks = extract_blocking_keys(
        name="Maure Williams Colombier Inc",
        address="85 Wayne Avenue, Ticonderoga, NY",
        country="US",
    )
    assert any(k.startswith("nm_US_maure williams colombier") for k in keys)
    assert any(k.startswith("cp_US_maurewilliamscolombier") for k in keys)
    assert any(k.startswith("cr_US_maurewilliamscolombier") for k in keys)
    assert "maure" in rare_toks or "williams" in rare_toks


def test_blocker_retrieval():
    blocker = CandidateBlocker(max_candidates_per_key=100, max_candidates_per_entity=10)

    # Index targets
    s2_records = [
        {"entity_id": "S2-001", "business_name": "Maure Wilblims Colombier Inc", "business_address": "", "country": "US"},
        {"entity_id": "S2-002", "business_name": "Something Completely Different", "business_address": "123 Elm St", "country": "US"},
    ]
    s3_records = [
        {"entity_id": "S3-001", "business_name": "maurewilliamscolombier.com", "business_address": "85 Wayne Ave, Ticonderoga", "country": "US"},
    ]
    blocker.index_target_records(s2_records + s3_records)

    # Query with S1
    cands = blocker.retrieve_candidates_for_entity(
        name="Maure Williams Colombier Inc",
        address="85 Wayne Avenue, Ticonderoga, NY",
        country="US",
    )

    assert "S3-001" in cands
    assert "S2-001" in cands
    assert "S2-002" not in cands or cands.index("S2-002") > cands.index("S3-001")
