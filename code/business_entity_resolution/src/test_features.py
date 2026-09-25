"""
Unit tests for features.py.
"""

from features import FEATURE_NAMES, RecordRepresentation, compute_pair_features


def test_compute_pair_features():
    s1 = RecordRepresentation(
        entity_id="S1-001",
        name="International Business Machines Corp",
        address="1 New Orchard Road, Armonk, NY 10504",
        country="US",
    )
    cand_match = RecordRepresentation(
        entity_id="S2-001",
        name="IBM Corporation",
        address="1 New Orchard Rd, Armonk, New York 10504",
        country="US",
    )
    cand_diff = RecordRepresentation(
        entity_id="S3-002",
        name="Acme Bakery LLC",
        address="99 Main St, Austin, TX",
        country="US",
    )

    f_match = compute_pair_features(s1, cand_match, cand_rank=0, blocking_score=10.0)
    f_diff = compute_pair_features(s1, cand_diff, cand_rank=5, blocking_score=1.0)

    assert len(f_match) == len(FEATURE_NAMES)
    assert len(f_diff) == len(FEATURE_NAMES)

    # IBM acronym check
    assert f_match[FEATURE_NAMES.index("name_acronym_match")] == 1.0

    # Pin match check
    assert f_match[FEATURE_NAMES.index("pin_match")] == 1.0
    assert f_match[FEATURE_NAMES.index("st_num_match")] == 1.0

    # Similarity should be much higher for match
    assert f_match[FEATURE_NAMES.index("name_jw")] > f_diff[FEATURE_NAMES.index("name_jw")]
    assert f_match[FEATURE_NAMES.index("addr_jw")] > f_diff[FEATURE_NAMES.index("addr_jw")]
