"""
Unit tests for assign.py.
"""

from assign import global_bipartite_assignment


def test_bipartite_conflict_resolution():
    # S1-A and S1-B both have S2-99 as candidate
    # S1-A confidence 0.90, S1-B confidence 0.75
    # S2-99 must only be assigned to S1-A!
    s1_probs = {
        "S1-A": [("S2-99", 0.90), ("S3-01", 0.85)],
        "S1-B": [("S2-99", 0.75), ("S3-02", 0.80)],
        "S1-C": [("S2-99", 0.60)], # below or lost conflict
    }

    assigned = global_bipartite_assignment(s1_probs, threshold=0.70)

    # S1-A gets S2-99 and S3-01
    assert "S2-99" in assigned["S1-A"]
    assert "S3-01" in assigned["S1-A"]

    # S1-B does NOT get S2-99, but gets S3-02
    assert "S2-99" not in assigned["S1-B"]
    assert "S3-02" in assigned["S1-B"]

    # S1-C gets nothing
    assert len(assigned["S1-C"]) == 0
