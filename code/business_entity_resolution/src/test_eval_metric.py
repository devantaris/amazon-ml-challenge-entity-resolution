"""
Unit tests for eval_metric.py.
Verifies against official worked example and corner cases.
"""

import pytest
from eval_metric import compute_entity_f05, compute_macro_f05, parse_id_string


def test_worked_example():
    """
    Test the worked example from student_resource/README.md:
    - true: S2-00047, S3-00812 (2 items)
    - pred: S2-00047, S2-00193, S3-00812 (3 items, 2 TP)
    - Precision: 2/3
    - Recall: 2/2 = 1.0
    - F_0.5 = (1.25 * (2/3) * 1) / (0.25 * (2/3) + 1) = (5/6) / (7/6) = 5/7 ≈ 0.7142857
    """
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    f05, prec, rec = compute_entity_f05(true_set, pred_set)

    assert pytest.approx(prec, 1e-4) == 2 / 3
    assert pytest.approx(rec, 1e-4) == 1.0
    assert pytest.approx(f05, 1e-3) == 0.714
    assert pytest.approx(f05, 1e-7) == 5 / 7


def test_singleton_correct():
    """Singleton with correctly predicted empty list -> score 1.0"""
    f05, prec, rec = compute_entity_f05(set(), set())
    assert f05 == 1.0
    assert prec == 1.0


def test_singleton_false_merge():
    """Singleton with predicted match -> false merge penalty -> score 0.0"""
    f05, prec, rec = compute_entity_f05(set(), {"S2-00001"})
    assert f05 == 0.0
    assert prec == 0.0


def test_matched_entity_empty_prediction():
    """Entity with true matches but model predicted none -> score 0.0"""
    f05, prec, rec = compute_entity_f05({"S2-00001"}, set())
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_matched_entity_wrong_prediction():
    """Entity with true match S2-00001 but model predicted S2-00002 -> score 0.0"""
    f05, prec, rec = compute_entity_f05({"S2-00001"}, {"S2-00002"})
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_macro_average():
    """Test macro average across 3 entities: worked example, correct singleton, false singleton."""
    gt = {
        "S1-00001": {"S2-00047", "S3-00812"},
        "S1-00002": set(),
        "S1-00003": set(),
    }
    preds = {
        "S1-00001": {"S2-00047", "S2-00193", "S3-00812"}, # 5/7
        "S1-00002": set(),                                 # 1.0
        "S1-00003": {"S3-00001"},                          # 0.0
    }
    res = compute_macro_f05(gt, preds)
    expected_macro = (5 / 7 + 1.0 + 0.0) / 3.0
    assert pytest.approx(res["macro_f05"], 1e-6) == expected_macro
    assert res["num_entities"] == 3
    assert res["num_singletons"] == 2
    assert res["singleton_accuracy"] == 0.5


def test_parse_id_string():
    assert parse_id_string("") == set()
    assert parse_id_string("   ") == set()
    assert parse_id_string("S2-001, S3-002,") == {"S2-001", "S3-002"}
