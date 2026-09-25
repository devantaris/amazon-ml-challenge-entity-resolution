"""
Evaluation Metric: Macro-averaged F_0.5 score for Business Entity Resolution.

As specified by the ML Challenge 2026 guidelines:
- F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Macro-averaged across all Source 1 entities in the evaluation set.
- Singletons (entities with 0 true matches):
    - Correctly predicting no matches (empty list) scores 1.0
    - Predicting any match scores 0.0
- Entities with true matches:
    - Predicting no matches scores 0.0
    - If True Positives (TP) == 0, score is 0.0
    - Otherwise F_0.5 is calculated using per-entity Precision and Recall.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def compute_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str],
) -> Tuple[float, float, float]:
    """
    Computes (F_0.5, precision, recall) for a single Source 1 entity.

    Parameters
    ----------
    true_matches : Set[str]
        Set of ground-truth entity IDs (from Source 2 / Source 3).
    pred_matches : Set[str]
        Set of predicted entity IDs (from Source 2 / Source 3).

    Returns
    -------
    Tuple[float, float, float]
        (f05, precision, recall)
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    if n_true == 0:
        # Ground truth is a singleton (no matches)
        if n_pred == 0:
            return 1.0, 1.0, 1.0
        else:
            # False merge on a singleton
            return 0.0, 0.0, 1.0

    # Entity has ground-truth matches
    if n_pred == 0:
        # Missed all matches
        return 0.0, 0.0, 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return f05, precision, recall


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    eval_ids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5 across all evaluated Source 1 entities.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Mapping: source1_entity_id -> set of true matching IDs
    predictions : Dict[str, Set[str]]
        Mapping: source1_entity_id -> set of predicted matching IDs
    eval_ids : Optional[Iterable[str]]
        Specific list/set of Source 1 IDs to evaluate over.
        Defaults to all keys in `ground_truth`.

    Returns
    -------
    Dict[str, float]
        {
            "macro_f05": ...,
            "macro_precision": ...,
            "macro_recall": ...,
            "num_entities": ...,
            "num_singletons": ...,
            "singleton_accuracy": ...,
            "non_singleton_f05": ...
        }
    """
    if eval_ids is None:
        target_ids = list(ground_truth.keys())
    else:
        target_ids = list(eval_ids)

    n_entities = len(target_ids)
    if n_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "num_entities": 0,
            "num_singletons": 0,
            "singleton_accuracy": 0.0,
            "non_singleton_f05": 0.0,
        }

    f05_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []

    singleton_scores: List[float] = []
    non_singleton_scores: List[float] = []

    empty_set = frozenset()

    for s1_id in target_ids:
        true_set = ground_truth.get(s1_id, empty_set)
        pred_set = predictions.get(s1_id, empty_set)

        f05, prec, rec = compute_entity_f05(true_set, pred_set)
        f05_scores.append(f05)
        precision_scores.append(prec)
        recall_scores.append(rec)

        if len(true_set) == 0:
            singleton_scores.append(f05)
        else:
            non_singleton_scores.append(f05)

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
        "num_entities": n_entities,
        "num_singletons": len(singleton_scores),
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
    }


def parse_id_string(s: Union[str, float]) -> Set[str]:
    """Parse comma-separated entity IDs string into a set."""
    if not isinstance(s, str) or not s.strip():
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def load_ground_truth_dict(tsv_path: str) -> Dict[str, Set[str]]:
    """Load ground truth TSV into a {source1_id: set_of_ids} dictionary."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    # Expected columns: source1_entity_id, matched_entity_ids
    s1_col = "source1_entity_id"
    match_col = "matched_entity_ids"
    if s1_col not in df.columns or match_col not in df.columns:
        raise ValueError(f"TSV {tsv_path} missing required columns {s1_col}, {match_col}")

    return {
        row[s1_col]: parse_id_string(row[match_col])
        for row in df.itertuples(index=False)
    }


def load_prediction_dict(tsv_path: str) -> Dict[str, Set[str]]:
    """Load matching_results.tsv into a {source1_id: set_of_ids} dictionary."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    s1_col = "source1_entity_id"
    match_col = "matched_entity_ids"
    if s1_col not in df.columns or match_col not in df.columns:
        raise ValueError(f"TSV {tsv_path} missing required columns {s1_col}, {match_col}")

    return {
        row[s1_col]: parse_id_string(row[match_col])
        for row in df.itertuples(index=False)
    }
