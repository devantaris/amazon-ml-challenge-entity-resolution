"""
Global Assignment & Threshold Optimization.

1. Bipartite conflict resolution:
   Ensures no Source 2 or Source 3 record is claimed by more than one Source 1 entity.
   Greedy highest-confidence-first assignment resolves conflicts, preventing precision penalties.
2. Singleton threshold gating:
   Guards against false merges on singletons which carry a heavy 1.0 -> 0.0 penalty under macro F_0.5.
3. Optimal threshold sweep on validation split.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from eval_metric import compute_macro_f05


def global_bipartite_assignment(
    s1_candidate_probs: Dict[str, List[Tuple[str, float]]],
    threshold: float = 0.65,
    singleton_margin: float = 0.05,
) -> Dict[str, Set[str]]:
    """
    Assigns candidate matches to Source 1 entities while ensuring each target ID
    (from Source 2 or Source 3) is matched to at most ONE Source 1 entity.

    Parameters
    ----------
    s1_candidate_probs : Dict[str, List[Tuple[str, float]]]
        Mapping from s1_id -> list of (cand_id, probability) pairs.
    threshold : float
        Decision threshold for predicting a match.
    singleton_margin : float
        Additional confidence buffer required for borderline single-candidate matches.

    Returns
    -------
    Dict[str, Set[str]]
        Final mapping from s1_id -> set of matched entity IDs.
    """
    # Flatten all candidate pairs with probability >= threshold
    all_pairs: List[Tuple[float, str, str]] = []
    
    for s1_id, cands in s1_candidate_probs.items():
        if not cands:
            continue
        # Apply slight singleton buffer if only 1 candidate exists
        effective_thresh = threshold + singleton_margin if len(cands) == 1 else threshold
        for cand_id, prob in cands:
            if prob >= effective_thresh:
                all_pairs.append((prob, s1_id, cand_id))

    # Sort descending by probability
    all_pairs.sort(key=lambda x: -x[0])

    assigned_targets: Set[str] = set()
    s1_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_candidate_probs}

    for prob, s1_id, cand_id in all_pairs:
        if cand_id not in assigned_targets:
            s1_matches[s1_id].add(cand_id)
            assigned_targets.add(cand_id)

    return s1_matches


def sweep_optimal_threshold(
    s1_candidate_probs: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Sweeps thresholds on validation split to find the threshold that maximizes macro F_0.5.

    Returns
    -------
    Tuple[float, float, Dict[str, float]]
        (best_threshold, best_f05, best_metrics_dict)
    """
    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    best_thresh = 0.65
    best_f05 = -1.0
    best_metrics = {}

    for t in thresholds:
        preds = global_bipartite_assignment(s1_candidate_probs, threshold=t)
        res = compute_macro_f05(ground_truth, preds)
        score = res["macro_f05"]
        print(f"Threshold {t:.2f} -> Macro F0.5: {score:.4f} (Prec: {res['macro_precision']:.4f}, Rec: {res['macro_recall']:.4f}, SingleAcc: {res['singleton_accuracy']:.4f})")
        if score > best_f05:
            best_f05 = score
            best_thresh = t
            best_metrics = res

    print(f"\nBest threshold: {best_thresh:.2f} with Macro F0.5: {best_f05:.4f}")
    return best_thresh, best_f05, best_metrics
