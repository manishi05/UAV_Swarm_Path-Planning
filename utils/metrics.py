"""
utils/metrics.py
=================
Performance metrics shared by the environment, and later by the PPO
evaluation script and baseline comparison harness (Day 3-4). Kept as
free functions (not tied to any specific env instance) so they can be
applied identically to PPO, Q-Learning, and A* rollout logs -- required
for a fair, reviewer-defensible comparison (Goal 4).

Metrics implemented here correspond directly to Section 3.5 of the
reference paper ("Performance measures"): coverage fraction and the
fraction of valid actions (PA, Eq. 4). Collision-rate metrics are our
Goal-2 extension, since the reference paper has no obstacle/collision
model to measure.
"""

from typing import Dict, List

import numpy as np


def coverage_fraction(visited_mask: np.ndarray, visitable_mask: np.ndarray) -> float:
    """
    Fraction of visitable cells that have been visited at least once.

    Corresponds to the reference paper's map-coverage measure
    (Section 3.5): "the fraction of cells that have been visited
    divided by the total number of cells" -- restricted here to
    visitable cells specifically, since non-visitable cells can never
    contribute to a meaningful coverage score.
    """
    n_visitable = visitable_mask.sum()
    if n_visitable == 0:
        return 0.0
    n_visited = np.logical_and(visited_mask, visitable_mask).sum()
    return float(n_visited / n_visitable)


def valid_action_fraction(valid_actions: int, total_actions: int) -> float:
    """
    Fraction of valid actions (PA, Eq. 4 in the reference paper): the
    ratio of actions that discovered a new visitable cell (without
    looping or entering non-visitable cells) to the total number of
    actions taken. Closer to 1.0 is better (fewer wasted moves).
    """
    if total_actions == 0:
        return 0.0
    return float(valid_actions / total_actions)


def collision_rate(collision_counts: Dict[str, int], total_steps: int) -> Dict[str, float]:
    """
    Normalizes raw per-episode collision counters (by type: UAV-UAV,
    UAV-static-obstacle, UAV-dynamic-obstacle, UAV-boundary) into a
    per-step rate, so episodes of different lengths remain comparable
    across PPO/Q-Learning/A* and across map sizes.

    Parameters
    ----------
    collision_counts : Dict[str, int]
        Mapping from collision-type name to raw event count for one episode.
    total_steps : int
        Number of environment steps taken in that episode.

    Returns
    -------
    Dict[str, float]
        Same keys, values divided by total_steps (0.0 if total_steps == 0).
    """
    if total_steps == 0:
        return {k: 0.0 for k in collision_counts}
    return {k: v / total_steps for k, v in collision_counts.items()}


def summarize_runs(values: List[float]) -> Dict[str, float]:
    """
    Computes mean/std/min/max across multiple independent runs (multiple
    seeds), which is the minimum statistical summary needed before
    running Shapiro-Wilk/ANOVA significance tests (Goal 4, matching the
    reference paper's Section 4 methodology in Tables 5-6).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
