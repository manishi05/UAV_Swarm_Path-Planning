"""
env/rewards.py
==============
Reward shaping module.

Centralises ALL reward computation for the UAV swarm environment.
No reward logic lives in the environment's step() — it delegates
entirely to RewardCalculator. This makes ablation studies trivial:
zero out any RewardConfig term and re-run.

Reward components (all configurable via RewardConfig):
    1. r_new_cell          — coverage progress reward
    2. r_revisit           — revisit penalty (exploration incentive)
    3. r_static_collision  — static obstacle collision penalty
    4. r_dynamic_collision — dynamic obstacle collision penalty
    5. r_uav_collision     — inter-UAV collision penalty
    6. r_out_of_bounds     — boundary violation penalty
    7. r_step              — per-step time pressure penalty
    8. r_completion        — episode completion bonus
    9. r_wind_drift        — wind-induced drift penalty

Architecture note:
    RewardCalculator is stateless — it only uses inputs passed to
    compute() and the config weights. This makes it thread-safe and
    trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple

from uav_swarm_ppo.configs.config import RewardConfig


@dataclass
class StepRewardInfo:
    """
    Structured breakdown of reward components for one UAV in one step.

    Returned by RewardCalculator.compute() so that logging can record
    each reward term individually for analysis and paper figures.

    Attributes
    ----------
    total : float
        Sum of all active reward components.
    new_cell : float
        Coverage reward (positive).
    revisit : float
        Revisit penalty (negative or zero).
    static_collision : float
        Static obstacle collision penalty (negative or zero).
    dynamic_collision : float
        Dynamic obstacle collision penalty (negative or zero).
    uav_collision : float
        Inter-UAV collision penalty (negative or zero).
    out_of_bounds : float
        Boundary violation penalty (negative or zero).
    step_penalty : float
        Per-step time cost (always negative).
    completion_bonus : float
        Episode completion bonus (positive, one-time).
    wind_penalty : float
        Wind drift penalty (negative or zero).
    """

    total: float = 0.0
    new_cell: float = 0.0
    revisit: float = 0.0
    static_collision: float = 0.0
    dynamic_collision: float = 0.0
    uav_collision: float = 0.0
    out_of_bounds: float = 0.0
    step_penalty: float = 0.0
    completion_bonus: float = 0.0
    wind_penalty: float = 0.0


class RewardCalculator:
    """
    Computes shaped rewards for each UAV at each environment step.

    Parameters
    ----------
    config : RewardConfig
        Reward weights loaded from the experiment configuration.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.cfg = config

    def compute(
        self,
        uav_id: int,
        intended_pos: Tuple[int, int],
        actual_pos: Tuple[int, int],
        cell_was_new: bool,
        hit_static: bool,
        hit_dynamic: bool,
        hit_uav: bool,
        went_out_of_bounds: bool,
        wind_caused_drift: bool,
        episode_complete: bool,
    ) -> StepRewardInfo:
        """
        Compute the full shaped reward for one UAV in one timestep.

        Parameters
        ----------
        uav_id : int
            Which UAV (used for logging — not needed for computation).
        intended_pos : Tuple[int, int]
            The (row, col) the UAV was trying to reach.
        actual_pos : Tuple[int, int]
            The (row, col) the UAV actually ended up at (may differ
            from intended if out-of-bounds was rejected).
        cell_was_new : bool
            True if actual_pos was previously uncovered.
        hit_static : bool
            True if the intended move collided with a static obstacle.
        hit_dynamic : bool
            True if the UAV's actual position is occupied by a dynamic obstacle.
        hit_uav : bool
            True if another UAV occupies the same cell after movement.
        went_out_of_bounds : bool
            True if the intended position was outside the field polygon.
        wind_caused_drift : bool
            True if wind pushed the UAV away from its intended action.
        episode_complete : bool
            True if coverage_target was reached this step.

        Returns
        -------
        StepRewardInfo
            Structured reward breakdown with `.total` summing all terms.
        """
        info = StepRewardInfo()

        # -- 1. Coverage reward -----------------------------------------
        if cell_was_new:
            info.new_cell = self.cfg.r_new_cell
        else:
            # Penalise revisiting already-covered cells
            info.revisit = self.cfg.r_revisit

        # -- 2. Collision penalties -------------------------------------
        if hit_static:
            info.static_collision = self.cfg.r_static_collision

        if hit_dynamic:
            info.dynamic_collision = self.cfg.r_dynamic_collision

        if hit_uav:
            info.uav_collision = self.cfg.r_uav_collision

        # -- 3. Boundary violation -------------------------------------
        if went_out_of_bounds:
            info.out_of_bounds = self.cfg.r_out_of_bounds

        # -- 4. Wind penalty -------------------------------------------
        if wind_caused_drift:
            info.wind_penalty = self.cfg.r_wind_drift

        # -- 5. Per-step time cost -------------------------------------
        # Always applied — encourages efficiency
        info.step_penalty = self.cfg.r_step

        # -- 6. Completion bonus ---------------------------------------
        # Only awarded once (the environment sets this flag once)
        if episode_complete:
            info.completion_bonus = self.cfg.r_completion

        # -- 7. Sum all components ------------------------------------
        info.total = (
            info.new_cell
            + info.revisit
            + info.static_collision
            + info.dynamic_collision
            + info.uav_collision
            + info.out_of_bounds
            + info.step_penalty
            + info.completion_bonus
            + info.wind_penalty
        )

        return info
