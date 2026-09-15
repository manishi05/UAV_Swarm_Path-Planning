"""
environment/reward.py
======================
Reward computation for Goal 3. Returns a structured `RewardBreakdown`
(not a bare scalar) so every one of the 8 required reward categories
-- Exploration, Coverage, Collision avoidance, Boundary violations,
Dynamic obstacle avoidance, UAV separation, Mission completion, Energy
efficiency -- is independently visible, loggable, and tunable.

The "Coverage" component reproduces Table 2 / Eq. 2 of the reference
paper exactly. Every other component is our extension -- the reference
paper's reward has no notion of safety, exploration shaping, mission
completion, or energy cost at all. See REWARD_ENGINEERING.md for full
design rationale, constant sourcing, and tuning guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional

from configs.config import RewardConfig
from environment.collision import CollisionType


@dataclass
class RewardBreakdown:
    """
    One UAV's (or, once summed, the whole swarm's) reward for one
    timestep, split into the 8 Goal-3 categories. Kept as a dataclass
    rather than a dict so category names are enforced at the type level
    (a typo in a dict key would silently create a new untracked bucket).

    `__add__` is implemented so per-UAV breakdowns can be summed into a
    swarm-level breakdown with `sum(breakdowns, RewardBreakdown())`,
    which is what `swarm_env.py` does each step before logging.
    """
    exploration: float = 0.0
    coverage: float = 0.0
    collision_avoidance: float = 0.0
    boundary: float = 0.0
    dynamic_obstacle_avoidance: float = 0.0
    uav_separation: float = 0.0
    mission_completion: float = 0.0
    energy_efficiency: float = 0.0

    @property
    def total(self) -> float:
        """Scalar sum of all categories -- this is what Gymnasium's step() returns as the reward."""
        return sum(getattr(self, f.name) for f in fields(self))

    def as_dict(self) -> Dict[str, float]:
        """Category name -> value, plus 'total'. Used directly by the episode logger."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["total"] = self.total
        return d

    def __add__(self, other: "RewardBreakdown") -> "RewardBreakdown":
        if not isinstance(other, RewardBreakdown):
            return NotImplemented
        return RewardBreakdown(**{
            f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)
        })

    def __radd__(self, other):
        # Enables sum([breakdown1, breakdown2, ...]) starting from int 0.
        if other == 0:
            return self
        return self.__add__(other)


class RewardCalculator:
    """
    Computes a `RewardBreakdown` for a single UAV's outcome at one
    timestep. `swarm_env.py` calls this once per alive UAV per step and
    sums the results (via `sum(breakdowns, RewardBreakdown())`) into a
    single swarm-level breakdown, whose `.total` is the scalar reward
    required by the global-policy PPO setup -- matching the reference
    paper's own finding (Section 4.2) that a single global controller
    for the whole swarm outperforms one network per UAV.
    """

    def __init__(self, config: RewardConfig):
        self._config = config

    def compute(
        self,
        entered_new_cell: bool,
        entered_visited_cell: bool,
        entered_boundary_violation: bool,
        entered_obstacle_zone: bool,
        n_visited_cells: int,
        grid_rows: int,
        grid_cols: int,
        local_novelty: float,
        collisions: List[CollisionType],
        nearest_dynamic_obstacle_dist: Optional[float],
        nearest_uav_dist: Optional[float],
        disturbance_magnitude: float,
        dt: float,
    ) -> RewardBreakdown:
        """
        Parameters
        ----------
        entered_new_cell : bool
            UAV's current cell is flyable and was not previously visited.
        entered_visited_cell : bool
            UAV's current cell is flyable and was already visited.
        entered_boundary_violation : bool
            UAV's current cell is outside the field boundary entirely
            (grid-level signal -- cell doesn't exist or `is_visitable`
            is False). Distinct from `CollisionType.UAV_BOUNDARY`, which
            is the continuous-position field-polygon check -- both
            measure the same underlying "outside the field" event at
            different resolutions. As of the R1 intervention, only the
            more severe (more negative) of the two triggered signals is
            applied per step, not their sum (see the boundary block
            below and REWARD_ENGINEERING.md's changelog for why the
            earlier additive behavior double-counted a single event).
        entered_obstacle_zone : bool
            UAV's current cell is flyable-by-boundary but blocked by a
            static obstacle (`blocked_static`); a softer, cell-level
            precursor to the discrete UAV_STATIC_OBSTACLE distance
            collision below.
        n_visited_cells, grid_rows, grid_cols : int
            Feed the reference paper's Eq. 2 scaling term.
        local_novelty : float, range [0, 1]
            1 - (fraction of a local window around the UAV's cell that
            is already visited). Drives the Exploration term: rewards
            moving toward locally under-covered regions, independent of
            whether the exact target cell itself is new.
        collisions : List[CollisionType]
            Discrete collision events attributed to this UAV this step.
        nearest_dynamic_obstacle_dist : Optional[float]
            Distance (m) to the nearest dynamic obstacle, or None if
            there are no dynamic obstacles in the environment.
        nearest_uav_dist : Optional[float]
            Distance (m) to the nearest other alive UAV, or None if this
            is the only alive UAV.
        disturbance_magnitude : float
            Magnitude of wind + wake disturbance this UAV experienced.
        dt : float
            Physics timestep (s), for the baseline energy-efficiency cost.

        Returns
        -------
        RewardBreakdown
        """
        cfg = self._config
        r = RewardBreakdown()

        # ---------------- Coverage (reference paper's Eq. 2) ----------------
        if entered_new_cell:
            scale = 1.0 + max(grid_rows, grid_cols) / max(n_visited_cells, 1)
            r.coverage += cfg.new_cell_base * scale
        elif entered_visited_cell:
            r.coverage += cfg.visited_cell

        # ---------------- Boundary violations ----------------
        # R1 intervention (see REWARD_ENGINEERING.md changelog): non_visitable
        # (grid-cell, coarse, pre-computed) and out_of_bounds (continuous
        # position, exact, live) both measure the same underlying physical
        # fact -- "is this UAV outside the true field boundary right now" --
        # via two different-resolution tests of the same event, not two
        # distinct violation types. Applying both additively double-counts
        # a single violation (confirmed empirically: the two signals
        # co-fire in the large majority of violating steps). Fixed here:
        # apply only the MORE SEVERE (more negative) of whichever signal(s)
        # actually triggered this step -- never both, never unconditionally.
        # Because both values are negative, "more severe" = min(), not max()
        # (max(-225.17, -300.0) == -225.17, the WEAKER penalty -- the wrong
        # direction for a "worse violation should cost more" rule).
        boundary_out_of_bounds = CollisionType.UAV_BOUNDARY in collisions
        if entered_boundary_violation or boundary_out_of_bounds:
            candidates = []
            if entered_boundary_violation:
                candidates.append(cfg.non_visitable)
            if boundary_out_of_bounds:
                candidates.append(cfg.out_of_bounds)
            r.boundary += min(candidates)

        # ---------------- Collision avoidance (obstacle, cell-level precursor) ----------------
        if entered_obstacle_zone:
            r.collision_avoidance += cfg.blocked_cell_penalty

        # ---------------- Exploration (dense, local-novelty shaping) ----------------
        # Only rewarded inside the field -- we don't want the policy
        # learning that flying out of bounds is a good way to find
        # "novel" (never-visited) territory.
        if not entered_boundary_violation:
            r.exploration += cfg.exploration_coeff * local_novelty

        # ---------------- Discrete collision events ----------------
        # Note: UAV_BOUNDARY is deliberately NOT handled in this loop --
        # it is consolidated into the single non-additive boundary block
        # above (R1 intervention) rather than added a second time here.
        for collision in collisions:
            if collision == CollisionType.UAV_STATIC_OBSTACLE:
                r.collision_avoidance += cfg.collision_static_obstacle
            elif collision == CollisionType.UAV_DYNAMIC_OBSTACLE:
                r.dynamic_obstacle_avoidance += cfg.collision_dynamic_obstacle
            elif collision == CollisionType.UAV_UAV:
                r.uav_separation += cfg.collision_uav_uav

        # ---------------- Dynamic obstacle avoidance (dense proximity) ----------------
        if (nearest_dynamic_obstacle_dist is not None
                and nearest_dynamic_obstacle_dist < cfg.dynamic_obstacle_warning_radius_m):
            proximity = (
                (cfg.dynamic_obstacle_warning_radius_m - nearest_dynamic_obstacle_dist)
                / cfg.dynamic_obstacle_warning_radius_m
            )
            r.dynamic_obstacle_avoidance += -cfg.dynamic_obstacle_proximity_coeff * proximity

        # ---------------- UAV separation (dense proximity) ----------------
        if (nearest_uav_dist is not None
                and nearest_uav_dist < cfg.uav_separation_warning_radius_m):
            proximity = (
                (cfg.uav_separation_warning_radius_m - nearest_uav_dist)
                / cfg.uav_separation_warning_radius_m
            )
            r.uav_separation += -cfg.uav_separation_proximity_coeff * proximity

        # ---------------- Energy efficiency ----------------
        r.energy_efficiency += cfg.energy_efficiency_coeff * dt
        r.energy_efficiency += cfg.wind_work_coeff * disturbance_magnitude

        # Mission completion is a one-time SWARM-level bonus, not a
        # per-UAV outcome -- deliberately NOT computed here. See
        # `swarm_env.py::step()`, which adds it once to the aggregated
        # breakdown when full coverage is first achieved.

        return r
