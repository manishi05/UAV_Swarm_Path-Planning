"""
environment/swarm_env.py
=========================
`SwarmFarmEnv`: the Gymnasium environment that assembles every module
built so far (polygon field, grid, static obstacles, dynamic obstacles,
wind, UAV physics, collision engine, reward) into a single RL-trainable
environment. This is the object Stable-Baselines3's PPO will be pointed
at on Day 3 -- everything upstream of this file exists to be consumed
here.

Action space design: MultiDiscrete([5] * n_uavs) -- one of the 5
ActionType values per UAV, matching the reference paper's simplified
straight-movement action set (Section 3.3.4), extended with STAY.
A single global policy outputs all n_uavs actions at once, which
mirrors the reference paper's own conclusion (Section 4.2) that a
single global controller for the whole swarm outperforms one network
per UAV -- so we are not arbitrarily choosing this structure, we are
carrying forward the paper's own best-supported result into the PPO
upgrade.

Observation space design: a Dict space (compatible with SB3's
MultiInputPolicy) with:
  - "flyable_map": (H, W) float32, 1 = flyable, 0 = not (field boundary
    + static obstacles combined -- Goal 1 + Goal 2 fused into one map,
    which is what a policy actually needs to know, vs. the paper's
    separate "flying map" input, Fig. 4).
  - "visited_map": (H, W) float32, 1 = visited at least once.
  - "dynamic_obstacle_map": (H, W) float32, rasterized proximity of
    moving obstacles (tractors/humans/animals) -- something the
    reference paper's static-map-only input (Fig. 4) cannot represent
    at all, since it has no dynamic obstacles.
  - "uav_position_map": (H, W) float32, count of UAVs currently in each
    cell (paper's "drones' actual positions map", Fig. 4, extended to
    real cell counts rather than a binary marker).
  - "wind_vector": (2,) float32, current ambient wind (vx, vy).
  - "battery_frac": (n_uavs,) float32, each UAV's remaining battery
    fraction (paper's Section 3.4 battery-limited episode concept, made
    observable to the policy instead of only bounding episode length).

Goal 3 additions in this file (vs. Goal 1/2 version):
  - Reward computation now uses `RewardBreakdown` (8 named categories,
    see environment/reward.py) instead of a bare float; the swarm-level
    breakdown is summed via `sum(breakdowns, RewardBreakdown())`.
  - Extra per-UAV signals computed each step to drive the new reward
    categories: local exploration novelty, nearest-dynamic-obstacle
    distance, nearest-other-UAV distance.
  - A one-time mission-completion bonus, applied at the swarm level
    (not per-UAV) the step full flyable-cell coverage is first achieved.
  - Optional `EpisodeLogger` integration (`utils/episode_logger.py`):
    if provided, `reset()`/`step()`/`close()` automatically drive
    `start_episode`/`log_step`/`end_episode`/`close`.
  - `render()` now follows the Gymnasium `render_mode` convention
    (`"rgb_array"` returns a numpy image via the renderer); the
    previous raw-state accessor is preserved as `get_render_state()`
    for the detailed on-disk figures `experiments/run_experiment.py`
    produces.
  - `close()` is implemented (previously missing), releasing any
    matplotlib figures and flushing the episode logger.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs.config import ExperimentConfig
from environment.collision import CollisionEngine, CollisionType
from environment.drone import UAV, ActionType
from environment.grid_map import GridMapGenerator
from environment.obstacle import (
    DynamicObstacleGenerator, DynamicObstacleManager, StaticObstacleGenerator,
)
from environment.polygon_field import FarmFieldPolygon, MinimumBoundingRectangle
from environment.reward import RewardBreakdown, RewardCalculator
from environment.wind import WindField, compute_wake_interference

logger = logging.getLogger("uav_swarm_ppo.environment.swarm_env")


class SwarmFarmEnv(gym.Env):
    """
    Multi-UAV coverage environment over an irregular farm field with
    static/dynamic obstacles, wind, and inter-UAV collision avoidance.

    One `step()` call advances every UAV in the swarm simultaneously
    by one physics timestep, consistent with the reference paper's
    swarm-level (not sequential per-UAV) episode structure.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, vertices: List[Tuple[float, float]], config: ExperimentConfig,
                 episode_logger: Optional["EpisodeLogger"] = None,
                 render_mode: Optional[str] = "rgb_array"):
        """
        Parameters
        ----------
        vertices : List[Tuple[float, float]]
            Field boundary vertices in meters (Goal 1 input).
        config : ExperimentConfig
            Full experiment configuration (see configs/config.py);
            controls every aspect of field, obstacle, wind, drone, and
            reward behavior, plus the master random seed.
        episode_logger : Optional[EpisodeLogger]
            If provided (see utils/episode_logger.py), `reset()`,
            `step()`, and `close()` automatically log per-step reward
            breakdowns and per-episode summaries (rewards, coverage,
            collisions, episode statistics -- the Goal 3 logging
            requirement). Left as `None` by default so unit tests /
            quick scripts don't need to construct a logger.
        render_mode : Optional[str]
            Gymnasium render mode. `"rgb_array"` (default) makes
            `render()` return a numpy image array. `"human"` shows an
            interactive matplotlib window. `None` disables `render()`.
        """
        super().__init__()
        self._vertices = vertices
        self._config = config
        self._episode_logger = episode_logger
        self.render_mode = render_mode
        self._rng = np.random.default_rng(config.seed)

        # --- Static geometry: built once, reused across episodes ---
        self._field = FarmFieldPolygon(vertices)
        self._mbr = MinimumBoundingRectangle(self._field, mode=config.field_config.mbr_mode)
        self._base_grid = GridMapGenerator(self._field, self._mbr, config.field_config, seed=config.seed)
        n_rows, n_cols = self._base_grid.shape

        self._static_obstacle_gen = StaticObstacleGenerator(config.obstacles, seed=config.seed)
        self._dynamic_obstacle_gen = DynamicObstacleGenerator(config.obstacles, seed=config.seed)
        self._collision_engine = CollisionEngine(
            self._field, config.drone.safety_radius_m, config.drone.collision_radius_m,
        )
        self._reward_calc = RewardCalculator(config.reward)
        self._renderer = None  # lazily constructed in render()/get_render_state() consumers

        # --- Per-episode mutable state (initialized in reset()) ---
        self._grid: Optional[GridMapGenerator] = None
        self._uavs: List[UAV] = []
        self._static_obstacles = []
        self._dynamic_manager: Optional[DynamicObstacleManager] = None
        self._wind: Optional[WindField] = None
        self._visited: Optional[np.ndarray] = None
        self._step_count = 0
        self._total_actions = 0
        self._valid_actions = 0
        self._episode_collision_counts = {t.value: 0 for t in CollisionType}
        self._mission_bonus_awarded = False

        # --- Gymnasium spaces ---
        n_uavs = config.env.n_uavs
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * n_uavs)
        self.observation_space = spaces.Dict({
            "flyable_map": spaces.Box(0, 1, shape=(n_rows, n_cols), dtype=np.float32),
            "visited_map": spaces.Box(0, 1, shape=(n_rows, n_cols), dtype=np.float32),
            "dynamic_obstacle_map": spaces.Box(0, 1, shape=(n_rows, n_cols), dtype=np.float32),
            "uav_position_map": spaces.Box(0, n_uavs, shape=(n_rows, n_cols), dtype=np.float32),
            "wind_vector": spaces.Box(-20, 20, shape=(2,), dtype=np.float32),
            "battery_frac": spaces.Box(0, 1, shape=(n_uavs,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Rebuilds all per-episode state: a fresh grid copy (so static
        obstacles from a previous episode don't persist), fresh static +
        dynamic obstacles, fresh UAV start positions/batteries, and a
        freshly reset wind field. Also starts a new episode in the
        `EpisodeLogger`, if one was provided.

        Note on grid reuse: the *labeling* (visitable/non-visitable) is
        deterministic given the field+config, so we do not recompute
        geometry every episode (expensive); we only need a fresh
        `blocked_static` / `is_start` state, so we regenerate a shallow
        grid from the cached field/MBR rather than from scratch.
        """
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self._config.seed
        self._rng = np.random.default_rng(episode_seed)

        self._grid = GridMapGenerator(self._field, self._mbr, self._config.field_config, seed=episode_seed)
        self._static_obstacles = self._static_obstacle_gen.generate(self._field, self._grid)

        dynamic_obstacles = self._dynamic_obstacle_gen.generate(self._field, self._grid)
        self._dynamic_manager = DynamicObstacleManager(dynamic_obstacles, seed=episode_seed)

        self._wind = WindField(self._config.wind, seed=episode_seed)

        start_cells = self._grid.assign_start_cells(self._config.env.n_uavs, seed=episode_seed)
        self._uavs = [
            UAV(
                uav_id=i,
                position=np.array([cell.center_x, cell.center_y]),
                battery_s=self._config.drone.battery_capacity_s,
                config=self._config.drone,
            )
            for i, cell in enumerate(start_cells)
        ]

        n_rows, n_cols = self._grid.shape
        self._visited = np.zeros((n_rows, n_cols), dtype=bool)
        for cell in start_cells:
            self._visited[cell.row, cell.col] = True

        self._step_count = 0
        self._total_actions = 0
        self._valid_actions = 0
        self._episode_collision_counts = {t.value: 0 for t in CollisionType}
        self._mission_bonus_awarded = False

        if self._episode_logger is not None:
            self._episode_logger.start_episode(episode_seed)

        logger.info(
            "Episode reset (seed=%d): %d UAVs, %d static obstacles, %d dynamic obstacles.",
            episode_seed, len(self._uavs), len(self._static_obstacles),
            len(self._dynamic_manager.obstacles),
        )

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """
        Advances the whole swarm by one timestep.

        Sequence per step (order matters and is deliberate):
        1. Advance ambient wind and dynamic obstacles.
        2. Compute inter-UAV wake interference from *current* positions
           (i.e., based on where UAVs were before this step's motion,
           avoiding a same-step feedback loop between motion and wake).
        3. Integrate each alive UAV's physics (commanded action + wind +
           wake), then hard-clamp to the MBR bounds.
        4. Run collision detection on the *post-motion* positions.
        5. Compute nearest-neighbor distances (dynamic obstacle, other
           UAV) and local exploration novelty, then compute each alive
           UAV's `RewardBreakdown` and sum into a swarm-level breakdown.
        6. Apply the one-time mission-completion bonus if full coverage
           was just achieved.
        7. Check termination (full coverage, all UAVs dead) and
           truncation (max episode steps); log to `EpisodeLogger` if set.
        """
        dt = self._config.env.dt_s
        wind_vector = self._wind.step(dt)
        self._dynamic_manager.step(dt)

        positions_before = np.array([u.position for u in self._uavs])
        wake_extra = compute_wake_interference(
            positions_before, wind_vector,
            self._config.wind.wake_radius_m, self._config.wind.wake_strength,
            self._rng,
        )

        min_x, min_y, max_x, max_y = self._mbr.bounds
        disturbance_mags = np.zeros(len(self._uavs))
        for i, uav in enumerate(self._uavs):
            if not uav.alive:
                continue
            act = ActionType(int(action[i]))
            uav.step_physics(act, dt, wind_vector, wake_extra[i])
            uav.clamp_to_bounds(min_x, min_y, max_x, max_y)
            disturbance_mags[i] = np.linalg.norm(
                self._config.drone.wind_coupling * wind_vector + wake_extra[i]
            )
            self._total_actions += 1

        report = self._collision_engine.detect(
            self._uavs, self._static_obstacles, self._dynamic_manager.obstacles,
        )
        for key, val in report.counts.items():
            self._episode_collision_counts[key] += val

        dynamic_positions = self._dynamic_manager.positions()
        alive_positions = {u.uav_id: u.position for u in self._uavs if u.alive}

        breakdowns: List[RewardBreakdown] = []
        for i, uav in enumerate(self._uavs):
            if not uav.alive:
                continue

            cell = self._grid.get_cell_at_point(uav.position[0], uav.position[1])
            entered_new = entered_visited = entered_boundary = entered_obstacle_zone = False
            if cell is None or not cell.is_visitable:
                entered_boundary = True
            elif cell.blocked_static:
                entered_obstacle_zone = True
            elif self._visited[cell.row, cell.col]:
                entered_visited = True
            else:
                entered_new = True
                self._visited[cell.row, cell.col] = True
                self._valid_actions += 1

            local_novelty = self._compute_local_novelty(cell) if cell is not None else 0.0

            nearest_dyn_dist = None
            if len(dynamic_positions) > 0:
                nearest_dyn_dist = float(np.min(np.linalg.norm(dynamic_positions - uav.position, axis=1)))

            nearest_uav_dist = None
            other_positions = [p for uid, p in alive_positions.items() if uid != uav.uav_id]
            if other_positions:
                nearest_uav_dist = float(np.min(np.linalg.norm(np.array(other_positions) - uav.position, axis=1)))

            breakdown = self._reward_calc.compute(
                entered_new_cell=entered_new,
                entered_visited_cell=entered_visited,
                entered_boundary_violation=entered_boundary,
                entered_obstacle_zone=entered_obstacle_zone,
                n_visited_cells=int(self._visited.sum()),
                grid_rows=self._grid.shape[0], grid_cols=self._grid.shape[1],
                local_novelty=local_novelty,
                collisions=report.for_uav(uav.uav_id),
                nearest_dynamic_obstacle_dist=nearest_dyn_dist,
                nearest_uav_dist=nearest_uav_dist,
                disturbance_magnitude=disturbance_mags[i],
                dt=dt,
            )
            breakdowns.append(breakdown)

        swarm_breakdown = sum(breakdowns, RewardBreakdown())

        self._step_count += 1

        flyable = self._grid.flyable_mask()
        full_coverage = bool(np.array_equal(self._visited & flyable, flyable))
        all_dead = all(not u.alive for u in self._uavs)

        # One-time swarm-level mission-completion bonus, awarded exactly
        # once (the step full coverage is first reached -- the episode
        # terminates immediately after, so "once" and "ever" coincide).
        if full_coverage and not self._mission_bonus_awarded:
            swarm_breakdown.mission_completion += self._config.reward.mission_completion_bonus
            self._mission_bonus_awarded = True

        terminated = full_coverage or all_dead
        truncated = self._step_count >= self._config.env.max_episode_steps

        info = self._get_info()
        info["reward_breakdown"] = swarm_breakdown.as_dict()

        if self._episode_logger is not None:
            self._episode_logger.log_step(swarm_breakdown, info)
            if terminated or truncated:
                self._episode_logger.end_episode()

        return self._get_obs(), swarm_breakdown.total, terminated, truncated, info

    def render(self):
        """
        Gymnasium-standard render entrypoint. Behavior depends on
        `self.render_mode` (set at construction):
          - "rgb_array": returns an (H, W, 3) uint8 numpy array via
            `visualization.renderer.EnvironmentRenderer.render_to_array`.
          - "human": opens an interactive matplotlib window (best used
            outside of headless training).
          - None: raises, per Gymnasium convention (render_mode must be
            set at construction to call render()).

        For the detailed, titled, saved-to-disk figures used in
        `experiments/run_experiment.py`, use `get_render_state()` +
        `EnvironmentRenderer.render_frame()` directly instead -- this
        method is the lightweight Gymnasium-API-compliant path (e.g.
        for a video-recording wrapper during PPO training).
        """
        if self.render_mode is None:
            raise RuntimeError(
                "Called render() but render_mode was not set at construction. "
                "Pass render_mode='rgb_array' or 'human' to SwarmFarmEnv()."
            )
        from visualization.renderer import EnvironmentRenderer
        if self._renderer is None:
            self._renderer = EnvironmentRenderer()

        state = self.get_render_state()
        if self.render_mode == "rgb_array":
            return self._renderer.render_to_array(state, title=f"Step {self._step_count}")
        elif self.render_mode == "human":
            self._renderer.render_frame(state, title=f"Step {self._step_count}", show=True)
            return None

    def get_render_state(self) -> dict:
        """
        Returns the raw environment state dict (field, MBR, grid,
        obstacles, UAVs, wind, visited mask) for external/custom
        rendering. Kept separate from `render()` (which now follows the
        Gymnasium API and returns an image/None) so callers that want
        the full detailed state -- e.g. `experiments/run_experiment.py`
        saving titled before/after figures -- still have direct access
        to it without re-deriving anything from an image array.
        """
        return {
            "field": self._field, "mbr": self._mbr, "grid": self._grid,
            "static_obstacles": self._static_obstacles,
            "dynamic_obstacles": self._dynamic_manager.obstacles if self._dynamic_manager else [],
            "uavs": self._uavs, "wind_vector": self._wind.vector if self._wind else np.zeros(2),
            "visited": self._visited,
        }

    def close(self):
        """
        Releases resources: closes any matplotlib figures opened by
        rendering, and flushes the `EpisodeLogger` (if one was provided)
        so its summary CSV/log file are guaranteed to be written even if
        `close()` is called mid-episode (e.g., a training run being
        interrupted).
        """
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except ImportError:
            pass  # matplotlib was never imported (headless training, no render() calls) -- nothing to close.

        if self._episode_logger is not None:
            self._episode_logger.close()

        logger.info("SwarmFarmEnv closed.")

    # ------------------------------------------------------------------
    # Reward-signal helper computations
    # ------------------------------------------------------------------

    def _compute_local_novelty(self, cell) -> float:
        """
        Computes the Exploration reward signal: 1 minus the fraction of
        flyable cells already visited within a square window of radius
        `RewardConfig.exploration_window_radius` around `cell`.

        Restricted to flyable cells within the window (non-flyable cells
        are excluded from both numerator and denominator) so a window
        near the field edge or a cluster of obstacles isn't penalized
        for having "unvisitable" neighbors -- novelty should reflect
        only how much of the *actually coverable* local area remains
        unexplored, which is what a policy can meaningfully act on.

        Returns 0.0 (no exploration bonus) if the window contains no
        flyable cells at all (fully boxed in by obstacles/boundary).
        """
        radius = self._config.reward.exploration_window_radius
        n_rows, n_cols = self._grid.shape
        flyable = self._grid.flyable_mask()

        r0, r1 = max(0, cell.row - radius), min(n_rows, cell.row + radius + 1)
        c0, c1 = max(0, cell.col - radius), min(n_cols, cell.col + radius + 1)

        window_flyable = flyable[r0:r1, c0:c1]
        window_visited = self._visited[r0:r1, c0:c1]

        n_flyable_in_window = window_flyable.sum()
        if n_flyable_in_window == 0:
            return 0.0

        n_visited_in_window = np.logical_and(window_visited, window_flyable).sum()
        return float(1.0 - n_visited_in_window / n_flyable_in_window)

    # ------------------------------------------------------------------
    # Observation / info construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        n_rows, n_cols = self._grid.shape
        flyable_map = self._grid.flyable_mask().astype(np.float32)
        visited_map = self._visited.astype(np.float32)

        dynamic_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        for obstacle in self._dynamic_manager.obstacles:
            cell = self._grid.get_cell_at_point(obstacle.x, obstacle.y)
            if cell is not None:
                dynamic_map[cell.row, cell.col] = 1.0

        uav_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        for uav in self._uavs:
            if not uav.alive:
                continue
            cell = self._grid.get_cell_at_point(uav.position[0], uav.position[1])
            if cell is not None:
                uav_map[cell.row, cell.col] += 1.0

        battery_frac = np.array([
            u.battery_s / self._config.drone.battery_capacity_s for u in self._uavs
        ], dtype=np.float32)

        return {
            "flyable_map": flyable_map,
            "visited_map": visited_map,
            "dynamic_obstacle_map": dynamic_map,
            "uav_position_map": uav_map,
            "wind_vector": self._wind.vector.astype(np.float32),
            "battery_frac": battery_frac,
        }

    def _get_info(self) -> dict:
        flyable = self._grid.flyable_mask()
        n_flyable = int(flyable.sum())
        n_visited = int((self._visited & flyable).sum())
        return {
            "coverage_fraction": n_visited / max(n_flyable, 1),
            "valid_action_fraction": self._valid_actions / max(self._total_actions, 1),
            "collision_counts": dict(self._episode_collision_counts),
            "step_count": self._step_count,
        }


def make_env(
    vertices: List[Tuple[float, float]],
    config: ExperimentConfig,
    episode_logger: Optional["EpisodeLogger"] = None,
    render_mode: Optional[str] = None,
    seed_offset: int = 0,
):
    """
    Factory returning a zero-argument callable that constructs one
    `SwarmFarmEnv` instance -- the shape Stable-Baselines3's `DummyVecEnv`/
    `SubprocVecEnv` require (`env_fns: List[Callable[[], gym.Env]]`).

    Added for Goal 4 (`ppo/train.py`, `ppo/evaluate.py`) but deliberately
    placed here rather than duplicated in each caller, since baselines
    (`baselines/qlearning.py`, `baselines/astar.py`, Day 4) will need the
    identical construction recipe -- one factory, reused everywhere,
    keeps every algorithm training/evaluating on a genuinely identical
    environment definition (the whole point of the shared `ExperimentConfig`).

    Parameters
    ----------
    vertices, config, episode_logger, render_mode
        Passed straight through to `SwarmFarmEnv.__init__`.
    seed_offset : int
        Added to `config.seed` for this particular env instance. Critical
        for vectorized training: `n_envs` copies of the *same* seed would
        each generate an identical field/obstacle/wind realization and
        defeat the purpose of parallel rollout diversity. `ppo/train.py`
        passes a different `seed_offset` (0, 1, 2, ...) per parallel env.

    Returns
    -------
    Callable[[], SwarmFarmEnv]
    """
    def _init() -> SwarmFarmEnv:
        import dataclasses
        env_config = dataclasses.replace(config, seed=config.seed + seed_offset)
        return SwarmFarmEnv(vertices, env_config, episode_logger=episode_logger, render_mode=render_mode)
    return _init
