"""
env/uav_swarm_env.py
====================
Main Gymnasium environment: UAVSwarmEnv.

This is the central module of the simulation. It wires together all
sub-modules (Field, ObstacleManager, WindModel, UAV, RewardCalculator)
into a standards-compliant Gymnasium environment that Stable-Baselines3
can train PPO on directly.

Environment summary:
    - Observation: flat vector per UAV containing coverage map, static
      obstacle map, dynamic obstacle map, UAV positions, wind vector.
    - Action: Discrete(5) per UAV — stay/N/S/E/W.
    - Reward: shaped per RewardConfig (see rewards.py).
    - Termination: coverage_target reached OR max_steps exceeded.

Multi-agent strategy:
    SB3 expects a single-agent Gymnasium interface. We handle multiple
    UAVs by treating the joint action as a single MultiDiscrete action
    space of shape (n_uavs,) and the joint observation as a concatenated
    flat vector. This is the standard centralised-training approach
    used in academic MARL baselines.

Reproducibility:
    All stochasticity routes through a single seeded np.random.Generator
    created from EnvConfig.seed. Calling env.reset(seed=N) re-seeds
    everything, guaranteeing identical episode sequences.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uav_swarm_ppo.configs.config import EnvConfig, ExperimentConfig, RewardConfig
from uav_swarm_ppo.env.field import Field
from uav_swarm_ppo.env.obstacles import ObstacleManager
from uav_swarm_ppo.env.rewards import RewardCalculator, StepRewardInfo
from uav_swarm_ppo.env.uav import UAV, N_ACTIONS
from uav_swarm_ppo.env.wind import WindModel


class UAVSwarmEnv(gym.Env):
    """
    Multi-UAV swarm coverage environment for irregular agricultural fields.

    Implements the full Gymnasium API: reset(), step(), render(), close().

    Observation space (flat vector per step):
        [coverage_map (G*G), static_obs_map (G*G), dynamic_obs_map (G*G),
         uav_positions (n_uavs * 2), wind_vector (2)]
        where G = grid_size.

    Action space:
        MultiDiscrete([N_ACTIONS] * n_uavs)
        Each UAV independently selects one of 5 discrete actions.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration bundle.
    """

    # Gymnasium metadata
    metadata: Dict[str, Any] = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()

        # Store config references for easy access
        self.cfg: EnvConfig = config.env
        self.reward_cfg: RewardConfig = config.reward

        # ----------------------------------------------------------------
        # Build sub-modules
        # ----------------------------------------------------------------

        # Farm field geometry (static for the whole experiment)
        self.field = Field(
            vertices=self.cfg.field_vertices,
            grid_size=self.cfg.grid_size,
        )

        # Seeded random generator — the single source of randomness
        self.rng: np.random.Generator = np.random.default_rng(self.cfg.seed)

        # Obstacle manager (re-places obstacles on each reset)
        self.obstacle_manager = ObstacleManager(
            field=self.field,
            n_static=self.cfg.n_static_obstacles,
            n_dynamic=self.cfg.n_dynamic_obstacles,
            dynamic_speed=self.cfg.dynamic_obstacle_speed,
            rng=self.rng,
        )

        # Wind model
        self.wind_model = WindModel(
            max_magnitude=self.cfg.wind_max_magnitude,
            rng=self.rng,
            enabled=self.cfg.wind_enabled,
        )

        # Reward calculator (stateless — just a function wrapper)
        self.reward_calc = RewardCalculator(config.reward)

        # UAV objects (positions set during reset)
        self.uavs: List[UAV] = [
            UAV(
                uav_id=i,
                init_row=0,
                init_col=0,
                obs_radius=self.cfg.uav_obs_radius,
            )
            for i in range(self.cfg.n_uavs)
        ]

        # ----------------------------------------------------------------
        # Gymnasium spaces
        # ----------------------------------------------------------------

        # Observation: flat concatenation of 3 grid maps + positions + wind
        G = self.cfg.grid_size
        n = self.cfg.n_uavs
        obs_dim = (
            G * G          # coverage map
            + G * G        # static obstacle map
            + G * G        # dynamic obstacle map
            + n * 2        # UAV (row, col) positions
            + 2            # wind vector (row_component, col_component)
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=float(G),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: each UAV picks one of N_ACTIONS discrete actions
        self.action_space = spaces.MultiDiscrete(
            [N_ACTIONS] * n, dtype=np.int64
        )

        # ----------------------------------------------------------------
        # Episode state (initialised in reset())
        # ----------------------------------------------------------------

        # Binary coverage map — 1 = cell has been visited
        self.coverage_map: np.ndarray = np.zeros(
            (G, G), dtype=np.float32
        )
        self.step_count: int = 0
        self.episode_complete: bool = False

        # Per-step reward info for logging (list over UAVs)
        self.last_reward_info: List[StepRewardInfo] = []

        # Episode-level accumulators for metrics
        self.episode_collisions: int = 0
        self.episode_cells_covered: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API — reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Places UAVs at random valid starting positions, re-distributes
        obstacles, resets wind, and clears the coverage map.

        Parameters
        ----------
        seed : int, optional
            If provided, re-seeds the RNG (overrides config seed).
        options : dict, optional
            Reserved for future use (ignored currently).

        Returns
        -------
        observation : np.ndarray
            Initial observation vector.
        info : dict
            Empty info dict (Gymnasium convention).
        """
        # Re-seed if requested (important for eval across multiple seeds)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.obstacle_manager.rng = self.rng
            self.wind_model.rng = self.rng

        # Reset coverage map
        self.coverage_map = np.zeros(
            (self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32
        )

        # Place UAVs at random non-overlapping valid positions
        uav_starts = self._sample_uav_starts()
        for i, uav in enumerate(self.uavs):
            row, col = uav_starts[i]
            uav.reset(row, col)

        # Place obstacles (not on UAV start positions)
        reserved = set(uav_starts)
        self.obstacle_manager.reset(reserved_cells=reserved)

        # Reset wind
        self.wind_model.reset()

        # Reset episode counters
        self.step_count = 0
        self.episode_complete = False
        self.episode_collisions = 0
        self.episode_cells_covered = 0
        self.last_reward_info = []

        # Mark initial UAV positions as covered
        for uav in self.uavs:
            if self.coverage_map[uav.row, uav.col] == 0:
                self.coverage_map[uav.row, uav.col] = 1.0
                self.episode_cells_covered += 1

        return self._get_observation(), {}

    # ------------------------------------------------------------------
    # Gymnasium API — step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the simulation by one timestep.

        Processes actions for all UAVs simultaneously:
            1. Evolve wind
            2. Advance dynamic obstacles
            3. Compute intended positions for all UAVs
            4. Validate and apply movements
            5. Check collisions
            6. Update coverage map
            7. Compute rewards
            8. Check termination conditions

        Parameters
        ----------
        actions : np.ndarray
            Shape (n_uavs,) array of integer action indices.

        Returns
        -------
        observation : np.ndarray
            New joint observation after all UAVs have moved.
        reward : float
            Sum of rewards across all UAVs this step (centralised reward).
        terminated : bool
            True if the episode ended due to a natural termination
            (coverage target reached).
        truncated : bool
            True if the episode ended due to max_steps being reached.
        info : dict
            Diagnostic information for logging.
        """
        self.step_count += 1

        # ---- 1. Evolve environment state ---------------------------

        # Wind evolves each step
        self.wind_model.step()
        # Sample wind drift for each UAV
        wind_drifts = self.wind_model.get_drift(self.cfg.n_uavs)

        # Dynamic obstacles move
        self.obstacle_manager.step()

        # ---- 2. Compute intended positions for all UAVs ------------

        intended_positions: List[Tuple[int, int]] = []
        for i, uav in enumerate(self.uavs):
            pos = uav.compute_intended_position(
                action=int(actions[i]),
                wind_drift=wind_drifts[i],
            )
            intended_positions.append(pos)

        # ---- 3. Resolve movements and compute rewards ---------------

        self.last_reward_info = []
        total_reward: float = 0.0

        # Track new positions for inter-UAV collision detection
        new_positions: List[Tuple[int, int]] = []

        for i, uav in enumerate(self.uavs):
            intended = intended_positions[i]
            int_row, int_col = intended

            # Flags for reward calculation
            went_oob = False
            hit_static = False
            hit_dynamic = False
            wind_drifted = False

            # Check if wind caused unintended movement
            base_dr, base_dc = (0, 0)  # default: stay
            from uav_swarm_ppo.env.uav import ACTION_DELTAS
            base_dr, base_dc = ACTION_DELTAS[int(actions[i])]
            wind_dr = int(wind_drifts[i][0])
            wind_dc = int(wind_drifts[i][1])
            wind_drifted = (wind_dr != 0 or wind_dc != 0)

            # Determine actual position after validation
            if not self.field.is_valid_cell(int_row, int_col):
                # Out of bounds — stay in place, apply penalty
                went_oob = True
                actual_row, actual_col = uav.row, uav.col
            elif (int_row, int_col) in self.obstacle_manager.static_cells:
                # Static obstacle — stay in place
                hit_static = True
                actual_row, actual_col = uav.row, uav.col
            else:
                # Valid move — proceed
                actual_row, actual_col = int_row, int_col

            # Check dynamic obstacle collision at new position
            if (actual_row, actual_col) in self.obstacle_manager.dynamic_cells:
                hit_dynamic = True

            new_positions.append((actual_row, actual_col))
            uav.move_to(actual_row, actual_col)

        # ---- 4. Inter-UAV collision detection ----------------------

        # A collision occurs when two UAVs occupy the same cell
        pos_count: Dict[Tuple[int, int], int] = {}
        for pos in new_positions:
            pos_count[pos] = pos_count.get(pos, 0) + 1
        uav_collision_cells: Set[Tuple[int, int]] = {
            pos for pos, count in pos_count.items() if count > 1
        }

        # ---- 5. Update coverage and compute rewards ----------------

        for i, uav in enumerate(self.uavs):
            cell_new = False
            if self.coverage_map[uav.row, uav.col] == 0:
                self.coverage_map[uav.row, uav.col] = 1.0
                cell_new = True
                self.episode_cells_covered += 1
                uav.cells_covered += 1

            hit_uav = uav.position in uav_collision_cells
            if hit_uav:
                uav.collisions += 1
                self.episode_collisions += 1

            # Re-derive flags (some were computed in loop above)
            # We need to retrieve stored flags — simplest approach:
            # re-check conditions using current uav state
            hit_static_flag = new_positions[i] != intended_positions[i] and \
                intended_positions[i] in self.obstacle_manager.static_cells
            went_oob_flag = not self.field.is_valid_cell(*intended_positions[i])
            hit_dyn_flag = uav.position in self.obstacle_manager.dynamic_cells

            # Check for completion BEFORE computing reward so bonus applies
            coverage_ratio = self._coverage_ratio()
            ep_complete_this_step = (
                not self.episode_complete
                and coverage_ratio >= self.cfg.coverage_target
            )
            if ep_complete_this_step:
                self.episode_complete = True

            wind_drift_flag = (
                wind_drifts[i][0] != 0 or wind_drifts[i][1] != 0
            )

            reward_info = self.reward_calc.compute(
                uav_id=i,
                intended_pos=intended_positions[i],
                actual_pos=uav.position,
                cell_was_new=cell_new,
                hit_static=hit_static_flag,
                hit_dynamic=hit_dyn_flag,
                hit_uav=hit_uav,
                went_out_of_bounds=went_oob_flag,
                wind_caused_drift=wind_drift_flag,
                episode_complete=ep_complete_this_step,
            )
            self.last_reward_info.append(reward_info)
            total_reward += reward_info.total

        # ---- 6. Termination conditions -----------------------------

        terminated = self.episode_complete
        truncated = self.step_count >= self.cfg.max_steps

        # ---- 7. Build info dict ------------------------------------

        info = self._build_info()

        return self._get_observation(), total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Gymnasium API — render
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current environment state.

        Returns an RGB image array of shape (grid_size*20, grid_size*20, 3)
        suitable for video recording or matplotlib display.

        Returns
        -------
        np.ndarray or None
            RGB array if render_mode is 'rgb_array', else None.
        """
        # Import here to avoid requiring matplotlib for non-visual runs
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import io
        from PIL import Image

        G = self.cfg.grid_size
        cell = 20  # pixels per cell

        fig, ax = plt.subplots(figsize=(G * cell / 72, G * cell / 72), dpi=72)
        ax.set_xlim(0, G)
        ax.set_ylim(0, G)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw field polygon
        from shapely.geometry import mapping
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon as MplPolygon
        verts = np.array(self.field.vertices)
        field_patch = MplPolygon(verts, closed=True, facecolor="#e8f5e9",
                                 edgecolor="#2e7d32", linewidth=1.5)
        ax.add_patch(field_patch)

        # Draw coverage
        for row in range(G):
            for col in range(G):
                if self.coverage_map[row, col] > 0:
                    rect = patches.Rectangle(
                        (col, row), 1, 1,
                        facecolor="#81c784", alpha=0.6, edgecolor="none"
                    )
                    ax.add_patch(rect)

        # Draw static obstacles
        for row, col in self.obstacle_manager.static_cells:
            rect = patches.Rectangle(
                (col, row), 1, 1, facecolor="#b71c1c", edgecolor="none"
            )
            ax.add_patch(rect)

        # Draw dynamic obstacles
        for row, col in self.obstacle_manager.dynamic_cells:
            rect = patches.Rectangle(
                (col, row), 1, 1, facecolor="#e65100",
                edgecolor="none", alpha=0.85
            )
            ax.add_patch(rect)

        # Draw UAVs
        colors = ["#1565c0", "#6a1b9a", "#00695c", "#f57f17"]
        for i, uav in enumerate(self.uavs):
            c = colors[i % len(colors)]
            ax.plot(
                uav.col + 0.5, uav.row + 0.5,
                marker="^", markersize=8, color=c,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=10
            )

        # Title with stats
        cov = self._coverage_ratio() * 100
        ax.set_title(
            f"Step {self.step_count} | Coverage: {cov:.1f}% | "
            f"Collisions: {self.episode_collisions}",
            fontsize=7, pad=2
        )

        # Convert to RGB array
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=72)
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        return img[:, :, :3]  # Drop alpha channel

    def close(self) -> None:
        """Clean up resources. Called by SB3 at the end of training."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_uav_starts(self) -> List[Tuple[int, int]]:
        """
        Sample n_uavs distinct valid starting positions.

        Returns
        -------
        List[Tuple[int, int]]
            List of (row, col) starting positions, one per UAV.
        """
        cells = list(self.field.valid_cells)
        indices = self.rng.choice(len(cells), size=self.cfg.n_uavs, replace=False)
        return [cells[i] for i in indices]

    def _coverage_ratio(self) -> float:
        """
        Compute the fraction of valid field cells that have been visited.

        Returns
        -------
        float
            Coverage ratio in [0, 1].
        """
        covered = float(np.sum(self.coverage_map[
            [r for r, c in self.field.valid_cells],
            [c for r, c in self.field.valid_cells],
        ]))
        return covered / max(self.field.n_valid_cells, 1)

    def _get_observation(self) -> np.ndarray:
        """
        Build and return the flat joint observation vector.

        Observation layout:
            [coverage_map (G²), static_map (G²), dynamic_map (G²),
             uav_positions (n_uavs × 2), wind_vector (2)]

        Returns
        -------
        np.ndarray
            Shape (obs_dim,) float32 observation.
        """
        G = self.cfg.grid_size

        # Grid maps — flattened
        cov_flat = self.coverage_map.flatten()
        static_flat = self.obstacle_manager.get_static_grid().flatten()
        dynamic_flat = self.obstacle_manager.get_dynamic_grid().flatten()

        # UAV positions — normalised to [0, 1]
        uav_pos = np.array(
            [[uav.row / G, uav.col / G] for uav in self.uavs],
            dtype=np.float32,
        ).flatten()

        # Wind vector — normalised by max magnitude
        wind = self.wind_model.vector / max(self.cfg.wind_max_magnitude, 1e-6)

        return np.concatenate(
            [cov_flat, static_flat, dynamic_flat, uav_pos, wind]
        ).astype(np.float32)

    def _build_info(self) -> Dict[str, Any]:
        """
        Build the info dictionary returned by step().

        Contains diagnostic metrics for logging and evaluation.

        Returns
        -------
        dict
            Metrics dictionary.
        """
        return {
            "coverage_ratio": self._coverage_ratio(),
            "step_count": self.step_count,
            "episode_collisions": self.episode_collisions,
            "episode_cells_covered": self.episode_cells_covered,
            "n_valid_cells": self.field.n_valid_cells,
            "reward_breakdown": [
                {
                    "uav": i,
                    "total": ri.total,
                    "new_cell": ri.new_cell,
                    "collision": ri.static_collision + ri.dynamic_collision,
                    "completion": ri.completion_bonus,
                }
                for i, ri in enumerate(self.last_reward_info)
            ],
        }
