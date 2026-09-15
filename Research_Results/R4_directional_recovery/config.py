"""
configs/config.py
==================

Single source of truth for every parameter used anywhere in the project.

Design rationale (why one file, not scattered constants):
A reviewer's first question about "your algorithm is more robust" claims
is usually "robust under which exact settings, and can I reproduce them?".
By funneling every numeric/behavioral choice (field geometry, obstacle
counts, wind strength, drone physics, reward weights, episode length,
random seed) through dataclasses that serialize to/from JSON, a single
file fully specifies one experiment. `ExperimentConfig.to_json()` /
`from_json()` are what `experiments/run_experiment.py` and the PPO
training script (Day 3) will use to log and replay exact configurations.

Every dataclass below is intentionally "dumb" (no logic, just typed,
documented, defaulted fields) -- logic that *uses* these values lives in
the corresponding environment/ module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Optional


class MBRMode(str, Enum):
    """Strategy for computing the field's Minimum Bounding Rectangle."""
    AXIS_ALIGNED = "axis_aligned"
    ROTATED_MIN_AREA = "rotated_min_area"


class CellLabelRule(str, Enum):
    """Strategy for labeling a grid cell visitable vs non-visitable."""
    CENTROID = "centroid"
    AREA_OVERLAP = "area_overlap"


@dataclass(frozen=True)
class FieldConfig:
    """Geometry/discretization parameters for the farm field (Goal 1)."""
    cell_size_m: float = 10.0
    mbr_mode: MBRMode = MBRMode.AXIS_ALIGNED
    label_rule: CellLabelRule = CellLabelRule.AREA_OVERLAP
    overlap_threshold: float = 0.5


@dataclass(frozen=True)
class ObstacleConfig:
    """
    Counts and sizing for static and dynamic obstacles (Goal 2).

    Static obstacle radii are footprint radii in meters (used both for
    blocking grid cells and for UAV collision-distance checks). Dynamic
    obstacle speeds are in m/s, chosen to be physically plausible:
    a tractor is slow (~2 m/s ≈ 7 km/h field speed), a human walking
    (~1.4 m/s), a bird/animal faster and more erratic (~5 m/s).
    """
    n_trees: int = 8
    n_poles: int = 4
    n_buildings: int = 2
    n_irrigation: int = 5
    tree_radius_m: float = 2.0
    pole_radius_m: float = 0.5
    building_radius_m: float = 6.0
    irrigation_radius_m: float = 1.5

    n_tractors: int = 1
    n_humans: int = 2
    n_animals: int = 3
    tractor_speed_mps: float = 2.0
    human_speed_mps: float = 1.4
    animal_speed_mps: float = 5.0
    tractor_radius_m: float = 2.0
    human_radius_m: float = 0.4
    animal_radius_m: float = 0.6


@dataclass(frozen=True)
class WindConfig:
    """
    Parameters for the stochastic wind field, modeled as an
    Ornstein-Uhlenbeck (OU) mean-reverting process rather than white
    noise, because real near-surface wind is gusty but temporally
    correlated (a gust doesn't fully reverse direction every timestep).

    mean_speed_mps : long-run average wind speed.
    mean_direction_rad : long-run average wind direction (radians, 0 = +x axis).
    theta : mean-reversion rate (higher = wind returns to the mean faster).
    sigma : volatility (higher = gustier, more erratic wind).
    wake_radius_m : distance within which one UAV's downwash/prop-wash can
        add turbulence to a nearby UAV (addresses inter-UAV air interaction).
    wake_strength : magnitude (m/s) of extra turbulence injected into a
        UAV that is within wake_radius_m of another UAV.
    """
    mean_speed_mps: float = 3.0
    mean_direction_rad: float = 0.0
    theta: float = 0.5
    sigma: float = 1.2
    wake_radius_m: float = 8.0
    wake_strength: float = 0.6


@dataclass(frozen=True)
class DroneConfig:
    """Physical parameters shared by all UAVs in the swarm."""
    max_speed_mps: float = 8.0
    collision_radius_m: float = 1.0
    safety_radius_m: float = 3.0  # UAV-UAV minimum separation before penalty
    battery_capacity_s: float = 1800.0  # 30 minutes, matching the reference paper
    wind_coupling: float = 0.8  # fraction of wind vector added to velocity


@dataclass(frozen=True)
class RewardConfig:
    """
    Reward shaping weights, organized into the 8 categories required by
    Goal 3: Exploration, Coverage, Collision avoidance, Boundary
    violations, Dynamic obstacle avoidance, UAV separation, Mission
    completion, Energy efficiency. See REWARD_ENGINEERING.md for the
    full design rationale behind every constant below.

    -- Coverage (reproduces Table 2 / Eq. 2 of the reference paper) --
    new_cell_base, visited_cell, non_visitable: paper's exact constants.

    -- Exploration (our extension, dense shaping distinct from coverage) --
    exploration_coeff, exploration_window_radius: rewards moving into
    locally under-visited neighborhoods, counteracting the "left-edge
    bias" failure mode the reference paper itself documents (Fig. 10).

    -- Collision avoidance (obstacle collisions, discrete + cell-level) --
    collision_static_obstacle, collision_dynamic_obstacle: sparse penalty
    at the moment of physical contact.
    blocked_cell_penalty: softer penalty for flying into a cell blocked
    by a static obstacle, even before physical contact distance is reached.

    -- Boundary violations --
    out_of_bounds: penalty for CollisionEngine's continuous-position
    boundary check (field polygon containment).
    non_visitable: penalty for the cell-grid-level equivalent (kept
    under Coverage historically but conceptually a boundary signal;
    see REWARD_ENGINEERING.md for why both signals are kept).

    -- Dynamic obstacle avoidance (dense proximity shaping, in ADDITION
       to the discrete collision_dynamic_obstacle penalty above) --
    dynamic_obstacle_proximity_coeff, dynamic_obstacle_warning_radius_m:
    ramps a penalty smoothly as a UAV approaches a moving obstacle,
    giving PPO a denser gradient than a sparse collision-only penalty.

    -- UAV separation (dense proximity shaping, in ADDITION to the
       discrete collision_uav_uav penalty) --
    uav_separation_proximity_coeff, uav_separation_warning_radius_m.

    -- Mission completion --
    mission_completion_bonus: one-time swarm-level bonus awarded when
    full flyable-cell coverage is achieved (applied at the env level,
    not per-UAV -- see swarm_env.py).

    -- Energy efficiency --
    energy_efficiency_coeff: small per-second baseline flight-time cost
    (encourages faster mission completion, matching the reference
    paper's flight-time-minimization objective).
    wind_work_coeff: extra cost proportional to disturbance fought
    (existing Goal-2 term, now formally bucketed as "energy efficiency").
    """
    # Coverage
    new_cell_base: float = 358.74
    visited_cell: float = -31.14
    non_visitable: float = -225.17

    # Exploration
    exploration_coeff: float = 5.0
    exploration_window_radius: int = 2

    # Collision avoidance
    collision_static_obstacle: float = -400.0
    collision_dynamic_obstacle: float = -450.0
    blocked_cell_penalty: float = -150.0

    # Boundary violations
    out_of_bounds: float = -100.0

    # Dynamic obstacle avoidance (dense)
    dynamic_obstacle_proximity_coeff: float = 50.0
    dynamic_obstacle_warning_radius_m: float = 10.0

    # UAV separation (discrete + dense)
    collision_uav_uav: float = -400.0
    uav_separation_proximity_coeff: float = 60.0
    uav_separation_warning_radius_m: float = 6.0

    # Mission completion
    mission_completion_bonus: float = 5000.0

    # Energy efficiency
    energy_efficiency_coeff: float = -0.5
    wind_work_coeff: float = -2.0


@dataclass(frozen=True)
class EnvConfig:
    """Episode-level and simulation-loop parameters."""
    n_uavs: int = 3
    max_episode_steps: int = 500
    dt_s: float = 1.0  # physics timestep in seconds


@dataclass(frozen=True)
class PPOConfig:
    """
    Every knob for Stable-Baselines3 PPO training (Goal 4). Grouped as:
    (a) core PPO hyperparameters -- the ones the task explicitly calls
    out to tune (learning rate, gamma, clip range, batch size, n_steps);
    (b) network architecture; (c) parallelism; (d) logging/checkpointing
    cadence. See PPO_TRAINING.md for what each one does and concrete
    tuning guidance -- this class only holds the numbers.

    A note on n_steps/batch_size/n_envs, since these three interact and
    are a common source of silent misconfiguration: PPO collects
    `n_steps * n_envs` transitions per rollout before each policy update,
    then trains on that rollout for `n_epochs` passes in minibatches of
    `batch_size`. `n_steps * n_envs` must be divisible by `batch_size`,
    or SB3 will raise at construction time -- `train.py` asserts this
    explicitly up front rather than letting PPO() fail with a less
    obvious error deep in its constructor.
    """
    # --- Core hyperparameters (explicitly called out to tune) ---
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    batch_size: int = 256
    n_steps: int = 2048          # rollout length per environment, per update
    n_epochs: int = 10           # gradient-descent passes per rollout

    # --- Regularization ---
    ent_coef: float = 0.01       # entropy bonus -- encourages exploration
    vf_coef: float = 0.5         # value-loss weight in the combined PPO loss
    max_grad_norm: float = 0.5   # gradient clipping

    # --- Network architecture (MultiInputPolicy, since obs is a Dict) ---
    net_arch_pi: List[int] = field(default_factory=lambda: [256, 256])
    net_arch_vf: List[int] = field(default_factory=lambda: [256, 256])

    # --- Training run length / parallelism ---
    total_timesteps: int = 200_000
    n_envs: int = 4              # parallel environments (DummyVecEnv)

    # --- Checkpointing / evaluation cadence (in environment timesteps) ---
    checkpoint_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5

    # --- Output locations ---
    tensorboard_log: str = "results/tensorboard"
    model_dir: str = "results/models"
    monitor_dir: str = "results/monitor"


@dataclass(frozen=True)
class VecNormalizeConfig:
    """
    Settings for Stable-Baselines3's `VecNormalize` wrapper, applied
    around the vectorized training/evaluation environments (see
    `ppo/train.py`, `ppo/evaluate.py`) to give PPO numerically
    well-scaled observations and rewards. This is a training-time
    statistical transform layered OUTSIDE the environment -- it does
    NOT alter `SwarmFarmEnv`'s observation representation, does NOT
    alter `environment/reward.py` or `RewardConfig`, and does NOT change
    any reward weight. The environment always computes and exposes the
    same raw reward/observations it always has; `VecNormalize` only
    changes what PPO itself is shown.

    Measured justification (see VECNORMALIZE_INTEGRATION.md for the full
    inspection): under a random policy with the current (unmodified)
    reward function, raw per-step reward has mean=-1395.90, std=678.51,
    with single-step outliers beyond +5000/-3300 -- a numerically large,
    imbalanced scale that a PPO value function fits poorly (this matches
    the near-zero `explained_variance` diagnostic already documented in
    PPO_TRAINING.md Sec. 5). Reward normalization is therefore enabled
    by default. Four of the six observation Dict keys (`flyable_map`,
    `visited_map`, `dynamic_obstacle_map`, `battery_frac`) are already
    naturally bounded in [0,1] at the environment level; only
    `wind_vector` (measured range roughly [-6, 8]) and `uav_position_map`
    (small integer counts) have a real scale mismatch. `norm_obs_keys=None`
    (Option A, approved) normalizes all six anyway via SB3's own default
    behavior, which is simpler and more standard to describe in the
    paper's methodology than a selective subset.
    """
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0     # SB3 default; generous relative to the ~unit-scale post-normalization values measured above
    clip_reward: float = 10.0  # SB3 default; ~10 std deviations of headroom given the measured reward std of 678.51 post-normalization
    norm_obs_keys: Optional[List[str]] = None  # None = normalize all Dict observation keys (Option A, approved)
    gamma: float = 0.99        # discount used internally by VecNormalize's own reward running-return estimate;
                                # kept as an explicit, separately-recorded field (not silently borrowed from
                                # PPOConfig.gamma) so the saved experiment config always shows exactly what
                                # VecNormalize itself was configured with, independent of any future PPOConfig changes


@dataclass(frozen=True)
class QLearningConfig:
    """
    Hyperparameters for the classical Q-Learning baseline
    (`baselines/qlearning.py`), reproducing Section 3.3.2-3.3.5 of the
    reference paper as closely as possible so it is a genuine baseline
    comparison, not a strawman.

    Values default to the paper's own reported constants where the
    paper reports one: `gamma=0.91` ("Through an initial exploration
    process, the chosen value for gamma is 0.91"), `epsilon_start=0.47`,
    `epsilon_decay=0.93`, `epsilon_min=0.05`, `memory_size=60`,
    `hidden_units=167` (Section 3.3.2/3.3.4). See BASELINES.md for the
    one deliberate deviation (output layer activation) and why it was
    necessary.
    """
    gamma: float = 0.91
    epsilon_start: float = 0.47
    epsilon_decay: float = 0.93
    epsilon_min: float = 0.05
    memory_size: int = 60          # per-UAV FIFO replay memory (paper Section 3.3.5)
    hidden_units: int = 167        # paper's first dense layer width
    learning_rate: float = 1.0e-3  # RMSprop learning rate (paper specifies RMSprop, not this exact rate -- unreported)
    batch_size: int = 32           # minibatch size sampled from replay memory per training step
    train_every_n_steps: int = 1   # how often (in env steps) to run one gradient update
    n_episodes: int = 200
    max_steps_per_episode: int = 500
    use_paper_faithful_first_layer: bool = True  # linear (no nonlinearity) first layer, matching the paper exactly
    model_dir: str = "results/models/qlearning"


@dataclass(frozen=True)
class AStarConfig:
    """
    Parameters for the A* coverage-planning baseline (`baselines/astar.py`).
    Not present in the reference paper at all (it only uses Q-Learning) --
    A* is included as a second, non-learning baseline per the task's own
    "Baseline Methods: A*, Classical Q-Learning" requirement, representing
    the classical (non-RL) path-planning approach it's most natural to
    compare a learned policy against.
    """
    dynamic_obstacle_buffer_m: float = 2.0  # extra clearance added around moving obstacles when planning
    replan_every_step: bool = True          # re-run A* every step (dynamic obstacles move) vs. only at path exhaustion
    n_episodes: int = 20
    max_steps_per_episode: int = 500


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Aggregates every sub-config plus the master random seed. This is the
    object that should be constructed once per experiment and threaded
    through field generation, obstacle placement, wind, drones, and the
    Gymnasium environment -- guaranteeing that two runs with the same
    ExperimentConfig produce identical environments.
    """
    seed: int = 42
    field_config: FieldConfig = field(default_factory=FieldConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    vecnormalize: VecNormalizeConfig = field(default_factory=VecNormalizeConfig)
    qlearning: QLearningConfig = field(default_factory=QLearningConfig)
    astar: AStarConfig = field(default_factory=AStarConfig)

    def to_json(self, path: str) -> None:
        """Serialize the full experiment configuration to JSON."""
        payload = asdict(self)
        payload["field_config"]["mbr_mode"] = self.field_config.mbr_mode.value
        payload["field_config"]["label_rule"] = self.field_config.label_rule.value
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ExperimentConfig":
        """Deserialize a previously saved experiment configuration."""
        with open(path, "r") as f:
            payload = json.load(f)
        payload["field_config"]["mbr_mode"] = MBRMode(payload["field_config"]["mbr_mode"])
        payload["field_config"]["label_rule"] = CellLabelRule(payload["field_config"]["label_rule"])
        return ExperimentConfig(
            seed=payload["seed"],
            field_config=FieldConfig(**payload["field_config"]),
            obstacles=ObstacleConfig(**payload["obstacles"]),
            wind=WindConfig(**payload["wind"]),
            drone=DroneConfig(**payload["drone"]),
            reward=RewardConfig(**payload["reward"]),
            env=EnvConfig(**payload["env"]),
            ppo=PPOConfig(**payload.get("ppo", {})),
            vecnormalize=VecNormalizeConfig(**payload.get("vecnormalize", {})),
            qlearning=QLearningConfig(**payload.get("qlearning", {})),
            astar=AStarConfig(**payload.get("astar", {})),
        )
