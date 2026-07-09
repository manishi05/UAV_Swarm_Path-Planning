"""
configs/config.py
=================
Central configuration module for the UAV Swarm PPO simulation.

All hyperparameters, environment settings, training settings, and
reproducibility seeds are defined here as a frozen dataclass.
This is the single source of truth — no magic numbers anywhere else
in the codebase. Every other module imports from here.

Design rationale:
    Using a dataclass (rather than a plain dict or .yaml) gives us
    type annotations, IDE autocomplete, and easy serialisation to JSON
    for logging experiment configurations alongside results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Field geometry
# ---------------------------------------------------------------------------

# Default irregular polygon vertices (x, y) in grid-cell coordinates.
# This approximates a realistic non-convex farm field.
# Vertices are listed counter-clockwise.
DEFAULT_FIELD_VERTICES: List[Tuple[float, float]] = [
    (2, 0), (18, 1), (22, 5), (24, 14),
    (20, 22), (12, 24), (4, 20), (0, 10), (2, 0)
]


@dataclass
class EnvConfig:
    """
    Configuration for the Gymnasium UAV swarm environment.

    Attributes
    ----------
    grid_size : int
        Side length of the square bounding grid (cells).
        The irregular polygon is inscribed within this grid.
    cell_size : float
        Physical size of each cell in metres (used for wind calculations).
    field_vertices : List[Tuple[float, float]]
        Ordered (x, y) vertices of the irregular polygon farm field.
    n_uavs : int
        Number of UAV agents in the swarm.
    n_static_obstacles : int
        Number of permanently fixed obstacles (e.g. trees, pylons).
    n_dynamic_obstacles : int
        Number of moving obstacles (e.g. tractors, animals).
    dynamic_obstacle_speed : int
        Number of cells a dynamic obstacle moves per timestep.
    max_steps : int
        Maximum timesteps per episode before forced termination.
    coverage_target : float
        Fraction of valid field cells that must be covered to
        trigger the completion bonus (0.0–1.0).
    uav_obs_radius : int
        Radius (in cells) of each UAV's local observation window.
    seed : int
        Master random seed for full reproducibility.
    wind_enabled : bool
        Whether stochastic wind disturbances are applied each step.
    wind_max_magnitude : float
        Maximum wind displacement in cells per step (float, applied
        probabilistically to UAV movement).
    """

    grid_size: int = 25
    cell_size: float = 1.0
    field_vertices: List[Tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_FIELD_VERTICES
    )
    n_uavs: int = 4
    n_static_obstacles: int = 5
    n_dynamic_obstacles: int = 3
    dynamic_obstacle_speed: int = 1
    max_steps: int = 500
    coverage_target: float = 0.90
    uav_obs_radius: int = 3
    seed: int = 42
    wind_enabled: bool = True
    wind_max_magnitude: float = 0.4


@dataclass
class RewardConfig:
    """
    Reward shaping weights and penalties.

    Each term is independently tuneable so ablation studies can be
    run by zeroing out individual components.

    Attributes
    ----------
    r_new_cell : float
        Reward for visiting a previously uncovered valid field cell.
    r_revisit : float
        Penalty for revisiting an already-covered cell (encourages exploration).
    r_static_collision : float
        Penalty for a UAV entering a static obstacle cell.
    r_dynamic_collision : float
        Penalty for a UAV colliding with a moving obstacle.
    r_uav_collision : float
        Penalty when two UAVs occupy the same cell simultaneously.
    r_out_of_bounds : float
        Penalty for attempting to leave the field polygon boundary.
    r_step : float
        Small negative reward applied every timestep (time pressure).
    r_completion : float
        Large one-time bonus when coverage_target is reached.
    r_wind_drift : float
        Small penalty when wind pushes a UAV off its intended path.
    """

    r_new_cell: float = 1.0
    r_revisit: float = -0.1
    r_static_collision: float = -5.0
    r_dynamic_collision: float = -5.0
    r_uav_collision: float = -2.0
    r_out_of_bounds: float = -3.0
    r_step: float = -0.01
    r_completion: float = 50.0
    r_wind_drift: float = -0.2


@dataclass
class PPOConfig:
    """
    Stable-Baselines3 PPO hyperparameters.

    These values follow best-practice defaults for continuous-action
    multi-step environments and are documented for the paper's
    Implementation Details section.

    Attributes
    ----------
    total_timesteps : int
        Total environment steps for the entire training run.
    n_steps : int
        Steps collected per environment per update rollout.
    batch_size : int
        Mini-batch size for gradient updates.
    n_epochs : int
        Number of passes over the collected rollout data per update.
    gamma : float
        Discount factor for future rewards.
    gae_lambda : float
        Generalised Advantage Estimation (GAE) lambda parameter.
    clip_range : float
        PPO clipping epsilon — prevents overly large policy updates.
    learning_rate : float
        Adam optimiser learning rate.
    ent_coef : float
        Entropy coefficient — encourages exploration.
    vf_coef : float
        Value function loss coefficient.
    max_grad_norm : float
        Gradient clipping threshold.
    policy : str
        SB3 policy architecture identifier.
    verbose : int
        SB3 verbosity level (0=silent, 1=info, 2=debug).
    """

    total_timesteps: int = 1_000_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MlpPolicy"
    verbose: int = 1


@dataclass
class LoggingConfig:
    """
    Paths and settings for experiment logging.

    Attributes
    ----------
    log_dir : str
        Directory for TensorBoard logs and SB3 model checkpoints.
    results_dir : str
        Directory for CSV metric exports and figure outputs.
    checkpoint_freq : int
        Save a model checkpoint every N timesteps.
    eval_freq : int
        Run evaluation every N timesteps during training.
    n_eval_episodes : int
        Number of episodes for each mid-training evaluation.
    """

    log_dir: str = "logs"
    results_dir: str = "results"
    checkpoint_freq: int = 50_000
    eval_freq: int = 25_000
    n_eval_episodes: int = 10


@dataclass
class ExperimentConfig:
    """
    Top-level container that bundles all sub-configs.

    This is the object passed around the entire codebase. Serialising
    this to JSON at the start of each run ensures full reproducibility
    — every experiment is self-documenting.

    Attributes
    ----------
    env : EnvConfig
        Environment settings.
    reward : RewardConfig
        Reward shaping weights.
    ppo : PPOConfig
        PPO training hyperparameters.
    logging : LoggingConfig
        Logging and checkpointing paths.
    experiment_name : str
        Human-readable tag for this run (used in filenames).
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment_name: str = "uav_swarm_ppo_v1"

    def save(self, path: str | Path) -> None:
        """
        Serialise the full config to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. 'results/config.json').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """
        Load a config from a previously saved JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the JSON config file.

        Returns
        -------
        ExperimentConfig
            Reconstructed config object.
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses from the flat dict
        return cls(
            env=EnvConfig(**data["env"]),
            reward=RewardConfig(**data["reward"]),
            ppo=PPOConfig(**data["ppo"]),
            logging=LoggingConfig(**data["logging"]),
            experiment_name=data["experiment_name"],
        )
