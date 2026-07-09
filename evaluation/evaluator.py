"""
evaluation/evaluator.py
=======================
Evaluation and baseline comparison module.

Provides:
    Evaluator     — runs a trained PPO model across multiple seeds
                    and computes publication-quality metrics.
    RandomBaseline — uniform random action policy (lower bound).
    GreedyBaseline — coverage-greedy heuristic policy (sanity check).

Results are exported as:
    - CSV of per-episode metrics (coverage, collisions, steps)
    - Summary statistics (mean ± std across seeds) for the paper table

Architecture note:
    Evaluator is decoupled from Trainer — it only needs a path to a
    saved model and an ExperimentConfig. This allows re-evaluation
    of any checkpoint without re-running training.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from uav_swarm_ppo.configs.config import ExperimentConfig
from uav_swarm_ppo.env.uav_swarm_env import UAVSwarmEnv


# ---------------------------------------------------------------------------
# Metric container
# ---------------------------------------------------------------------------

class EpisodeMetrics:
    """
    Holds scalar metrics for a single evaluation episode.

    Attributes
    ----------
    coverage_ratio : float
        Final coverage ratio (0–1) at episode end.
    n_collisions : int
        Total collision events across all UAVs.
    steps : int
        Total timesteps taken.
    total_reward : float
        Cumulative reward.
    completed : bool
        Whether coverage_target was reached.
    """

    def __init__(
        self,
        coverage_ratio: float,
        n_collisions: int,
        steps: int,
        total_reward: float,
        completed: bool,
    ) -> None:
        self.coverage_ratio = coverage_ratio
        self.n_collisions = n_collisions
        self.steps = steps
        self.total_reward = total_reward
        self.completed = completed


# ---------------------------------------------------------------------------
# PPO Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Multi-seed evaluator for a trained PPO model.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration (same one used for training).
    model_path : str
        Path to the saved SB3 model (.zip, without extension).
    results_dir : str or Path, optional
        Directory to save CSV outputs. Defaults to config.logging.results_dir.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model_path: str,
        results_dir: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.results_dir = Path(results_dir or config.logging.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load trained model (env will be set below per seed)
        self._model: Optional[PPO] = None

    def run(
        self,
        n_seeds: int = 5,
        n_episodes_per_seed: int = 5,
        label: str = "PPO",
    ) -> Dict[str, float]:
        """
        Evaluate the PPO model across multiple seeds and episodes.

        Parameters
        ----------
        n_seeds : int
            Number of distinct random seeds to evaluate on.
        n_episodes_per_seed : int
            Episodes per seed.
        label : str
            Name tag for CSV export (e.g. 'PPO', 'Random', 'Greedy').

        Returns
        -------
        dict
            Summary statistics: mean and std for each metric.
        """
        all_metrics: List[EpisodeMetrics] = []

        for seed_offset in range(n_seeds):
            seed = self.config.env.seed + seed_offset * 1000

            # Build fresh env with this seed
            from copy import deepcopy
            cfg = deepcopy(self.config)
            cfg.env.seed = seed
            env = UAVSwarmEnv(cfg)

            # Load model (first time) or just reset env
            if self._model is None:
                self._model = PPO.load(self.model_path)

            for ep in range(n_episodes_per_seed):
                metrics = self._run_episode(env, use_model=True)
                all_metrics.append(metrics)

            env.close()

        # Export to CSV
        self._export_csv(all_metrics, label)

        # Compute summary statistics
        return self._summarise(all_metrics, label)

    def _run_episode(
        self,
        env: UAVSwarmEnv,
        use_model: bool = True,
        policy_fn=None,
    ) -> EpisodeMetrics:
        """
        Run one evaluation episode.

        Parameters
        ----------
        env : UAVSwarmEnv
            Environment instance.
        use_model : bool
            If True, use the loaded PPO model. If False, use policy_fn.
        policy_fn : callable, optional
            Alternative policy function: obs -> action array.

        Returns
        -------
        EpisodeMetrics
            Metrics for this episode.
        """
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if use_model and self._model is not None:
                # SB3 expects (1, obs_dim) for single env
                action, _ = self._model.predict(
                    obs.reshape(1, -1), deterministic=True
                )
                action = action[0]
            elif policy_fn is not None:
                action = policy_fn(obs, env)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        return EpisodeMetrics(
            coverage_ratio=info["coverage_ratio"],
            n_collisions=info["episode_collisions"],
            steps=info["step_count"],
            total_reward=total_reward,
            completed=env.episode_complete,
        )

    def _export_csv(
        self,
        metrics: List[EpisodeMetrics],
        label: str,
    ) -> None:
        """
        Export per-episode metrics to CSV.

        Parameters
        ----------
        metrics : List[EpisodeMetrics]
            All episode results.
        label : str
            Policy label for filename.
        """
        path = self.results_dir / f"{label}_episodes.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode", "coverage_ratio", "n_collisions",
                    "steps", "total_reward", "completed"
                ]
            )
            writer.writeheader()
            for i, m in enumerate(metrics):
                writer.writerow({
                    "episode": i,
                    "coverage_ratio": round(m.coverage_ratio, 4),
                    "n_collisions": m.n_collisions,
                    "steps": m.steps,
                    "total_reward": round(m.total_reward, 3),
                    "completed": int(m.completed),
                })
        print(f"[Evaluator] Episode metrics saved to {path}")

    def _summarise(
        self,
        metrics: List[EpisodeMetrics],
        label: str,
    ) -> Dict[str, float]:
        """
        Compute mean ± std summary statistics across all episodes.

        Parameters
        ----------
        metrics : List[EpisodeMetrics]
            All episode results.
        label : str
            Policy label for printing.

        Returns
        -------
        dict
            Keys: metric_mean, metric_std for each metric.
        """
        cov = [m.coverage_ratio for m in metrics]
        col = [m.n_collisions for m in metrics]
        steps = [m.steps for m in metrics]
        rew = [m.total_reward for m in metrics]
        comp = [m.completed for m in metrics]

        summary = {
            "coverage_mean": float(np.mean(cov)),
            "coverage_std": float(np.std(cov)),
            "collisions_mean": float(np.mean(col)),
            "collisions_std": float(np.std(col)),
            "steps_mean": float(np.mean(steps)),
            "steps_std": float(np.std(steps)),
            "reward_mean": float(np.mean(rew)),
            "reward_std": float(np.std(rew)),
            "completion_rate": float(np.mean(comp)),
        }

        print(f"\n[{label}] Results over {len(metrics)} episodes:")
        print(f"  Coverage:   {summary['coverage_mean']*100:.1f}% "
              f"± {summary['coverage_std']*100:.1f}%")
        print(f"  Collisions: {summary['collisions_mean']:.1f} "
              f"± {summary['collisions_std']:.1f}")
        print(f"  Steps:      {summary['steps_mean']:.0f} "
              f"± {summary['steps_std']:.0f}")
        print(f"  Completion: {summary['completion_rate']*100:.0f}%")

        return summary


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

class RandomBaseline:
    """
    Uniform random action policy — the minimum performance baseline.

    Usage:
        baseline = RandomBaseline(config)
        results = baseline.evaluate(n_seeds=5, n_episodes_per_seed=5)
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def _policy(self, obs: np.ndarray, env: UAVSwarmEnv) -> np.ndarray:
        """Sample uniformly from the action space."""
        return env.action_space.sample()

    def evaluate(
        self,
        n_seeds: int = 5,
        n_episodes_per_seed: int = 5,
    ) -> Dict[str, float]:
        """
        Run random policy evaluation.

        Parameters
        ----------
        n_seeds : int
            Number of seeds.
        n_episodes_per_seed : int
            Episodes per seed.

        Returns
        -------
        dict
            Summary statistics.
        """
        # Reuse Evaluator infrastructure but with random policy
        evaluator = Evaluator(
            config=self.config,
            model_path="",  # unused
            results_dir=Path(self.config.logging.results_dir) / "baselines",
        )
        all_metrics = []
        for seed_offset in range(n_seeds):
            from copy import deepcopy
            cfg = deepcopy(self.config)
            cfg.env.seed = self.config.env.seed + seed_offset * 1000
            env = UAVSwarmEnv(cfg)
            for _ in range(n_episodes_per_seed):
                m = evaluator._run_episode(env, use_model=False,
                                          policy_fn=self._policy)
                all_metrics.append(m)
            env.close()

        evaluator._export_csv(all_metrics, "Random")
        return evaluator._summarise(all_metrics, "Random")


class GreedyBaseline:
    """
    Coverage-greedy baseline: each UAV always moves toward the nearest
    uncovered cell, ignoring obstacles.

    This is stronger than random but simpler than RL — a useful
    intermediate baseline for the paper's comparison table.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def _policy(self, obs: np.ndarray, env: UAVSwarmEnv) -> np.ndarray:
        """
        For each UAV, move one step toward the nearest uncovered cell.

        Uses Manhattan distance heuristic. Ties broken randomly.
        """
        from uav_swarm_ppo.env.uav import ACTION_DELTAS
        actions = []

        for uav in env.uavs:
            # Find nearest uncovered valid cell
            best_dist = float("inf")
            best_action = 0  # default: stay

            for row, col in env.field.valid_cells:
                if env.coverage_map[row, col] == 0:
                    dist = abs(uav.row - row) + abs(uav.col - col)
                    if dist < best_dist:
                        best_dist = dist
                        # Choose action that reduces distance
                        dr = row - uav.row
                        dc = col - uav.col
                        if abs(dr) >= abs(dc):
                            best_action = 1 if dr < 0 else 2  # N or S
                        else:
                            best_action = 3 if dc > 0 else 4  # E or W

            actions.append(best_action)

        return np.array(actions, dtype=np.int64)

    def evaluate(
        self,
        n_seeds: int = 5,
        n_episodes_per_seed: int = 5,
    ) -> Dict[str, float]:
        """Run greedy baseline evaluation."""
        evaluator = Evaluator(
            config=self.config,
            model_path="",
            results_dir=Path(self.config.logging.results_dir) / "baselines",
        )
        all_metrics = []
        for seed_offset in range(n_seeds):
            from copy import deepcopy
            cfg = deepcopy(self.config)
            cfg.env.seed = self.config.env.seed + seed_offset * 1000
            env = UAVSwarmEnv(cfg)
            for _ in range(n_episodes_per_seed):
                m = evaluator._run_episode(env, use_model=False,
                                          policy_fn=self._policy)
                all_metrics.append(m)
            env.close()

        evaluator._export_csv(all_metrics, "Greedy")
        return evaluator._summarise(all_metrics, "Greedy")
