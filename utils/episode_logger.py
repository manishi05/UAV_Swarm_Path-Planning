"""
utils/episode_logger.py
=========================
Logging framework for Goal 3: rewards (full per-category breakdown),
coverage, collisions, and episode statistics.

Two outputs per experiment run:
  1. A human-readable log (via `utils.logger.get_logger`) with one INFO
     line per episode summary -- for watching training progress live.
  2. A CSV file (`<experiment_name>_episodes.csv`) with one row per
     episode -- for the Day 4 evaluation harness and statistical tests
     (Shapiro-Wilk/ANOVA across seeds, matching the reference paper's
     own methodology), and for plotting reward-component curves to
     diagnose reward-shaping problems during PPO training.

Deliberately does NOT depend on pandas (keeps the dependency list from
Goal 1/2 unchanged) -- uses the stdlib `csv` module.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, Optional

from environment.collision import CollisionType
from environment.reward import RewardBreakdown
from utils.logger import get_logger

# Fixed, stable set of CSV columns, determined up front from the known
# RewardBreakdown categories and CollisionType enum (rather than
# discovered dynamically per episode). This lets `_write_summary_row`
# append in O(1) per episode instead of reading back and rewriting the
# whole file -- important because PPO training (Day 3) will produce
# thousands of episodes, and an O(n^2) rewrite-per-episode strategy
# would become the dominant cost of training.
_REWARD_CATEGORY_COLUMNS = [
    "reward_exploration", "reward_coverage", "reward_collision_avoidance",
    "reward_boundary", "reward_dynamic_obstacle_avoidance", "reward_uav_separation",
    "reward_mission_completion", "reward_energy_efficiency",
]
_COLLISION_COLUMNS = [f"collisions_{t.value}" for t in CollisionType]
_SUMMARY_FIELDNAMES = (
    ["episode", "seed", "steps", "total_reward", "coverage_fraction", "valid_action_fraction"]
    + _REWARD_CATEGORY_COLUMNS + _COLLISION_COLUMNS
)


class EpisodeLogger:
    """
    Accumulates per-step data across one episode and flushes an
    episode-level summary row on `end_episode()`. One instance is
    typically created per experiment/training run and passed into
    `SwarmFarmEnv(..., episode_logger=...)`, which calls
    `start_episode` / `log_step` / `end_episode` automatically.
    """

    def __init__(self, log_dir: str, experiment_name: str = "experiment",
                 verbose_step_logging: bool = False):
        """
        Parameters
        ----------
        log_dir : str
            Directory for the CSV summary file and the human-readable
            log file (created if it doesn't exist).
        experiment_name : str
            Prefix for output filenames, so multiple experiments'
            outputs can coexist in the same `results/` directory
            without overwriting each other.
        verbose_step_logging : bool
            If True, also buffers every individual step's reward
            breakdown in memory and writes a separate per-step CSV per
            episode on `end_episode()` -- useful for deep-diving into a
            single episode's reward dynamics, but generates one file per
            episode, so leave False for large-scale multi-seed runs.
        """
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self._experiment_name = experiment_name
        self._verbose = verbose_step_logging

        self._summary_path = os.path.join(log_dir, f"{experiment_name}_episodes.csv")
        self._logger = get_logger(
            "uav_swarm_ppo.episode_logger",
            log_file=os.path.join(log_dir, f"{experiment_name}.log"),
        )

        self._episode_idx = -1
        self._current: Optional[dict] = None
        self._step_records: list = []

    def start_episode(self, seed: int) -> int:
        """
        Begins a new episode's accumulator. Must be called once per
        `env.reset()`. Returns the episode index (0-based, incrementing).
        """
        self._episode_idx += 1
        self._current = {
            "episode": self._episode_idx,
            "seed": seed,
            "steps": 0,
            "breakdown_sums": defaultdict(float),
            "collision_counts": defaultdict(int),
            "coverage_fraction": 0.0,
            "valid_action_fraction": 0.0,
        }
        self._step_records = []
        self._logger.info("Episode %d started (seed=%d).", self._episode_idx, seed)
        return self._episode_idx

    def log_step(self, reward_breakdown: RewardBreakdown, info: Dict) -> None:
        """
        Records one step's outcome. Called once per `env.step()`.

        Parameters
        ----------
        reward_breakdown : RewardBreakdown
            The swarm-level (already-summed-across-UAVs) reward breakdown
            for this step.
        info : Dict
            The `info` dict returned by `SwarmFarmEnv.step()`, containing
            `coverage_fraction`, `valid_action_fraction`, and
            `collision_counts` (cumulative counts for the episode so far).
        """
        if self._current is None:
            raise RuntimeError("log_step() called before start_episode(). "
                                "Call start_episode() once per env.reset().")

        c = self._current
        c["steps"] += 1
        for key, value in reward_breakdown.as_dict().items():
            c["breakdown_sums"][key] += value

        # collision_counts in `info` are already cumulative for the
        # episode (see SwarmFarmEnv._episode_collision_counts), so we
        # take the latest snapshot rather than summing again.
        c["collision_counts"] = dict(info.get("collision_counts", {}))
        c["coverage_fraction"] = info.get("coverage_fraction", c["coverage_fraction"])
        c["valid_action_fraction"] = info.get("valid_action_fraction", c["valid_action_fraction"])

        if self._verbose:
            record = {"step": c["steps"], **reward_breakdown.as_dict()}
            self._step_records.append(record)

    def end_episode(self) -> Dict:
        """
        Finalizes the current episode: writes one summary row to the
        CSV, logs a human-readable summary line, and (if
        verbose_step_logging) writes the per-step detail CSV.

        Returns
        -------
        Dict
            The summary row that was written, for programmatic use
            (e.g., by an evaluation script collecting results in memory
            without re-reading the CSV).
        """
        c = self._current
        summary = {
            "episode": c["episode"],
            "seed": c["seed"],
            "steps": c["steps"],
            "total_reward": c["breakdown_sums"].get("total", 0.0),
            "coverage_fraction": c["coverage_fraction"],
            "valid_action_fraction": c["valid_action_fraction"],
        }
        for key, value in c["breakdown_sums"].items():
            if key != "total":
                summary[f"reward_{key}"] = value
        for key, value in c["collision_counts"].items():
            summary[f"collisions_{key}"] = value

        self._write_summary_row(summary)
        self._logger.info(
            "Episode %d ended: steps=%d total_reward=%.2f coverage=%.3f "
            "valid_action_frac=%.3f collisions=%s",
            c["episode"], c["steps"], summary["total_reward"],
            c["coverage_fraction"], c["valid_action_fraction"], c["collision_counts"],
        )

        if self._verbose:
            self._write_step_csv(c["episode"])

        self._current = None
        return summary

    def _write_summary_row(self, summary: Dict) -> None:
        """
        Appends one row to the episode-summary CSV in O(1), using the
        fixed `_SUMMARY_FIELDNAMES` column set. Any key in `summary` not
        in that fixed set (e.g., a collision type that was never
        registered in `CollisionType`) is silently dropped by
        `extrasaction="ignore"` rather than raising -- a defensive
        choice so a future new collision type doesn't crash a long
        training run; add it to `CollisionType` AND `_COLLISION_COLUMNS`
        together if you extend the collision system.
        """
        file_exists = os.path.exists(self._summary_path)
        with open(self._summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

    def _write_step_csv(self, episode_idx: int) -> None:
        """Writes the full per-step reward breakdown for one episode (verbose mode only)."""
        path = os.path.join(self._log_dir, f"{self._experiment_name}_episode{episode_idx}_steps.csv")
        if not self._step_records:
            return
        fieldnames = list(self._step_records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._step_records)
        self._logger.info("Per-step detail written to %s", path)

    def close(self) -> None:
        """
        Finalizes logging. Safe to call even if an episode is mid-flight
        (e.g., training interrupted) -- does not raise if there's no
        open episode to flush.
        """
        if self._current is not None:
            self._logger.warning(
                "EpisodeLogger.close() called with an episode still in "
                "progress (episode %d); it will not be written to the "
                "summary CSV.", self._current["episode"],
            )
        self._logger.info("EpisodeLogger closed. Summary at %s", self._summary_path)
