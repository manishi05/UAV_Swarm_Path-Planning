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

import numpy as np

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
            "cumulative_reward": 0.0,
        }
        self._step_records = []
        self._logger.info("Episode %d started (seed=%d).", self._episode_idx, seed)
        return self._episode_idx

    @property
    def verbose_step_logging(self) -> bool:
        """
        Whether this logger will actually use a per-step `step_context`
        if given one. Exposed publicly so `SwarmFarmEnv.step()` can skip
        building the (moderately expensive: per-UAV state extraction,
        wind magnitude, etc.) rich context entirely when verbose logging
        is off -- which is the default, and what bulk PPO/Q-Learning
        training runs use, keeping runtime overhead at effectively zero
        for the common case per the "minimize runtime overhead" requirement.
        """
        return self._verbose

    def log_step(self, reward_breakdown: RewardBreakdown, info: Dict,
                 step_context: Optional[Dict] = None) -> None:
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
            `collision_counts` (cumulative counts for the episode so far
            -- used only for the episode-summary row; per-step collision
            counts, if you need "when did boundary violations begin"
            granularity, come from `step_context` instead, see below).
        step_context : Optional[Dict]
            Full per-timestep simulator state, built by
            `SwarmFarmEnv.step()` (see the `_build_step_context` call
            site there). When provided AND `verbose_step_logging=True`,
            this is what makes the per-step CSV publication-grade:
            UAV positions/velocities/battery, wind vector, per-step (not
            cumulative) collision counts, obstacle counts, and episode
            termination flags -- everything needed to independently
            reconstruct and verify any reported metric without re-running
            the simulation. If `None` (e.g. a caller not yet updated, or
            verbose logging used outside `SwarmFarmEnv`), falls back to
            the minimal step+reward-only record for backward compatibility.
        """
        if self._current is None:
            raise RuntimeError("log_step() called before start_episode(). "
                                "Call start_episode() once per env.reset().")

        c = self._current
        c["steps"] += 1
        breakdown_dict = reward_breakdown.as_dict()
        for key, value in breakdown_dict.items():
            c["breakdown_sums"][key] += value
        c["cumulative_reward"] += breakdown_dict["total"]

        # collision_counts in `info` are already cumulative for the
        # episode (see SwarmFarmEnv._episode_collision_counts), so we
        # take the latest snapshot rather than summing again -- this
        # feeds the episode-SUMMARY row only. Per-step collision counts
        # (for the verbose per-step CSV) come from step_context instead,
        # which carries this step's raw CollisionReport.counts, not the
        # running total -- the two are deliberately different sources
        # for two different questions ("how many total" vs "when did they start").
        c["collision_counts"] = dict(info.get("collision_counts", {}))
        c["coverage_fraction"] = info.get("coverage_fraction", c["coverage_fraction"])
        c["valid_action_fraction"] = info.get("valid_action_fraction", c["valid_action_fraction"])

        if self._verbose:
            if step_context is not None:
                record = self._build_verbose_record(c, breakdown_dict, step_context)
            else:
                # Backward-compatible minimal fallback (pre-upgrade behavior).
                record = {"step": c["steps"], **breakdown_dict}
            self._step_records.append(record)

    @staticmethod
    def _build_verbose_record(episode_state: Dict, breakdown_dict: Dict, ctx: Dict) -> Dict:
        """
        Flattens one timestep's full simulator state into a single,
        pandas-ready dict (no nested lists/tuples/objects -- every value
        is a plain scalar, per the "analysis-friendly, not a Python
        object dump" requirement). Per-UAV vector quantities (position,
        velocity, battery) are expanded into `uav{i}_x`, `uav{i}_y`, etc.
        columns rather than stored as a single stringified list, so a
        reviewer (or `pandas`) can select/plot e.g. `uav0_x` directly
        without a parsing step.
        """
        record = {
            # --- General ---
            "step": episode_state["steps"],
            "episode": episode_state["episode"],
            "seed": episode_state["seed"],
            "time_seconds": ctx["time_seconds"],

            # --- Coverage ---
            "coverage_fraction": ctx["coverage_fraction"],
            "visited_cells": ctx["visited_cells"],
            "total_flyable_cells": ctx["total_flyable_cells"],
            "new_cells_visited_this_step": ctx["new_cells_visited_this_step"],
        }

        # --- UAV state (flattened, one set of columns per UAV) ---
        batteries = []
        for i, uav_state in enumerate(ctx["uav_states"]):
            record[f"uav{i}_x"] = uav_state["x"]
            record[f"uav{i}_y"] = uav_state["y"]
            record[f"uav{i}_vx"] = uav_state["vx"]
            record[f"uav{i}_vy"] = uav_state["vy"]
            record[f"uav{i}_battery_frac"] = uav_state["battery_frac"]
            record[f"uav{i}_alive"] = uav_state["alive"]
            # Raw ActionType ordinal this UAV was commanded this step
            # (0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT -- see
            # environment/drone.py::ActionType). Added so a reviewer can
            # distinguish "policy chose not to move" from "policy moved
            # but was deflected/blocked/collided" when new cells discovered
            # is zero -- new_cells_visited_this_step alone cannot make
            # that distinction, which matters directly for diagnosing a
            # coverage plateau (one of the explicit research questions
            # this logger is required to answer).
            record[f"uav{i}_action"] = uav_state["action"]
            batteries.append(uav_state["battery_frac"])
        # Swarm-level battery summary -- added because "when did batteries
        # become limiting" (an explicit research question in the request)
        # is awkward to answer from N separate per-UAV columns alone when
        # swarm size varies across experiments; these two columns answer
        # it directly and are comparable across different n_uavs settings.
        record["mean_battery_fraction"] = float(np.mean(batteries)) if batteries else 0.0
        record["min_battery_fraction"] = float(np.min(batteries)) if batteries else 0.0
        record["n_alive_uavs"] = sum(1 for u in ctx["uav_states"] if u["alive"])

        # --- Environment ---
        record["wind_x"] = ctx["wind_vector"][0]
        record["wind_y"] = ctx["wind_vector"][1]
        # Added: wind speed as a scalar magnitude. "Did wind affect
        # coverage" (an explicit research question) is a direct
        # correlation between this one column and coverage_fraction /
        # new_cells_visited_this_step -- with only wind_x/wind_y a
        # researcher would have to compute this themselves every time.
        record["wind_speed_mps"] = float(np.hypot(ctx["wind_vector"][0], ctx["wind_vector"][1]))
        record["number_dynamic_obstacles"] = ctx["number_dynamic_obstacles"]
        record["number_static_obstacles"] = ctx["number_static_obstacles"]

        # --- Collision information (PER-STEP counts, not cumulative --
        # this is what makes "when did boundary violations begin"
        # answerable: cumulative counts alone can only tell you the final
        # total, not the onset step) ---
        record["uav_uav_collisions"] = ctx["collisions_this_step"].get("uav_uav", 0)
        record["uav_static_collisions"] = ctx["collisions_this_step"].get("uav_static_obstacle", 0)
        record["uav_dynamic_collisions"] = ctx["collisions_this_step"].get("uav_dynamic_obstacle", 0)
        record["boundary_violations"] = ctx["collisions_this_step"].get("uav_boundary", 0)

        # --- Reward breakdown (all 8 categories + total) ---
        record["exploration"] = breakdown_dict["exploration"]
        record["coverage"] = breakdown_dict["coverage"]
        record["collision_avoidance"] = breakdown_dict["collision_avoidance"]
        record["boundary"] = breakdown_dict["boundary"]
        record["dynamic_obstacle_avoidance"] = breakdown_dict["dynamic_obstacle_avoidance"]
        record["uav_separation"] = breakdown_dict["uav_separation"]
        record["mission_completion"] = breakdown_dict["mission_completion"]
        record["energy_efficiency"] = breakdown_dict["energy_efficiency"]
        record["total_reward"] = breakdown_dict["total"]
        # Added: running cumulative reward within the episode. Lets a
        # reviewer plot the reward curve for a single episode directly
        # from one column (`df.plot(x='step', y='cumulative_reward')`)
        # instead of computing `total_reward.cumsum()` themselves --
        # cheap to store, and doubles as an independent cross-check
        # (their own cumsum should equal this column exactly, or something
        # is wrong with either the logger or their reconstruction).
        record["cumulative_reward"] = episode_state["cumulative_reward"]

        # --- Episode statistics ---
        record["valid_action_fraction"] = ctx["valid_action_fraction"]
        record["mission_completed"] = ctx["mission_completed"]
        record["terminated"] = ctx["terminated"]
        record["truncated"] = ctx["truncated"]

        return record

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
