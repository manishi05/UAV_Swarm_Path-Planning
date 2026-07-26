"""
ppo/callbacks.py
=================
Custom Stable-Baselines3 callback for Goal 4.

Only one custom callback is needed: `SwarmMetricsCallback`. Checkpointing
and evaluation reuse SB3's own well-tested `CheckpointCallback` and
`EvalCallback` directly in `ppo/train.py` -- no need to reinvent those.

Why this callback exists: SB3's own logger only knows about generic RL
quantities (episode reward/length, losses) by default. It has no idea
our `info` dict (returned every `step()` by `SwarmFarmEnv`) carries
`coverage_fraction`, `valid_action_fraction`, `collision_counts`, and an
8-category `reward_breakdown` (Goal 3). This callback reads those
directly from the vectorized environment's `infos` list every step and
logs rolling-window means into the *same* SB3 Logger that already writes
to TensorBoard + CSV -- so "coverage curve" and per-category reward
curves show up right alongside the standard PPO curves, on the same
`time/total_timesteps` x-axis, with no separate logging pipeline.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SwarmMetricsCallback(BaseCallback):
    """
    Logs `SwarmFarmEnv`-specific scalars to SB3's Logger every rollout.

    Uses a fixed-size rolling window (not a per-rollout reset) so the
    signal is already smoothed, matching how SB3 itself smooths
    `rollout/ep_rew_mean` via its internal `ep_info_buffer`.
    """

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._window_size = window_size
        self._coverage: Deque[float] = deque(maxlen=window_size)
        self._valid_action: Deque[float] = deque(maxlen=window_size)
        self._reward_categories: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._collision_counts: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def _on_step(self) -> bool:
        """
        Called once per vectorized environment step (i.e., once per
        `n_envs` transitions collected together). `self.locals["infos"]`
        is the list of `info` dicts, one per sub-environment, for this step.

        Returning True means "continue training" -- returning False would
        stop `model.learn()` early, which this callback must never do.
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "coverage_fraction" in info:
                self._coverage.append(info["coverage_fraction"])
            if "valid_action_fraction" in info:
                self._valid_action.append(info["valid_action_fraction"])
            if "reward_breakdown" in info:
                for category, value in info["reward_breakdown"].items():
                    self._reward_categories[category].append(float(value))
            if "collision_counts" in info:
                for ctype, count in info["collision_counts"].items():
                    self._collision_counts[ctype].append(float(count))
        return True

    def _on_rollout_end(self) -> None:
        """
        Flushes windowed means into the SB3 Logger at the end of each
        rollout (every `n_steps * n_envs` transitions, immediately before
        each policy update) -- the same cadence SB3 itself uses for
        `rollout/ep_rew_mean`, so every curve stays aligned on the same
        x-axis when plotted together (see visualization/training_plots.py).
        """
        if self._coverage:
            self.logger.record("custom/coverage_fraction_mean", float(np.mean(self._coverage)))
        if self._valid_action:
            self.logger.record("custom/valid_action_fraction_mean", float(np.mean(self._valid_action)))
        for category, values in self._reward_categories.items():
            if values:
                self.logger.record(f"custom/reward_{category}_mean", float(np.mean(values)))
        for ctype, counts in self._collision_counts.items():
            if counts:
                self.logger.record(f"custom/collisions_{ctype}_mean", float(np.mean(counts)))
