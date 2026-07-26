"""
visualization/training_plots.py
================================
Reads Stable-Baselines3's CSV training log (`progress.csv`, written
because `ppo/train.py` configures the logger with the "csv" format) and
produces the three curves Goal 4 requires: reward, coverage, and loss.

Deliberately reads the CSV (via pandas) rather than TensorBoard's binary
event files -- avoids adding a tensorboard-log-parsing dependency for
something as simple as three line plots, and the CSV is directly
human-inspectable (open it in Excel/pandas yourself if a plot looks odd).
TensorBoard itself remains available too (`tensorboard --logdir
results/tensorboard`) for interactive exploration during training.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def plot_training_curves(log_dir: str = "results/tensorboard",
                          output_dir: str = "results/plots") -> None:
    """
    Parameters
    ----------
    log_dir : str
        Directory passed as `tensorboard_log` to PPO -- SB3 creates an
        auto-incremented subfolder (e.g. `PPO_1/`) inside it containing
        `progress.csv`. This function finds the most recently modified one.
    output_dir : str
        Where to save the resulting PNG figures.
    """
    import matplotlib.pyplot as plt

    csv_path = _find_progress_csv(log_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"No progress.csv found under {log_dir}. Did ppo/train.py run with "
            "the 'csv' logger format enabled (see the configure() call in train.py)?"
        )
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    x = df["time/total_timesteps"] if "time/total_timesteps" in df else df.index

    # --- 1. Reward curve ---
    if "rollout/ep_rew_mean" in df:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, df["rollout/ep_rew_mean"], color="tab:blue")
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Mean episode reward")
        ax.set_title("PPO Training: Reward Curve")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        print("Warning: 'rollout/ep_rew_mean' not found (no episodes completed yet?) -- skipping reward curve.")

    # --- 2. Coverage curve (from SwarmMetricsCallback, ppo/callbacks.py) ---
    if "custom/coverage_fraction_mean" in df:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, df["custom/coverage_fraction_mean"], color="tab:green")
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Mean coverage fraction")
        ax.set_title("PPO Training: Coverage Curve")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(output_dir, "coverage_curve.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        print("Warning: 'custom/coverage_fraction_mean' not found -- was SwarmMetricsCallback attached?")

    # --- 3. Loss curves (policy gradient, value function, entropy) ---
    loss_cols = [c for c in ["train/policy_gradient_loss", "train/value_loss", "train/entropy_loss"]
                 if c in df]
    if loss_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in loss_cols:
            ax.plot(x, df[col], label=col.split("/")[-1])
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Loss")
        ax.set_title("PPO Training: Loss Curves")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        print("Warning: no train/*_loss columns found -- has at least one policy update happened?")

    # --- 4. Bonus: per-category reward breakdown (all 8 Goal-3 categories on one plot) ---
    category_cols = [c for c in df.columns if c.startswith("custom/reward_") and c.endswith("_mean")]
    if category_cols:
        fig, ax = plt.subplots(figsize=(9, 6))
        for col in category_cols:
            label = col.replace("custom/reward_", "").replace("_mean", "")
            ax.plot(x, df[col], label=label)
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Mean reward contribution")
        ax.set_title("PPO Training: Reward Breakdown by Category")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(output_dir, "reward_breakdown_curves.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved training curve plots to {output_dir}/")


def _find_progress_csv(log_dir: str) -> Optional[str]:
    """
    Locates progress.csv under `log_dir`. Checks two layouts, since both
    occur depending on how the logger was configured:
      1. Directly at `log_dir/progress.csv` -- what `ppo/train.py`
         produces (it calls `configure(log_dir, ...)` directly).
      2. Nested under an auto-incremented `log_dir/PPO_N/progress.csv`
         -- what happens if `tensorboard_log` is instead passed straight
         to `PPO(...)` and SB3 manages its own logger internally.
    Returns the most recently modified match if there are several.
    """
    if not os.path.isdir(log_dir):
        return None
    candidates = []
    direct = os.path.join(log_dir, "progress.csv")
    if os.path.isfile(direct):
        candidates.append(direct)
    for name in os.listdir(log_dir):
        nested = os.path.join(log_dir, name, "progress.csv")
        if os.path.isfile(nested):
            candidates.append(nested)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plot_training_curves()
