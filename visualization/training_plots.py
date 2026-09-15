"""
visualization/training_plots.py
================================

Reads Stable-Baselines3's CSV training log (`progress.csv`) and produces
publication-ready PPO training-analysis figures:

1. PPO episode reward curve
2. PPO coverage curve with moving-average trend
3. PPO loss curves with independent y-axes
4. PPO reward-component breakdown with moving-average trends

The source CSV is never modified.

IMPORTANT:
Stable-Baselines3 does not record every metric at every logging row.
Therefore, NaN values are expected for some metrics.

Missing values are NOT:
    - replaced
    - interpolated
    - forward-filled
    - backward-filled
    - converted to zero

Each metric is filtered independently before plotting so x/y alignment
remains correct.

Moving averages are used only for visual trend presentation. The raw
observations remain visible wherever appropriate.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================================
# Configuration
# =========================================================================

DEFAULT_LOG_DIR = "results/tensorboard"
DEFAULT_OUTPUT_DIR = "results/plots"

FIGURE_DPI = 300

# Number of valid plotted observations used for trend smoothing.
# This does NOT alter the underlying dataset.
MOVING_AVERAGE_WINDOW = 10


# =========================================================================
# Helper: clean one x/y series
# =========================================================================

def _valid_series(
    x: pd.Series,
    y: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Return only finite x/y observations.

    No interpolation or imputation is performed.
    """

    x_numeric = pd.to_numeric(
        x,
        errors="coerce",
    )

    y_numeric = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid = (
        x_numeric.notna()
        & y_numeric.notna()
        & np.isfinite(x_numeric)
        & np.isfinite(y_numeric)
    )

    return (
        x_numeric.loc[valid],
        y_numeric.loc[valid],
    )


# =========================================================================
# Helper: moving average
# =========================================================================

def _moving_average(
    y: pd.Series,
    window: int = MOVING_AVERAGE_WINDOW,
) -> pd.Series:
    """
    Calculate a trailing moving average.

    The moving average is calculated only from valid observations.

    min_periods=1 means the beginning of the curve is not discarded.
    """

    if window <= 1:
        return y.copy()

    return y.rolling(
        window=window,
        min_periods=1,
    ).mean()


# =========================================================================
# Main plotting function
# =========================================================================

def plot_training_curves(
    log_dir: str = DEFAULT_LOG_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Read an SB3 progress.csv and generate publication-ready plots.

    Parameters
    ----------
    log_dir : str
        Directory containing progress.csv.

        Supported layouts:

            log_dir/progress.csv

        or:

            log_dir/PPO_N/progress.csv

        If multiple candidates are found, the most recently modified
        progress.csv is selected.

    output_dir : str
        Directory where PNG figures are saved.
    """

    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------
    # 1. Locate progress.csv
    # ---------------------------------------------------------------------

    csv_path = _find_progress_csv(log_dir)

    if csv_path is None:
        raise FileNotFoundError(
            f"No progress.csv found under '{log_dir}'. "
            "Check that PPO training was run with the CSV logger enabled."
        )

    print(
        f"Reading training log: {csv_path}"
    )

    # ---------------------------------------------------------------------
    # 2. Read CSV
    # ---------------------------------------------------------------------

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(
            f"The training log is empty: {csv_path}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    # ---------------------------------------------------------------------
    # 3. X-axis
    # ---------------------------------------------------------------------

    if "time/total_timesteps" in df.columns:

        x = pd.to_numeric(
            df["time/total_timesteps"],
            errors="coerce",
        )

        x_label = "Timesteps"

        valid_x = x.dropna()

        if not valid_x.empty:
            print(
                f"Timesteps: "
                f"{valid_x.min():.0f} -> "
                f"{valid_x.max():.0f}"
            )

    else:

        print(
            "Warning: 'time/total_timesteps' not found. "
            "Using dataframe index as x-axis."
        )

        x = pd.Series(
            df.index,
            index=df.index,
            dtype="float64",
        )

        x_label = "Training Log Entry"

    # =====================================================================
    # FIGURE 1
    # PPO EPISODE REWARD
    # =====================================================================

    reward_col = "rollout/ep_rew_mean"

    if reward_col in df.columns:

        reward_x, reward_y = _valid_series(
            x,
            df[reward_col],
        )

        if len(reward_y) > 0:

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            # Raw observations.
            ax.plot(
                reward_x,
                reward_y,
                linewidth=1.8,
                label="Mean episode reward",
            )

            ax.set_xlabel(
                x_label
            )

            ax.set_ylabel(
                "Mean Episode Reward"
            )

            ax.set_title(
                "PPO Training: Reward Curve"
            )

            ax.legend()

            ax.grid(
                alpha=0.25
            )

            fig.tight_layout()

            path = os.path.join(
                output_dir,
                "reward_curve.png",
            )

            fig.savefig(
                path,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                f"Saved reward curve: {path} "
                f"({len(reward_y)} valid observations)"
            )

        else:

            print(
                f"Warning: '{reward_col}' has no valid observations. "
                "Skipping reward curve."
            )

    else:

        print(
            f"Warning: '{reward_col}' not found. "
            "Skipping reward curve."
        )

    # =====================================================================
    # FIGURE 2
    # PPO COVERAGE
    # =====================================================================

    coverage_col = "custom/coverage_fraction_mean"

    if coverage_col in df.columns:

        coverage_x, coverage_y = _valid_series(
            x,
            df[coverage_col],
        )

        if len(coverage_y) > 0:

            coverage_ma = _moving_average(
                coverage_y
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            # Raw coverage observations.
            ax.plot(
                coverage_x,
                coverage_y,
                linewidth=1.0,
                alpha=0.35,
                label="Mean coverage fraction",
            )

            # Trend line.
            ax.plot(
                coverage_x,
                coverage_ma,
                linewidth=2.5,
                label=(
                    f"{MOVING_AVERAGE_WINDOW}-point "
                    "moving average"
                ),
            )

            ax.set_xlabel(
                x_label
            )

            ax.set_ylabel(
                "Mean Coverage Fraction"
            )

            ax.set_title(
                "PPO Training: Coverage Curve"
            )

            ax.set_ylim(
                0,
                1
            )

            ax.legend()

            ax.grid(
                alpha=0.25
            )

            fig.tight_layout()

            path = os.path.join(
                output_dir,
                "coverage_curve.png",
            )

            fig.savefig(
                path,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                f"Saved coverage curve: {path} "
                f"({len(coverage_y)} valid observations)"
            )

        else:

            print(
                f"Warning: '{coverage_col}' has no valid observations. "
                "Skipping coverage curve."
            )

    else:

        print(
            f"Warning: '{coverage_col}' not found. "
            "Was SwarmMetricsCallback attached?"
        )

    # =====================================================================
    # FIGURE 3
    # PPO LOSS CURVES
    # =====================================================================

    loss_cols = [
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
    ]

    available_loss_cols = [
        col
        for col in loss_cols
        if col in df.columns
    ]

    if available_loss_cols:

        # Three independent panels are used because the three losses
        # have substantially different numerical scales.
        fig, axes = plt.subplots(
            nrows=len(available_loss_cols),
            ncols=1,
            figsize=(10, 8),
            sharex=True,
        )

        # When only one axis exists, matplotlib does not return a list.
        if len(available_loss_cols) == 1:
            axes = [axes]

        plotted_any = False

        for ax, col in zip(
            axes,
            available_loss_cols,
        ):

            loss_x, loss_y = _valid_series(
                x,
                df[col],
            )

            label = col.split(
                "/",
                1,
            )[-1]

            if len(loss_y) == 0:

                ax.text(
                    0.5,
                    0.5,
                    "No valid observations",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

                ax.set_ylabel(
                    label
                )

                continue

            ax.plot(
                loss_x,
                loss_y,
                linewidth=1.8,
            )

            ax.set_ylabel(
                label
            )

            ax.grid(
                alpha=0.25
            )

            plotted_any = True

            print(
                f"Loss '{label}': "
                f"{len(loss_y)} valid observations"
            )

        axes[-1].set_xlabel(
            x_label
        )

        fig.suptitle(
            "PPO Training: Loss Curves"
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.96]
        )

        path = os.path.join(
            output_dir,
            "loss_curves.png",
        )

        if plotted_any:

            fig.savefig(
                path,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )

            print(
                f"Saved loss curves: {path}"
            )

        plt.close(fig)

    else:

        print(
            "Warning: none of the expected PPO loss columns "
            "were found. Skipping loss curves."
        )

    # =====================================================================
    # FIGURE 4
    # PPO REWARD BREAKDOWN
    # =====================================================================

    category_cols = [
        col
        for col in df.columns
        if (
            col.startswith("custom/reward_")
            and col.endswith("_mean")
            and col != "custom/reward_total_mean"
        )
    ]

    if category_cols:

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        plotted_any = False

        for col in category_cols:

            component_x, component_y = _valid_series(
                x,
                df[col],
            )

            if len(component_y) == 0:

                print(
                    f"Warning: '{col}' contains no valid observations. "
                    "Skipping this component."
                )

                continue

            label = (
                col
                .replace(
                    "custom/reward_",
                    "",
                )
                .replace(
                    "_mean",
                    "",
                )
            )

            # Raw component observations.
            ax.plot(
                component_x,
                component_y,
                linewidth=0.8,
                alpha=0.18,
            )

            # Smoothed trend.
            component_ma = _moving_average(
                component_y
            )

            ax.plot(
                component_x,
                component_ma,
                linewidth=1.8,
                label=label,
            )

            plotted_any = True

        if plotted_any:

            ax.set_xlabel(
                x_label
            )

            ax.set_ylabel(
                "Mean Reward Contribution"
            )

            ax.set_title(
                "PPO Training: Reward Breakdown by Category"
            )

            ax.axhline(
                0,
                linewidth=0.8,
            )

            ax.legend(
                fontsize=8,
                ncol=2,
            )

            ax.grid(
                alpha=0.25
            )

            fig.tight_layout()

            path = os.path.join(
                output_dir,
                "reward_breakdown_curves.png",
            )

            fig.savefig(
                path,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                f"Saved reward breakdown: {path} "
                f"({len(category_cols)} components)"
            )

        else:

            plt.close(fig)

            print(
                "Warning: reward-component columns were found, "
                "but none contained valid observations."
            )

    else:

        print(
            "Warning: no individual custom/reward_*_mean "
            "columns found. Skipping reward breakdown."
        )

    # =====================================================================
    # COMPLETE
    # =====================================================================

    print(
        "\n===== TRAINING CURVE ANALYSIS COMPLETE ====="
    )

    print(
        f"Output directory: {output_dir}"
    )


# =========================================================================
# Locate progress.csv
# =========================================================================

def _find_progress_csv(
    log_dir: str,
) -> Optional[str]:
    """
    Locate progress.csv under log_dir.

    Supported layouts:

        1. log_dir/progress.csv

        2. log_dir/PPO_N/progress.csv

    If multiple candidates are found, the most recently modified file
    is selected.

    The function intentionally searches only the specified logging
    directory and its immediate subdirectories.
    """

    if not os.path.isdir(log_dir):
        return None

    candidates: list[str] = []

    # ---------------------------------------------------------------------
    # Direct layout
    # ---------------------------------------------------------------------

    direct = os.path.join(
        log_dir,
        "progress.csv",
    )

    if os.path.isfile(direct):
        candidates.append(
            direct
        )

    # ---------------------------------------------------------------------
    # Nested layout
    # ---------------------------------------------------------------------

    try:

        for name in os.listdir(log_dir):

            nested_dir = os.path.join(
                log_dir,
                name,
            )

            if not os.path.isdir(nested_dir):
                continue

            nested = os.path.join(
                nested_dir,
                "progress.csv",
            )

            if os.path.isfile(nested):
                candidates.append(
                    nested
                )

    except OSError:

        return None

    # ---------------------------------------------------------------------
    # No files found
    # ---------------------------------------------------------------------

    if not candidates:
        return None

    # ---------------------------------------------------------------------
    # Most recently modified candidate
    # ---------------------------------------------------------------------

    return max(
        candidates,
        key=os.path.getmtime,
    )


# =========================================================================
# Direct execution
# =========================================================================

if __name__ == "__main__":

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    if project_root not in sys.path:
        sys.path.insert(
            0,
            project_root,
        )

    plot_training_curves(
        DEFAULT_LOG_DIR,
        DEFAULT_OUTPUT_DIR,
    )