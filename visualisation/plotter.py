"""
visualization/plotter.py
========================
Publication-quality plotting module.

Generates all figures needed for the paper:
    1. Training reward curve (with rolling average)
    2. Coverage map heatmap at episode end
    3. Comparison bar chart: PPO vs Greedy vs Random
    4. Collision rate comparison
    5. Field polygon + valid cell visualisation

All figures are saved as high-DPI PNG and PDF for LaTeX inclusion.

Architecture note:
    Plotter has no dependency on the environment or training modules.
    It only reads CSV files produced by Evaluator and numpy arrays.
    This means figures can be regenerated from saved results without
    re-running training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Jupyter use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Style constants (publication-ready)
# ---------------------------------------------------------------------------

COLORS = {
    "ppo": "#1565c0",      # Deep blue — primary algorithm
    "greedy": "#e65100",   # Deep orange — greedy baseline
    "random": "#757575",   # Grey — random baseline
    "coverage": "#43a047", # Green — coverage maps
    "field": "#e8f5e9",    # Light green — field background
    "obstacle_s": "#b71c1c",  # Red — static obstacle
    "obstacle_d": "#ff6f00",  # Amber — dynamic obstacle
}

FIGURE_DPI = 300      # Publication-quality DPI
FONT_SIZE = 10        # Base font size for paper figures


def _setup_style() -> None:
    """Apply consistent matplotlib style for all figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "legend.fontsize": FONT_SIZE - 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


class Plotter:
    """
    Generates all paper figures from training and evaluation data.

    Parameters
    ----------
    results_dir : str or Path
        Directory where CSV files are located and figures will be saved.
    """

    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.fig_dir = self.results_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        _setup_style()

    # ------------------------------------------------------------------
    # Figure 1 — Training reward curve
    # ------------------------------------------------------------------

    def plot_training_curve(
        self,
        rewards: List[float],
        window: int = 50,
        save_name: str = "fig1_training_curve",
    ) -> None:
        """
        Plot episodic rewards with a rolling mean overlay.

        Parameters
        ----------
        rewards : List[float]
            Episode reward values in chronological order.
        window : int
            Rolling average window size.
        save_name : str
            Output filename (without extension).
        """
        fig, ax = plt.subplots(figsize=(5, 3))

        episodes = np.arange(len(rewards))
        rewards_arr = np.array(rewards)

        # Raw rewards (low opacity)
        ax.plot(
            episodes, rewards_arr,
            color=COLORS["ppo"], alpha=0.25, linewidth=0.7,
            label="Episode reward"
        )

        # Rolling mean
        if len(rewards) >= window:
            roll = np.convolve(
                rewards_arr, np.ones(window) / window, mode="valid"
            )
            ax.plot(
                np.arange(window - 1, len(rewards)), roll,
                color=COLORS["ppo"], linewidth=1.8,
                label=f"Rolling mean (n={window})"
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("PPO Training Reward Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        self._save(fig, save_name)

    # ------------------------------------------------------------------
    # Figure 2 — Field and coverage heatmap
    # ------------------------------------------------------------------

    def plot_coverage_map(
        self,
        coverage_map: np.ndarray,
        field_vertices: List[Tuple[float, float]],
        uav_positions: Optional[List[Tuple[int, int]]] = None,
        static_obstacles: Optional[List[Tuple[int, int]]] = None,
        dynamic_obstacles: Optional[List[Tuple[int, int]]] = None,
        save_name: str = "fig2_coverage_map",
    ) -> None:
        """
        Visualise the field polygon with coverage overlay.

        Parameters
        ----------
        coverage_map : np.ndarray
            Binary (grid_size, grid_size) coverage array.
        field_vertices : list of (float, float)
            Polygon vertices for field boundary.
        uav_positions : list of (int, int), optional
            Final UAV positions to overlay as markers.
        static_obstacles : list of (int, int), optional
            Static obstacle positions.
        dynamic_obstacles : list of (int, int), optional
            Dynamic obstacle positions.
        save_name : str
            Output filename.
        """
        G = coverage_map.shape[0]
        fig, ax = plt.subplots(figsize=(5, 5))

        # Field boundary
        verts = np.array(field_vertices)
        from matplotlib.patches import Polygon as MplPolygon
        field_patch = MplPolygon(
            verts, closed=True,
            facecolor=COLORS["field"], edgecolor="#2e7d32",
            linewidth=1.5, zorder=1
        )
        ax.add_patch(field_patch)

        # Coverage heatmap
        masked = np.ma.masked_where(coverage_map == 0, coverage_map)
        ax.imshow(
            masked, origin="lower", cmap="Greens",
            vmin=0, vmax=1, alpha=0.7,
            extent=[0, G, 0, G], zorder=2
        )

        # Static obstacles
        if static_obstacles:
            for row, col in static_obstacles:
                rect = mpatches.Rectangle(
                    (col, row), 1, 1,
                    facecolor=COLORS["obstacle_s"], zorder=3
                )
                ax.add_patch(rect)

        # Dynamic obstacles
        if dynamic_obstacles:
            for row, col in dynamic_obstacles:
                rect = mpatches.Rectangle(
                    (col, row), 1, 1,
                    facecolor=COLORS["obstacle_d"], zorder=3, alpha=0.85
                )
                ax.add_patch(rect)

        # UAV positions
        if uav_positions:
            uav_colors = [COLORS["ppo"], "#6a1b9a", "#00695c", "#f57f17"]
            for i, (row, col) in enumerate(uav_positions):
                ax.plot(
                    col + 0.5, row + 0.5, "^",
                    markersize=9, color=uav_colors[i % len(uav_colors)],
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=5, label=f"UAV {i+1}"
                )

        # Coverage percentage annotation
        cov_pct = coverage_map.sum() / max((coverage_map > -1).sum(), 1) * 100
        ax.text(
            0.02, 0.97, f"Coverage: {cov_pct:.1f}%",
            transform=ax.transAxes, fontsize=9,
            va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )

        ax.set_xlim(0, G)
        ax.set_ylim(0, G)
        ax.set_xlabel("Column (x)")
        ax.set_ylabel("Row (y)")
        ax.set_title("UAV Swarm Coverage Map")
        if uav_positions:
            ax.legend(loc="lower right", fontsize=8)

        # Legend for obstacles
        legend_patches = [
            mpatches.Patch(color=COLORS["obstacle_s"], label="Static obstacle"),
            mpatches.Patch(color=COLORS["obstacle_d"], label="Dynamic obstacle"),
            mpatches.Patch(color=COLORS["coverage"], label="Covered cell"),
        ]
        ax.legend(handles=legend_patches, loc="lower left", fontsize=7)

        fig.tight_layout()
        self._save(fig, save_name)

    # ------------------------------------------------------------------
    # Figure 3 — Algorithm comparison bar chart
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        summaries: Dict[str, Dict[str, float]],
        save_name: str = "fig3_comparison",
    ) -> None:
        """
        Bar chart comparing PPO, Greedy, and Random baselines.

        Parameters
        ----------
        summaries : dict
            Keys are policy names ('PPO', 'Greedy', 'Random').
            Values are summary dicts from Evaluator._summarise().
        save_name : str
            Output filename.
        """
        policies = list(summaries.keys())
        color_map = {
            "PPO": COLORS["ppo"],
            "Greedy": COLORS["greedy"],
            "Random": COLORS["random"],
        }

        # Two subplots: coverage and collisions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

        # -- Coverage ---
        means = [summaries[p]["coverage_mean"] * 100 for p in policies]
        stds = [summaries[p]["coverage_std"] * 100 for p in policies]
        bars = ax1.bar(
            policies, means, yerr=stds,
            color=[color_map.get(p, "#888") for p in policies],
            capsize=5, edgecolor="white", linewidth=0.8
        )
        ax1.set_ylabel("Coverage (%)")
        ax1.set_title("Field Coverage")
        ax1.set_ylim(0, 110)
        for bar, m, s in zip(bars, means, stds):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                m + s + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=8
            )

        # -- Collisions ---
        means_c = [summaries[p]["collisions_mean"] for p in policies]
        stds_c = [summaries[p]["collisions_std"] for p in policies]
        bars2 = ax2.bar(
            policies, means_c, yerr=stds_c,
            color=[color_map.get(p, "#888") for p in policies],
            capsize=5, edgecolor="white", linewidth=0.8
        )
        ax2.set_ylabel("Collision Count")
        ax2.set_title("Collision Events")
        for bar, m, s in zip(bars2, means_c, stds_c):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                m + s + 0.2,
                f"{m:.1f}", ha="center", va="bottom", fontsize=8
            )

        fig.suptitle(
            "Algorithm Comparison (mean ± std, 5 seeds × 5 episodes)",
            fontsize=9
        )
        fig.tight_layout()
        self._save(fig, save_name)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> None:
        """
        Save figure as both PNG (for viewing) and PDF (for LaTeX).

        Parameters
        ----------
        fig : plt.Figure
            The figure to save.
        name : str
            Filename without extension.
        """
        for ext in ["png", "pdf"]:
            path = self.fig_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plotter] Saved {name}.png / .pdf to {self.fig_dir}")
