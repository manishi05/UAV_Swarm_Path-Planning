"""
visualization/renderer.py
==========================
Renders the full SwarmFarmEnv state: field boundary, grid, static
obstacles (color-coded by type), dynamic obstacles (marker-coded by
type), UAV positions, wind direction, and visited-cell shading.

Two output modes, sharing one figure-building routine:
  - `render_frame()`: saves to disk / shows interactively -- used by
    `experiments/run_experiment.py` for paper-quality saved figures.
  - `render_to_array()`: returns an (H, W, 3) uint8 RGB numpy array --
    used by `SwarmFarmEnv.render()` to satisfy Gymnasium's
    `render_mode="rgb_array"` convention (Goal 3 requirement), e.g. for
    feeding into a video recorder wrapper during PPO training.

Matplotlib is imported only inside this module (not in environment/*),
so headless PPO training never pays the import cost or risks a backend
error on a machine without a display.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_OBSTACLE_COLORS = {
    "tree": "forestgreen",
    "pole": "dimgray",
    "building": "saddlebrown",
    "irrigation": "royalblue",
}
_DYNAMIC_MARKERS = {
    "tractor": ("s", "orange"),
    "human": ("o", "red"),
    "animal": ("^", "purple"),
}


class EnvironmentRenderer:
    """
    Stateless renderer: takes the dict returned by
    `SwarmFarmEnv.get_render_state()` and produces a matplotlib figure,
    either saved/shown or returned as an array. Kept separate from the
    env class itself (single-responsibility) and reusable for both live
    rendering during rollouts and static after-the-fact figure
    generation for the paper.
    """

    def _build_figure(self, state: dict, title: str):
        """
        Shared figure-construction logic used by both `render_frame`
        and `render_to_array`, so the two output paths can never
        visually drift apart from each other.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend, safe for headless/array rendering
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon, Circle

        field, mbr, grid = state["field"], state["mbr"], state["grid"]
        visited = state["visited"]

        fig, ax = plt.subplots(figsize=(9, 9))

        # --- Grid cells: visitable/non-visitable/visited/blocked shading ---
        for cell in grid.cells:
            cx, cy = cell.polygon.exterior.xy
            if cell.blocked_static:
                color = "black"
            elif not cell.is_visitable:
                color = "lightgray"
            elif visited[cell.row, cell.col]:
                color = "#cfe8cf"  # pale green = covered
            else:
                color = "white"
            ax.add_patch(MplPolygon(np.column_stack([cx, cy]), closed=True,
                                     facecolor=color, edgecolor="#dddddd", linewidth=0.3, zorder=1))

        # --- Field boundary ---
        field_xy = np.array(field.polygon.exterior.coords)
        ax.plot(field_xy[:, 0], field_xy[:, 1], color="tab:blue", lw=1.5, zorder=2)

        # --- Static obstacles ---
        for obstacle in state["static_obstacles"]:
            color = _OBSTACLE_COLORS.get(obstacle.obstacle_type.value, "black")
            ax.add_patch(Circle((obstacle.x, obstacle.y), obstacle.radius_m,
                                 facecolor=color, edgecolor="black", alpha=0.8, zorder=3))

        # --- Dynamic obstacles ---
        for obstacle in state["dynamic_obstacles"]:
            marker, color = _DYNAMIC_MARKERS.get(obstacle.obstacle_type.value, ("x", "black"))
            ax.plot(obstacle.x, obstacle.y, marker=marker, color=color,
                    markersize=10, zorder=4, markeredgecolor="black")

        # --- UAVs ---
        for uav in state["uavs"]:
            color = "green" if uav.alive else "gray"
            ax.plot(uav.position[0], uav.position[1], marker="*", color=color,
                    markersize=16, zorder=5, markeredgecolor="black")
            ax.annotate(f"UAV{uav.uav_id}", uav.position, textcoords="offset points",
                        xytext=(6, 6), fontsize=8, zorder=5)

        # --- Wind direction arrow ---
        min_x, min_y, max_x, max_y = mbr.bounds
        wind = state["wind_vector"]
        arrow_origin = (min_x + 0.05 * (max_x - min_x), max_y - 0.05 * (max_y - min_y))
        if np.linalg.norm(wind) > 1e-6:
            ax.annotate("", xy=(arrow_origin[0] + wind[0] * 2, arrow_origin[1] + wind[1] * 2),
                        xytext=arrow_origin,
                        arrowprops=dict(arrowstyle="->", color="black", lw=2), zorder=6)
            ax.text(arrow_origin[0], arrow_origin[1] + 3, "wind", fontsize=8)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.tight_layout()
        return fig

    def render_frame(self, state: dict, save_path: Optional[str] = None,
                      title: str = "UAV Swarm Farm Environment", show: bool = False):
        """Builds the figure and saves to disk and/or shows it interactively."""
        import matplotlib.pyplot as plt
        fig = self._build_figure(state, title)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def render_to_array(self, state: dict, title: str = "UAV Swarm Farm Environment") -> np.ndarray:
        """
        Builds the figure and returns it as an (H, W, 3) uint8 RGB array,
        without touching disk. This is what `SwarmFarmEnv.render()` calls
        under `render_mode="rgb_array"` to satisfy the Gymnasium API.
        """
        import matplotlib.pyplot as plt
        fig = self._build_figure(state, title)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = buffer.reshape(height, width, 4)[:, :, :3].copy()  # drop alpha channel
        plt.close(fig)
        return image
