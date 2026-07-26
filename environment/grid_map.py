"""
environment/grid_map.py
========================
Converts a FarmFieldPolygon + MinimumBoundingRectangle into a grid of
GridCell objects, labeled visitable/non-visitable. Reproduces Fig. 3c
of the reference paper (Section 3.2, steps 2-3), with the labeling rule
made explicit (see CellLabelRule design decision below) since the paper
does not specify one.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, box, Point

from configs.config import FieldConfig, MBRMode, CellLabelRule
from environment.cell import GridCell
from environment.polygon_field import FarmFieldPolygon, MinimumBoundingRectangle

logger = logging.getLogger("uav_swarm_ppo.environment.grid_map")


class GridMapGenerator:
    """
    Discretizes the MBR into GridCell objects and labels each visitable
    or non-visitable relative to the field boundary.

    Design decision (why AREA_OVERLAP is the default, not CENTROID):
    a naive centroid-in-polygon test can mislabel cells that are mostly
    inside an irregular/concave field boundary but whose exact center
    happens to fall outside (or the reverse). AREA_OVERLAP thresholds on
    the fraction of a cell's own area that intersects the field, which
    is more geometrically faithful for the non-square, non-convex fields
    this project targets (unlike the reference paper's square maps,
    where the distinction barely matters).
    """

    def __init__(self, field: FarmFieldPolygon, mbr: MinimumBoundingRectangle,
                 config: FieldConfig, seed: int = 42):
        self._field = field
        self._mbr = mbr
        self._config = config
        self._seed = seed
        self._cells: List[GridCell] = []
        self._n_rows = 0
        self._n_cols = 0
        self._generate()

    def _generate(self) -> None:
        """Dispatches to axis-aligned or rotated grid construction."""
        cell_size = self._config.cell_size_m
        rect = self._mbr.rectangle

        if self._mbr.mode == MBRMode.AXIS_ALIGNED:
            self._generate_axis_aligned(rect, cell_size)
        else:
            self._generate_rotated(rect, cell_size)

        n_visitable = sum(c.is_visitable for c in self._cells)
        logger.info(
            "Grid generated: %d rows x %d cols = %d cells (%d visitable, %.1f%%)",
            self._n_rows, self._n_cols, len(self._cells), n_visitable,
            100.0 * n_visitable / max(len(self._cells), 1),
        )

    def _generate_axis_aligned(self, rect: Polygon, cell_size: float) -> None:
        """Row-major cell sweep over an axis-aligned rectangle."""
        min_x, min_y, max_x, max_y = rect.bounds
        width, height = max_x - min_x, max_y - min_y
        self._n_cols = int(np.ceil(width / cell_size))
        self._n_rows = int(np.ceil(height / cell_size))

        for row in range(self._n_rows):
            for col in range(self._n_cols):
                cx0 = min_x + col * cell_size
                cy0 = min_y + row * cell_size
                cell_poly = box(cx0, cy0, cx0 + cell_size, cy0 + cell_size)
                center_x, center_y = cx0 + cell_size / 2, cy0 + cell_size / 2
                is_visitable = self._label_cell(cell_poly, center_x, center_y)
                self._cells.append(GridCell(
                    row=row, col=col, center_x=center_x, center_y=center_y,
                    polygon=cell_poly, is_visitable=is_visitable,
                ))

    def _generate_rotated(self, rect: Polygon, cell_size: float) -> None:
        """
        Cell sweep in the rotated rectangle's own local (u, v) edge
        basis, so cells tile the rectangle regardless of its rotation
        relative to the world x/y axes.
        """
        corners = list(rect.exterior.coords)[:4]
        origin = np.array(corners[0])
        u_vec, v_vec = np.array(corners[1]) - origin, np.array(corners[3]) - origin
        width, height = np.linalg.norm(u_vec), np.linalg.norm(v_vec)
        u_hat, v_hat = u_vec / width, v_vec / height

        self._n_cols = int(np.ceil(width / cell_size))
        self._n_rows = int(np.ceil(height / cell_size))

        for row in range(self._n_rows):
            for col in range(self._n_cols):
                base = origin + col * cell_size * u_hat + row * cell_size * v_hat
                p00, p10 = base, base + cell_size * u_hat
                p11, p01 = base + cell_size * (u_hat + v_hat), base + cell_size * v_hat
                cell_poly = Polygon([p00, p10, p11, p01])
                center = (p00 + p11) / 2.0
                is_visitable = self._label_cell(cell_poly, center[0], center[1])
                self._cells.append(GridCell(
                    row=row, col=col, center_x=float(center[0]), center_y=float(center[1]),
                    polygon=cell_poly, is_visitable=is_visitable,
                ))

    def _label_cell(self, cell_poly: Polygon, center_x: float, center_y: float) -> bool:
        """Applies the configured CellLabelRule to one cell."""
        field_poly = self._field.polygon
        if self._config.label_rule == CellLabelRule.CENTROID:
            return field_poly.contains(Point(center_x, center_y))
        if not cell_poly.intersects(field_poly):
            return False
        overlap_fraction = cell_poly.intersection(field_poly).area / cell_poly.area
        return overlap_fraction >= self._config.overlap_threshold

    # ------------------------------------------------------------------
    @property
    def cells(self) -> List[GridCell]:
        return self._cells

    @property
    def shape(self) -> Tuple[int, int]:
        return self._n_rows, self._n_cols

    @property
    def cell_size_m(self) -> float:
        """
        Physical cell edge length in meters. Public accessor (added for
        Goal 4's A* baseline, `baselines/astar.py`, which needs this to
        convert a physical obstacle-avoidance buffer distance into a
        number of grid cells) -- avoids external code reaching into the
        private `_config` attribute directly.
        """
        return self._config.cell_size_m

    def get_cell(self, row: int, col: int) -> Optional[GridCell]:
        """O(1) lookup by (row, col); returns None if out of bounds."""
        if 0 <= row < self._n_rows and 0 <= col < self._n_cols:
            idx = row * self._n_cols + col
            if idx < len(self._cells):
                return self._cells[idx]
        return None

    def get_cell_at_point(self, x: float, y: float) -> Optional[GridCell]:
        """
        Locate the GridCell containing world-space point (x, y). Used by
        drone.py to map a UAV's continuous physical position back to a
        discrete cell for coverage bookkeeping every physics step.
        """
        min_x, min_y, _, _ = self._mbr.bounds
        cell_size = self._config.cell_size_m
        if self._mbr.mode == MBRMode.AXIS_ALIGNED:
            col = int((x - min_x) / cell_size)
            row = int((y - min_y) / cell_size)
            return self.get_cell(row, col)
        # For rotated MBRs, fall back to an exact geometric search since
        # simple index arithmetic doesn't hold in a rotated frame. This
        # path is called once per UAV per step, so a linear scan is
        # acceptable at the grid sizes used in this project (a KD-tree
        # index would be the scalable follow-up noted in Future Work).
        pt = Point(x, y)
        for cell in self._cells:
            if cell.polygon.contains(pt):
                return cell
        return None

    def visitable_mask(self) -> np.ndarray:
        """(n_rows, n_cols) boolean array, True = inside field boundary."""
        mask = np.zeros((self._n_rows, self._n_cols), dtype=bool)
        for c in self._cells:
            mask[c.row, c.col] = c.is_visitable
        return mask

    def flyable_mask(self) -> np.ndarray:
        """
        (n_rows, n_cols) boolean array, True = visitable AND not blocked
        by a static obstacle. This is the mask PPO's observation and the
        reward function actually care about (Goal 2 extension of
        visitable_mask).
        """
        mask = np.zeros((self._n_rows, self._n_cols), dtype=bool)
        for c in self._cells:
            mask[c.row, c.col] = c.is_flyable
        return mask

    def assign_start_cells(self, n_uavs: int, seed: Optional[int] = None) -> List[GridCell]:
        """
        Deterministically (seeded) assigns n_uavs distinct flyable cells
        as UAV start positions. Called AFTER obstacle placement in the
        pipeline (see swarm_env.py) so start cells never coincide with a
        static obstacle.
        """
        candidates = [c for c in self._cells if c.is_flyable]
        if n_uavs > len(candidates):
            raise ValueError(f"Requested {n_uavs} start cells but only {len(candidates)} flyable cells exist.")
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        chosen_idx = rng.choice(len(candidates), size=n_uavs, replace=False)
        chosen = [candidates[i] for i in chosen_idx]
        for c in chosen:
            c.is_start = True
        logger.info("Assigned %d start cell(s).", n_uavs)
        return chosen

    def to_json(self, path: str) -> None:
        """Serializes cell labels (not full geometry) for caching/reuse."""
        payload = {
            "n_rows": self._n_rows, "n_cols": self._n_cols,
            "cells": [
                {"row": c.row, "col": c.col, "center_x": c.center_x, "center_y": c.center_y,
                 "is_visitable": c.is_visitable, "is_start": c.is_start,
                 "blocked_static": c.blocked_static}
                for c in self._cells
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Grid map written to %s", path)
