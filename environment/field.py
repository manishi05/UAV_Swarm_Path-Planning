"""
env/field.py
============
Field geometry module.

Defines the irregular polygon farm field using Shapely and provides
utilities for:
    - Checking whether a grid cell is inside the field boundary
    - Computing the set of all valid cells
    - Converting between grid indices and world coordinates
    - Visualising the field boundary for debugging

Architecture note:
    This module has zero dependency on Gymnasium or SB3. It is a pure
    geometry utility so it can be tested and used independently.
    The UAVSwarmEnv imports Field and delegates all boundary queries to it.
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np
from shapely.geometry import Point, Polygon


class Field:
    """
    Represents the irregular polygon farm field.

    The field is defined by a list of (x, y) vertices that form a
    closed polygon. The polygon is discretised into a grid of cells,
    and each cell is classified as either inside (valid) or outside
    (invalid) the field boundary.

    Parameters
    ----------
    vertices : List[Tuple[float, float]]
        Ordered (x, y) polygon vertices in grid-cell coordinates.
        The polygon will be automatically closed (first == last is not
        required but is accepted).
    grid_size : int
        Side length of the square bounding grid.
    """

    def __init__(
        self,
        vertices: List[Tuple[float, float]],
        grid_size: int,
    ) -> None:
        # Store raw configuration
        self.vertices = vertices
        self.grid_size = grid_size

        # Build the Shapely polygon used for all containment checks.
        # Shapely handles non-convex (concave) polygons correctly,
        # which is essential for realistic field shapes.
        self.polygon: Polygon = Polygon(vertices)

        if not self.polygon.is_valid:
            raise ValueError(
                "Field polygon is invalid (self-intersecting or degenerate). "
                "Check your vertex list."
            )

        # Pre-compute the set of valid cells once at construction time.
        # This avoids repeated Shapely point-in-polygon queries during
        # training, which would be too slow for a tight simulation loop.
        self._valid_cells: Set[Tuple[int, int]] = self._compute_valid_cells()

        # Expose as a sorted list for deterministic iteration order
        self._valid_cells_list: List[Tuple[int, int]] = sorted(self._valid_cells)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def valid_cells(self) -> Set[Tuple[int, int]]:
        """Return the frozenset of all (row, col) cells inside the field."""
        return self._valid_cells

    @property
    def valid_cells_list(self) -> List[Tuple[int, int]]:
        """Return a deterministically ordered list of valid (row, col) cells."""
        return self._valid_cells_list

    @property
    def n_valid_cells(self) -> int:
        """Total number of valid field cells."""
        return len(self._valid_cells)

    def is_valid_cell(self, row: int, col: int) -> bool:
        """
        Check whether a given grid cell is inside the field polygon.

        Uses the pre-computed lookup set for O(1) performance.

        Parameters
        ----------
        row : int
            Row index (y-axis) in the grid.
        col : int
            Column index (x-axis) in the grid.

        Returns
        -------
        bool
            True if the cell centre lies within the field polygon.
        """
        return (row, col) in self._valid_cells

    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check whether indices are within the bounding grid dimensions.

        This is a cheaper check than is_valid_cell and is used as a
        first-pass filter before the polygon containment check.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        bool
            True if 0 <= row < grid_size and 0 <= col < grid_size.
        """
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert a grid cell (row, col) to world (x, y) coordinates.

        The centre of cell (row, col) maps to world point (col + 0.5, row + 0.5).

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        Tuple[float, float]
            World (x, y) coordinates of the cell centre.
        """
        return (col + 0.5, row + 0.5)

    def get_boundary_array(self) -> np.ndarray:
        """
        Return a 2D binary array marking valid cells.

        Shape: (grid_size, grid_size).
        Value 1 = valid field cell, 0 = outside the field.

        Useful for visualisation and for encoding the field
        shape in the agent observation.

        Returns
        -------
        np.ndarray
            Binary grid of shape (grid_size, grid_size).
        """
        boundary = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for row, col in self._valid_cells:
            boundary[row, col] = 1
        return boundary

    def random_valid_cell(self, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Sample a uniformly random valid field cell.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random generator (for reproducibility, always pass
            the environment's seeded generator).

        Returns
        -------
        Tuple[int, int]
            A random (row, col) cell guaranteed to be inside the field.
        """
        idx = rng.integers(0, len(self._valid_cells_list))
        return self._valid_cells_list[idx]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_valid_cells(self) -> Set[Tuple[int, int]]:
        """
        Iterate over every cell in the bounding grid and test whether
        its centre point falls inside the field polygon.

        Uses Shapely's ``contains`` method which correctly handles
        cells on the boundary (they are included via ``touches``).

        Returns
        -------
        Set[Tuple[int, int]]
            Set of (row, col) tuples for all cells inside the polygon.
        """
        valid: Set[Tuple[int, int]] = set()

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Cell centre in world coordinates
                world_x, world_y = self.cell_to_world(row, col)
                point = Point(world_x, world_y)

                # A cell is valid if its centre is inside OR on the boundary
                if self.polygon.contains(point) or self.polygon.touches(point):
                    valid.add((row, col))

        return valid
