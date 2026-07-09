"""
env/obstacles.py
================
Obstacle management module.

Handles two obstacle types:
    StaticObstacle  — fixed position (trees, pylons, buildings).
    DynamicObstacle — moves along a cardinal direction each step,
                      bouncing off field boundaries and other obstacles
                      (tractors, animals, vehicles).

Architecture note:
    ObstacleManager owns all obstacles and exposes a clean interface
    to the environment's step() and reset() methods. The environment
    never manipulates obstacle state directly — it always goes through
    ObstacleManager. This keeps the environment's step() clean and
    makes obstacle logic independently testable.
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from uav_swarm_ppo.env.field import Field


# Cardinal movement directions: (delta_row, delta_col)
DIRECTIONS: List[Tuple[int, int]] = [
    (-1, 0),  # North
    (1, 0),   # South
    (0, 1),   # East
    (0, -1),  # West
]


class StaticObstacle:
    """
    A permanently fixed obstacle occupying a single grid cell.

    Parameters
    ----------
    row : int
        Row position of the obstacle.
    col : int
        Column position of the obstacle.
    """

    def __init__(self, row: int, col: int) -> None:
        self.row = row
        self.col = col

    @property
    def position(self) -> Tuple[int, int]:
        """Return (row, col) position."""
        return (self.row, self.col)

    def __repr__(self) -> str:
        return f"StaticObstacle(row={self.row}, col={self.col})"


class DynamicObstacle:
    """
    A moving obstacle that steps one cell per timestep.

    The obstacle travels in a fixed cardinal direction and reverses
    when it would leave the valid field area or collide with a static
    obstacle. This creates a bouncing patrol pattern that tests the
    UAV agents' collision avoidance capabilities.

    Parameters
    ----------
    row : int
        Initial row position.
    col : int
        Initial column position.
    direction_idx : int
        Index into DIRECTIONS list (0=N, 1=S, 2=E, 3=W).
    speed : int
        Number of cells to move per timestep.
    """

    def __init__(
        self,
        row: int,
        col: int,
        direction_idx: int,
        speed: int = 1,
    ) -> None:
        self.row = row
        self.col = col
        self.direction_idx = direction_idx % len(DIRECTIONS)
        self.speed = speed

        # Store initial state for reset
        self._init_row = row
        self._init_col = col
        self._init_dir = direction_idx

    @property
    def position(self) -> Tuple[int, int]:
        """Return current (row, col) position."""
        return (self.row, self.col)

    def step(
        self,
        field: Field,
        blocked_cells: Set[Tuple[int, int]],
    ) -> None:
        """
        Advance the obstacle by ``speed`` cells in its current direction.

        If the next cell is invalid (outside field) or blocked (static
        obstacle), the direction is reversed and the obstacle stays put
        for this timestep. This mirrors realistic object behaviour
        (a tractor reaches the field edge and turns around).

        Parameters
        ----------
        field : Field
            The farm field — used to validate candidate positions.
        blocked_cells : Set[Tuple[int, int]]
            Cells containing static obstacles (cannot pass through).
        """
        for _ in range(self.speed):
            dr, dc = DIRECTIONS[self.direction_idx]
            next_row = self.row + dr
            next_col = self.col + dc

            # Check if next position is valid and unblocked
            if (
                field.is_valid_cell(next_row, next_col)
                and (next_row, next_col) not in blocked_cells
            ):
                # Move forward
                self.row = next_row
                self.col = next_col
            else:
                # Bounce: reverse direction
                self.direction_idx = (self.direction_idx + 2) % len(DIRECTIONS)
                # Do not move this timestep

    def reset(self) -> None:
        """Restore the obstacle to its initial position and direction."""
        self.row = self._init_row
        self.col = self._init_col
        self.direction_idx = self._init_dir

    def __repr__(self) -> str:
        dir_name = ["N", "S", "E", "W"][self.direction_idx]
        return (
            f"DynamicObstacle(row={self.row}, col={self.col}, "
            f"dir={dir_name}, speed={self.speed})"
        )


class ObstacleManager:
    """
    Manages all static and dynamic obstacles in the environment.

    Responsibilities:
        - Placing obstacles randomly within valid field cells at reset
        - Advancing dynamic obstacles each timestep
        - Providing fast collision-check lookups via sets
        - Exposing obstacle positions for observation encoding

    Parameters
    ----------
    field : Field
        The farm field geometry.
    n_static : int
        Number of static obstacles to place.
    n_dynamic : int
        Number of dynamic obstacles to place.
    dynamic_speed : int
        Movement speed for all dynamic obstacles (cells/step).
    rng : np.random.Generator
        Seeded random generator for reproducible placement.
    """

    def __init__(
        self,
        field: Field,
        n_static: int,
        n_dynamic: int,
        dynamic_speed: int,
        rng: np.random.Generator,
    ) -> None:
        self.field = field
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self.dynamic_speed = dynamic_speed
        self.rng = rng

        # These are populated by reset()
        self.static_obstacles: List[StaticObstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # Fast lookup sets — updated each step
        self._static_cells: Set[Tuple[int, int]] = set()
        self._dynamic_cells: Set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def static_cells(self) -> Set[Tuple[int, int]]:
        """Set of cells occupied by static obstacles."""
        return self._static_cells

    @property
    def dynamic_cells(self) -> Set[Tuple[int, int]]:
        """Set of cells currently occupied by dynamic obstacles."""
        return self._dynamic_cells

    @property
    def all_obstacle_cells(self) -> Set[Tuple[int, int]]:
        """Union of static and dynamic obstacle cells."""
        return self._static_cells | self._dynamic_cells

    def reset(self, reserved_cells: Set[Tuple[int, int]]) -> None:
        """
        Randomly place all obstacles within the valid field.

        Obstacles will not be placed on reserved_cells (e.g. UAV
        starting positions) to avoid spawning inside agents.

        Parameters
        ----------
        reserved_cells : Set[Tuple[int, int]]
            Cells that must remain obstacle-free after placement.
        """
        # Determine the pool of candidate cells
        candidate_cells = list(
            self.field.valid_cells - reserved_cells
        )
        self.rng.shuffle(candidate_cells)

        # Place static obstacles
        self.static_obstacles = []
        for i in range(self.n_static):
            if i >= len(candidate_cells):
                break  # Fewer valid cells than requested obstacles
            row, col = candidate_cells[i]
            self.static_obstacles.append(StaticObstacle(row, col))

        # Update static lookup set immediately
        self._static_cells = {obs.position for obs in self.static_obstacles}

        # Remove static obstacle cells from dynamic pool
        used = set(candidate_cells[: self.n_static])
        dynamic_pool = [c for c in candidate_cells[self.n_static:] if c not in used]

        # Place dynamic obstacles
        self.dynamic_obstacles = []
        for i in range(self.n_dynamic):
            if i >= len(dynamic_pool):
                break
            row, col = dynamic_pool[i]
            dir_idx = int(self.rng.integers(0, len(DIRECTIONS)))
            self.dynamic_obstacles.append(
                DynamicObstacle(row, col, dir_idx, speed=self.dynamic_speed)
            )

        # Update dynamic lookup set
        self._dynamic_cells = {obs.position for obs in self.dynamic_obstacles}

    def step(self) -> None:
        """
        Advance all dynamic obstacles by one timestep.

        Must be called once per environment step, before collision
        checks are performed.
        """
        for obs in self.dynamic_obstacles:
            obs.step(self.field, self._static_cells)

        # Recompute the dynamic cell lookup set after movement
        self._dynamic_cells = {obs.position for obs in self.dynamic_obstacles}

    def get_static_grid(self) -> np.ndarray:
        """
        Return a binary grid with 1s at static obstacle positions.

        Shape: (grid_size, grid_size). Used for observation encoding.

        Returns
        -------
        np.ndarray
            Binary array of shape (grid_size, grid_size).
        """
        grid = np.zeros(
            (self.field.grid_size, self.field.grid_size), dtype=np.float32
        )
        for row, col in self._static_cells:
            grid[row, col] = 1.0
        return grid

    def get_dynamic_grid(self) -> np.ndarray:
        """
        Return a binary grid with 1s at current dynamic obstacle positions.

        Shape: (grid_size, grid_size). Used for observation encoding.

        Returns
        -------
        np.ndarray
            Binary array of shape (grid_size, grid_size).
        """
        grid = np.zeros(
            (self.field.grid_size, self.field.grid_size), dtype=np.float32
        )
        for row, col in self._dynamic_cells:
            grid[row, col] = 1.0
        return grid
