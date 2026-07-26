"""
environment/obstacle.py
========================
Static and dynamic obstacles for a realistic agricultural environment.

Why this exists (gap vs. the reference paper):
Section 7 ("Future work") of Puente-Castro et al. (2022) explicitly
names this as unsolved: "Experiments will be made with maps with
obstacles ... Obstacles can be fixed obstacles (trees, poles, etc.) or
dynamic obstacles (birds, other UAVs, etc.)." This module is our
implementation of that gap, extended to the specific real-world entities
relevant to a farm: trees, electric poles, buildings, and irrigation
equipment (static); tractors, human workers, and birds/animals (dynamic).

Design pattern: obstacles are represented uniformly (position + radius +
type) so collision.py and reward.py can treat all static obstacles (and
all dynamic obstacles) polymorphically, while a Strategy pattern
(`MovementStrategy`) lets each dynamic obstacle type move differently
without subclassing `DynamicObstacle` itself -- keeping the obstacle
count/type list in `ObstacleConfig` the only place new obstacle types
need to be registered.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

from configs.config import ObstacleConfig
from environment.cell import GridCell
from environment.grid_map import GridMapGenerator
from environment.polygon_field import FarmFieldPolygon

logger = logging.getLogger("uav_swarm_ppo.environment.obstacle")


class ObstacleType(str, Enum):
    """All obstacle categories modeled in this project."""
    TREE = "tree"
    POLE = "pole"
    BUILDING = "building"
    IRRIGATION = "irrigation"
    TRACTOR = "tractor"
    HUMAN = "human"
    ANIMAL = "animal"


STATIC_TYPES = {ObstacleType.TREE, ObstacleType.POLE, ObstacleType.BUILDING, ObstacleType.IRRIGATION}
DYNAMIC_TYPES = {ObstacleType.TRACTOR, ObstacleType.HUMAN, ObstacleType.ANIMAL}


@dataclass
class StaticObstacle:
    """
    An immovable obstacle (tree, electric pole, building, irrigation
    equipment). Represented as a circular footprint of given radius --
    sufficient for collision distance checks and cell-blocking, and a
    deliberate simplification vs. exact building footprints (documented
    limitation, extensible to Shapely Polygons per-obstacle if a later
    revision needs exact building shapes).
    """
    obstacle_id: int
    obstacle_type: ObstacleType
    x: float
    y: float
    radius_m: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def footprint(self) -> Polygon:
        """Circular footprint as a Shapely polygon, for exact intersection tests."""
        return Point(self.x, self.y).buffer(self.radius_m)


class MovementStrategy(ABC):
    """
    Strategy interface for how a dynamic obstacle moves each timestep.
    Separating movement behavior from the `DynamicObstacle` data class
    (rather than subclassing DynamicObstacle per type) means a new
    behavior (e.g., a herding animal flock) can be added without
    touching obstacle placement, collision, or reward code.
    """

    @abstractmethod
    def step(self, position: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Returns the obstacle's new position after advancing by dt seconds."""
        raise NotImplementedError


class WaypointPatrolStrategy(MovementStrategy):
    """
    Moves toward a cyclic list of waypoints at a fixed speed, looping
    back to the first waypoint after reaching the last. Models a
    tractor's back-and-forth row-following pattern in a field.
    """

    def __init__(self, waypoints: List[Tuple[float, float]], speed_mps: float, arrival_tolerance_m: float = 1.0):
        if len(waypoints) < 2:
            raise ValueError("WaypointPatrolStrategy requires at least 2 waypoints.")
        self._waypoints = [np.array(w) for w in waypoints]
        self._speed = speed_mps
        self._tolerance = arrival_tolerance_m
        self._target_idx = 1

    def step(self, position: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        target = self._waypoints[self._target_idx]
        direction = target - position
        distance = np.linalg.norm(direction)
        if distance < self._tolerance:
            self._target_idx = (self._target_idx + 1) % len(self._waypoints)
            target = self._waypoints[self._target_idx]
            direction = target - position
            distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return position
        step_dist = min(self._speed * dt, distance)
        return position + (direction / distance) * step_dist


class BoundedRandomWalkStrategy(MovementStrategy):
    """
    Random-direction walk at approximately constant speed, reflecting
    off the field's bounding box when it would exit. Models human
    workers (lower speed, smoother heading changes) and birds/animals
    (higher speed, more erratic heading changes) via different
    `heading_noise_std` and `speed_mps` values.
    """

    def __init__(self, speed_mps: float, bounds: Tuple[float, float, float, float],
                 heading_noise_std: float = 0.6):
        self._speed = speed_mps
        self._bounds = bounds  # (min_x, min_y, max_x, max_y)
        self._heading = None  # radians; initialized lazily on first step
        self._heading_noise_std = heading_noise_std

    def step(self, position: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        if self._heading is None:
            self._heading = rng.uniform(0, 2 * np.pi)
        # Heading follows a correlated random walk (smoother than pure
        # white noise, avoiding physically implausible instant U-turns).
        self._heading += rng.normal(0, self._heading_noise_std)
        velocity = self._speed * np.array([np.cos(self._heading), np.sin(self._heading)])
        new_pos = position + velocity * dt

        min_x, min_y, max_x, max_y = self._bounds
        if new_pos[0] < min_x or new_pos[0] > max_x:
            self._heading = np.pi - self._heading  # reflect horizontally
            new_pos[0] = np.clip(new_pos[0], min_x, max_x)
        if new_pos[1] < min_y or new_pos[1] > max_y:
            self._heading = -self._heading  # reflect vertically
            new_pos[1] = np.clip(new_pos[1], min_y, max_y)
        return new_pos


@dataclass
class DynamicObstacle:
    """
    A moving obstacle (tractor, human, animal/bird). Position is updated
    each timestep by delegating to its `MovementStrategy`, keeping this
    class itself strategy-agnostic (composition over inheritance).
    """
    obstacle_id: int
    obstacle_type: ObstacleType
    x: float
    y: float
    radius_m: float
    strategy: MovementStrategy

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def step(self, dt: float, rng: np.random.Generator) -> None:
        """Advances this obstacle's position in place by dt seconds."""
        new_pos = self.strategy.step(self.position, dt, rng)
        self.x, self.y = float(new_pos[0]), float(new_pos[1])


class StaticObstacleGenerator:
    """
    Places static obstacles on flyable, non-start grid cells, and marks
    those cells `blocked_static = True`. Placement is rejection-sampled
    against the field polygon (so obstacles never land outside the
    actual field boundary, even though the MBR is larger than the field)
    and against already-placed obstacles (minimum separation), for
    physical plausibility.
    """

    def __init__(self, config: ObstacleConfig, seed: int = 42):
        self._config = config
        self._rng = np.random.default_rng(seed)

    def generate(self, field: FarmFieldPolygon, grid: GridMapGenerator) -> List[StaticObstacle]:
        """
        Parameters
        ----------
        field : FarmFieldPolygon
            Used to reject candidate positions outside the true field boundary.
        grid : GridMapGenerator
            Provides candidate visitable cells and is mutated in place:
            cells overlapped by a placed obstacle get `blocked_static = True`.

        Returns
        -------
        List[StaticObstacle]
        """
        specs = [
            (ObstacleType.TREE, self._config.n_trees, self._config.tree_radius_m),
            (ObstacleType.POLE, self._config.n_poles, self._config.pole_radius_m),
            (ObstacleType.BUILDING, self._config.n_buildings, self._config.building_radius_m),
            (ObstacleType.IRRIGATION, self._config.n_irrigation, self._config.irrigation_radius_m),
        ]

        obstacles: List[StaticObstacle] = []
        obstacle_id = 0
        min_separation_m = 2.0  # minimum gap between obstacle centers, avoids overlapping footprints

        candidate_cells = [c for c in grid.cells if c.is_visitable and not c.is_start]
        self._rng.shuffle(candidate_cells)

        for obstacle_type, count, radius in specs:
            placed = 0
            for cell in candidate_cells:
                if placed >= count:
                    break
                if cell.blocked_static:
                    continue
                if not field.contains_point(cell.center_x, cell.center_y):
                    continue
                if any(np.linalg.norm(o.position - np.array([cell.center_x, cell.center_y])) < min_separation_m
                       for o in obstacles):
                    continue

                obstacle = StaticObstacle(
                    obstacle_id=obstacle_id, obstacle_type=obstacle_type,
                    x=cell.center_x, y=cell.center_y, radius_m=radius,
                )
                obstacles.append(obstacle)
                obstacle_id += 1
                placed += 1

                # Mark every grid cell whose polygon intersects the
                # obstacle's footprint as blocked (a large building can
                # span multiple cells; a pole typically blocks only one).
                for c in grid.cells:
                    if c.polygon.intersects(obstacle.footprint):
                        c.blocked_static = True

            if placed < count:
                logger.warning(
                    "Only placed %d/%d obstacles of type %s (ran out of valid candidate cells).",
                    placed, count, obstacle_type.value,
                )

        logger.info("Placed %d static obstacles total.", len(obstacles))
        return obstacles


class DynamicObstacleGenerator:
    """
    Creates DynamicObstacle instances (tractors, humans, animals) with
    physically appropriate movement strategies, initialized at random
    valid positions inside the field boundary.
    """

    def __init__(self, config: ObstacleConfig, seed: int = 42):
        self._config = config
        self._rng = np.random.default_rng(seed)

    def generate(self, field: FarmFieldPolygon, grid: GridMapGenerator) -> List[DynamicObstacle]:
        bounds = grid._mbr.bounds  # (min_x, min_y, max_x, max_y); used for random-walk reflection
        obstacles: List[DynamicObstacle] = []
        obstacle_id = 1000  # offset from static IDs to keep IDs globally unique

        flyable_cells = [c for c in grid.cells if c.is_flyable and not c.is_start]

        # --- Tractors: patrol a straight row-like path across the field ---
        for _ in range(self._config.n_tractors):
            start_cell = self._rng.choice(flyable_cells)
            end_cell = self._rng.choice(flyable_cells)
            waypoints = [
                (start_cell.center_x, start_cell.center_y),
                (end_cell.center_x, end_cell.center_y),
            ]
            strategy = WaypointPatrolStrategy(waypoints, self._config.tractor_speed_mps)
            obstacles.append(DynamicObstacle(
                obstacle_id=obstacle_id, obstacle_type=ObstacleType.TRACTOR,
                x=waypoints[0][0], y=waypoints[0][1],
                radius_m=self._config.tractor_radius_m, strategy=strategy,
            ))
            obstacle_id += 1

        # --- Humans: slow bounded random walk ---
        for _ in range(self._config.n_humans):
            cell = self._rng.choice(flyable_cells)
            strategy = BoundedRandomWalkStrategy(
                self._config.human_speed_mps, bounds, heading_noise_std=0.4)
            obstacles.append(DynamicObstacle(
                obstacle_id=obstacle_id, obstacle_type=ObstacleType.HUMAN,
                x=cell.center_x, y=cell.center_y,
                radius_m=self._config.human_radius_m, strategy=strategy,
            ))
            obstacle_id += 1

        # --- Animals/birds: fast, erratic bounded random walk ---
        for _ in range(self._config.n_animals):
            cell = self._rng.choice(flyable_cells)
            strategy = BoundedRandomWalkStrategy(
                self._config.animal_speed_mps, bounds, heading_noise_std=1.0)
            obstacles.append(DynamicObstacle(
                obstacle_id=obstacle_id, obstacle_type=ObstacleType.ANIMAL,
                x=cell.center_x, y=cell.center_y,
                radius_m=self._config.animal_radius_m, strategy=strategy,
            ))
            obstacle_id += 1

        logger.info("Created %d dynamic obstacles.", len(obstacles))
        return obstacles


class DynamicObstacleManager:
    """
    Owns the list of DynamicObstacle instances and steps all of them
    together each environment timestep. Kept separate from the
    generator (which only constructs the initial set) so `swarm_env.py`
    has one clear call (`manager.step(dt)`) per simulation tick.
    """

    def __init__(self, obstacles: List[DynamicObstacle], seed: int = 42):
        self._obstacles = obstacles
        self._rng = np.random.default_rng(seed)

    @property
    def obstacles(self) -> List[DynamicObstacle]:
        return self._obstacles

    def step(self, dt: float) -> None:
        """Advances every dynamic obstacle's position by dt seconds."""
        for obstacle in self._obstacles:
            obstacle.step(dt, self._rng)

    def positions(self) -> np.ndarray:
        """(N, 2) array of all dynamic obstacle positions, for rasterizing into observations."""
        if not self._obstacles:
            return np.zeros((0, 2))
        return np.array([o.position for o in self._obstacles])
