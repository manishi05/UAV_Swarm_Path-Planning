"""
environment/collision.py
=========================
Collision detection engine. This is the module that directly answers
your stated gap: "The paper assumes multiple UAVs but doesn't
realistically solve UAV-UAV collisions." The reference paper has no
collision model whatsoever -- UAVs simply occupy grid cells, and two
UAVs in the same cell are never flagged or penalized. This module adds:

  - UAV-UAV collision/near-miss detection (pairwise distance vs. a
    configurable safety radius, not just "same cell").
  - UAV-static-obstacle collision (distance to tree/pole/building/
    irrigation-equipment footprints).
  - UAV-dynamic-obstacle collision (distance to tractor/human/animal
    positions, which move every step).
  - UAV-boundary violation (leaving the field polygon or the MBR).

Detected events are returned as a structured report consumed by
reward.py (Goal 2: "tune the reward function so drones learn to avoid
collisions") and by swarm_env.py for episode termination logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import pdist, squareform

from environment.drone import UAV
from environment.obstacle import StaticObstacle, DynamicObstacle
from environment.polygon_field import FarmFieldPolygon


class CollisionType(str, Enum):
    UAV_UAV = "uav_uav"
    UAV_STATIC_OBSTACLE = "uav_static_obstacle"
    UAV_DYNAMIC_OBSTACLE = "uav_dynamic_obstacle"
    UAV_BOUNDARY = "uav_boundary"


@dataclass
class CollisionReport:
    """
    Per-step collision results, keyed by UAV id, plus raw counts by type
    for episode-level logging (fed into utils.metrics.collision_rate).
    """
    events: Dict[int, List[CollisionType]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=lambda: {t.value: 0 for t in CollisionType})

    def add(self, uav_id: int, collision_type: CollisionType) -> None:
        self.events.setdefault(uav_id, []).append(collision_type)
        self.counts[collision_type.value] += 1

    def for_uav(self, uav_id: int) -> List[CollisionType]:
        return self.events.get(uav_id, [])


class CollisionEngine:
    """
    Stateless (per-call) collision detector: takes the current swarm and
    obstacle state and returns a CollisionReport. Kept stateless (no
    internal memory across steps) so it can be unit-tested with
    hand-constructed positions independent of a running environment.
    """

    def __init__(self, field: FarmFieldPolygon, safety_radius_m: float,
                 collision_radius_m: float):
        """
        Parameters
        ----------
        field : FarmFieldPolygon
            Used for the UAV_BOUNDARY check (must remain inside the true
            field polygon, not just the rectangular MBR).
        safety_radius_m : float
            UAV-UAV minimum separation distance. Violating this is
            flagged as UAV_UAV even before a "physical" collision,
            because in real operations you want the policy penalized
            for entering an unsafe separation, not just for an actual
            crash -- this is what makes the check "realistic" rather
            than the paper's binary same-cell-or-not test.
        collision_radius_m : float
            Physical UAV radius, used together with an obstacle's own
            radius to determine UAV-obstacle contact distance.
        """
        self._field = field
        self._safety_radius = safety_radius_m
        self._collision_radius = collision_radius_m

    def detect(
        self,
        uavs: List[UAV],
        static_obstacles: List[StaticObstacle],
        dynamic_obstacles: List[DynamicObstacle],
    ) -> CollisionReport:
        """
        Runs all four collision checks for the current state and returns
        a combined CollisionReport. Dead UAVs are skipped (they cannot
        collide further, and continuing to penalize/report on them would
        distort per-episode collision-rate metrics).
        """
        report = CollisionReport()
        alive_uavs = [u for u in uavs if u.alive]

        self._check_uav_uav(alive_uavs, report)
        self._check_uav_static(alive_uavs, static_obstacles, report)
        self._check_uav_dynamic(alive_uavs, dynamic_obstacles, report)
        self._check_uav_boundary(alive_uavs, report)

        return report

    def _check_uav_uav(self, uavs: List[UAV], report: CollisionReport) -> None:
        """
        Pairwise Euclidean distance check via scipy's pdist (vectorized,
        O(n^2) but negligible at swarm sizes used in this project;
        flagged in Future Work as a KD-tree candidate for large swarms).
        """
        if len(uavs) < 2:
            return
        positions = np.array([u.position for u in uavs])
        dist_matrix = squareform(pdist(positions))
        n = len(uavs)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] < self._safety_radius:
                    report.add(uavs[i].uav_id, CollisionType.UAV_UAV)
                    report.add(uavs[j].uav_id, CollisionType.UAV_UAV)

    def _check_uav_static(self, uavs: List[UAV], obstacles: List[StaticObstacle],
                           report: CollisionReport) -> None:
        for uav in uavs:
            for obstacle in obstacles:
                dist = np.linalg.norm(uav.position - obstacle.position)
                if dist < (self._collision_radius + obstacle.radius_m):
                    report.add(uav.uav_id, CollisionType.UAV_STATIC_OBSTACLE)

    def _check_uav_dynamic(self, uavs: List[UAV], obstacles: List[DynamicObstacle],
                            report: CollisionReport) -> None:
        for uav in uavs:
            for obstacle in obstacles:
                dist = np.linalg.norm(uav.position - obstacle.position)
                if dist < (self._collision_radius + obstacle.radius_m):
                    report.add(uav.uav_id, CollisionType.UAV_DYNAMIC_OBSTACLE)

    def _check_uav_boundary(self, uavs: List[UAV], report: CollisionReport) -> None:
        for uav in uavs:
            if not self._field.contains_point(uav.position[0], uav.position[1]):
                report.add(uav.uav_id, CollisionType.UAV_BOUNDARY)
