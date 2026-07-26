"""
environment/polygon_field.py
=============================
Field boundary geometry: reading polygon vertices and computing the
Minimum Bounding Rectangle (MBR). Reproduces Section 3.2 / Fig. 3(a,b)
of Puente-Castro et al. (2022) -- see the module-level discussion in
`grid_map.py` for the labeling-rule design decision that pairs with this.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from shapely.geometry import Polygon, box, Point
from shapely.validation import make_valid

from configs.config import MBRMode

logger = logging.getLogger("uav_swarm_ppo.environment.polygon_field")


class FarmFieldPolygon:
    """
    Represents and validates the irregular polygon boundary of a farm
    field. Single responsibility: geometric validity of the boundary
    only (not MBR, not grid, not obstacles) -- reused by obstacle.py to
    keep obstacle placement inside the field, and by collision.py for
    boundary-violation checks.
    """

    def __init__(self, vertices: List[Tuple[float, float]]):
        """
        Parameters
        ----------
        vertices : List[Tuple[float, float]]
            Ordered (x, y) coordinates in meters. At least 3 required.

        Raises
        ------
        ValueError
            If fewer than 3 vertices are given, or the polygon is
            invalid and cannot be repaired.
        """
        if len(vertices) < 3:
            raise ValueError(f"A polygon requires >= 3 vertices, got {len(vertices)}.")

        raw_polygon = Polygon(vertices)
        if not raw_polygon.is_valid:
            logger.warning("Input polygon is invalid; attempting repair via make_valid().")
            repaired = make_valid(raw_polygon)
            if repaired.geom_type != "Polygon":
                raise ValueError(
                    f"Polygon repair produced a {repaired.geom_type}, not a Polygon. "
                    "Supply a simple (non-self-intersecting) boundary."
                )
            raw_polygon = repaired

        self._polygon: Polygon = raw_polygon
        logger.info(
            "FarmFieldPolygon initialized: %d vertices, area=%.2f m^2, perimeter=%.2f m",
            len(vertices), self._polygon.area, self._polygon.length,
        )

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def area_m2(self) -> float:
        return self._polygon.area

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._polygon.bounds

    def contains_point(self, x: float, y: float) -> bool:
        """True iff (x, y) lies inside the field boundary."""
        return self._polygon.contains(Point(x, y))


class MinimumBoundingRectangle:
    """
    Computes the field's Minimum Bounding Rectangle (Fig. 3b). Supports
    both the paper's axis-aligned interpretation and a true rotated
    minimum-area rectangle (our improvement -- see MODULE_GUIDE.md for
    the quantitative comparison).
    """

    def __init__(self, field: FarmFieldPolygon, mode: MBRMode = MBRMode.AXIS_ALIGNED):
        self.mode = mode
        self._rectangle: Polygon = self._compute(field.polygon, mode)
        logger.info(
            "MBR computed (mode=%s): area=%.2f m^2 (field=%.2f m^2, efficiency=%.1f%%)",
            mode.value, self._rectangle.area, field.area_m2,
            100.0 * field.area_m2 / self._rectangle.area,
        )

    @staticmethod
    def _compute(polygon: Polygon, mode: MBRMode) -> Polygon:
        if mode == MBRMode.AXIS_ALIGNED:
            min_x, min_y, max_x, max_y = polygon.bounds
            return box(min_x, min_y, max_x, max_y)
        elif mode == MBRMode.ROTATED_MIN_AREA:
            return polygon.minimum_rotated_rectangle
        raise ValueError(f"Unsupported MBRMode: {mode}")

    @property
    def rectangle(self) -> Polygon:
        return self._rectangle

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._rectangle.bounds
