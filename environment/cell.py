"""
environment/cell.py
====================
Defines `GridCell`, the atomic unit of the discretized flight
environment. Split into its own module (rather than living inside
polygon_field.py as before) because obstacle.py, drone.py, and
swarm_env.py all need this type without needing to import the heavier
Shapely-based field/MBR generation logic -- keeping the dependency graph
shallow, per the project's modularity requirement.
"""

from dataclasses import dataclass
from shapely.geometry import Polygon


@dataclass
class GridCell:
    """
    A single cell of the discretized flight environment.

    Attributes
    ----------
    row, col : int
        Integer grid coordinates (row 0 = topmost, matching Fig. 3c of
        the reference paper).
    center_x, center_y : float
        Physical-space (meters) coordinates of the cell's center.
    polygon : Polygon
        The cell's own square geometry in physical space, used for exact
        intersection tests against obstacles and the field boundary.
    is_visitable : bool
        True if the cell lies inside the field polygon (Goal 1 concept:
        purely a function of field geometry, set once at grid-generation
        time and never changed afterward).
    is_start : bool
        True if this cell is a designated UAV start position.
    blocked_static : bool
        True if a static obstacle (tree/pole/building/irrigation
        equipment) occupies this cell (Goal 2 concept: set by
        `StaticObstacleGenerator` after grid generation, independent of
        field-boundary visitability).
    """
    row: int
    col: int
    center_x: float
    center_y: float
    polygon: Polygon
    is_visitable: bool
    is_start: bool = False
    blocked_static: bool = False

    @property
    def is_flyable(self) -> bool:
        """
        True iff a UAV is currently permitted to occupy this cell: it
        must be inside the field boundary AND not blocked by a static
        obstacle. Kept as a derived property (not a stored flag) so it
        can never drift out of sync with its two source flags.
        """
        return self.is_visitable and not self.blocked_static
