"""
env/uav.py
==========
UAV state management module.

Defines the UAV class which tracks each drone's position, heading,
and per-step state within the swarm simulation. The environment
holds a list of UAV objects and delegates position updates to them.

Architecture note:
    UAV is a lightweight data + behaviour class. It does not contain
    any RL logic — it simply manages physical state. Reward computation
    is handled separately in rewards.py to keep concerns separated.
    This makes it easy to add UAV-specific properties (battery,
    altitude, sensor range) in future without touching reward logic.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Action definitions: each action maps to a (delta_row, delta_col) move
# Action 0: Stay (hover in place)
# Action 1: Move North (row - 1)
# Action 2: Move South (row + 1)
# Action 3: Move East  (col + 1)
# Action 4: Move West  (col - 1)
ACTION_DELTAS: dict[int, Tuple[int, int]] = {
    0: (0, 0),   # Stay / hover
    1: (-1, 0),  # North
    2: (1, 0),   # South
    3: (0, 1),   # East
    4: (0, -1),  # West
}
N_ACTIONS: int = len(ACTION_DELTAS)


class UAV:
    """
    Represents a single UAV agent in the swarm.

    Tracks position, heading, and episode-level statistics.
    Position is stored as (row, col) grid indices.

    Parameters
    ----------
    uav_id : int
        Unique integer identifier for this UAV within the swarm.
    init_row : int
        Initial row position at episode start.
    init_col : int
        Initial column position at episode start.
    obs_radius : int
        Radius of this UAV's local observation window (in cells).
    """

    def __init__(
        self,
        uav_id: int,
        init_row: int,
        init_col: int,
        obs_radius: int,
    ) -> None:
        self.uav_id = uav_id
        self.obs_radius = obs_radius

        # Current position
        self.row: int = init_row
        self.col: int = init_col

        # Initial position (for reset)
        self._init_row: int = init_row
        self._init_col: int = init_col

        # Episode statistics (reset each episode)
        self.cells_covered: int = 0       # New cells this UAV covered
        self.collisions: int = 0          # Total collision events
        self.steps_taken: int = 0         # Steps this UAV has taken

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> Tuple[int, int]:
        """Return current (row, col) position."""
        return (self.row, self.col)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, row: int, col: int) -> None:
        """
        Reset UAV to a new starting position and clear episode stats.

        Parameters
        ----------
        row : int
            New starting row.
        col : int
            New starting column.
        """
        self.row = row
        self.col = col
        self._init_row = row
        self._init_col = col

        # Clear per-episode statistics
        self.cells_covered = 0
        self.collisions = 0
        self.steps_taken = 0

    def compute_intended_position(
        self,
        action: int,
        wind_drift: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Compute the cell the UAV would move to given an action and wind.

        This does NOT update the UAV's position — the environment calls
        this first, validates the position, then calls move_to() only
        if the position is valid. This separation allows the environment
        to handle boundary rejection cleanly.

        Parameters
        ----------
        action : int
            Integer action index (0–4, see ACTION_DELTAS).
        wind_drift : np.ndarray
            Shape (2,) array of (delta_row, delta_col) wind displacement.

        Returns
        -------
        Tuple[int, int]
            Intended (row, col) after action + wind.
        """
        # Base movement from chosen action
        dr, dc = ACTION_DELTAS[action]

        # Add wind drift (integer displacement)
        new_row = self.row + dr + int(wind_drift[0])
        new_col = self.col + dc + int(wind_drift[1])

        return (new_row, new_col)

    def move_to(self, row: int, col: int) -> None:
        """
        Update position to (row, col) and increment step counter.

        Parameters
        ----------
        row : int
            Target row.
        col : int
            Target column.
        """
        self.row = row
        self.col = col
        self.steps_taken += 1

    def get_local_observation(
        self,
        grid: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a local observation window centred on this UAV.

        Returns a (2*obs_radius+1, 2*obs_radius+1) sub-grid centred
        at the UAV's current position. Cells outside the grid boundary
        are padded with -1 (indicating "wall / out of bounds").

        Parameters
        ----------
        grid : np.ndarray
            Full grid array of shape (grid_size, grid_size).

        Returns
        -------
        np.ndarray
            Flattened local observation window.
        """
        r, c = self.row, self.col
        rad = self.obs_radius
        gs = grid.shape[0]

        # Window size
        win_size = 2 * rad + 1
        # Pad the full grid so we can safely slice around any cell
        padded = np.full(
            (gs + 2 * rad, gs + 2 * rad),
            fill_value=-1.0,
            dtype=np.float32,
        )
        # Copy real grid values into padded array
        padded[rad: rad + gs, rad: rad + gs] = grid.astype(np.float32)

        # Extract window centred at UAV position (accounting for padding offset)
        window = padded[r: r + win_size, c: c + win_size]

        return window.flatten()
