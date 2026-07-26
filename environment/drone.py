"""
environment/drone.py
=====================
UAV agent: continuous-space physics (position/velocity/battery), driven
by discrete high-level actions.

Why continuous physics, not pure cell-teleportation (gap vs. reference
paper): the reference paper's UAVs move by "codified straight movements"
between grid cells with no notion of velocity, momentum, or external
disturbance (Section 3.3.4) -- a UAV simply occupies a new cell each
action. That makes it structurally impossible to represent wind
pushing a UAV off its intended path, or two UAVs' physical proximity
mattering between discrete steps. Here, each discrete action sets a
*commanded* velocity toward a neighboring cell; actual displacement each
physics step is the commanded velocity plus wind and wake disturbances,
integrated over dt. The UAV's discrete "current cell" (used for coverage
reward, matching the paper's Eq. 2) is then derived from its continuous
position, not the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from configs.config import DroneConfig


class ActionType(IntEnum):
    """
    Discrete action set, matching the reference paper's simplification
    to straight (non-curved) movements (Section 3.3.4, Fig. 5) plus a
    STAY action (needed here because, unlike the paper's turn-based grid
    stepping, our continuous-time simulation needs an explicit
    "hold position" command, e.g., for a UAV that has finished covering
    its area or is waiting out a nearby collision risk).
    """
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


# Unit direction vectors in world (x, y) space for each action.
_ACTION_VECTORS = {
    ActionType.STAY: np.array([0.0, 0.0]),
    ActionType.UP: np.array([0.0, 1.0]),
    ActionType.DOWN: np.array([0.0, -1.0]),
    ActionType.LEFT: np.array([-1.0, 0.0]),
    ActionType.RIGHT: np.array([1.0, 0.0]),
}


@dataclass
class UAV:
    """
    A single UAV agent's physical state.

    Attributes
    ----------
    uav_id : int
        Unique identifier within the swarm.
    position : np.ndarray, shape (2,)
        Current continuous (x, y) position in meters.
    velocity : np.ndarray, shape (2,)
        Current effective (post-disturbance) velocity in m/s.
    battery_s : float
        Remaining battery, in seconds of flight, matching the reference
        paper's assumption of a fixed maximum flight time (Section 3.4).
    alive : bool
        False once battery is exhausted or the UAV has been removed
        after a critical collision -- dead UAVs are excluded from
        further action-taking but remain in the swarm's position list
        so their last known location can still be rendered/logged
        (relevant for the paper's fault-tolerance framing: the swarm
        should be able to continue with the survivors).
    config : DroneConfig
        Shared physical parameters (max speed, battery capacity, etc.).
    """
    uav_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    battery_s: float = 0.0
    alive: bool = True
    config: Optional[DroneConfig] = None

    def commanded_velocity(self, action: ActionType) -> np.ndarray:
        """
        Maps a discrete action to the UAV's *intended* velocity vector
        (before wind/wake disturbance is applied), scaled to max speed.
        """
        direction = _ACTION_VECTORS[action]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return np.zeros(2)
        return (direction / norm) * self.config.max_speed_mps

    def step_physics(self, action: ActionType, dt: float,
                      wind_vector: np.ndarray, extra_turbulence: np.ndarray) -> None:
        """
        Integrates this UAV's position forward by one physics timestep.

        Effective velocity = commanded velocity (from the discrete
        action) + wind_coupling * ambient wind + inter-UAV wake
        turbulence. This is where the reference paper's pure grid
        teleportation is replaced with a genuine (if simplified) physics
        integration step.

        Parameters
        ----------
        action : ActionType
            The discrete action chosen by the policy for this UAV.
        dt : float
            Physics timestep, seconds.
        wind_vector : np.ndarray, shape (2,)
            Current ambient wind vector (m/s) from WindField.
        extra_turbulence : np.ndarray, shape (2,)
            Additional per-UAV turbulence from `compute_wake_interference`.
        """
        if not self.alive:
            return

        commanded = self.commanded_velocity(action)
        disturbance = self.config.wind_coupling * wind_vector + extra_turbulence
        effective_velocity = commanded + disturbance

        self.velocity = effective_velocity
        self.position = self.position + effective_velocity * dt

        # Battery cost: baseline hovering/flight cost plus extra cost
        # proportional to how much the UAV must fight the disturbance
        # (a UAV holding a straight line through a headwind burns more
        # battery than one flying with calm air) -- this ties the wind
        # model to a measurable consequence rather than being cosmetic.
        disturbance_cost = 0.1 * np.linalg.norm(disturbance)
        self.battery_s -= dt * (1.0 + disturbance_cost)
        if self.battery_s <= 0.0:
            self.battery_s = 0.0
            self.alive = False

    def clamp_to_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float) -> bool:
        """
        Clamps this UAV's position to the given axis-aligned bounds
        (used as a hard physical safety net after collision detection
        flags an out-of-bounds event, so a UAV can never be simulated
        as physically leaving the MBR even if the policy commands it to).

        Returns
        -------
        bool
            True if clamping was necessary (i.e., a boundary violation
            occurred), for the caller to attribute a boundary-collision
            event.
        """
        clamped = np.clip(self.position, [min_x, min_y], [max_x, max_y])
        violated = not np.allclose(clamped, self.position)
        self.position = clamped
        return violated
