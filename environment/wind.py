"""
env/wind.py
===========
Stochastic wind disturbance module.

Models wind as a time-varying vector that applies probabilistic
displacement to UAV movement. This adds realistic environmental
noise that the agents must learn to compensate for.

Wind model:
    - Direction changes slowly (Gaussian walk on angle each step).
    - Magnitude is sampled from a truncated normal distribution.
    - Each UAV receives the wind vector independently — the actual
      displacement is whether the wind exceeds a threshold and
      pushes the UAV one cell off its intended path.

Architecture note:
    WindModel is stateful (direction evolves over the episode) and
    is reset at the start of each episode. The environment calls
    apply() once per step to get per-UAV drift flags.
"""

from __future__ import annotations

import numpy as np


class WindModel:
    """
    Stochastic wind disturbance model for a UAV swarm environment.

    Wind is represented as a 2D vector (magnitude, direction) that
    evolves via a Gaussian random walk each timestep. When the wind
    component along a UAV's intended movement direction is large enough,
    it probabilistically deflects the UAV to an adjacent cell.

    Parameters
    ----------
    max_magnitude : float
        Maximum wind displacement in cells per step.
        A value of 0.4 means wind can deflect a UAV ~40% of the time.
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    enabled : bool
        If False, apply() always returns zero drift (wind disabled).
    """

    def __init__(
        self,
        max_magnitude: float,
        rng: np.random.Generator,
        enabled: bool = True,
    ) -> None:
        self.max_magnitude = max_magnitude
        self.rng = rng
        self.enabled = enabled

        # Wind state: angle (radians) and magnitude (cells/step)
        self._angle: float = 0.0
        self._magnitude: float = 0.0

    def reset(self) -> None:
        """
        Initialise wind to a random direction and low magnitude.

        Called at the start of each episode to ensure varied but
        reproducible wind conditions when using fixed seeds.
        """
        # Start with a random wind direction
        self._angle = float(self.rng.uniform(0, 2 * np.pi))
        # Start with low magnitude — it builds gradually
        self._magnitude = float(self.rng.uniform(0, self.max_magnitude * 0.3))

    def step(self) -> None:
        """
        Evolve the wind vector by one timestep.

        Direction undergoes a Gaussian random walk (±15° std dev).
        Magnitude is re-sampled with mean-reversion toward zero.
        """
        if not self.enabled:
            return

        # Random walk on direction (wrap around 2π)
        angle_noise = self.rng.normal(0, np.deg2rad(15))
        self._angle = (self._angle + angle_noise) % (2 * np.pi)

        # Mean-reverting magnitude: tends back toward 0 with noise
        mag_noise = self.rng.normal(0, self.max_magnitude * 0.1)
        self._magnitude = float(
            np.clip(self._magnitude * 0.95 + mag_noise, 0, self.max_magnitude)
        )

    def get_drift(self, n_uavs: int) -> np.ndarray:
        """
        Compute wind drift for each UAV this timestep.

        Returns a (n_uavs, 2) array of (delta_row, delta_col) values.
        Each value is either -1, 0, or +1 — the wind-induced
        displacement in that axis for this UAV.

        The probability of a non-zero drift in any axis is proportional
        to the wind component along that axis relative to max_magnitude.

        Parameters
        ----------
        n_uavs : int
            Number of UAVs to compute drift for.

        Returns
        -------
        np.ndarray
            Array of shape (n_uavs, 2) with integer drift values.
        """
        drift = np.zeros((n_uavs, 2), dtype=np.int32)

        if not self.enabled or self._magnitude < 1e-6:
            return drift

        # Wind vector in (row, col) space
        # angle=0 → East, increasing angle → counter-clockwise
        wind_col = self._magnitude * np.cos(self._angle)  # East/West component
        wind_row = -self._magnitude * np.sin(self._angle)  # North/South component

        # Each UAV independently experiences wind
        for i in range(n_uavs):
            # Row axis drift
            prob_row = abs(wind_row) / self.max_magnitude
            if self.rng.random() < prob_row:
                drift[i, 0] = int(np.sign(wind_row))

            # Col axis drift
            prob_col = abs(wind_col) / self.max_magnitude
            if self.rng.random() < prob_col:
                drift[i, 1] = int(np.sign(wind_col))

        return drift

    @property
    def vector(self) -> np.ndarray:
        """
        Current wind as a (row_component, col_component) array.

        Useful for including wind state in the observation vector.

        Returns
        -------
        np.ndarray
            Shape (2,) wind vector in grid-cell units.
        """
        col_component = self._magnitude * np.cos(self._angle)
        row_component = -self._magnitude * np.sin(self._angle)
        return np.array([row_component, col_component], dtype=np.float32)
