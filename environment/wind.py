"""
environment/wind.py
====================
Wind disturbance model, covering two distinct physical effects:

1. Ambient atmospheric wind (`WindField`): a spatially-uniform,
   temporally-correlated stochastic wind vector affecting every UAV.
2. Inter-UAV wake interference (`compute_wake_interference`): additional
   localized turbulence a UAV experiences when flying close to another
   UAV, due to prop-wash/downwash -- this directly answers the
   requirement "how the movement/air from one drone affects another".

Why an Ornstein-Uhlenbeck (OU) process for ambient wind, not white noise:
real near-surface wind is gusty but autocorrelated in time -- a gust
doesn't fully reverse direction from one instant to the next. The OU
process is the standard mean-reverting stochastic model used for this
(a discretized relative of the Dryden turbulence model used in flight
simulation), which is both more realistic and gives the wind curve a
recognizable "gustiness" rather than jittery noise, matching what an
IEEE reviewer would expect a "wind disturbance model" to actually mean.
"""

from __future__ import annotations

import logging

import numpy as np

from configs.config import WindConfig

logger = logging.getLogger("uav_swarm_ppo.environment.wind")


class WindField:
    """
    Ornstein-Uhlenbeck stochastic wind vector, shared (spatially
    uniform) across the whole field at any given timestep. State is a
    2D vector (vx, vy) in m/s.

    Update rule (discretized OU process):
        v(t+dt) = v(t) + theta * (mean - v(t)) * dt + sigma * sqrt(dt) * N(0, I)
    where `mean` is the long-run wind vector derived from
    (mean_speed_mps, mean_direction_rad), `theta` controls how fast wind
    reverts to that mean, and `sigma` controls gust intensity.
    """

    def __init__(self, config: WindConfig, seed: int = 42):
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._mean_vector = config.mean_speed_mps * np.array([
            np.cos(config.mean_direction_rad), np.sin(config.mean_direction_rad),
        ])
        # Initialize current wind at the mean rather than zero, so
        # episodes don't all start in an artificial calm.
        self._current_vector = self._mean_vector.copy()

    def reset(self) -> None:
        """Resets wind state to the mean vector at the start of a new episode."""
        self._current_vector = self._mean_vector.copy()

    def step(self, dt: float) -> np.ndarray:
        """
        Advances the wind state by dt seconds and returns the new vector.

        Parameters
        ----------
        dt : float
            Timestep in seconds.

        Returns
        -------
        np.ndarray
            Current wind vector (vx, vy) in m/s, after the update.
        """
        theta, sigma = self._config.theta, self._config.sigma
        drift = theta * (self._mean_vector - self._current_vector) * dt
        diffusion = sigma * np.sqrt(dt) * self._rng.normal(size=2)
        self._current_vector = self._current_vector + drift + diffusion
        return self._current_vector

    @property
    def vector(self) -> np.ndarray:
        """Current wind vector without advancing state."""
        return self._current_vector

    @property
    def speed(self) -> float:
        """Current wind speed magnitude (m/s)."""
        return float(np.linalg.norm(self._current_vector))


def compute_wake_interference(
    positions: np.ndarray,
    wind_vector: np.ndarray,
    wake_radius_m: float,
    wake_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Computes an additional per-UAV turbulence vector caused by nearby
    UAVs' prop-wash/downwash, modeling the requirement that one drone's
    movement/air affects nearby drones in the swarm.

    Physical simplification: any UAV within `wake_radius_m` of another
    UAV receives extra turbulence proportional to `wake_strength`,
    directionally biased downwind of the disturbing UAV relative to the
    ambient wind (i.e., you're more affected by a neighbor's wake if you
    are downwind of them), with a small random component reflecting the
    genuinely chaotic nature of turbulent prop-wash. This is a
    deliberately lightweight aerodynamic model -- a full computational
    fluid dynamics wake model is out of scope for a path-planning RL
    paper, and is flagged in Limitations/Future Work.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Current positions of all N UAVs.
    wind_vector : np.ndarray, shape (2,)
        Current ambient wind vector, used to determine downwind bias.
    wake_radius_m : float
        Distance within which one UAV's wake affects another.
    wake_strength : float
        Base magnitude (m/s) of injected turbulence at zero distance,
        decaying linearly to zero at wake_radius_m.
    rng : np.random.Generator
        Seeded RNG for the stochastic component.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Extra turbulence vector to add to each UAV's effective wind,
        in the same order as `positions`.
    """
    n = positions.shape[0]
    extra = np.zeros((n, 2))
    if n < 2:
        return extra

    wind_speed = np.linalg.norm(wind_vector)
    wind_dir = wind_vector / wind_speed if wind_speed > 1e-6 else np.array([1.0, 0.0])

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = positions[i] - positions[j]  # from j (disturber) to i (affected)
            dist = np.linalg.norm(delta)
            if dist < 1e-6 or dist >= wake_radius_m:
                continue
            # Downwind bias: i is more strongly affected by j's wake if
            # i lies roughly in the direction the wind is blowing from j.
            downwind_alignment = max(0.0, np.dot(delta / dist, wind_dir))
            proximity_factor = 1.0 - (dist / wake_radius_m)
            magnitude = wake_strength * proximity_factor * (0.5 + 0.5 * downwind_alignment)
            random_component = rng.normal(scale=0.3, size=2)
            extra[i] += magnitude * wind_dir + random_component * proximity_factor

    return extra
