"""
utils/seed.py
==============
Central seeding utility. Every source of randomness in this project
(field/obstacle/UAV placement, wind noise, PPO's own RNG, baseline
algorithms) must be seeded from a single call to `set_global_seed` at
the start of an experiment -- otherwise "run N seeds and compare means"
style robustness claims (which is what Goal 4 / the paper's ANOVA
analysis needs) are not reproducible.
"""

import os
import random
import logging

import numpy as np

logger = logging.getLogger("uav_swarm_ppo.seed")


def set_global_seed(seed: int) -> None:
    """
    Seeds Python's `random`, NumPy, and (if installed) PyTorch, and sets
    the PYTHONHASHSEED environment variable for deterministic hashing.

    Parameters
    ----------
    seed : int
        The seed value to apply everywhere.

    Notes
    -----
    This does NOT seed per-object `np.random.default_rng()` instances
    created independently downstream (e.g., inside GridMapGenerator or
    WindField) -- those take their own seed explicitly from
    `ExperimentConfig.seed` so each subsystem's stream is independent
    and inspectable, while still being fully determined by one root seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        # PyTorch is a Stable-Baselines3 dependency and will be installed
        # by Day 3; this module must not hard-fail before that.
        pass
    logger.info("Global random seed set to %d", seed)
