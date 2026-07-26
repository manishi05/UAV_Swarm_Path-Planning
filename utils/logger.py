"""
utils/logger.py
================
Single point of logging configuration for the whole project, so every
module logs in the same format and (optionally) to the same file,
instead of each script calling `logging.basicConfig` independently
(which silently no-ops after the first call and causes confusing
inconsistent logs across modules).
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None,
                level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger. Safe to call multiple times with the
    same name (e.g., once per module import) -- handlers are only
    attached once per logger instance to avoid duplicate log lines.

    Parameters
    ----------
    name : str
        Logger name, conventionally the module's dotted path, e.g.
        "uav_swarm_ppo.environment.obstacle".
    log_file : Optional[str]
        If provided, logs are additionally written to this file (append
        mode), which is what `experiments/run_experiment.py` uses to
        keep a permanent record per experiment run for the paper's
        reproducibility appendix.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Prevent double-logging via the root logger.
        logger.propagate = False

    return logger
