"""
ppo/evaluate.py
================
Loads a trained PPO model and runs deterministic evaluation episodes
against `SwarmFarmEnv`, logging full per-category reward/coverage/
collision statistics via `EpisodeLogger` (Goal 3) -- the same structured
CSV output Day 4's baseline comparison (Q-Learning, A*) will also
produce, so PPO and the baselines can be compared on identical columns.

Run with:
    python -m ppo.evaluate --model results/models/ppo_swarm_final.zip --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from configs.config import ExperimentConfig
from environment.swarm_env import make_env
from ppo.train import FIELD_VERTICES
from utils.episode_logger import EpisodeLogger
from utils.logger import get_logger

logger = get_logger("uav_swarm_ppo.ppo.evaluate", log_file="results/evaluate.log")


def evaluate(model_path: str, n_episodes: int = 10, seed: int = 999,
             deterministic: bool = True, experiment_name: str = "ppo_eval"):
    """
    Runs `n_episodes` rollouts of a trained model against a single
    (non-vectorized) `SwarmFarmEnv` instance, deterministic by default
    (PPO's action = argmax of the policy distribution, not sampled --
    the standard choice for reporting final performance, as opposed to
    the stochastic sampling used during training for exploration).

    A fresh `EpisodeLogger` is attached directly to the env so every
    episode's full reward breakdown, coverage, and collision counts are
    written to `results/episode_logs/<experiment_name>_episodes.csv` --
    safe here (unlike during vectorized training) because this env runs
    single-instance, sequential episodes with no interleaving risk.

    Returns
    -------
    Tuple[List[float], List[float]]
        Per-episode (total_reward, coverage_fraction) lists.
    """
    config = ExperimentConfig(seed=seed)
    episode_logger = EpisodeLogger("results/episode_logs", experiment_name=experiment_name)

    env_fn = make_env(FIELD_VERTICES, config, episode_logger=episode_logger, render_mode=None)
    env = env_fn()

    model = PPO.load(model_path)
    logger.info("Loaded model from %s", model_path)

    episode_rewards, episode_coverages = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
        episode_coverages.append(info["coverage_fraction"])
        logger.info("Episode %d: reward=%.2f coverage=%.3f collisions=%s",
                    ep, total_reward, info["coverage_fraction"], info["collision_counts"])

    env.close()

    logger.info(
        "Evaluation over %d episodes: reward mean=%.2f std=%.2f | coverage mean=%.3f std=%.3f",
        n_episodes, float(np.mean(episode_rewards)), float(np.std(episode_rewards)),
        float(np.mean(episode_coverages)), float(np.std(episode_coverages)),
    )
    return episode_rewards, episode_coverages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on SwarmFarmEnv.")
    parser.add_argument("--model", type=str, default="results/models/ppo_swarm_final.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed)
