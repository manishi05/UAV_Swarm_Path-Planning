"""
ppo/evaluate.py
================
Loads a trained PPO model and runs deterministic evaluation episodes
against `SwarmFarmEnv`, logging full per-category reward/coverage/
collision statistics via `EpisodeLogger` (Goal 3) -- the same structured
CSV output Day 4's baseline comparison (Q-Learning, A*) will also
produce, so PPO and the baselines can be compared on identical columns.

VecNormalize integration (see VECNORMALIZE_INTEGRATION.md for the full
inspection/rationale): if the model was trained with VecNormalize-
wrapped observations (the default as of this change, see ppo/train.py),
evaluating it against raw, unnormalized observations would feed it
inputs far outside its training distribution and produce meaningless
results. This module now loads the SAVED VecNormalize statistics from
training and applies them to observations before every `model.predict()`
call.

Design note -- why this does NOT drive the episode loop through the
VecEnv's own step()/reset() API: SB3's VecEnv wrappers (including
VecNormalize) auto-reset the underlying environment the instant an
episode ends, returning the *next* episode's initial observation from
the same step() call that ended the previous episode. This project's
`EpisodeLogger` has its own start_episode()/end_episode() hooks tied
directly to `SwarmFarmEnv.reset()`/`step()` (see environment/swarm_env.py)
-- driving the loop through a VecEnv's auto-reset semantics risks
silently double-triggering those hooks (an extra phantom reset()
immediately after the real one) and corrupting the per-episode CSV.
To avoid this entirely, `VecNormalize` is used here ONLY as a stateless
observation-transform function (`.normalize_obs()`), applied directly to
this script's own raw, non-vectorized `env.step()`/`env.reset()` loop --
identical control flow to before this change, with only "what happens to
obs before model.predict()" now different. Raw reward is used directly;
`VecNormalize` is never applied to reward at evaluation time at all
(consistent with norm_reward=False for evaluation).

Run with:
    python -m ppo.evaluate --model results/models/ppo_swarm_final.zip \\
        --vecnormalize results/models/vecnormalize_final.pkl --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.config import ExperimentConfig
from environment.swarm_env import make_env
from ppo.train import FIELD_VERTICES
from utils.episode_logger import EpisodeLogger
from utils.logger import get_logger

logger = get_logger("uav_swarm_ppo.ppo.evaluate", log_file="results/evaluate.log")


def _load_vec_normalize(vecnormalize_path: str, config: ExperimentConfig) -> VecNormalize:
    """
    Loads saved VecNormalize statistics from training and forces the
    evaluation-appropriate flags.

    A throwaway `DummyVecEnv` (built with `episode_logger=None`, entirely
    separate from the real evaluation env constructed in `evaluate()`) is
    used purely to satisfy `VecNormalize.load()`'s required `venv`
    argument for observation/action-space validation -- it is never
    stepped or reset, and is closed immediately after loading, so it
    cannot interact with the real `EpisodeLogger` in any way.

    Critical correctness point (confirmed by reading SB3's source, not
    assumed): `VecNormalize.save()` pickles the ENTIRE object, including
    whatever `training`/`norm_reward` flags were set at save time. Since
    this file is always saved from the TRAINING env (`training=True`,
    see ppo/train.py), the loaded object initially has `training=True`
    too -- `.load()` does NOT reset these flags itself. They are forced
    to the evaluation-correct values explicitly below; relying on
    defaults after loading would silently keep updating statistics and
    normalizing reward during evaluation, which is exactly what the
    verification report checks for.
    """
    loader_env_fn = make_env(FIELD_VERTICES, config, episode_logger=None, render_mode=None)
    loader_venv = DummyVecEnv([loader_env_fn])
    vec_normalize = VecNormalize.load(vecnormalize_path, loader_venv)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    loader_venv.close()
    logger.info(
        "Loaded VecNormalize statistics from %s (forced training=False, norm_reward=False).",
        vecnormalize_path,
    )
    return vec_normalize


def evaluate(model_path: str, vecnormalize_path: str = None, n_episodes: int = 10, seed: int = 999,
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

    Parameters
    ----------
    model_path : str
        Path to a saved PPO model (.zip).
    vecnormalize_path : Optional[str]
        Path to the matching saved VecNormalize statistics (.pkl), as
        written by ppo/train.py alongside the model checkpoint it
        corresponds to (e.g. `results/models/vecnormalize_final.pkl` for
        `ppo_swarm_final.zip`, or `results/models/best_model/vecnormalize.pkl`
        for `best_model/best_model.zip`). If None, observations are NOT
        normalized at all -- only correct for a model trained with
        `ExperimentConfig().vecnormalize` effectively disabled. Passing
        no path (or a mismatched path) for a normalization-trained model
        will silently produce meaningless results, since the policy's
        inputs would not match what it learned on -- this function logs
        a clear warning in that case rather than failing silently.

    Returns
    -------
    Tuple[List[float], List[float]]
        Per-episode (total_reward, coverage_fraction) lists -- always
        RAW environment values; VecNormalize is never applied to reward
        or to any reported metric here, only (optionally) to the
        observations fed into `model.predict()`.
    """
    config = ExperimentConfig(seed=seed)
    episode_logger = EpisodeLogger("results/episode_logs", experiment_name=experiment_name)

    env_fn = make_env(FIELD_VERTICES, config, episode_logger=episode_logger, render_mode=None)
    env = env_fn()

    vec_normalize = None
    if vecnormalize_path is not None:
        vec_normalize = _load_vec_normalize(vecnormalize_path, config)
    else:
        logger.warning(
            "No vecnormalize_path provided -- evaluating with RAW (unnormalized) observations. "
            "If this model was trained with VecNormalize enabled (the default, see PPOConfig/"
            "VecNormalizeConfig), this will produce meaningless results. Pass --vecnormalize "
            "explicitly unless you are certain this model was trained without normalization."
        )

    model = PPO.load(model_path)
    logger.info("Loaded model from %s", model_path)

    episode_rewards, episode_coverages = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            # normalize_obs() operates correctly on a single (unbatched)
            # Dict observation -- RunningMeanStd's mean/var arrays are
            # shaped to match each observation key's own (unbatched)
            # shape, not a batched (num_envs, ...) shape, so no manual
            # batching/unbatching is needed here (confirmed by reading
            # VecNormalize._normalize_obs's source directly).
            model_input = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
            action, _ = model.predict(model_input, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            # `reward` here is always the RAW environment reward --
            # VecNormalize is never applied to reward during evaluation
            # (norm_reward=False was forced in _load_vec_normalize, and
            # in any case this reward comes directly from env.step(),
            # never passing through any VecNormalize reward transform at
            # all in this control flow).
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
    parser.add_argument("--vecnormalize", type=str, default=None,
                         help="Path to matching VecNormalize .pkl (e.g. results/models/vecnormalize_final.pkl). "
                              "Required for correct evaluation of a model trained with normalization enabled.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()
    evaluate(args.model, args.vecnormalize, args.episodes, args.seed)
