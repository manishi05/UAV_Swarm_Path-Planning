"""
ppo/train.py
=============
PPO training pipeline (Goal 4), built on Stable-Baselines3, training
directly against `SwarmFarmEnv` (Goals 1-3) -- **zero changes to any
environment/ module were needed to reach this point**; that was the
whole design intent of building the Gymnasium environment first.

Run with:
    python -m ppo.train

What this script does, in order:
  1. Builds `ExperimentConfig` (field/obstacle/wind/drone/reward/env/ppo)
     and saves it to disk immediately -- every trained model has an
     exact, reproducible record of what it was trained with.
  2. Builds `n_envs` parallel training environments (`DummyVecEnv`), each
     wrapped in SB3's `Monitor` for standard episode reward/length
     tracking, each independently seeded via `make_env(seed_offset=i)`
     (see environment/swarm_env.py) so parallel rollouts see genuinely
     different field/obstacle/wind realizations, not `n_envs` copies of
     the same episode.
  3. Builds one separate evaluation environment for periodic
     deterministic evaluation during training.
  4. Constructs the PPO model (`MultiInputPolicy`, since our observation
     space is a `Dict` -- see swarm_env.py) with every hyperparameter
     read from `PPOConfig` (configs/config.py).
  5. Attaches three callbacks: `CheckpointCallback` (periodic model
     saves), `EvalCallback` (periodic deterministic evaluation, saves
     the best model separately), and `SwarmMetricsCallback` (our custom
     coverage/reward-breakdown/collision logging -- see ppo/callbacks.py).
  6. Trains for `PPOConfig.total_timesteps`, with TensorBoard + CSV
     logging active throughout.
  7. Saves the final model and closes all environments.

See PPO_TRAINING.md for the full explanation of every step, what to
watch in the logs, and how to tune each hyperparameter.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from configs.config import ExperimentConfig, PPOConfig
from environment.swarm_env import make_env
from ppo.callbacks import SwarmMetricsCallback
from utils.logger import get_logger
from utils.seed import set_global_seed

logger = get_logger("uav_swarm_ppo.ppo.train", log_file="results/train.log")

# NOTE: kept identical to experiments/run_experiment.py's field so Day 1-4
# results stay comparable. Replace with your real field's vertices before
# generating paper results (update in both places, or better, factor out
# to a shared configs/field_presets.py -- see PPO_TRAINING.md improvement #1).
FIELD_VERTICES = [
    (10, 10), (120, 5), (210, 60), (195, 130),
    (150, 110), (110, 175), (55, 150), (20, 90),
]


def build_train_vec_env(vertices, config: ExperimentConfig, n_envs: int, monitor_dir: str) -> DummyVecEnv:
    """
    Builds `n_envs` independently-seeded training environments wrapped
    in `DummyVecEnv`.

    DummyVecEnv (sequential, single-process) rather than SubprocVecEnv:
    at the environment sizes used here (small grids, a handful of UAVs),
    per-step Python/Shapely overhead dominates over any parallelism gain
    from separate processes, and DummyVecEnv avoids the added complexity
    of cross-process pickling of `SwarmFarmEnv`'s Shapely geometry
    objects. If you scale to much larger fields or many more parallel
    envs, SubprocVecEnv is a documented upgrade path (PPO_TRAINING.md
    §6) -- `make_env` already returns the exact zero-argument callables
    SubprocVecEnv requires, so switching is a one-line change here.
    """
    os.makedirs(monitor_dir, exist_ok=True)
    env_fns = []
    for i in range(n_envs):
        env_fn = make_env(vertices, config, episode_logger=None, render_mode=None, seed_offset=i)

        def _monitored(env_fn=env_fn, idx=i):
            env = env_fn()
            return Monitor(env, filename=os.path.join(monitor_dir, f"train_env{idx}"))

        env_fns.append(_monitored)
    return DummyVecEnv(env_fns)


def train(config: ExperimentConfig, verbose: int = 1) -> str:
    """
    Runs one full PPO training job from a given `ExperimentConfig`.

    Returns
    -------
    str
        Path to the saved final model (`.zip`, SB3's save format).
    """
    ppo_cfg = config.ppo

    # PPO's internal invariant: n_steps * n_envs must be divisible by
    # batch_size, or minibatch construction fails deep inside SB3 with a
    # less obvious error. Checked explicitly here with a clear message.
    rollout_size = ppo_cfg.n_steps * ppo_cfg.n_envs
    assert rollout_size % ppo_cfg.batch_size == 0, (
        f"n_steps ({ppo_cfg.n_steps}) * n_envs ({ppo_cfg.n_envs}) = {rollout_size} "
        f"must be divisible by batch_size ({ppo_cfg.batch_size}). "
        f"Adjust one of these three values in PPOConfig."
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs(ppo_cfg.model_dir, exist_ok=True)
    config.to_json(os.path.join("results", "ppo_experiment_config.json"))
    set_global_seed(config.seed)

    logger.info("Building %d parallel training environment(s)...", ppo_cfg.n_envs)
    train_env = build_train_vec_env(FIELD_VERTICES, config, ppo_cfg.n_envs, ppo_cfg.monitor_dir)

    logger.info("Building evaluation environment...")
    eval_env_fn = make_env(FIELD_VERTICES, config, episode_logger=None, render_mode=None, seed_offset=1000)
    eval_env = DummyVecEnv([
        lambda: Monitor(eval_env_fn(), filename=os.path.join(ppo_cfg.monitor_dir, "eval_env"))
    ])

    policy_kwargs = dict(net_arch=dict(pi=list(ppo_cfg.net_arch_pi), vf=list(ppo_cfg.net_arch_vf)))

    model = PPO(
        policy="MultiInputPolicy",  # required for Dict observation spaces (see swarm_env.py)
        env=train_env,
        learning_rate=ppo_cfg.learning_rate,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=ppo_cfg.tensorboard_log,
        seed=config.seed,
        verbose=verbose,
    )

    # Attach a CSV-format logger in addition to the default stdout +
    # tensorboard loggers, so training curves can be plotted directly
    # from <tensorboard_log>/progress.csv with pandas
    # (visualization/training_plots.py), without needing to parse
    # TensorBoard's binary event-file format.
    sb3_logger = configure(ppo_cfg.tensorboard_log, ["stdout", "csv", "tensorboard"])
    model.set_logger(sb3_logger)

    # SB3 convention for vectorized envs: *Callback save_freq/eval_freq
    # are counted in calls to the callback (once per vectorized step),
    # not in raw environment timesteps -- dividing by n_envs converts a
    # "timesteps" cadence (max(x // n_envs, 1)) into the correct callback
    # cadence. Documented explicitly since this is a common silent bug
    # source (checkpoints/evals firing n_envs times more/less often than intended).
    checkpoint_callback = CheckpointCallback(
        save_freq=max(ppo_cfg.checkpoint_freq // ppo_cfg.n_envs, 1),
        save_path=ppo_cfg.model_dir,
        name_prefix="ppo_swarm",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(ppo_cfg.model_dir, "best_model"),
        log_path=os.path.join(ppo_cfg.model_dir, "eval_logs"),
        eval_freq=max(ppo_cfg.eval_freq // ppo_cfg.n_envs, 1),
        n_eval_episodes=ppo_cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    metrics_callback = SwarmMetricsCallback()
    callback = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    logger.info("Starting PPO training for %d timesteps (n_envs=%d, n_steps=%d, batch_size=%d)...",
                ppo_cfg.total_timesteps, ppo_cfg.n_envs, ppo_cfg.n_steps, ppo_cfg.batch_size)
    model.learn(total_timesteps=ppo_cfg.total_timesteps, callback=callback, progress_bar=False)

    final_path = os.path.join(ppo_cfg.model_dir, "ppo_swarm_final")
    model.save(final_path)
    logger.info("Training complete. Final model saved to %s.zip", final_path)

    train_env.close()
    eval_env.close()
    return final_path + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Train PPO on SwarmFarmEnv.")
    parser.add_argument("--total-timesteps", type=int, default=None,
                         help="Override PPOConfig.total_timesteps (useful for a quick pipeline check).")
    parser.add_argument("--n-envs", type=int, default=None, help="Override PPOConfig.n_envs.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override PPOConfig.n_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override PPOConfig.batch_size.")
    args = parser.parse_args()

    ppo_kwargs = {}
    if args.total_timesteps is not None:
        ppo_kwargs["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        ppo_kwargs["n_envs"] = args.n_envs
    if args.n_steps is not None:
        ppo_kwargs["n_steps"] = args.n_steps
    if args.batch_size is not None:
        ppo_kwargs["batch_size"] = args.batch_size

    config = ExperimentConfig(seed=42, ppo=PPOConfig(**ppo_kwargs))
    train(config)


if __name__ == "__main__":
    main()
