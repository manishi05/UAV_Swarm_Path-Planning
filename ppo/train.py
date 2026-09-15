"""
ppo/train.py
=============
PPO training pipeline (Goal 4), built on Stable-Baselines3, training
directly against `SwarmFarmEnv` (Goals 1-3) -- **zero changes to any
environment/ module were needed to reach this point**; that was the
whole design intent of building the Gymnasium environment first.

VecNormalize integration (see VECNORMALIZE_INTEGRATION.md for the full
inspection/rationale): the vectorized training env, and separately the
evaluation env used by `EvalCallback`, are now wrapped in SB3's
`VecNormalize`, which gives PPO numerically well-scaled observations and
rewards. This is a training-time statistical transform layered OUTSIDE
`SwarmFarmEnv` -- it does not change the environment's observation
representation, does not change `environment/reward.py` or
`RewardConfig`, and does not change any reward weight. The environment
always computes and exposes the same raw reward/observations it always
has (confirmed by inspection: `EpisodeLogger` and `SwarmMetricsCallback`
both read from the environment's `info` dict, which `VecNormalize` never
touches -- see the module docstring in `ppo/callbacks.py`).

Run with:
    python -m ppo.train

What this script does, in order:
  1. Builds `ExperimentConfig` (field/obstacle/wind/drone/reward/env/ppo/
     vecnormalize) and saves it to disk immediately -- every trained
     model has an exact, reproducible record of what it was trained
     with, including whether/how VecNormalize was configured.
  2. Builds `n_envs` parallel training environments (`DummyVecEnv`), each
     wrapped in SB3's `Monitor` for standard episode reward/length
     tracking, each independently seeded via `make_env(seed_offset=i)`
     (see environment/swarm_env.py) so parallel rollouts see genuinely
     different field/obstacle/wind realizations, not `n_envs` copies of
     the same episode. The resulting `DummyVecEnv` is then wrapped in
     `VecNormalize(training=True)`.
  3. Builds one separate evaluation environment for periodic
     deterministic evaluation during training, also wrapped in
     `VecNormalize`, but with `training=False, norm_reward=False`.
     SB3's `EvalCallback` automatically synchronizes this eval env's
     normalization statistics from the training env before every
     evaluation call (`sync_envs_normalization`, confirmed by reading
     SB3's own source -- see VECNORMALIZE_INTEGRATION.md) -- no manual
     syncing code was added here, since SB3 already does this correctly.
  4. Constructs the PPO model (`MultiInputPolicy`, since our observation
     space is a `Dict` -- see swarm_env.py) with every hyperparameter
     read from `PPOConfig` (configs/config.py).
  5. Attaches five callbacks: `CheckpointCallback` (periodic model
     saves), `SaveVecNormalizeCallback` (periodic paired normalization-
     statistics saves, new), `EvalCallback` (periodic deterministic
     evaluation, saves the best model separately, with
     `SaveVecNormalizeOnNewBestCallback` attached via
     `callback_on_new_best` so the best model is never orphaned from its
     matching statistics), and `SwarmMetricsCallback` (our custom
     coverage/reward-breakdown/collision logging -- reads raw values
     from `info`, unaffected by VecNormalize, see ppo/callbacks.py).
  6. Trains for `PPOConfig.total_timesteps`, with TensorBoard + CSV
     logging active throughout.
  7. Saves the final model AND the final VecNormalize statistics
     (paired, same directory) and closes all environments.

See PPO_TRAINING.md for the full explanation of every PPO step, and
VECNORMALIZE_INTEGRATION.md for the normalization-specific design,
inspection findings, and verification evidence.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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


class SaveVecNormalizeCallback(BaseCallback):
    """
    Saves the training env's `VecNormalize` running statistics alongside
    each periodic model checkpoint, at the same cadence as
    `CheckpointCallback`.

    Why this is needed: SB3's built-in `CheckpointCallback` has no
    knowledge of `VecNormalize` and only saves policy weights. Without
    this, a periodic checkpoint `.zip` could never be correctly
    re-evaluated later, since the observation-normalization statistics
    it was trained against would be lost -- a checkpoint trained with
    normalized observations is not meaningfully separable from the
    exact statistics used to normalize them (see the "checkpoints" and
    "best-model compatibility" requirements in
    VECNORMALIZE_INTEGRATION.md).

    Deliberately implemented here, not added to `ppo/callbacks.py`
    (`SwarmMetricsCallback`'s home), to keep this VecNormalize-
    integration change confined to exactly the files reported as
    needing modification (`configs/config.py`, `ppo/train.py`,
    `ppo/evaluate.py`).
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self._save_freq = save_freq
        self._save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            vec_normalize_env = self.model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                path = os.path.join(self._save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl")
                vec_normalize_env.save(path)
                if self.verbose > 0:
                    logger.info("VecNormalize statistics saved to %s", path)
        return True


class SaveVecNormalizeOnNewBestCallback(BaseCallback):
    """
    Paired with `EvalCallback`'s `callback_on_new_best` hook: saves the
    eval env's `VecNormalize` statistics exactly when `EvalCallback`
    decides the current policy is the new best -- so `best_model.zip` is
    never orphaned from the normalization statistics needed to evaluate
    it correctly.

    At the moment this fires, the eval env's statistics are guaranteed
    to already be synchronized with the training env's current
    statistics: `EvalCallback._on_step()` calls SB3's own
    `sync_envs_normalization(training_env, eval_env)` immediately before
    running the evaluation episodes that determine whether this is a new
    best (confirmed by reading SB3's `EvalCallback` source directly, not
    assumed -- see VECNORMALIZE_INTEGRATION.md). So saving the eval env's
    statistics here is equivalent to saving the training env's
    statistics at this exact instant, and is the more semantically
    correct choice: it's literally the normalization state that produced
    this best-model result.
    """

    def __init__(self, eval_vec_normalize_env: VecNormalize, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self._eval_vec_normalize_env = eval_vec_normalize_env
        self._save_path = save_path

    def _on_step(self) -> bool:
        os.makedirs(self._save_path, exist_ok=True)
        path = os.path.join(self._save_path, "vecnormalize.pkl")
        self._eval_vec_normalize_env.save(path)
        if self.verbose > 0:
            logger.info("Best-model VecNormalize statistics saved to %s", path)
        return True


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

    # --- VecNormalize wrapping (see module docstring for full rationale) ---
    vn_cfg = config.vecnormalize
    logger.info(
        "Wrapping train/eval envs with VecNormalize (norm_obs=%s, norm_reward=%s, "
        "clip_obs=%.1f, clip_reward=%.1f, norm_obs_keys=%s)...",
        vn_cfg.norm_obs, vn_cfg.norm_reward, vn_cfg.clip_obs, vn_cfg.clip_reward, vn_cfg.norm_obs_keys,
    )
    train_env = VecNormalize(
        train_env,
        training=True,
        norm_obs=vn_cfg.norm_obs,
        norm_reward=vn_cfg.norm_reward,
        clip_obs=vn_cfg.clip_obs,
        clip_reward=vn_cfg.clip_reward,
        norm_obs_keys=vn_cfg.norm_obs_keys,
        gamma=vn_cfg.gamma,
    )
    # Eval env: training=False (statistics must never update during
    # evaluation -- requirement), norm_reward=False (evaluation must
    # report/act on RAW reward for research metrics -- requirement).
    # norm_obs stays enabled: the policy still needs in-distribution
    # (normalized) observations to act sensibly, since it was trained on
    # normalized observations -- only the *reward* semantics differ
    # between training and evaluation here, not what the policy sees.
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=vn_cfg.norm_obs,
        norm_reward=False,
        clip_obs=vn_cfg.clip_obs,
        clip_reward=vn_cfg.clip_reward,
        norm_obs_keys=vn_cfg.norm_obs_keys,
        gamma=vn_cfg.gamma,
    )

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
    # Paired VecNormalize statistics for periodic checkpoints (see class
    # docstring) -- same save_freq as checkpoint_callback above, so every
    # ppo_swarm_{N}_steps.zip has a matching vecnormalize_{N}_steps.pkl.
    vecnormalize_checkpoint_callback = SaveVecNormalizeCallback(
        save_freq=max(ppo_cfg.checkpoint_freq // ppo_cfg.n_envs, 1),
        save_path=ppo_cfg.model_dir,
    )
    # Paired VecNormalize statistics for the best-model checkpoint (see
    # class docstring) -- fires exactly when EvalCallback finds a new best.
    save_best_vecnormalize_callback = SaveVecNormalizeOnNewBestCallback(
        eval_vec_normalize_env=eval_env,
        save_path=os.path.join(ppo_cfg.model_dir, "best_model"),
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=save_best_vecnormalize_callback,
        best_model_save_path=os.path.join(ppo_cfg.model_dir, "best_model"),
        log_path=os.path.join(ppo_cfg.model_dir, "eval_logs"),
        eval_freq=max(ppo_cfg.eval_freq // ppo_cfg.n_envs, 1),
        n_eval_episodes=ppo_cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    metrics_callback = SwarmMetricsCallback()
    callback = CallbackList([
        checkpoint_callback, vecnormalize_checkpoint_callback, eval_callback, metrics_callback,
    ])

    logger.info("Starting PPO training for %d timesteps (n_envs=%d, n_steps=%d, batch_size=%d)...",
                ppo_cfg.total_timesteps, ppo_cfg.n_envs, ppo_cfg.n_steps, ppo_cfg.batch_size)
    model.learn(total_timesteps=ppo_cfg.total_timesteps, callback=callback, progress_bar=False)

    final_path = os.path.join(ppo_cfg.model_dir, "ppo_swarm_final")
    model.save(final_path)
    final_vecnormalize_path = os.path.join(ppo_cfg.model_dir, "vecnormalize_final.pkl")
    train_env.save(final_vecnormalize_path)
    logger.info("Training complete. Final model saved to %s.zip", final_path)
    logger.info("Final VecNormalize statistics saved to %s", final_vecnormalize_path)

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
