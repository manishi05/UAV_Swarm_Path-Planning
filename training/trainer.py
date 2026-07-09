"""
training/trainer.py
===================
PPO training orchestration module.

Wires together UAVSwarmEnv + Stable-Baselines3 PPO into a clean,
reproducible training pipeline with:
    - TensorBoard logging
    - Periodic model checkpointing
    - Mid-training evaluation callbacks
    - Config serialisation at run start

Architecture note:
    Trainer owns the training lifecycle. It creates the environment,
    instantiates the SB3 PPO model with config-driven hyperparameters,
    attaches callbacks, and calls model.learn(). All paths and
    hyperparameters come from ExperimentConfig — no hardcoded values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from uav_swarm_ppo.configs.config import ExperimentConfig
from uav_swarm_ppo.env.uav_swarm_env import UAVSwarmEnv


class Trainer:
    """
    Manages the full PPO training lifecycle.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.log_cfg = config.logging
        self.ppo_cfg = config.ppo

        # Resolve output directories
        self.log_dir = Path(self.log_cfg.log_dir) / config.experiment_name
        self.results_dir = Path(self.log_cfg.results_dir) / config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Save config alongside results for full reproducibility
        config.save(self.results_dir / "config.json")
        print(f"[Trainer] Config saved to {self.results_dir / 'config.json'}")

        # Build environments
        self.train_env = self._make_env(seed=config.env.seed)
        self.eval_env = self._make_env(seed=config.env.seed + 1000)

        # Build PPO model
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> PPO:
        """
        Run the full training loop.

        Returns
        -------
        PPO
            The trained SB3 PPO model.
        """
        print(
            f"[Trainer] Starting training: {self.ppo_cfg.total_timesteps:,} "
            f"timesteps | Experiment: {self.config.experiment_name}"
        )

        callbacks = self._build_callbacks()
        self.model.learn(
            total_timesteps=self.ppo_cfg.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        final_path = str(self.results_dir / "final_model")
        self.model.save(final_path)
        print(f"[Trainer] Final model saved to {final_path}.zip")

        return self.model

    def load(self, path: str) -> PPO:
        """
        Load a previously saved PPO model.

        Parameters
        ----------
        path : str
            Path to the .zip model file (without extension).

        Returns
        -------
        PPO
            Loaded model, ready for evaluation or continued training.
        """
        self.model = PPO.load(path, env=self.train_env)
        print(f"[Trainer] Model loaded from {path}")
        return self.model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_env(self, seed: int) -> DummyVecEnv:
        """
        Create a vectorised, monitored training environment.

        Wraps UAVSwarmEnv in SB3's Monitor (for episode stats) and
        DummyVecEnv (required by SB3 even for single-env training).

        Parameters
        ----------
        seed : int
            Random seed for this environment instance.

        Returns
        -------
        DummyVecEnv
            Wrapped environment compatible with SB3.
        """
        def _make() -> Monitor:
            # Create config copy with updated seed for this env
            from copy import deepcopy
            from uav_swarm_ppo.configs.config import EnvConfig
            cfg_copy = deepcopy(self.config)
            cfg_copy.env.seed = seed
            env = UAVSwarmEnv(cfg_copy)
            env = Monitor(env, str(self.log_dir))
            return env

        return DummyVecEnv([_make])

    def _build_model(self) -> PPO:
        """
        Instantiate the SB3 PPO model with config-driven hyperparameters.

        All values come from PPOConfig — zero hardcoded numbers here.

        Returns
        -------
        PPO
            Initialised (untrained) PPO model.
        """
        cfg = self.ppo_cfg

        model = PPO(
            policy=cfg.policy,
            env=self.train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            verbose=cfg.verbose,
            tensorboard_log=str(self.log_dir),
            seed=self.config.env.seed,
        )

        print(
            f"[Trainer] PPO model built | Policy: {cfg.policy} | "
            f"LR: {cfg.learning_rate} | clip: {cfg.clip_range}"
        )
        return model

    def _build_callbacks(self) -> CallbackList:
        """
        Build the SB3 callback stack for training.

        Callbacks:
            CheckpointCallback — saves model every N timesteps.
            EvalCallback       — evaluates on eval_env every N timesteps,
                                 saves best model automatically.

        Returns
        -------
        CallbackList
            Combined callback list for model.learn().
        """
        checkpoint_cb = CheckpointCallback(
            save_freq=self.log_cfg.checkpoint_freq,
            save_path=str(self.log_dir / "checkpoints"),
            name_prefix="ppo_uav",
            verbose=1,
        )

        eval_cb = EvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=str(self.results_dir / "best_model"),
            log_path=str(self.results_dir / "eval_logs"),
            eval_freq=self.log_cfg.eval_freq,
            n_eval_episodes=self.log_cfg.n_eval_episodes,
            deterministic=True,
            verbose=1,
        )

        return CallbackList([checkpoint_cb, eval_cb])
