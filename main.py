"""
main.py
=======
Main entry point for the UAV Swarm PPO experiment pipeline.

Runs the full pipeline in order:
    1. Load (or create) experiment config
    2. Validate environment
    3. Train PPO agent
    4. Evaluate PPO vs Greedy vs Random baselines
    5. Generate all paper figures

Usage:
    # Full training run (default config):
    python main.py

    # Quick smoke-test (2000 timesteps):
    python main.py --smoke-test

    # Evaluate only (skip training, load existing model):
    python main.py --eval-only --model-path results/uav_swarm_ppo_v1/final_model

    # Custom config:
    python main.py --config results/my_run/config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path so all imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from uav_swarm_ppo.configs.config import ExperimentConfig, PPOConfig
from uav_swarm_ppo.env.uav_swarm_env import UAVSwarmEnv
from uav_swarm_ppo.evaluation.evaluator import Evaluator, GreedyBaseline, RandomBaseline
from uav_swarm_ppo.training.trainer import Trainer
from uav_swarm_ppo.visualization.plotter import Plotter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="UAV Swarm PPO — Training and Evaluation Pipeline"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a 2000-timestep smoke test to verify the setup."
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and run evaluation only."
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to a saved model for --eval-only mode."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file to load."
    )
    return parser.parse_args()


def validate_env(config: ExperimentConfig) -> None:
    """
    Run a quick sanity check on the environment.

    Verifies that:
        - The environment resets without error
        - Observations match the declared observation space
        - Step returns the correct number of values
        - 10 random-action steps complete without crashing

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to build the environment with.
    """
    print("\n[Validate] Running environment sanity check...")
    env = UAVSwarmEnv(config)
    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape, (
        f"Obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    )
    assert env.observation_space.contains(obs), "Obs outside observation_space bounds"

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print(
        f"[Validate] ✓ Environment OK | "
        f"Obs dim: {env.observation_space.shape[0]} | "
        f"Valid field cells: {env.field.n_valid_cells}"
    )


def main() -> None:
    """Main experiment pipeline."""
    args = parse_args()

    # ----------------------------------------------------------------
    # 1. Load or create config
    # ----------------------------------------------------------------

    if args.config:
        print(f"[Main] Loading config from {args.config}")
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()

    # Smoke test: override to very short training
    if args.smoke_test:
        print("[Main] SMOKE TEST mode — 2000 timesteps only")
        config.ppo.total_timesteps = 2000
        config.ppo.n_steps = 64
        config.ppo.batch_size = 32
        config.logging.checkpoint_freq = 1000
        config.logging.eval_freq = 1000
        config.logging.n_eval_episodes = 2
        config.experiment_name = "smoke_test"

    print(f"\n[Main] Experiment: {config.experiment_name}")
    print(f"       UAVs: {config.env.n_uavs} | "
          f"Grid: {config.env.grid_size}x{config.env.grid_size} | "
          f"Timesteps: {config.ppo.total_timesteps:,}")

    # ----------------------------------------------------------------
    # 2. Validate environment
    # ----------------------------------------------------------------

    validate_env(config)

    # ----------------------------------------------------------------
    # 3. Train (unless --eval-only)
    # ----------------------------------------------------------------

    results_dir = Path(config.logging.results_dir) / config.experiment_name
    model_path = args.model_path

    if not args.eval_only:
        trainer = Trainer(config)
        trainer.train()
        model_path = str(results_dir / "final_model")
    else:
        if model_path is None:
            model_path = str(results_dir / "final_model")
            print(f"[Main] --eval-only: defaulting to {model_path}")

    # ----------------------------------------------------------------
    # 4. Evaluate
    # ----------------------------------------------------------------

    print("\n[Main] Running evaluations...")

    # PPO evaluation
    evaluator = Evaluator(config, model_path, results_dir=results_dir)
    ppo_results = evaluator.run(n_seeds=5, n_episodes_per_seed=5, label="PPO")

    # Greedy baseline
    greedy = GreedyBaseline(config)
    greedy_results = greedy.evaluate(n_seeds=5, n_episodes_per_seed=5)

    # Random baseline
    random_bl = RandomBaseline(config)
    random_results = random_bl.evaluate(n_seeds=5, n_episodes_per_seed=5)

    # ----------------------------------------------------------------
    # 5. Generate paper figures
    # ----------------------------------------------------------------

    print("\n[Main] Generating figures...")
    plotter = Plotter(results_dir)

    # Comparison bar chart
    plotter.plot_comparison(
        summaries={
            "PPO": ppo_results,
            "Greedy": greedy_results,
            "Random": random_results,
        },
        save_name="fig3_algorithm_comparison",
    )

    # Field visualisation (one episode)
    env = UAVSwarmEnv(config)
    env.reset()
    plotter.plot_coverage_map(
        coverage_map=env.coverage_map,
        field_vertices=config.env.field_vertices,
        uav_positions=[uav.position for uav in env.uavs],
        static_obstacles=list(env.obstacle_manager.static_cells),
        dynamic_obstacles=list(env.obstacle_manager.dynamic_cells),
        save_name="fig2_field_layout",
    )
    env.close()

    print(f"\n[Main] ✓ Pipeline complete.")
    print(f"       Results saved to: {results_dir}")
    print(f"       Figures saved to: {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
