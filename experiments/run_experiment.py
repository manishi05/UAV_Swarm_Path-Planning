"""
experiments/run_experiment.py
==============================
End-to-end smoke test / demo of `SwarmFarmEnv`, now exercising the full
Goal 3 additions: structured reward breakdown, episode logging, and the
Gymnasium-compliant render()/close() lifecycle.

Run with:
    python -m experiments.run_experiment
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs.config import ExperimentConfig
from environment.swarm_env import SwarmFarmEnv
from utils.episode_logger import EpisodeLogger
from utils.logger import get_logger
from utils.seed import set_global_seed

logger = get_logger("uav_swarm_ppo.experiments.run_experiment",
                     log_file="results/run_experiment.log")

OUTPUT_DIR = "results"

FIELD_VERTICES = [
    (10, 10), (120, 5), (210, 60), (195, 130),
    (150, 110), (110, 175), (55, 150), (20, 90),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = ExperimentConfig(seed=42)
    config.to_json(os.path.join(OUTPUT_DIR, "experiment_config.json"))
    set_global_seed(config.seed)

    # Goal 3: EpisodeLogger wired directly into the env -- every step's
    # reward breakdown and every episode's summary (reward, coverage,
    # collisions, episode stats) is logged automatically.
    episode_logger = EpisodeLogger(log_dir=OUTPUT_DIR, experiment_name="smoke_test",
                                    verbose_step_logging=True)

    logger.info("Building SwarmFarmEnv...")
    env = SwarmFarmEnv(FIELD_VERTICES, config, episode_logger=episode_logger,
                        render_mode="rgb_array")

    obs, info = env.reset(seed=config.seed)
    logger.info("Initial info: %s", info)

    # Detailed, titled, saved-to-disk figure -- uses get_render_state()
    # + the renderer directly (not the Gymnasium render() path).
    from visualization.renderer import EnvironmentRenderer
    renderer = EnvironmentRenderer()
    renderer.render_frame(env.get_render_state(),
                           save_path=os.path.join(OUTPUT_DIR, "frame_initial.png"),
                           title="Initial state (t=0)")

    # Gymnasium-compliant render() path -- returns a numpy RGB array.
    frame_array = env.render()
    logger.info("Gymnasium render() returned array of shape %s dtype %s",
                frame_array.shape, frame_array.dtype)

    rng = np.random.default_rng(config.seed)
    # Run for the full configured episode length so termination/truncation
    # actually fires and EpisodeLogger.end_episode() writes a CSV row --
    # a partial demo loop (as in the Goal 1/2 version) never closes the
    # episode, so the logging framework's output would never be exercised.
    n_steps = config.env.max_episode_steps
    episode_reward = 0.0

    for step in range(n_steps):
        action = env.action_space.sample()  # random policy -- PPO replaces this on Day 3
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if step % 20 == 0 or terminated or truncated:
            logger.info(
                "step=%3d reward=%8.2f cum_reward=%10.2f coverage=%.3f valid_action_frac=%.3f "
                "breakdown=%s",
                step, reward, episode_reward, info["coverage_fraction"],
                info["valid_action_fraction"],
                {k: round(v, 1) for k, v in info["reward_breakdown"].items() if k != "total"},
            )

        if terminated or truncated:
            logger.info("Episode ended at step %d (terminated=%s, truncated=%s)",
                        step, terminated, truncated)
            break

    renderer.render_frame(env.get_render_state(),
                           save_path=os.path.join(OUTPUT_DIR, "frame_final.png"),
                           title=f"State after {step + 1} steps (random policy)")

    logger.info("Smoke test complete. Final info: %s", info)
    print("\n=== Smoke test summary ===")
    print(f"Steps executed:        {step + 1}")
    print(f"Cumulative reward:     {episode_reward:.2f}")
    print(f"Coverage fraction:     {info['coverage_fraction']:.3f}")
    print(f"Valid action fraction: {info['valid_action_fraction']:.3f}")
    print(f"Collision counts:      {info['collision_counts']}")
    print(f"Reward breakdown:      {info['reward_breakdown']}")
    print(f"\nEpisode summary CSV:   {OUTPUT_DIR}/smoke_test_episodes.csv")

    # Goal 3: close() releases matplotlib resources and flushes the logger.
    env.close()


if __name__ == "__main__":
    main()
