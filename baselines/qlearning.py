"""
baselines/qlearning.py
========================
Classical Q-Learning baseline, reproducing Section 3.3.2-3.3.5 of
Puente-Castro et al. (2022) as closely as possible, adapted to run
against `SwarmFarmEnv` (Goals 1-3) instead of the paper's fixed square
grid -- so the comparison against PPO (Goal 4) is on the *same*
environment, not a simplified stand-in.

What's reproduced faithfully from the paper:
  - A small dense ANN approximating Q-values (Section 3.3.2): first
    layer 167 units, "linear activation" (i.e. no nonlinearity -- see
    the deviation note below for why this is honored, not silently fixed).
  - Bellman update, Eq. 1: Q(s,a) <- r + gamma * max_a' Q(s',a'), gamma=0.91.
  - Epsilon-greedy exploration: epsilon=0.47, decayed by x0.93 per
    episode, floored at 0.05 (Section 3.3.4).
  - Per-UAV FIFO experience replay memory, size 60 (Section 3.3.5):
    "Each UAV in the group has its own memory... At no time the actions
    of other UAVs are stored."
  - A single shared (global) network across all UAVs (Section 4.2's own
    finding: "using a single ANN for the whole swarm is the best
    configuration") -- consistent with PPO's own single-global-policy
    MultiDiscrete design (swarm_env.py), so all three algorithms compared
    in Day 4 share this same "one controller for the whole swarm" structure.

One deliberate, documented deviation from the paper (everything else is
paper-faithful):
  - Output layer: the paper specifies a *softmax* activation on the
    Q-value output layer (Section 3.3.2, Fig. 4). Q-values are not a
    probability distribution -- they are estimates of expected future
    reward, which is unbounded in both sign and magnitude (our own
    reward scale reaches the hundreds of thousands, see
    PPO_TRAINING.md Sec. 5). A softmax output would compress and
    distort the *relative* magnitude of Q-values action-to-action,
    which is exactly the information argmax-based action selection
    depends on. This is used here as a genuine value-based baseline
    (linear/identity output activation, the standard choice for a
    Q-network, e.g. the original DQN architecture), not to make the
    paper look worse -- softmax on Q-values is not standard practice
    anywhere in the RL literature and reproducing it exactly would make
    this a strictly worse (and non-standard) baseline for no
    scientific reason. `QLearningConfig.use_paper_faithful_first_layer`
    still keeps the *first* layer's literal "linear activation" choice,
    which is faithful and harmless (it only removes one nonlinearity,
    it doesn't break value semantics the way a softmax output would).
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import ExperimentConfig, QLearningConfig
from environment.drone import ActionType
from environment.swarm_env import SwarmFarmEnv, make_env
from utils.episode_logger import EpisodeLogger
from utils.logger import get_logger

logger = get_logger("uav_swarm_ppo.baselines.qlearning", log_file="results/qlearning.log")

N_ACTIONS = len(ActionType)


def flatten_observation_for_uav(obs: Dict[str, np.ndarray], uav_index: int, n_uavs: int) -> np.ndarray:
    """
    Builds the per-UAV state vector fed to the shared Q-network.

    The reference paper's ANN input is three flattened maps ("flying
    map", "visited cells map", "drones' positions map", Fig. 4). We
    extend this with the `dynamic_obstacle_map` (a genuine addition
    since the paper has no dynamic obstacles at all) and the scalar
    `wind_vector`/this-UAV's `battery_frac` (both Goal-2/3 concepts the
    paper has no equivalent of). Everything is flattened and
    concatenated into one 1D vector, matching the paper's own "Flatten
    Layer" step (Fig. 4) before the dense layers.

    Parameters
    ----------
    obs : Dict[str, np.ndarray]
        One observation dict as returned by `SwarmFarmEnv._get_obs()`.
    uav_index : int
        Which UAV this state vector is being built for -- only
        `battery_frac[uav_index]` is included (a UAV only observes its
        own battery, not its teammates'), matching the paper's
        per-UAV-memory philosophy of not leaking one UAV's private state
        into another's decision-making.
    n_uavs : int
        Total UAV count, needed to index `battery_frac` safely.

    Returns
    -------
    np.ndarray, shape (input_dim,), dtype float32
    """
    parts = [
        obs["flyable_map"].flatten(),
        obs["visited_map"].flatten(),
        obs["dynamic_obstacle_map"].flatten(),
        obs["uav_position_map"].flatten(),
        obs["wind_vector"].flatten(),
        np.array([obs["battery_frac"][uav_index]], dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


class QNetwork(nn.Module):
    """
    The shared (global) Q-network, architecture matching the reference
    paper's Section 3.3.2 as closely as is sensible for a genuine
    Q-learning baseline (see module docstring for the one deviation).

    Two dense layers: `hidden_units` (167 by default, paper's value)
    with an optional linear (identity) first-layer activation matching
    the paper's literal spec, then a `N_ACTIONS`-unit linear output
    layer producing raw Q-values, one per discrete action.
    """

    def __init__(self, input_dim: int, hidden_units: int = 167,
                 use_paper_faithful_first_layer: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, N_ACTIONS)
        # Paper Section 3.3.2: "the first one with 167 neurons and linear
        # activation function". A "linear activation" is mathematically
        # equivalent to no activation at all (identity) -- honored
        # exactly when use_paper_faithful_first_layer=True. Setting it to
        # False swaps in a ReLU, a standard/better-performing choice for
        # a real trained baseline; kept configurable so you can report
        # either as a documented ablation if useful for the paper.
        self._use_identity = use_paper_faithful_first_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        if not self._use_identity:
            h = torch.relu(h)
        return self.fc2(h)  # raw Q-values, no output activation (see module docstring)


class ReplayMemory:
    """
    Per-UAV FIFO experience replay buffer, `maxlen=memory_size` (60 by
    default, paper Section 3.3.5). One instance per UAV -- the paper is
    explicit that UAVs never share transitions ("At no time the actions
    of other UAVs are stored... an action is not correct for one UAV
    does not imply that it is incorrect for the others since they can be
    in different positions on the map").
    """

    def __init__(self, capacity: int):
        self._buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Uniform random sample, or the whole buffer if it's smaller than batch_size."""
        n = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, n)

    def __len__(self) -> int:
        return len(self._buffer)


class QLearningAgent:
    """
    Owns the single shared `QNetwork` and one `ReplayMemory` per UAV,
    and implements epsilon-greedy action selection plus the Bellman
    (Eq. 1) TD update via RMSprop -- the paper's specified optimizer
    (Section 3.3.2).

    This is genuinely "classical" Q-Learning with function approximation
    (a small ANN standing in for the Q-table, exactly as the paper
    frames it -- Section 3.3.2's own contrast with "Deep Q-learning" as
    a *different*, fancier method they are not using), not DQN: no
    target network, no double-Q correction, no prioritized replay. This
    is a deliberate fidelity choice, not an oversight -- see BASELINES.md
    for why, and what upgrading to DQN would look like if you want a
    stronger (but less paper-faithful) baseline later.
    """

    def __init__(self, input_dim: int, n_uavs: int, config: QLearningConfig, seed: int = 42):
        self._config = config
        self._n_uavs = n_uavs
        torch.manual_seed(seed)
        self._rng = random.Random(seed)

        self.network = QNetwork(input_dim, config.hidden_units, config.use_paper_faithful_first_layer)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memories: List[ReplayMemory] = [ReplayMemory(config.memory_size) for _ in range(n_uavs)]
        self.epsilon = config.epsilon_start

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection: with probability `epsilon`,
        choose uniformly at random over the 5 actions; otherwise choose
        argmax_a Q(state, a) from the current network (paper Section 3.3.4).
        """
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(N_ACTIONS)
        with torch.no_grad():
            q_values = self.network(torch.from_numpy(state).unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, uav_index: int, state, action, reward, next_state, done) -> None:
        self.memories[uav_index].push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        One gradient update: samples `batch_size` transitions from each
        UAV's memory (kept separate per UAV per the paper, then pooled
        only at the *batch* level for one shared-network gradient step --
        the weights being updated are shared, even though the
        transitions contributing to any one update are drawn respecting
        each UAV's own memory boundary, matching the paper's "no
        cross-UAV noise in the stored data" principle while still
        training one global network from all of it).

        Returns
        -------
        float
            The scalar MSE loss for this update (for logging).
        """
        batch = []
        per_uav_batch = max(self._config.batch_size // self._n_uavs, 1)
        for memory in self.memories:
            batch.extend(memory.sample(per_uav_batch))
        if len(batch) < 2:
            return 0.0  # not enough data yet to form a meaningful gradient step

        states = torch.from_numpy(np.stack([b[0] for b in batch]))
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch]))
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        # Bellman target (Eq. 1): r + gamma * max_a' Q(s', a'), zeroed
        # out for terminal transitions (no bootstrapping past episode end).
        with torch.no_grad():
            next_q = self.network(next_states)
            max_next_q = torch.max(next_q, dim=1).values
            target = rewards + self._config.gamma * max_next_q * (1.0 - dones)

        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Called once per episode (paper Section 3.3.4: decay applies per episode, not per step)."""
        self.epsilon = max(self._config.epsilon_min, self.epsilon * self._config.epsilon_decay)


def train_qlearning(config: ExperimentConfig, vertices: List[Tuple[float, float]]) -> str:
    """
    Full training loop: runs `QLearningConfig.n_episodes` episodes
    against `SwarmFarmEnv`, storing transitions and updating the shared
    network every `train_every_n_steps` environment steps.

    Uses the *same* `SwarmFarmEnv` and `make_env` factory PPO training
    uses (environment/swarm_env.py) -- no separate/simplified
    environment was built for this baseline, so the comparison against
    PPO is on identical dynamics, obstacles, wind, and reward function.

    Returns
    -------
    str
        Path to the saved model weights (.pt file).
    """
    qcfg = config.qlearning
    os.makedirs(qcfg.model_dir, exist_ok=True)

    episode_logger = EpisodeLogger("results/episode_logs", experiment_name="qlearning_train")
    env: SwarmFarmEnv = make_env(vertices, config, episode_logger=episode_logger)()

    n_uavs = config.env.n_uavs
    obs, info = env.reset(seed=config.seed)
    input_dim = flatten_observation_for_uav(obs, 0, n_uavs).shape[0]
    logger.info("Q-network input dimension: %d (hidden_units=%d)", input_dim, qcfg.hidden_units)

    agent = QLearningAgent(input_dim, n_uavs, qcfg, seed=config.seed)

    step_counter = 0
    for episode in range(qcfg.n_episodes):
        obs, info = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        episode_loss = []

        while not (terminated or truncated):
            per_uav_states = [flatten_observation_for_uav(obs, i, n_uavs) for i in range(n_uavs)]
            actions = np.array([agent.select_action(s) for s in per_uav_states])

            next_obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward

            next_per_uav_states = [flatten_observation_for_uav(next_obs, i, n_uavs) for i in range(n_uavs)]
            done = terminated or truncated
            for i in range(n_uavs):
                # Shared team reward is stored against every UAV's own
                # (state, action) pair -- consistent with our reward
                # function's design (RewardCalculator sums per-UAV
                # contributions into one scalar team reward each step,
                # see environment/reward.py), and with what PPO's single
                # global policy also optimizes against, keeping the
                # learning signal identical in kind across both algorithms.
                agent.store_transition(i, per_uav_states[i], int(actions[i]), reward,
                                        next_per_uav_states[i], done)

            step_counter += 1
            if step_counter % qcfg.train_every_n_steps == 0:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)

            obs = next_obs

        agent.decay_epsilon()
        mean_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
        logger.info(
            "Episode %d/%d: reward=%.2f coverage=%.3f epsilon=%.3f mean_loss=%.4f",
            episode + 1, qcfg.n_episodes, episode_reward, info["coverage_fraction"],
            agent.epsilon, mean_loss,
        )

    env.close()

    model_path = os.path.join(qcfg.model_dir, "qlearning_final.pt")
    torch.save(agent.network.state_dict(), model_path)
    logger.info("Q-Learning training complete. Model saved to %s", model_path)
    return model_path


def evaluate_qlearning(model_path: str, config: ExperimentConfig, vertices: List[Tuple[float, float]],
                        n_episodes: int = 10, seed: int = 999) -> Tuple[List[float], List[float]]:
    """
    Loads a trained Q-network and runs `n_episodes` greedy (epsilon=0)
    evaluation episodes, logging full Goal-3-style episode statistics --
    same CSV columns `ppo/evaluate.py` produces, for direct comparison.
    """
    qcfg = config.qlearning
    n_uavs = config.env.n_uavs

    episode_logger = EpisodeLogger("results/episode_logs", experiment_name="qlearning_eval")
    env: SwarmFarmEnv = make_env(vertices, config, episode_logger=episode_logger)()

    obs, info = env.reset(seed=seed)
    input_dim = flatten_observation_for_uav(obs, 0, n_uavs).shape[0]
    network = QNetwork(input_dim, qcfg.hidden_units, qcfg.use_paper_faithful_first_layer)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    episode_rewards, episode_coverages = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            with torch.no_grad():
                actions = []
                for i in range(n_uavs):
                    state = flatten_observation_for_uav(obs, i, n_uavs)
                    q_values = network(torch.from_numpy(state).unsqueeze(0))
                    actions.append(int(torch.argmax(q_values, dim=1).item()))
            obs, reward, terminated, truncated, info = env.step(np.array(actions))
            total_reward += reward
        episode_rewards.append(total_reward)
        episode_coverages.append(info["coverage_fraction"])
        logger.info("Eval episode %d: reward=%.2f coverage=%.3f", ep, total_reward, info["coverage_fraction"])

    env.close()
    return episode_rewards, episode_coverages


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the classical Q-Learning baseline on SwarmFarmEnv.")
    parser.add_argument("--episodes", type=int, default=None, help="Override QLearningConfig.n_episodes.")
    args = parser.parse_args()

    from ppo.train import FIELD_VERTICES  # shared field definition (see make_env note in GOAL4_MODULE_GUIDE.md)

    qcfg_kwargs = {}
    if args.episodes is not None:
        qcfg_kwargs["n_episodes"] = args.episodes
    exp_config = ExperimentConfig(seed=42, qlearning=QLearningConfig(**qcfg_kwargs))
    exp_config.to_json("results/qlearning_experiment_config.json")
    train_qlearning(exp_config, FIELD_VERTICES)
