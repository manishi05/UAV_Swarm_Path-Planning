"""
baselines/astar.py
====================
A* coverage-planning baseline. Not present in the reference paper at all
(it only evaluates Q-Learning) -- included per the task's explicit
"Baseline Methods: A*, Classical Q-Learning" requirement, as the natural
classical (non-learning) path-planning comparison point for a learned
policy.

A* itself is a single-target shortest-path algorithm, not a coverage
algorithm -- it has no native notion of "visit every cell." The standard
way to turn it into a coverage baseline (and the one used here) is
**greedy nearest-frontier replanning**: at each decision point, find the
nearest not-yet-visited flyable cell (the "frontier"), run A* to compute
the shortest obstacle-avoiding path to it, and take one step along that
path. Replan whenever the target is reached, the path is blocked by a
newly-arrived dynamic obstacle, or (by default, `AStarConfig.replan_every_step`)
every single step -- necessary because tractors/humans/animals move every
timestep, so a path computed even one step ago can already be stale.

Design decision -- privileged state access: unlike PPO and the Q-Learning
baseline, which must act purely from `SwarmFarmEnv`'s `observation_space`
(the Dict of grids/vectors a real onboard policy would have), this
controller reads the environment's full internal state directly via
`env.get_render_state()` (exact continuous obstacle positions, exact grid cell
objects, etc.). This is standard practice for classical planning
baselines -- A* is not a *learned* perception-to-action policy, it's a
planner that is assumed to have accurate localization/mapping already
solved, which is a reasonable (and typically necessary) assumption for
this class of algorithm. This is stated explicitly here so it's not
mistaken for an unfair advantage silently smuggled into the comparison.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from configs.config import AStarConfig, ExperimentConfig
from environment.drone import ActionType
from environment.grid_map import GridMapGenerator
from environment.swarm_env import SwarmFarmEnv, make_env
from utils.episode_logger import EpisodeLogger
from utils.logger import get_logger

logger = get_logger("uav_swarm_ppo.baselines.astar", log_file="results/astar.log")

Coord = Tuple[int, int]  # (row, col)

# Grid-index deltas for the 4 movement actions, matching drone.py's
# world-space convention: row increases with +y (UP), col increases
# with +x (RIGHT) -- see grid_map.py's axis-aligned cell indexing.
_ACTION_DELTAS: Dict[ActionType, Coord] = {
    ActionType.STAY: (0, 0),
    ActionType.UP: (1, 0),
    ActionType.DOWN: (-1, 0),
    ActionType.LEFT: (0, -1),
    ActionType.RIGHT: (0, 1),
}
_DELTA_TO_ACTION: Dict[Coord, ActionType] = {v: k for k, v in _ACTION_DELTAS.items()}


def _manhattan(a: Coord, b: Coord) -> int:
    """Admissible heuristic for 4-connected unit-cost grid movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(start: Coord, goal: Coord, blocked: Set[Coord],
                   grid_shape: Tuple[int, int]) -> Optional[List[Coord]]:
    """
    Standard A* search over the 4-connected grid graph.

    Parameters
    ----------
    start, goal : Coord
        (row, col) start and target cells.
    blocked : Set[Coord]
        Cells that cannot be entered (non-flyable, or currently occupied
        by a static/dynamic obstacle -- see `_compute_blocked_cells`).
    grid_shape : Tuple[int, int]
        (n_rows, n_cols), for bounds checking.

    Returns
    -------
    Optional[List[Coord]]
        The path from `start` to `goal` inclusive, or None if
        unreachable given the current `blocked` set.
    """
    n_rows, n_cols = grid_shape
    if goal in blocked:
        return None

    open_heap: List[Tuple[int, Coord]] = [(_manhattan(start, goal), start)]
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, int] = {start: 0}
    visited: Set[Coord] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        r, c = current
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (r + dr, c + dc)
            if not (0 <= neighbor[0] < n_rows and 0 <= neighbor[1] < n_cols):
                continue
            if neighbor in blocked or neighbor in visited:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None  # no path exists (goal unreachable given current obstacles)


def nearest_unvisited_cell(start: Coord, visited_mask: np.ndarray, blocked: Set[Coord],
                            grid_shape: Tuple[int, int]) -> Optional[Coord]:
    """
    Breadth-first search from `start` over the (currently) unblocked
    4-connected grid graph, returning the first not-yet-visited cell
    encountered. BFS explores in strictly increasing hop-distance order,
    so the first unvisited cell found is guaranteed nearest by hop
    count -- exactly the "frontier" target A* then plans a path to.

    Returns None if every reachable cell has already been visited (or
    nothing is reachable at all from `start`).
    """
    n_rows, n_cols = grid_shape
    queue: deque = deque([start])
    seen: Set[Coord] = {start}

    while queue:
        current = queue.popleft()
        r, c = current
        if 0 <= r < n_rows and 0 <= c < n_cols and not visited_mask[r, c] and current != start:
            return current
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (r + dr, c + dc)
            if (0 <= neighbor[0] < n_rows and 0 <= neighbor[1] < n_cols
                    and neighbor not in blocked and neighbor not in seen):
                seen.add(neighbor)
                queue.append(neighbor)
    return None


def _compute_blocked_cells(grid: GridMapGenerator, dynamic_obstacles, buffer_m: float) -> Set[Coord]:
    """
    Builds the set of currently-blocked grid cells: every cell that is
    not flyable (outside the field boundary or occupied by a static
    obstacle -- Goal 1/2 concepts, static across an episode), plus every
    cell within `buffer_m` of a dynamic obstacle's *current* position
    (recomputed fresh every call, since tractors/humans/animals move
    every step -- this is what makes A* here a replanning baseline, not
    a one-shot planner).
    """
    n_rows, n_cols = grid.shape
    blocked: Set[Coord] = set()
    for cell in grid.cells:
        if not cell.is_flyable:
            blocked.add((cell.row, cell.col))

    cell_size = grid.cell_size_m
    buffer_cells = max(1, int(np.ceil(buffer_m / cell_size)))
    for obstacle in dynamic_obstacles:
        center_cell = grid.get_cell_at_point(obstacle.x, obstacle.y)
        if center_cell is None:
            continue
        for dr in range(-buffer_cells, buffer_cells + 1):
            for dc in range(-buffer_cells, buffer_cells + 1):
                r, c = center_cell.row + dr, center_cell.col + dc
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    blocked.add((r, c))
    return blocked


class AStarSwarmController:
    """
    Decentralized greedy coverage controller: each UAV independently
    targets its own nearest unvisited cell (excluding cells already
    claimed by a teammate this decision round, to reduce trivially
    redundant targeting) and follows an A*-planned path to it, replanning
    as configured by `AStarConfig`.

    "Decentralized" here means each UAV's target/path decision doesn't
    depend on a jointly-optimized swarm-wide assignment (which would be
    a much more expensive combinatorial assignment problem) -- it's a
    reasonable, standard simplification for a classical baseline, and
    intentionally not the more sophisticated method a full coverage-path-
    planning paper might use, since the point of this baseline is to be
    a credible non-learning reference point, not to win.
    """

    def __init__(self, config: AStarConfig, n_uavs: int):
        self._config = config
        self._n_uavs = n_uavs
        self._paths: List[Optional[List[Coord]]] = [None] * n_uavs
        self._path_indices: List[int] = [0] * n_uavs

    def reset(self) -> None:
        self._paths = [None] * self._n_uavs
        self._path_indices = [0] * self._n_uavs

    def decide_actions(self, env: SwarmFarmEnv) -> np.ndarray:
        """
        Computes one discrete action per UAV for the current environment
        state, called once per `env.step()`.
        """
        state = env.get_render_state()
        grid: GridMapGenerator = state["grid"]
        uavs = state["uavs"]
        visited_mask = state["visited"]
        dynamic_obstacles = state["dynamic_obstacles"]

        blocked = _compute_blocked_cells(grid, dynamic_obstacles, self._config.dynamic_obstacle_buffer_m)
        claimed_targets: Set[Coord] = set()
        actions = np.zeros(self._n_uavs, dtype=int)

        for i, uav in enumerate(uavs):
            if not uav.alive:
                actions[i] = int(ActionType.STAY)
                continue

            current_cell = grid.get_cell_at_point(uav.position[0], uav.position[1])
            if current_cell is None:
                actions[i] = int(ActionType.STAY)
                continue
            current_coord = (current_cell.row, current_cell.col)

            needs_replan = (
                self._config.replan_every_step
                or self._paths[i] is None
                or self._path_indices[i] >= len(self._paths[i]) - 1
                or self._paths[i][self._path_indices[i] + 1] in blocked
            )

            if needs_replan:
                target = nearest_unvisited_cell(
                    current_coord, visited_mask,
                    blocked | claimed_targets, grid.shape,
                )
                if target is None:
                    self._paths[i] = None
                    actions[i] = int(ActionType.STAY)
                    continue
                path = a_star_search(current_coord, target, blocked, grid.shape)
                self._paths[i] = path
                self._path_indices[i] = 0
                if path is not None:
                    claimed_targets.add(target)

            path = self._paths[i]
            if path is None or len(path) < 2:
                actions[i] = int(ActionType.STAY)
                continue

            idx = self._path_indices[i]
            current_step = path[idx]
            next_step = path[idx + 1]
            delta = (next_step[0] - current_step[0], next_step[1] - current_step[1])
            actions[i] = int(_DELTA_TO_ACTION.get(delta, ActionType.STAY))
            self._path_indices[i] += 1

        return actions


def run_astar(config: ExperimentConfig, vertices: List[Tuple[float, float]],
              n_episodes: Optional[int] = None, seed: int = 999) -> Tuple[List[float], List[float]]:
    """
    Runs the A* baseline for `n_episodes` (default: `AStarConfig.n_episodes`)
    episodes against `SwarmFarmEnv`, logging full Goal-3-style episode
    statistics via `EpisodeLogger` -- same CSV columns as
    `ppo/evaluate.py` and `baselines/qlearning.py::evaluate_qlearning`,
    for direct comparison.

    A* has no training phase (it's a planner, not a learner), so this is
    the only entry point for this baseline -- there is no separate
    `train_astar`.

    Returns
    -------
    Tuple[List[float], List[float]]
        Per-episode (total_reward, coverage_fraction).
    """
    acfg = config.astar
    n_episodes = n_episodes if n_episodes is not None else acfg.n_episodes
    n_uavs = config.env.n_uavs

    episode_logger = EpisodeLogger("results/episode_logs", experiment_name="astar_eval")
    env: SwarmFarmEnv = make_env(vertices, config, episode_logger=episode_logger)()
    controller = AStarSwarmController(acfg, n_uavs)

    episode_rewards, episode_coverages = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        controller.reset()
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            actions = controller.decide_actions(env)
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_coverages.append(info["coverage_fraction"])
        logger.info("Episode %d/%d: reward=%.2f coverage=%.3f collisions=%s",
                    ep + 1, n_episodes, total_reward, info["coverage_fraction"], info["collision_counts"])

    env.close()
    logger.info(
        "A* baseline over %d episodes: reward mean=%.2f | coverage mean=%.3f",
        n_episodes, float(np.mean(episode_rewards)), float(np.mean(episode_coverages)),
    )
    return episode_rewards, episode_coverages


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the A* coverage-planning baseline on SwarmFarmEnv.")
    parser.add_argument("--episodes", type=int, default=None, help="Override AStarConfig.n_episodes.")
    args = parser.parse_args()

    from ppo.train import FIELD_VERTICES

    exp_config = ExperimentConfig(seed=42)
    exp_config.to_json("results/astar_experiment_config.json")
    run_astar(exp_config, FIELD_VERTICES, n_episodes=args.episodes)
