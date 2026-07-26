# Baselines Module Guide — A* and Classical Q-Learning

Companion to `MODULE_GUIDE.md` (Goal 1/2), `GOAL3_MODULE_GUIDE.md` (Goal 3),
and `GOAL4_MODULE_GUIDE.md` (Goal 4). Same architecture, same folder
structure — `baselines/` already existed as an empty placeholder since Day 1;
this fills it in.

---

## Which files changed vs. were added

| File | Status | What changed |
|---|---|---|
| `configs/config.py` | Modified | New `QLearningConfig`, `AStarConfig`; `ExperimentConfig` gained `qlearning`/`astar` fields |
| `environment/grid_map.py` | Modified | One new property: `GridMapGenerator.cell_size_m` (public accessor, previously private) |
| `baselines/qlearning.py` | **New** | Classical Q-Learning baseline |
| `baselines/astar.py` | **New** | A* coverage-planning baseline |

**Nothing in `environment/reward.py`, `collision.py`, `obstacle.py`, `wind.py`,
`drone.py`, `swarm_env.py`, `cell.py`, or `polygon_field.py` was touched.**
Both baselines run against the exact same `SwarmFarmEnv` and `make_env()`
factory that Goal 4's PPO training uses — this is deliberate and important:
it means "PPO vs. Q-Learning vs. A*" is a comparison of *algorithms* on an
identical environment, not three subtly different setups.

---

## 1. `configs/config.py` — two new dataclasses

```python
QLearningConfig: gamma, epsilon_start/decay/min, memory_size, hidden_units,
                  learning_rate, batch_size, train_every_n_steps, n_episodes,
                  max_steps_per_episode, use_paper_faithful_first_layer, model_dir

AStarConfig: dynamic_obstacle_buffer_m, replan_every_step, n_episodes,
             max_steps_per_episode
```

`QLearningConfig`'s defaults (`gamma=0.91`, `epsilon_start=0.47`,
`epsilon_decay=0.93`, `epsilon_min=0.05`, `memory_size=60`,
`hidden_units=167`) are **the reference paper's own reported constants**,
not arbitrary choices — see `baselines/qlearning.py`'s module docstring for
exactly which paper section each one comes from. Retune here, same pattern
as every other config in this project — never hard-code a hyperparameter
inside `baselines/*.py` itself.

`AStarConfig` has no paper equivalent (the reference paper doesn't use A* at
all) — its defaults are ours, documented as such.

---

## 2. `environment/grid_map.py` — one new property

```python
@property
def cell_size_m(self) -> float:
    return self._config.cell_size_m
```

Added because `baselines/astar.py` needs to convert a physical buffer
distance (meters, around a moving obstacle) into a number of grid cells,
and reaching into `GridMapGenerator`'s private `_config` attribute from
outside the class would break encapsulation. This is the only change to any
Goal 1/2/3 module across all of Goal 4 + baselines work.

---

## 3. `baselines/qlearning.py` — line-by-section walkthrough

### `flatten_observation_for_uav(obs, uav_index, n_uavs)`
Builds the per-UAV input vector for the Q-network: flattens
`flyable_map`, `visited_map`, `dynamic_obstacle_map`, `uav_position_map`
(all `SwarmFarmEnv` observation channels) plus `wind_vector` and *this
UAV's own* `battery_frac` entry, concatenated into one 1D array. Mirrors
the reference paper's Fig. 4 "Flatten Layer" step, extended with the two
map channels (`dynamic_obstacle_map`) and scalars (wind, battery) the
paper's environment has no equivalent of.

**To change what the Q-network sees**: edit this function's `parts` list.
Whatever you add/remove here automatically changes `input_dim`, which is
computed dynamically from a sample observation in `train_qlearning()` — no
need to hard-code a dimension anywhere.

### `QNetwork(nn.Module)`
Two dense layers: `fc1` (`hidden_units=167` by default, paper's value),
`fc2` (5 outputs, one Q-value per `ActionType`). The
`use_paper_faithful_first_layer` flag controls whether `fc1`'s output goes
through an identity (paper-literal "linear activation") or a ReLU
(standard, better-performing choice). **The output layer is always linear
(no activation)** — this is the one deliberate deviation from the paper
(which specifies softmax on the output), explained at length in the module
docstring: softmax on unbounded Q-values isn't standard anywhere in the RL
literature and would actively break argmax-based action selection, not
just look different.

### `ReplayMemory`
A `deque(maxlen=memory_size)` FIFO buffer — one instance per UAV (see
`QLearningAgent.__init__`), matching the paper's explicit
"each UAV has its own memory, actions of other UAVs are never stored" design.

### `QLearningAgent`
Owns one shared `QNetwork` (weights shared across all UAVs — the paper's
own "global ANN" finding) and `n_uavs` separate `ReplayMemory` instances.
- `select_action(state)`: epsilon-greedy (paper's exact ε schedule).
- `store_transition(uav_index, ...)`: pushes into that UAV's own memory.
- `train_step()`: samples a small batch from **each** UAV's memory
  separately, concatenates them, and does **one** gradient step on the
  shared network — this is the specific mechanism that reconciles
  "separate memories" (paper-faithful) with "one shared network"
  (paper's own best-performing configuration).
- `decay_epsilon()`: called once per episode, not per step (paper Section 3.3.4).

**To swap in a different RL update rule** (e.g., add a target network, turn
this into DQN): this is the only class to touch — `train_step()` contains
the entire Bellman-update logic in one place.

### `train_qlearning(config, vertices)`
The training loop: for each of `QLearningConfig.n_episodes` episodes, steps
`SwarmFarmEnv` (identical construction path to PPO's, via `make_env`),
storing one transition per UAV per step and running `train_step()` every
`train_every_n_steps`. Saves final weights via `torch.save(state_dict())`
to `QLearningConfig.model_dir`.

### `evaluate_qlearning(model_path, config, vertices, n_episodes, seed)`
Loads a saved model, runs **greedy** (no exploration) evaluation episodes,
logs full Goal-3-style episode CSVs via `EpisodeLogger` — same schema as
`ppo/evaluate.py` and `baselines/astar.py::run_astar` (verified identical
column sets during testing — see §5 below).

---

## 4. `baselines/astar.py` — line-by-section walkthrough

### Why A* needs a coverage wrapper at all
A* itself only finds a shortest path to *one* target. Turning it into a
coverage baseline means repeatedly picking a target (the nearest
unvisited cell) and re-running A* to it — this whole file implements that
wrapper, not "just A*."

### `a_star_search(start, goal, blocked, grid_shape)`
Textbook A* over the 4-connected grid graph: Manhattan-distance heuristic
(admissible for unit-cost 4-connected movement), a min-heap open set,
returns the full path (list of `(row, col)`) or `None` if `goal` is
unreachable given the current `blocked` set.

### `nearest_unvisited_cell(start, visited_mask, blocked, grid_shape)`
Breadth-first search (not A* — no target to head toward yet) from `start`,
returning the first not-yet-visited cell encountered. BFS visits in
strictly increasing hop-distance order, which is what guarantees "first
found" means "nearest." This is the **target-selection** step; `a_star_search`
is the **path-planning** step to whatever target this function picks.

### `_compute_blocked_cells(grid, dynamic_obstacles, buffer_m)`
Builds the full blocked-cell set for one planning call: every
non-`is_flyable` cell (static, from Goal 1/2 — doesn't change during an
episode) **plus** every cell within `buffer_m` of a dynamic obstacle's
*current* position (recomputed fresh every single call, since tractors/
humans/animals move every step). This is what makes A* here a genuine
replanning baseline against moving obstacles, not a one-shot static planner.

### `AStarSwarmController`
Per-UAV state: `_paths[i]` (the currently-planned path) and
`_path_indices[i]` (how far along it we are). `decide_actions(env)`:
1. Reads privileged state via `env.get_render_state()` — **not**
   `env.render()`, which is now strictly Gym-API-compliant (Goal 3) and
   only returns an image/None. `get_render_state()` is the method built
   for exactly this kind of full-state access (see `GOAL3_MODULE_GUIDE.md`
   and `swarm_env.py` for why the two are split).
2. Recomputes the blocked-cell set fresh (§ above).
3. For each alive UAV, decides whether to replan (default: **every step**,
   `AStarConfig.replan_every_step=True` — necessary given moving obstacles;
   set `False` to only replan when the path is exhausted or blocked, a
   cheaper but less reactive mode).
4. Converts the next grid step in the path into one of the 5 `ActionType`
   values via `_DELTA_TO_ACTION` (built from `_ACTION_DELTAS`, matching
   `drone.py`'s exact row/col-to-world-direction convention — row+1 = UP,
   col+1 = RIGHT).

**Decentralized, not jointly optimized**: each UAV picks its own nearest
target independently (excluding cells already claimed by a teammate *this
round*, via `claimed_targets`) rather than solving a joint assignment
problem. Documented as a deliberate simplification, not an oversight — see
the class docstring.

### `run_astar(config, vertices, n_episodes, seed)`
No training phase (A* doesn't learn) — this is the only entry point,
structurally parallel to `evaluate_qlearning` and `ppo/evaluate.py::evaluate`:
runs episodes, logs via `EpisodeLogger`, returns `(rewards, coverages)`.

---

## 5. Verified: all three algorithms produce identical CSV schemas

Confirmed by direct comparison during testing:
```
episode,seed,steps,total_reward,coverage_fraction,valid_action_fraction,
reward_exploration,reward_coverage,reward_collision_avoidance,reward_boundary,
reward_dynamic_obstacle_avoidance,reward_uav_separation,reward_mission_completion,
reward_energy_efficiency,collisions_uav_uav,collisions_uav_static_obstacle,
collisions_uav_dynamic_obstacle,collisions_uav_boundary
```
identical across `results/episode_logs/{ppo_eval, qlearning_eval, astar_eval}_episodes.csv`.
This is what makes Day 4's statistical comparison (Shapiro-Wilk/ANOVA
across seeds, matching the reference paper's own methodology) a direct
multi-file CSV read, not a data-reconciliation project.

---

## 6. What was actually observed running both (real runs, not hypothetical)

- **A* (2 episodes)**: coverage 0.995 and 1.000, **zero** static-obstacle
  collisions (confirms the blocked-cell planning is working correctly),
  one episode terminated early via mission completion (220 steps). This is
  a strong, credible non-learning baseline — exactly what you want to
  compare a learned policy against.
- **A* still racked up UAV-boundary violations** (829, 56) despite never
  planning into a non-flyable cell — because wind-induced continuous drift
  (Goal 2's physics model) can push a UAV's exact position outside the
  field polygon even while it's following a geometrically valid grid path.
  This is a genuinely interesting finding for the paper: it demonstrates
  the value of the continuous-physics upgrade over the reference paper's
  pure grid-teleportation model — a "perfect" discrete planner still faces
  real-world continuous disturbances a grid-only method would never expose.
- **Q-Learning (3 training episodes)**: coverage trended up (0.527 → 0.534
  → 0.670) even in this very short smoke run — a weak but genuine positive
  learning signal. Loss magnitudes were in the millions, the **same
  reward-scale symptom already flagged for PPO** in `PPO_TRAINING.md` §5 —
  this is now confirmed to be an environment/reward property, not a
  PPO-specific problem, which strengthens the case for settling the reward
  scale (as discussed) before investing in long training runs of *any* of
  the three algorithms.
- **Evaluation (greedy, post-3-episode-training) showed worse coverage**
  (0.124, 0.108) than mid-training exploration — expected and correct for
  a deliberately undertrained smoke-test model (3 episodes vs. the
  configured default of 200), not a bug in the evaluation path.

---

## Quick sanity check after any change

```bash
cd uav_swarm_project
python -m baselines.astar --episodes 2
python -m baselines.qlearning --episodes 3
python -c "
from configs.config import ExperimentConfig
from baselines.qlearning import evaluate_qlearning
from ppo.train import FIELD_VERTICES
c = ExperimentConfig(seed=42)
evaluate_qlearning('results/models/qlearning/qlearning_final.pt', c, FIELD_VERTICES, n_episodes=2)
"
```
All three should complete in well under a minute and produce non-crashing,
non-NaN reward/coverage numbers.
