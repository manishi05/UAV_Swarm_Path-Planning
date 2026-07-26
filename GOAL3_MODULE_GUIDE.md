# Goal 3 Module Guide — Gymnasium Environment + Reward Engineering

This is a companion to `MODULE_GUIDE.md` (Goal 1/2). It only covers what
**changed or was added** for Goal 3. Nothing in `polygon_field.py`,
`grid_map.py`, `cell.py`, `obstacle.py`, `wind.py`, `drone.py`, or
`collision.py` was touched — Goal 3 builds strictly on top of them.

**Same architecture, same folder structure** — no new top-level folders.
Everything new lives inside the existing `environment/`, `utils/`, and
`visualization/` packages, plus one updated file in `experiments/`.

---

## What changed, file by file

| File | Status | What changed |
|---|---|---|
| `configs/config.py` | Modified | `RewardConfig` expanded from 8 to 20 fields, covering all 8 Goal-3 categories |
| `environment/reward.py` | Rewritten | Returns `RewardBreakdown` (8 named components) instead of a bare float |
| `environment/swarm_env.py` | Modified | New reward-signal computation, episode logger hooks, formal `render()`/`close()` |
| `utils/episode_logger.py` | **New** | The Goal 3 logging framework |
| `visualization/renderer.py` | Modified | Split into `render_frame()` (disk/interactive) and `render_to_array()` (Gymnasium `rgb_array`) |
| `experiments/run_experiment.py` | Modified | Wired up `EpisodeLogger`, uses `get_render_state()`, calls `env.close()` |

---

## 1. `configs/config.py` — `RewardConfig` (edit here to retune anything)

Grouped by the 8 Goal-3 categories. If you want to change how strongly any
behavior is rewarded/punished, **this is the only file you should need to
touch** — nothing else has hard-coded reward numbers.

```python
# Coverage (reference paper's exact Eq. 2 constants — leave alone unless
# you want to deviate from the paper's baseline on purpose)
new_cell_base, visited_cell, non_visitable

# Exploration (our extension — dense local-novelty shaping)
exploration_coeff, exploration_window_radius

# Collision avoidance (obstacle collisions — discrete + cell-level precursor)
collision_static_obstacle, collision_dynamic_obstacle, blocked_cell_penalty

# Boundary violations
out_of_bounds

# Dynamic obstacle avoidance (dense proximity shaping)
dynamic_obstacle_proximity_coeff, dynamic_obstacle_warning_radius_m

# UAV separation (discrete + dense proximity shaping)
collision_uav_uav, uav_separation_proximity_coeff, uav_separation_warning_radius_m

# Mission completion
mission_completion_bonus

# Energy efficiency
energy_efficiency_coeff, wind_work_coeff
```

See `REWARD_ENGINEERING.md` for the full rationale behind every default
value and concrete guidance on how to retune them.

---

## 2. `environment/reward.py` — the reward engine, rewritten

**`RewardBreakdown`** (new dataclass): one UAV's (or, summed, the swarm's)
reward for one step, split into 8 named fields (`exploration`, `coverage`,
`collision_avoidance`, `boundary`, `dynamic_obstacle_avoidance`,
`uav_separation`, `mission_completion`, `energy_efficiency`), plus a
`.total` property (sum of all 8) and `.as_dict()` (for logging).
`__add__`/`__radd__` are implemented so you can do
`sum(list_of_breakdowns, RewardBreakdown())` to combine per-UAV breakdowns
into a swarm-level one — this is exactly what `swarm_env.py` does.

**`RewardCalculator.compute()`** — signature grew from the Goal-2 version.
New parameters and what drives them:

| Parameter | Source | Feeds category |
|---|---|---|
| `entered_boundary_violation` | cell doesn't exist or `is_visitable=False` | `boundary` |
| `entered_obstacle_zone` | cell is visitable but `blocked_static=True` | `collision_avoidance` |
| `local_novelty` | `swarm_env._compute_local_novelty()` | `exploration` |
| `nearest_dynamic_obstacle_dist` | min distance to any dynamic obstacle | `dynamic_obstacle_avoidance` |
| `nearest_uav_dist` | min distance to any other alive UAV | `uav_separation` (dense part) |
| `dt` | `EnvConfig.dt_s` | `energy_efficiency` (baseline flight cost) |

**If you want to add a 9th reward category**: add a field to
`RewardBreakdown`, add the corresponding weight(s) to `RewardConfig`, add
the computation inside `RewardCalculator.compute()`, and pass whatever new
signal it needs from `swarm_env.py::step()`. That's the complete pattern —
follow it exactly for consistency.

**Mission completion is deliberately NOT computed inside `compute()`** —
it's a swarm-level, one-time bonus, not a per-UAV outcome. It's added
directly in `swarm_env.py::step()` after summing the per-UAV breakdowns.
If you move to a different termination condition, look there, not here.

---

## 3. `environment/swarm_env.py` — what actually changed

### `__init__()`
- New params: `episode_logger: Optional[EpisodeLogger] = None`,
  `render_mode: Optional[str] = "rgb_array"`.
- `self._mission_bonus_awarded` flag added (per-episode, reset in `reset()`).

### `reset()`
- One new line: `if self._episode_logger is not None:
  self._episode_logger.start_episode(episode_seed)`.

### `step()` — the main change
Order of operations (unchanged from Goal 2) with new work inserted at step 5:
1. Advance wind + dynamic obstacles (unchanged).
2. Compute wake interference from pre-motion positions (unchanged).
3. Integrate UAV physics + clamp to bounds (unchanged).
4. Run collision detection (unchanged).
5. **New**: for each alive UAV, compute `local_novelty` (via
   `_compute_local_novelty()`), `nearest_dyn_dist`, `nearest_uav_dist`,
   determine the 4-way cell classification (`entered_new` /
   `entered_visited` / `entered_boundary` / `entered_obstacle_zone` —
   this replaced the old 3-way `entered_non_visitable` flag), call
   `RewardCalculator.compute()`, collect all per-UAV `RewardBreakdown`s.
6. **New**: `swarm_breakdown = sum(breakdowns, RewardBreakdown())`.
7. **New**: if `full_coverage` just became true and the bonus hasn't been
   awarded yet this episode, add `mission_completion_bonus` directly to
   `swarm_breakdown.mission_completion`.
8. **New**: `info["reward_breakdown"] = swarm_breakdown.as_dict()`.
9. **New**: if an `episode_logger` is set, call `log_step()` every step,
   and `end_episode()` on the step that terminates/truncates.
10. Returns `swarm_breakdown.total` as the scalar reward (was `total_reward`
    float accumulator before — same numeric result, now derived from the
    structured breakdown instead of accumulated ad hoc).

### `_compute_local_novelty()` — new private method
Looks at a square window (radius = `RewardConfig.exploration_window_radius`
cells) around the UAV's current cell in the visited mask, restricted to
flyable cells only, and returns `1 - (visited flyable cells in window /
total flyable cells in window)`. This is what makes "Exploration" a
genuinely distinct signal from "Coverage" — Coverage only cares about the
UAV's *exact* current cell; Exploration cares about the *neighborhood*,
which is what counters the paper's own documented "got stuck exploring one
edge" failure mode (Fig. 10 in the reference paper).

### `render()` — rewritten to be Gymnasium-standard
Previously `render()` returned the raw state dict. Now it follows the
actual Gymnasium API: with `render_mode="rgb_array"` it returns a numpy
`(H, W, 3)` uint8 image (via the renderer's new `render_to_array()`); with
`"human"` it opens an interactive window. **The old behavior (raw state
dict) is preserved under a new name: `get_render_state()`.** If any of your
own code called `env.render()` expecting a dict, change it to
`env.get_render_state()`.

### `close()` — new, was missing before
Closes any open matplotlib figures and calls `episode_logger.close()` if
one was provided. Always call this at the end of a script/training run.

---

## 4. `utils/episode_logger.py` — the logging framework (new file)

`EpisodeLogger(log_dir, experiment_name, verbose_step_logging=False)`.

- **`start_episode(seed)`** — call once per `env.reset()`. `SwarmFarmEnv`
  does this automatically if you pass `episode_logger=` to its constructor.
- **`log_step(reward_breakdown, info)`** — call once per `env.step()`.
  Also automatic if wired into the env.
- **`end_episode()`** — call once when an episode terminates/truncates.
  Also automatic. Writes one row to `<experiment_name>_episodes.csv`
  (columns: episode, seed, steps, total_reward, coverage_fraction,
  valid_action_fraction, one `reward_<category>` column per Goal-3
  category, one `collisions_<type>` column per collision type) and one
  human-readable summary line to `<experiment_name>.log`.
- **`close()`** — call at the end of your script (or let `env.close()` do
  it for you). Warns (doesn't crash) if called mid-episode.

**If `verbose_step_logging=True`**, every individual step's full reward
breakdown is also written to `<experiment_name>_episode<N>_steps.csv` — one
file per episode. Useful for debugging a single episode's reward dynamics
in detail; leave `False` for large multi-episode/multi-seed PPO training
runs, or you'll get thousands of small files.

**CSV columns are fixed up front** (not discovered dynamically per
episode) specifically so appending a row is O(1) — this matters once PPO
training (Day 3) produces thousands of episodes; an O(n²) rewrite-the-whole-file
strategy would become a real bottleneck. If you add a 9th reward category
or a new collision type, you must also add its column name to
`_REWARD_CATEGORY_COLUMNS` / `_COLLISION_COLUMNS` at the top of the file,
or its value will be silently dropped from the CSV (not crash — see
`extrasaction="ignore"` in `_write_summary_row`).

---

## 5. `visualization/renderer.py` — split into two output modes

The old single `render_frame()` method is now backed by a shared
`_build_figure()` helper, with two public entry points:
- **`render_frame(state, save_path, title, show)`** — unchanged behavior,
  saves to disk and/or shows interactively. Used by
  `experiments/run_experiment.py` for the detailed titled figures.
- **`render_to_array(state, title)`** (new) — returns an `(H, W, 3)` uint8
  numpy array, no disk I/O. This is what `SwarmFarmEnv.render()` calls
  under `render_mode="rgb_array"`.

If you want to change what the rendered figure looks like (colors, marker
styles, legend, etc.), edit `_build_figure()` — both output paths will
pick up the change automatically, since they can't visually drift apart
from each other.

---

## 6. `experiments/run_experiment.py` — what to change if you rerun it

- Now constructs an `EpisodeLogger` and passes it into `SwarmFarmEnv(...)`.
- Runs for `config.env.max_episode_steps` (not a fixed 120) so the episode
  actually terminates/truncates and you can see `end_episode()` fire and
  the CSV get written — if you shorten this again for a quick test, you
  just won't see a CSV row (the logger will warn on `close()`, not crash).
- Calls `env.get_render_state()` (not `env.render()`) for the detailed
  saved figures, and separately calls `env.render()` once to demonstrate
  the Gymnasium-compliant array path.
- Ends with `env.close()`.

---

## Quick sanity check after any reward/logging change

```bash
cd uav_swarm_project
python -m experiments.run_experiment
cat results/smoke_test_episodes.csv    # one row, all 8 reward_* columns populated
```
If a category is always 0.0 in that row when you expected it to fire (e.g.
you added dynamic obstacles but `reward_dynamic_obstacle_avoidance` is
always 0), check `ObstacleConfig` counts first — it's usually that no
obstacles of that type were configured/placed, not a bug in the reward code.
