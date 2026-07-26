# UAV Swarm PPO — Module Guide

This document walks through every file in the project: what it does, why it's
structured that way, and — most importantly — **where to make changes** if you
want to tune something. Read this alongside the code; it is organized in the
same dependency order the code is built in (configs → geometry → obstacles →
physics → integration → visualization).

---

## 0. Gap analysis: what Goal 1 / Goal 2 required vs. what's implemented

| Requirement | Status | Where |
|---|---|---|
| Read polygon vertices | ✅ | `environment/polygon_field.py::FarmFieldPolygon` |
| Minimum Bounding Rectangle | ✅ (axis-aligned + rotated) | `environment/polygon_field.py::MinimumBoundingRectangle` |
| Grid generation | ✅ | `environment/grid_map.py::GridMapGenerator` |
| Cell classification (visitable/non-visitable) | ✅ | `grid_map.py::_label_cell` |
| Visualization | ✅ | `visualization/renderer.py` (full env), earlier `FieldVisualizer` (Goal-1 only, superseded) |
| Seeds / config / logging | ✅ | `configs/config.py`, `utils/seed.py`, `utils/logger.py` |
| Static obstacles (trees, poles, buildings, irrigation) | ✅ | `environment/obstacle.py::StaticObstacleGenerator` |
| Dynamic obstacles (tractors, humans, animals) | ✅ | `environment/obstacle.py::DynamicObstacleGenerator/Manager` |
| Wind disturbance model | ✅ (Ornstein-Uhlenbeck) | `environment/wind.py::WindField` |
| Inter-UAV air/wake interference | ✅ | `environment/wind.py::compute_wake_interference` |
| UAV physics (position, velocity, boundary) | ✅ | `environment/drone.py::UAV` |
| Collision detection (UAV-UAV, UAV-obstacle, UAV-boundary) | ✅ | `environment/collision.py::CollisionEngine` |
| Reward tuned for collision avoidance | ✅ | `environment/reward.py::RewardCalculator` |
| Gymnasium environment | ✅ | `environment/swarm_env.py::SwarmFarmEnv` |

**Not yet built (by design — Day 3/4 of the roadmap, not part of Goal 1/2):**
PPO training script (`ppo/train.py`), Q-Learning/A* baselines (`baselines/`),
evaluation harness, statistical testing. These consume `SwarmFarmEnv` exactly
as built — no changes to the environment should be needed to add them.

**Scarecrows** were in your obstacle list but are functionally identical to a
static obstacle with a small radius (like a pole) — rather than add a
redundant class, add `n_scarecrows` to `ObstacleConfig` and one more tuple in
`StaticObstacleGenerator.generate()`'s `specs` list if you want them
visually/statistically distinct from poles. Two-line change, shown in §4.

---

## 1. `configs/config.py` — every tunable parameter, in one place

Six dataclasses, aggregated into `ExperimentConfig`:

- `FieldConfig` — grid cell size, MBR strategy, cell-labeling rule.
- `ObstacleConfig` — counts and radii/speeds for every obstacle type.
- `WindConfig` — wind strength/gustiness, wake-interference strength.
- `DroneConfig` — UAV max speed, collision radius, safety radius, battery.
- `RewardConfig` — every reward/penalty weight.
- `EnvConfig` — number of UAVs, episode length, physics timestep.

**To change any experiment parameter, edit here — never hard-code a number
elsewhere in the codebase.** E.g., to make wind stronger:
```python
config = ExperimentConfig(wind=WindConfig(sigma=2.5, mean_speed_mps=5.0))
```
`ExperimentConfig.to_json()`/`from_json()` let you save/reload the exact
config used for a run — use this for every experiment you report in the paper.

One implementation note: the dataclass field is named `field_config` (not
`field`) inside `ExperimentConfig`, because `field` collides with
`dataclasses.field`, the factory function used for mutable defaults. This bit
us once during testing (see §7) — don't rename it back.

---

## 2. `environment/cell.py` — the atomic grid unit

`GridCell` holds: grid index (`row`, `col`), physical center, its own Shapely
polygon, and two independent boolean flags:
- `is_visitable` — set once at grid-generation time, purely a function of
  field geometry (Goal 1).
- `blocked_static` — set later by `StaticObstacleGenerator` (Goal 2).

`is_flyable` is a **derived property** (`is_visitable and not blocked_static`),
not a stored flag — this guarantees it can never go stale if you change one of
the two source flags. If you add a third reason a cell might be unflyable
(e.g., a no-fly zone), add it as another independent flag and extend this
property — don't overload `is_visitable`.

---

## 3. `environment/polygon_field.py` + `grid_map.py` — Goal 1

- `FarmFieldPolygon`: wraps and validates a Shapely polygon from raw vertices.
  Auto-repairs invalid (self-intersecting) input via `shapely.make_valid`,
  and **logs a warning when it does** — if you see this warning, your input
  vertices are malformed and the repaired shape may not match your intent;
  check it visually before trusting results.
- `MinimumBoundingRectangle`: two modes via `FieldConfig.mbr_mode`.
  `AXIS_ALIGNED` matches the reference paper's Fig. 3b. `ROTATED_MIN_AREA`
  uses Shapely's true minimum-area rectangle — use this if your field is
  diagonally oriented; it measurably reduces wasted non-visitable cells (see
  the comparison table our Day-1 demo script printed: axis-aligned MBR was
  34,000 m² vs. 33,455 m² rotated, for the same test field).
- `GridMapGenerator._label_cell()`: **this is the function to edit if you want
  a different visitable/non-visitable rule.** Currently: `AREA_OVERLAP` mode
  marks a cell visitable if ≥50% of its area intersects the field polygon
  (`FieldConfig.overlap_threshold`). Lower the threshold to be more permissive
  near edges; switch to `CellLabelRule.CENTROID` for the cheaper (and less
  accurate near concave boundaries) point-in-polygon test.
- `get_cell_at_point()`: maps a UAV's continuous (x, y) position back to a
  discrete cell. For `AXIS_ALIGNED` this is O(1) index arithmetic; for
  `ROTATED_MIN_AREA` it's a linear scan over all cells (documented as a
  scalability TODO — fine at these grid sizes, replace with a spatial index
  like `shapely.STRtree` if you scale to very large fields).

---

## 4. `environment/obstacle.py` — Goal 2 static + dynamic obstacles

**Static obstacles** (`StaticObstacle` + `StaticObstacleGenerator`):
- Each obstacle is a circular footprint (position + radius). Buildings use a
  larger radius, so their footprint can span multiple grid cells — all cells
  intersecting the footprint get `blocked_static = True`.
- Placement is rejection-sampled: candidates must be visitable, not a start
  cell, actually inside the true field polygon (not just the MBR), and at
  least 2 m from any already-placed obstacle.

**To add a new static obstacle type** (e.g., scarecrows): add counts/radius to
`ObstacleConfig`, add `SCARECROW = "scarecrow"` to `ObstacleType`, add one more
tuple to the `specs` list in `StaticObstacleGenerator.generate()`:
```python
(ObstacleType.SCARECROW, self._config.n_scarecrows, self._config.scarecrow_radius_m),
```
That's the entire change — collision detection and rendering already handle
any `ObstacleType` polymorphically (add a render color in
`visualization/renderer.py::_OBSTACLE_COLORS` too, or it'll default to black).

**Dynamic obstacles** (`DynamicObstacle` + `MovementStrategy` subclasses):
- `WaypointPatrolStrategy` — used for tractors; moves back and forth between
  two points at constant speed. Extend the waypoint list for a more complex
  patrol route (e.g., a multi-row coverage pattern).
- `BoundedRandomWalkStrategy` — used for humans (slower, smoother heading
  changes via `heading_noise_std=0.4`) and animals/birds (faster, more erratic,
  `heading_noise_std=1.0`). Reflects off the field's bounding box.
- **To add a new movement behavior**, subclass `MovementStrategy` and
  implement `step()`. Nothing else needs to change — `DynamicObstacleManager`
  calls `obstacle.step(dt, rng)` polymorphically regardless of strategy type.

---

## 5. `environment/wind.py` — ambient wind + inter-UAV wake

- `WindField`: an Ornstein-Uhlenbeck process, not white noise — wind is gusty
  but temporally correlated (a gust doesn't fully reverse every timestep).
  Tune `theta` (higher = snaps back to the mean wind faster / calmer) and
  `sigma` (higher = gustier) in `WindConfig`.
- `compute_wake_interference()`: **this is what answers "how does one drone's
  air affect another."** Any UAV within `wake_radius_m` of another UAV picks
  up extra turbulence, biased to be stronger if it's downwind of the other UAV
  relative to the ambient wind direction, decaying linearly to zero at
  `wake_radius_m`. This is a deliberately simplified aerodynamic model (no
  real CFD) — documented as a limitation, easy to extend if you have a more
  detailed downwash model to plug in (it's a pure function: positions + wind
  in, per-UAV turbulence vectors out — swap the internals freely).

---

## 6. `environment/drone.py` — UAV physics

- `ActionType`: `STAY, UP, DOWN, LEFT, RIGHT` — the paper's straight-movement
  action set (Section 3.3.4) plus `STAY`.
- `UAV.commanded_velocity()`: discrete action → intended velocity vector.
- `UAV.step_physics()`: **effective velocity = commanded + wind_coupling ×
  ambient wind + wake turbulence**, integrated over `dt`. This is the key
  structural change from the paper's pure grid-teleportation: a UAV's actual
  displacement can now differ from what the policy commanded, because of
  wind — which is exactly what makes collision avoidance and wind-awareness
  meaningful things for PPO to learn.
- Battery drains faster when fighting a stronger disturbance
  (`disturbance_cost` term) — ties the wind model to a measurable consequence
  rather than being cosmetic.
- `clamp_to_bounds()`: hard safety net — a UAV can never be physically
  simulated outside the MBR, regardless of what the policy commands.

---

## 7. `environment/collision.py` — the UAV-UAV collision gap, closed

This is the direct answer to *"the paper doesn't realistically solve UAV-UAV
collisions."* Four checks, all vectorized/simple distance comparisons:
- **UAV-UAV**: pairwise distance < `DroneConfig.safety_radius_m` (default 3m)
  — this is a *safety margin* violation, not just a literal same-point crash,
  which is what makes it realistic (real UAV ops care about minimum
  separation, not just avoiding exact collisions).
- **UAV-static obstacle** / **UAV-dynamic obstacle**: distance < sum of UAV
  and obstacle radii.
- **UAV-boundary**: UAV position outside the true field polygon (not just
  outside the MBR — a UAV can be inside the rectangle but outside the actual
  farm boundary, e.g. in a "gray" non-visitable corner).

`CollisionEngine.detect()` is stateless per call — feed it any list of UAVs/
obstacles and it returns a `CollisionReport`, no hidden state. This makes it
directly unit-testable with hand-built positions if you want to add tests.

---

## 8. `environment/reward.py` — Goal 2's "tune the reward function"

`RewardCalculator.compute()` is called once per UAV per step. Two parts:
1. **Reference paper's exact coverage term** (Table 2 + Eq. 2): new-cell
   reward scales up as fewer cells remain unvisited; visited-cell and
   non-visitable penalties are the paper's literal constants
   (358.74 / -31.14 / -225.17).
2. **Our safety extension**: a penalty per collision type, all independently
   tunable in `RewardConfig` (`collision_uav_uav`, `collision_static_obstacle`,
   `collision_dynamic_obstacle`, `out_of_bounds`), plus a small continuous
   `wind_work_coeff` penalty proportional to disturbance magnitude (encourages
   the policy to avoid unnecessarily flying through high-turbulence wake
   zones, not just avoid hard collisions).

**If PPO training on Day 3 looks unstable or collision-avoidance isn't being
learned, this is the first file to tune** — try scaling collision penalties
relative to the coverage reward's typical magnitude (currently collision
penalties are already an order of magnitude below the ~358 new-cell reward's
scaled value, which can reach several thousand early in an episode — you may
need to increase penalty magnitudes further once PPO is actually optimizing
against them).

---

## 9. `environment/swarm_env.py` — the Gymnasium integration

`SwarmFarmEnv(gym.Env)` ties everything together.

- **Action space**: `MultiDiscrete([5] * n_uavs)` — a single global policy
  outputs one action per UAV simultaneously. This mirrors the reference
  paper's own finding (Section 4.2) that one global network for the whole
  swarm beats one network per UAV, so we're not inventing this structure, we're
  carrying forward the paper's best-supported result.
- **Observation space**: a `Dict` (compatible with SB3's `MultiInputPolicy`):
  `flyable_map`, `visited_map`, `dynamic_obstacle_map`, `uav_position_map`
  (all H×W grids), plus `wind_vector` (2,) and `battery_frac` (n_uavs,).
- **`step()` order of operations matters** (documented in the docstring):
  wind/obstacles advance first, then wake interference is computed from
  *pre-motion* positions (avoiding same-step feedback), then physics
  integrates, then collisions are detected on *post-motion* positions, then
  reward is computed. If you reorder this, re-check for feedback loops.
- **Termination**: full flyable-cell coverage OR all UAVs out of battery.
  **Truncation**: `EnvConfig.max_episode_steps` reached.

**If you need to change what the policy observes** (e.g., add a per-UAV
relative-position feature instead of only grid maps), extend
`observation_space` and `_get_obs()` together — they must stay in sync or
SB3 will raise a shape-mismatch error at training time.

---

## 10. `visualization/renderer.py` — full-environment rendering

`EnvironmentRenderer.render_frame()` takes the dict returned by
`SwarmFarmEnv.render()` and draws: grid (white/gray/pale-green-visited/black-
blocked), field boundary, static obstacles (color per type), dynamic
obstacles (marker per type), UAVs (star marker, gray if battery-dead), and a
wind-direction arrow. Matplotlib is imported only inside this file, so
headless PPO training never pays that import cost.

---

## 11. `utils/` — cross-cutting helpers

- `seed.py::set_global_seed()` — call once at the start of every experiment
  script, before constructing any config or env.
- `logger.py::get_logger()` — safe to call repeatedly (won't duplicate log
  handlers); pass `log_file=...` to also persist logs for a run.
- `metrics.py` — `coverage_fraction`, `valid_action_fraction`,
  `collision_rate`, `summarize_runs` (mean/std/min/max across seeds) — these
  are what Day 3/4's evaluation harness and baseline comparison will call;
  built now so PPO, Q-Learning, and A* all get measured identically.

---

## 12. `experiments/run_experiment.py` — the smoke test

Builds a full `SwarmFarmEnv`, runs 120 steps of a **random** policy (not
PPO — that's Day 3), logs metrics every 20 steps, and renders before/after
frames to `results/`. Run it any time you change a module to sanity-check
nothing broke:
```bash
cd uav_swarm_project
python -m experiments.run_experiment
```
Expect large negative cumulative reward with a random policy — that's
correct, not a bug: a random policy flies out of bounds and near other UAVs
constantly, and the reward function is supposed to punish that heavily. Once
PPO is trained (Day 3), cumulative reward should climb substantially as it
learns to avoid these penalties while still maximizing coverage.

---

## Known simplifications (be ready to state these to a reviewer)

1. Static obstacles are circular footprints, not exact building polygons.
2. Wake interference is a simplified proximity+downwind heuristic, not CFD.
3. `get_cell_at_point()` for rotated MBRs is a linear scan (fine at current
   grid sizes; flagged as a scalability item for very large maps).
4. Wind is spatially uniform across the whole field at a given instant
   (real wind varies with position/altitude); reasonable for a field-sized
   area at constant altitude, but worth stating explicitly in Limitations.
