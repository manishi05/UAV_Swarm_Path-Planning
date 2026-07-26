# Reward Engineering — Design, Rationale, and Improvement Guide

This document explains the *entire* reward design process used in
`environment/reward.py`, why every constant is what it is, what its known
weaknesses are (found via an actual smoke-test run, not speculation), and
concrete ways you can improve it. Read this before tuning anything in
`RewardConfig` — it's written so you can defend every choice to a reviewer,
and so you know exactly what to change if PPO training doesn't behave.

---

## 1. Starting point: what the reference paper actually specifies

The reference paper (Puente-Castro et al., 2022) defines exactly **one**
reward mechanism (Table 2, Eq. 2, Section 3.3.3):

```
new_cell_reward = 358.74 × (1 + max(rows, cols) / visited_cells)
visited_cell_reward = -31.14      (revisiting an already-covered cell)
non_visitable_reward = -225.17    (flying over a cell outside the field)
```

This is a pure **coverage-maximization** signal: get to unvisited cells,
and get there faster as fewer cells remain (the scaling term grows as
`visited_cells` grows relative to map size — wait, more precisely, as
`visited_cells` is the *denominator*, the term actually *shrinks* as more
cells get visited, which the paper frames as "increases as fewer cells are
left undiscovered" in the inverse sense: early in an episode `visited_cells`
is small, so the multiplier is large, giving a strong initial pull toward
covering ground fast). This mechanism is preserved **exactly** in our
`coverage` category — same constants, same formula. This is a deliberate
choice: since we're comparing PPO against the paper's own Q-Learning
approach, the underlying task definition (cover the field efficiently) has
to stay identical, or a "PPO is better" result wouldn't be measuring the
same thing the paper measured.

**Everything else in our reward function is new** — the paper has no
concept of safety, obstacles, exploration shaping, mission completion
bonuses, or energy cost, because it has none of those environment features
to begin with (Section 7, "Future work", explicitly defers all of this).

---

## 2. The 8-category design (Goal 3 requirement)

Rather than one flat scalar, reward is computed as a `RewardBreakdown` —
8 independently named, independently weighted, independently loggable
components. This is a deliberate engineering choice, not just bookkeeping:

**Why decompose at all, instead of one tuned scalar reward?**
1. **Debuggability.** A policy that isn't learning collision avoidance
   could be failing for many different reasons (coverage reward dominates;
   collision penalty too small; collision penalty too *large* and swamps
   the learning signal entirely). Without decomposition you only see the
   symptom (total reward), not the cause. With decomposition, you can look
   at `results/<name>_episodes.csv` and see, e.g., that
   `reward_collision_avoidance` never moves while `reward_coverage`
   dominates — that's a scale-imbalance diagnosis you can act on directly.
2. **Reviewer defensibility.** "We tuned the reward function to avoid
   collisions" is a weak claim without evidence. "Here is the per-category
   reward curve over training, showing collision-avoidance reward rising
   from -X to near 0 while coverage reward continues to grow" is a strong,
   falsifiable claim you can put directly in a results figure.
3. **Independent tunability.** You can scale UAV separation without
   touching coverage, or add a curriculum that ramps up dynamic-obstacle
   penalties over training without redesigning the whole function.

### The 8 categories, what drives each, and what it's for

| # | Category | Discrete or dense? | What triggers it | Purpose |
|---|---|---|---|---|
| 1 | **Coverage** | Discrete (per-cell) | Entering a new / already-visited flyable cell | Reference paper's core task objective, unchanged |
| 2 | **Exploration** | Dense (every step) | Local neighborhood novelty around current cell | Counters the paper's own documented "one-edge bias" failure (Fig. 10) |
| 3 | **Collision avoidance** | Discrete + soft | Physical contact with static/dynamic obstacle, or flying into an obstacle-blocked cell | Directly implements Goal 2's obstacle-safety requirement |
| 4 | **Boundary violations** | Discrete (two signals) | Cell outside field OR continuous position outside field polygon | Answers "does the swarm know it crossed the boundary" |
| 5 | **Dynamic obstacle avoidance** | Dense (proximity ramp) | Distance to nearest moving obstacle within a warning radius | Denser gradient than a sparse collision-only signal — important for PPO |
| 6 | **UAV separation** | Discrete + dense | Physical UAV-UAV proximity violation, plus a proximity ramp | Directly implements the paper's unsolved UAV-UAV collision gap |
| 7 | **Mission completion** | One-time bonus | Full flyable-cell coverage first achieved (swarm-level) | Gives PPO an unambiguous "you won" signal distinct from per-step shaping |
| 8 | **Energy efficiency** | Dense (every step) | Baseline flight-time cost + disturbance-fighting cost | Ties to the paper's own flight-time-minimization framing (Section 3.4) |

---

## 3. Why boundary checking has **two separate signals**, not one

You asked specifically whether the swarm knows when it's crossed the
boundary. It has two independent checks, deliberately kept separate rather
than merged:

1. **Grid-level** (`entered_boundary_violation` in `reward.py`): is the
   UAV's current *cell* outside `is_visitable`? This is checked against
   the discretized grid, the same representation the observation space
   uses — so it's the signal a policy can most directly correlate with
   what it *observes* (the `flyable_map` channel).
2. **Continuous-position-level** (`CollisionType.UAV_BOUNDARY` from
   `collision.py`): is the UAV's exact (x, y) position, in meters, outside
   the true field polygon? This catches cases the grid-level check can
   miss — e.g., a UAV whose cell center is technically inside a
   boundary-adjacent cell but whose continuous position (pushed by wind)
   has drifted just past the true polygon edge before the next cell lookup.

Both map to the same `boundary` reward category, so in practice they
usually fire together — this redundancy is intentional (a real UAV
autopilot would also want two independent boundary checks, a cheap
grid/geofence check and a precise geometric one) but it does mean the
`boundary` category can be double-penalized in a single step. This is a
known, documented behavior — see the tuning note in §5.

---

## 4. Full formulas (what's actually computed)

```python
# Coverage (paper's Eq. 2, exact)
if entered_new_cell:
    scale = 1 + max(rows, cols) / max(n_visited_cells, 1)
    coverage += new_cell_base * scale        # new_cell_base = 358.74
elif entered_visited_cell:
    coverage += visited_cell                  # = -31.14

# Boundary
if entered_boundary_violation:                # grid-level
    boundary += non_visitable                 # = -225.17
if UAV_BOUNDARY in collisions:                 # continuous-position-level
    boundary += out_of_bounds                  # = -300.0

# Collision avoidance
if entered_obstacle_zone:                      # cell-level precursor
    collision_avoidance += blocked_cell_penalty # = -150.0
if UAV_STATIC_OBSTACLE in collisions:
    collision_avoidance += collision_static_obstacle   # = -400.0
if UAV_DYNAMIC_OBSTACLE in collisions:
    collision_avoidance += collision_dynamic_obstacle  # = -450.0

# Exploration (only inside the field)
if not entered_boundary_violation:
    exploration += exploration_coeff * local_novelty   # coeff = 5.0, novelty in [0,1]

# Dynamic obstacle avoidance (dense ramp)
if nearest_dynamic_dist < warning_radius:      # warning_radius = 10.0 m
    proximity = (warning_radius - dist) / warning_radius
    dynamic_obstacle_avoidance += -proximity_coeff * proximity  # coeff = 50.0

# UAV separation (discrete + dense ramp)
if UAV_UAV in collisions:
    uav_separation += collision_uav_uav        # = -400.0
if nearest_uav_dist < warning_radius:          # warning_radius = 6.0 m
    proximity = (warning_radius - dist) / warning_radius
    uav_separation += -proximity_coeff * proximity   # coeff = 60.0

# Mission completion (swarm-level, one-time, computed in swarm_env.py not reward.py)
if full_coverage and not already_awarded:
    mission_completion += mission_completion_bonus    # = 5000.0

# Energy efficiency
energy_efficiency += energy_efficiency_coeff * dt      # coeff = -0.5 per second
energy_efficiency += wind_work_coeff * disturbance_magnitude  # coeff = -2.0
```

---

## 5. What we actually observed, and what it tells you to fix

A 500-step random-policy rollout (`experiments/run_experiment.py`, seed 42,
3 UAVs) produced this episode-level breakdown:

| Category | Total over episode |
|---|---|
| Coverage | +29,607 |
| Exploration | +571 |
| Collision avoidance | -37,350 |
| **Boundary** | **-684,670** |
| Dynamic obstacle avoidance | -1,180 |
| UAV separation | -14,773 |
| Mission completion | 0 |
| Energy efficiency | -8,636 |
| **Total** | **-716,430** |

**This is a random policy, so large negative reward is expected and
correct** — but the *proportions* are diagnostic, and they tell you
something concrete: **boundary violations are, by a wide margin, the
dominant term.** This happens because the MBR of an irregular polygon
always contains substantial area outside the true field (visible as the
gray regions in the rendered frames), and a random walker spends a lot of
time in that zone. Two implications for you:

1. **This is likely appropriate, not a bug** — a real UAV swarm absolutely
   should be heavily punished for leaving its assigned field, more than for
   almost anything else. But:
2. **Before starting PPO training, check whether this scale imbalance
   prevents the policy from ever learning the *other* 7 categories.** If
   boundary penalties are 20-50x larger than every other signal, gradient
   updates early in training may be dominated entirely by "don't leave the
   field," and collision-avoidance / exploration / energy-efficiency
   learning could be starved until boundary violations are mostly solved.
   This is a real risk with PPO given how much larger this term is.

**Recommended fix if you observe this during training** (do this
empirically, don't guess): plot `reward_boundary` vs. the other 7
categories from the episode CSV over the first ~10% of training. If
`reward_boundary` converges near zero quickly (policy learns to stay in
bounds fast) before other categories start moving, you're fine as-is. If
it doesn't converge, or converges but other categories stay flat the whole
time, reduce `non_visitable` and `out_of_bounds` in magnitude, or apply a
curriculum (see §7.3 below).

---

## 6. Why some constants are what they are (sourcing)

- **Coverage constants (358.74, -31.14, -225.17)**: taken verbatim from the
  reference paper's Table 2, chosen there via "initial random exploration
  where the best combinations of rewards have been selected" (their
  wording, Section 3.3.3) — i.e., the paper itself doesn't derive these
  analytically either, they're empirically tuned. We inherit them for
  comparability, not because they're provably optimal.
- **Collision penalties (-400 / -450 / -150)**: set to be *larger in
  magnitude* than the largest single-step coverage reward achievable early
  in an episode (`358.74 × (1 + max(rows,cols)/1) ≈ 358.74 × (1 + ~30) ≈
  11,000` at the very first new cell when `n_visited_cells=1` — so actually
  collision penalties are *not* larger than the largest possible coverage
  reward; this is worth re-examining, see §7.1).
- **Proximity warning radii (10m dynamic, 6m UAV)**: chosen relative to
  `DroneConfig.max_speed_mps=8` — roughly one to two physics timesteps of
  travel at max speed, giving the dense shaping term time to actually
  influence a decision before a hard collision could occur at `dt=1s`.
- **Mission completion bonus (5000)**: set to exceed the total plausible
  negative reward from a *reasonably good* (not perfect) episode, so
  finishing the mission is unambiguously good even if a few collisions
  happened along the way. Not yet empirically validated against actual PPO
  training — treat as a starting point.
- **Energy efficiency coefficients (-0.5/s baseline, -2.0 × disturbance)**:
  deliberately small relative to every other category — energy efficiency
  is meant to be a tie-breaker between otherwise-similar strategies (prefer
  the faster/calmer path), not a dominant objective. If PPO ignores energy
  efficiency entirely, that's probably correct behavior, not a bug.

---

## 7. How to improve this (concrete next steps)

### 7.1 Reward scale auditing (do this first, before anything else)
Compute the *typical magnitude* of each category empirically (from the
episode CSV, across several random-policy rollouts) rather than reasoning
about it from constants alone — the coverage term's scaling factor
(`1 + max(rows,cols)/visited_cells`) makes its effective magnitude highly
non-obvious and state-dependent, exactly like we found with the boundary
term. Build a small script using `utils/metrics.py::summarize_runs()` over
multiple seeds' `reward_<category>` columns and look at relative
magnitudes before touching any constant.

### 7.2 Reward normalization
Consider normalizing each category to a comparable scale (e.g., z-score
using running mean/std, or a fixed-scale clip) before summing, rather than
relying on hand-picked constants to naturally balance. Stable-Baselines3
has built-in reward normalization wrappers (`VecNormalize`) — worth
enabling for Day 3 training regardless of how well-tuned the raw constants
are, since PPO's advantage estimation is sensitive to reward scale.

### 7.3 Curriculum learning
Rather than fixed penalty magnitudes from step 1, consider ramping
`collision_*` and `*_proximity_coeff` penalties up over the course of
training (e.g., linearly over the first N timesteps) — lets the policy
first learn basic coverage behavior, then progressively learn safety
constraints on top of it, rather than fighting both objectives from a
randomly-initialized policy simultaneously. This is a common and
defensible technique to cite in a methods section.

### 7.4 Potential-based reward shaping (theoretical rigor upgrade)
The `exploration` term as currently implemented (local-novelty bonus) is
**not** proven policy-invariant — it's a heuristic dense shaping term that
could, in principle, distort the optimal policy relative to the sparse
coverage-only objective. If you want a theoretically rigorous version for
the paper, reformulate it as **potential-based reward shaping** (Ng, Harada
& Russell, 1999): define a potential function `Φ(state)` and add
`γΦ(s') - Φ(s)` as the exploration bonus, which is provably policy-invariant
(does not change the optimal policy, only the learning dynamics). A natural
potential function here: `Φ(state) = -mean(distance from each cell to
nearest visited cell)` or similar. Left as a documented improvement, not
implemented now, given the time budget — but this is exactly the kind of
thing an IEEE reviewer might ask about your exploration term, so having the
answer ready ("we use heuristic shaping; a PBRS reformulation is
straightforward future work") is valuable even if you don't implement it.

### 7.5 Ablation methodology for the paper
Once PPO training works, a strong reviewer-facing result is an ablation
table: train with each reward category individually disabled (set its
coefficient(s) to 0) and report the effect on final coverage %, collision
rate, and mission-completion rate. This directly demonstrates each
category's necessity, rather than just asserting it. `RewardConfig`'s
dataclass structure makes this easy: construct N configs, each zeroing one
category's weight(s), run identical training/eval for each.

### 7.6 Multi-objective consideration
Right now all 8 categories are linearly summed into one scalar (standard
scalarization). If PPO training reveals the categories trade off in ways
a single linear combination can't represent well (e.g., you can't get both
high coverage AND zero collisions no matter how you weight them), consider
a constrained-RL formulation instead (e.g., Lagrangian penalty methods that
adapt collision-penalty weights automatically to satisfy a target collision
rate) — a heavier lift, but worth knowing as an option if simple
reweighting plateaus.

---

## 8. Summary: what to actually do next

1. Run `experiments/run_experiment.py`, inspect
   `results/smoke_test_episodes.csv`, and look at the per-category
   magnitudes yourself with your actual field/obstacle config (not just
   trust the numbers above, which are from my test field).
2. Before Day 3 PPO training starts, decide whether the boundary-dominance
   issue found above needs pre-emptive rebalancing, or whether you want to
   let PPO's own training curves tell you (safer, more empirical — start
   training, watch the per-category curves, adjust if needed).
3. Keep every `RewardConfig` you actually train with saved via
   `ExperimentConfig.to_json()` — you'll want to report exact reward
   weights in the paper's Methods section, and be able to defend each one
   using the reasoning in §6 above.
