# Goal 4 Module Guide — PPO Integration

Companion to `MODULE_GUIDE.md` (Goal 1/2) and `GOAL3_MODULE_GUIDE.md` (Goal 3).
Only covers what changed/was added for Goal 4. **Same architecture, same
folder structure** — `ppo/` and `results/` already existed as empty
placeholders since Day 1; this is the first goal that actually fills them in.

---

## What changed, file by file

| File | Status | What changed |
|---|---|---|
| `configs/config.py` | Modified | New `PPOConfig` dataclass; `ExperimentConfig` gained a `ppo` field |
| `environment/swarm_env.py` | Modified | One new function appended: `make_env()` (env factory for SB3's vectorized envs) |
| `ppo/callbacks.py` | **New** | `SwarmMetricsCallback` — logs coverage/reward-breakdown/collisions to TensorBoard+CSV |
| `ppo/train.py` | **New** | The training pipeline |
| `ppo/evaluate.py` | **New** | Loads a saved model, runs deterministic evaluation episodes |
| `visualization/training_plots.py` | **New** | Parses SB3's CSV log, produces reward/coverage/loss curve PNGs |

**Nothing in `environment/reward.py`, `collision.py`, `obstacle.py`, `wind.py`,
`drone.py`, `grid_map.py`, `polygon_field.py`, or `cell.py` was touched.**
This is the payoff of building the Gymnasium environment properly in Goals
1–3: PPO training required zero environment changes, only one small
addition (`make_env`) for SB3's vectorization API.

---

## 1. `configs/config.py` — `PPOConfig` (edit here to retune anything)

```python
# Core hyperparameters (explicitly called out in the task)
learning_rate, gamma, gae_lambda, clip_range, batch_size, n_steps, n_epochs

# Regularization
ent_coef, vf_coef, max_grad_norm

# Network architecture
net_arch_pi, net_arch_vf   # both default to [256, 256]

# Run length / parallelism
total_timesteps, n_envs

# Checkpointing / evaluation cadence
checkpoint_freq, eval_freq, n_eval_episodes

# Output locations
tensorboard_log, model_dir, monitor_dir
```

**This is the only file you should need to touch to retune training.**
Nothing else in `ppo/` has hard-coded hyperparameter values. See
`PPO_TRAINING.md` for what each one does and concrete tuning guidance for
the five the task specifically calls out (learning rate, gamma, clip range,
batch size, n_steps).

One correctness note baked into `ppo/train.py`: it asserts
`n_steps * n_envs` is divisible by `batch_size` before training starts (PPO
requires this to form minibatches cleanly) — if you change any of the three,
re-run and let the assertion tell you immediately if the combination is invalid,
rather than debugging a cryptic error from inside SB3.

---

## 2. `environment/swarm_env.py` — one addition: `make_env()`

```python
def make_env(vertices, config, episode_logger=None, render_mode=None, seed_offset=0):
    def _init():
        ...
        return SwarmFarmEnv(vertices, env_config, episode_logger=episode_logger, render_mode=render_mode)
    return _init
```

A factory returning a zero-argument callable — the exact shape SB3's
`DummyVecEnv`/`SubprocVecEnv` require (`env_fns: List[Callable[[], gym.Env]]`).
The important detail: **each call applies a different `seed_offset` to
`config.seed`**. Without this, `n_envs` parallel training environments would
each generate an *identical* field/obstacle/wind realization (same seed →
same everything), defeating the entire purpose of parallel rollout diversity.

This was placed in `swarm_env.py` (not duplicated inside `ppo/train.py`)
because Day 4's baselines (`baselines/qlearning.py`, `baselines/astar.py`,
still to come) will need the identical construction recipe — one factory,
reused by every algorithm, is what keeps "PPO vs. Q-Learning vs. A*" a fair
comparison rather than three subtly different environments.

---

## 3. `ppo/callbacks.py` — `SwarmMetricsCallback`

The **only** custom callback. Checkpointing and evaluation reuse SB3's own
`CheckpointCallback` and `EvalCallback` directly in `train.py` — no need to
reinvent well-tested infrastructure.

**Why this callback exists**: SB3's logger only knows generic RL quantities
(episode reward/length, losses) by default. It has no idea our `info` dict
carries `coverage_fraction`, `valid_action_fraction`, `collision_counts`,
and the 8-category `reward_breakdown` from Goal 3. Every `_on_step()` call,
it reads `self.locals["infos"]` (list of per-sub-env info dicts) and buffers
values into fixed-size rolling windows (`deque(maxlen=100)`). Every
`_on_rollout_end()` (i.e., right before each policy update), it flushes
windowed means into `self.logger.record(...)` — the *same* SB3 Logger
object already writing to TensorBoard + CSV, so these show up on the same
`time/total_timesteps` x-axis as `rollout/ep_rew_mean` and `train/loss`.

**To log a 9th custom metric**: add a `deque` for it in `__init__`, append
to it in `_on_step()` if the key is present in `info`, and record its mean
in `_on_rollout_end()`. Follow the existing pattern exactly.

---

## 4. `ppo/train.py` — the training pipeline

Read `train(config: ExperimentConfig) -> str` top to bottom; it's written
as a linear sequence matching the task's own listed steps:

1. **Assert** `n_steps * n_envs % batch_size == 0` (fail fast, clear message).
2. **Save config to disk** (`results/ppo_experiment_config.json`) *before*
   training starts — so even a crashed/interrupted run leaves a record of
   exactly what was attempted.
3. **`build_train_vec_env()`**: `n_envs` `SwarmFarmEnv` instances, each
   `Monitor`-wrapped (SB3's standard episode reward/length tracker — this
   is what feeds `rollout/ep_rew_mean`), each independently seeded via
   `make_env(seed_offset=i)`, combined into a `DummyVecEnv`.
   **Why `DummyVecEnv` not `SubprocVecEnv`**: at this environment's size
   (small grids, few UAVs), per-step Python/Shapely overhead dominates over
   any multiprocessing gain, and `DummyVecEnv` avoids cross-process pickling
   of Shapely geometry objects entirely. `make_env` already returns
   `SubprocVecEnv`-compatible callables — switching is a one-line change if
   you scale up (see `PPO_TRAINING.md` §6).
4. **One separate eval env** (same construction, `seed_offset=1000` so it
   never coincides with a training env's seed) for `EvalCallback`.
5. **`PPO(...)` construction**: `policy="MultiInputPolicy"` (required — our
   observation space is a `Dict`, see `swarm_env.py`), every hyperparameter
   read from `PPOConfig`, `policy_kwargs` setting the `[256, 256]` MLP
   architecture for both policy and value networks.
6. **`configure(...)` + `model.set_logger(...)`**: attaches `["stdout",
   "csv", "tensorboard"]` output formats. The `"csv"` format is what
   `visualization/training_plots.py` reads — without this line, only
   TensorBoard's binary format would exist and you'd need a tensorboard log
   parser to plot anything.
7. **Three callbacks** combined via `CallbackList`:
   - `CheckpointCallback`: periodic `.zip` saves to `model_dir`.
   - `EvalCallback`: periodic deterministic evaluation on the separate eval
     env; saves the best-performing checkpoint separately to
     `model_dir/best_model/`.
   - `SwarmMetricsCallback`: see §3 above.

   **`save_freq`/`eval_freq` are divided by `n_envs`** — SB3 counts callback
   invocations (once per vectorized step), not raw environment timesteps.
   `PPOConfig.checkpoint_freq=10_000` with `n_envs=4` means a checkpoint every
   2,500 *callback calls*, which corresponds to 10,000 *environment*
   timesteps — the division makes the config value mean what it says.
8. **`model.learn(...)`**, then **`model.save(...)`** to `model_dir/ppo_swarm_final`.

**Command-line overrides** (`--total-timesteps`, `--n-envs`, `--n-steps`,
`--batch-size`) exist specifically so you can run a fast pipeline-check
(e.g., `python -m ppo.train --total-timesteps 2000 --n-envs 2`) without
editing `PPOConfig`'s defaults, which should stay at production-scale values.

---

## 5. `ppo/evaluate.py` — post-training evaluation

`evaluate(model_path, n_episodes, seed) -> (rewards, coverages)`. Loads a
saved model via `PPO.load()`, runs `n_episodes` **deterministic** rollouts
(`model.predict(obs, deterministic=True)` — argmax of the policy
distribution, the standard choice for reporting final performance rather
than the stochastic sampling used during training exploration) against a
single non-vectorized `SwarmFarmEnv`.

**Important difference from training**: this attaches a real `EpisodeLogger`
directly to the env (safe here — single sequential episodes, no
interleaving risk, unlike vectorized training; see `PPO_TRAINING.md` §4 for
why training doesn't do this). Every evaluation episode gets a full
Goal-3-style CSV row (`results/episode_logs/ppo_eval_episodes.csv`) — the
same columns Day 4's Q-Learning/A* baseline evaluation will produce, so
comparison is a direct CSV join, not a reconciliation exercise.

---

## 6. `visualization/training_plots.py` — the three required curves

`plot_training_curves(log_dir="results/tensorboard", output_dir="results/plots")`
reads `progress.csv` (pandas) and saves four PNGs:
- `reward_curve.png` — `rollout/ep_rew_mean` vs. timesteps
- `coverage_curve.png` — `custom/coverage_fraction_mean` vs. timesteps (from `SwarmMetricsCallback`)
- `loss_curves.png` — policy gradient / value / entropy loss, three lines
- `reward_breakdown_curves.png` (bonus, not required but essentially free
  given the data is already there) — all 8 Goal-3 reward categories on one
  plot, directly useful for diagnosing reward-imbalance problems during training

Each plot is skipped with a warning (not a crash) if its source column
isn't in the CSV yet — e.g., loss curves need at least one policy update
to have happened, which needs at least one full rollout (`n_steps * n_envs`
timesteps) to complete first.

**Why CSV, not parsing TensorBoard's binary event files**: three line plots
don't justify a `tensorboard`-log-parsing dependency; the CSV is also
directly human-inspectable (`pandas.read_csv` or even a text editor) if a
plot ever looks wrong. TensorBoard itself remains fully available for
interactive exploration: `tensorboard --logdir results/tensorboard`.

---

## Quick sanity check after any change

```bash
cd uav_swarm_project
python -m ppo.train --total-timesteps 2000 --n-envs 2 --n-steps 256 --batch-size 128
python -m visualization.training_plots
python -m ppo.evaluate --model results/models/ppo_swarm_final.zip --episodes 2
```
All three should complete without error in well under a minute. If
`ppo.train` fails immediately with an assertion error, it's telling you
`n_steps * n_envs` isn't divisible by `batch_size` — fix the three numbers,
not the assertion.
