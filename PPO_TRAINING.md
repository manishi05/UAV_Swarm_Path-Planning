# PPO Training & Logging — Process, Diagnostics, and Improvement Guide

This document explains the *entire* PPO training pipeline used in `ppo/train.py`
and the logging pipeline that feeds `results/tensorboard/`, in plain English,
plus **actual diagnostics from a real verification run** (not hypothetical) —
read §5 before you invest compute in a long training run.

---

## 1. What the reference paper did, vs. what we do now

The reference paper (Puente-Castro et al., 2022) trains via classic
Q-Learning: a Q-table approximated by a small 2-layer dense ANN (167 → 4
neurons, RMSprop optimizer), epsilon-greedy exploration (ε=0.47, decayed by
×0.93 per episode, floor 0.05), a FIFO experience-replay memory of 60
actions per UAV, trained "from scratch" each episode with no persistent
policy checkpointing or held-out evaluation protocol beyond picking the
fastest of 30 episodes per configuration (their Table 4).

PPO replaces this entirely: an actor-critic policy trained via clipped
surrogate policy-gradient updates (Schulman et al., 2017), with a proper
train/checkpoint/evaluate lifecycle, TensorBoard+CSV logging, and
reproducible saved models — none of which the paper's methodology has (it
reports single-run results per configuration, explicitly because of
"computational and time costs of averaging the results of multiple runs,"
their words, Table 4 footnote). Day 4 will exploit this: PPO's saved,
reloadable models let us run proper multi-seed statistical comparisons the
paper itself couldn't afford to.

---

## 2. The training pipeline, step by step (what actually happens when you run it)

```bash
python -m ppo.train
```

1. **`ExperimentConfig` is built and saved to disk first**, before anything
   else happens (`results/ppo_experiment_config.json`). This is deliberate:
   even a training run that crashes partway through leaves behind an exact
   record of what was attempted, so nothing is ever "we trained a model but
   don't know with what settings."

2. **Four (default `n_envs=4`) parallel copies of `SwarmFarmEnv` are built**,
   each with a *different* random seed (`config.seed + seed_offset`,
   `seed_offset = 0, 1, 2, 3`). This matters more than it might look: PPO
   collects one large batch of experience (`n_steps × n_envs` transitions)
   before every policy update. If all 4 environments were identical (same
   seed), that batch would just be 4 copies of the same trajectory
   distribution — you'd get 4× the compute cost for close to 1× the
   information. Different seeds mean different field-obstacle-wind
   realizations, so each policy update is informed by genuinely diverse
   experience.

3. **One more environment is built for evaluation only** (`seed_offset=1000`,
   guaranteed not to collide with any training env's seed). This env is
   *never* used for gradient updates — only for periodically checking "how
   good is the current policy, really" without the noise of an
   still-exploring, still-updating training environment.

4. **The PPO model is constructed** with a `MultiInputPolicy` (because our
   observation space is a `Dict` of several grids + vectors — see
   `swarm_env.py`), two 256-unit hidden layers for both the policy and
   value networks, and every hyperparameter read from `PPOConfig`.

5. **Training runs via `model.learn(total_timesteps=...)`.** Internally,
   PPO alternates between two phases, repeated until `total_timesteps` is
   reached:
   - **Rollout collection**: all `n_envs` environments are stepped forward
     `n_steps` times each (`n_steps × n_envs` transitions total), using the
     *current* policy to choose actions (stochastically — sampled from the
     policy's action distribution, not deterministic, so the policy
     explores).
   - **Policy update**: the collected rollout is used to compute advantage
     estimates (via GAE, controlled by `gamma`/`gae_lambda`), then the
     policy and value networks are updated for `n_epochs` passes over the
     data, in minibatches of `batch_size`, using PPO's clipped surrogate
     objective (`clip_range` controls how far a single update is allowed to
     move the policy from its pre-update behavior — this is PPO's
     signature stability mechanism).
   Each such alternation is one "iteration" in the training log.

6. **Three things happen automatically during training** (the callbacks):
   checkpoints are saved periodically, the policy is evaluated
   deterministically on the held-out eval env periodically (with the
   single best-performing checkpoint kept separately), and our custom
   coverage/reward-breakdown/collision metrics are logged every rollout.

7. **At the end, the final model is saved** to
   `results/models/ppo_swarm_final.zip` — a single self-contained file
   (policy weights + hyperparameters + observation/action space metadata)
   that `ppo/evaluate.py` or any future script can reload with
   `PPO.load(path)`, no need to reconstruct the training setup from scratch.

---

## 3. The five hyperparameters the task calls out, explained

| Hyperparameter | Default | What it controls | Tuning intuition |
|---|---|---|---|
| **Learning rate** | 3e-4 | Step size for gradient updates to both policy and value networks | Standard PPO default, works across many tasks. Too high → unstable/diverging policy (watch `approx_kl` spiking). Too low → painfully slow learning. If training plateaus early, try a linear decay schedule (SB3 supports passing a callable) rather than just lowering the constant. |
| **Gamma (discount factor)** | 0.99 | How much future reward is valued vs. immediate reward | Our episodes are up to 500 steps — 0.99 means reward ~100 steps away still matters (0.99^100 ≈ 0.37), appropriate for a coverage task where good early positioning pays off many steps later. Lower (e.g. 0.9) would make the policy myopic, likely worse at whole-field coverage planning. |
| **Clip range** | 0.2 | Maximum fractional change PPO allows in the policy's action probabilities per update | The paper's own namesake mechanism. 0.2 is the standard default from the original PPO paper and works well in most Gym environments. If `train/clip_fraction` in the logs stays near 0 (as it did in our verification run, see §5), updates are timid — could mean the learning rate is too low, or genuinely converged. If clip_fraction is very high (>0.3-0.4 consistently), updates are too aggressive; lower the clip range or learning rate. |
| **Batch size** | 256 | Minibatch size for each of the `n_epochs` gradient-descent passes over one rollout | Must divide `n_steps × n_envs` evenly (enforced by an assertion in `train.py`). Larger batches → smoother, lower-variance gradients but fewer updates per rollout; smaller → noisier but more frequent updates. 256 is a reasonable middle ground for our rollout sizes (2048×4=8192 by default). |
| **n_steps** | 2048 | How many steps each environment contributes to one rollout before a policy update | This is *per environment* — with `n_envs=4`, one rollout = 8,192 transitions. Larger `n_steps` = more stable advantage estimates (especially important with `gamma=0.99` and long episodes) but less frequent policy updates. Given our 500-step episodes, `n_steps=2048` means each rollout spans ~4 full episodes per environment — enough to see complete episodes' worth of coverage/collision behavior before every update. |

**n_epochs (10, not explicitly called out but tightly coupled to the above)**:
how many gradient passes are made over each rollout before discarding it and
collecting a new one. More epochs extract more learning signal per rollout
(sample-efficient) but risk overfitting to that specific batch of
experience, which is part of why `clip_range` exists — it limits how far
repeated epochs can push the policy away from what it was when the rollout
was collected.

---

## 4. The logging pipeline, explained

Three destinations, all fed by the same underlying SB3 `Logger` object:

1. **`stdout`** — the tabular output you see live during training (what's
   shown in the tool-call outputs during our verification run below).
2. **TensorBoard** (`results/tensorboard/events.out.tfevents.*`) — for
   interactive exploration: `tensorboard --logdir results/tensorboard`,
   then open `http://localhost:6006`. Best for zooming into specific
   regions of training, comparing multiple runs side by side (each
   `model.learn()` call's logger writes to the same directory; use
   different `tensorboard_log` paths per run to compare them).
3. **CSV** (`results/tensorboard/progress.csv`) — one row per logged
   iteration, every scalar as a column. This is what
   `visualization/training_plots.py` reads with pandas to produce the
   three required curve plots (plus a bonus 4th, the reward-breakdown
   curve) without needing a TensorBoard-log parser.

**Two categories of logged scalars:**
- **SB3's own** (`rollout/*`, `train/*`, `time/*`): standard PPO
  diagnostics — episode reward/length, policy/value/entropy loss, KL
  divergence, clip fraction, explained variance, timesteps/sec.
- **Ours** (`custom/*`, from `SwarmMetricsCallback`, Goal 4's one custom
  addition): `coverage_fraction_mean`, `valid_action_fraction_mean`,
  `reward_<category>_mean` for all 8 Goal-3 categories, and
  `collisions_<type>_mean` for all 4 collision types — logged as rolling
  100-sample-window means, flushed once per rollout (same cadence as
  `rollout/ep_rew_mean`, so every curve lines up on the same x-axis).

**Episode-level detail** (separate from the above, which are all
step/rollout-level running averages): `ppo/evaluate.py` additionally
attaches a Goal-3 `EpisodeLogger`, producing one full CSV row per
evaluation episode (`results/episode_logs/ppo_eval_episodes.csv`) — the
same format Q-Learning/A* baselines will produce in Day 4, for direct
comparison.

---

## 5. What we actually observed (real verification run, not a hypothetical)

**Important context**: given the time budget for this session, the run
below used `total_timesteps=8000` (`n_envs=4, n_steps=256, batch_size=256`)
— **this is a pipeline-correctness check, not a trained model.** The
default `PPOConfig` shipped in the code (`total_timesteps=200_000`) is what
you should actually run for real results; expect that to take
meaningfully longer (see §7.3 for a runtime estimate).

All four deliverables were confirmed working:
- ✅ Training ran to completion, `fps≈210-230` on this machine's CPU.
- ✅ Checkpoints saved every 500 steps as configured
  (`ppo_swarm_2000/4000/6000/8000_steps.zip`).
- ✅ `EvalCallback` fired, reported `episode_reward=-632664.30 +/- 13283.04`,
  saved a `best_model.zip`.
- ✅ `model.save()` / `PPO.load()` round-tripped correctly (confirmed via
  `ppo/evaluate.py` loading the final model and running 2 deterministic
  episodes without error).
- ✅ All four plots (`reward_curve.png`, `coverage_curve.png`,
  `loss_curves.png`, `reward_breakdown_curves.png`) generated correctly
  from the CSV log.

**What the numbers are actually telling us (read this before training longer):**

1. **`entropy_loss` stayed flat at exactly -4.83 the entire run.** For a
   `MultiDiscrete([5,5,5])` action space (3 UAVs × 5 actions), a uniform
   random policy has entropy `3 × ln(5) ≈ 4.83` — i.e., **the policy barely
   moved from its random initialization** in 8,000 steps. This is expected
   at this timestep count (only ~1 policy update happens per ~2,048
   timesteps), not a bug — but it means none of the "is PPO learning
   anything yet" question can be answered from this run. You need
   thousands of updates, not 3-4, before entropy should visibly decrease.

2. **`approx_kl` was tiny (~1e-7) and `clip_fraction` was 0 throughout.**
   Consistent with point 1 — updates were barely perturbing the policy at
   all. Not concerning on its own at this timestep count, but worth
   re-checking once you run the full 200k-step training: if `clip_fraction`
   is *still* ~0 after tens of thousands of steps, the learning rate is
   likely too low for this reward scale (see point 3).

3. **`value_loss` was enormous (~2.6-5.6e8) and `explained_variance` was
   ~0 (essentially the value function explains none of the return
   variance yet).** This is the most actionable finding. Our reward scale
   is large — single-episode cumulative rewards were in the
   -500,000 to -700,000 range in this run (dominated by the boundary
   penalty issue already flagged in `REWARD_ENGINEERING.md` §5). PPO's
   value function has to learn to predict returns on this raw scale, and
   squared-error value loss on numbers this large produces exactly the
   huge loss values seen here. This isn't necessarily catastrophic — PPO
   can still learn with poorly-scaled rewards, just less efficiently — but
   it is the single highest-leverage thing to fix before spending a full
   200k-step training budget. **See §7.1 for the concrete fix
   (`VecNormalize`).**

4. **`coverage_fraction_mean` and `reward_boundary_mean` moved a little
   (coverage ~0.17-0.24, boundary reward improving from -1.4k toward -668
   in the best window) even in this very short run** — a weak but genuine
   signal that the policy is capable of learning *something* in this
   environment, which is reassuring for the pipeline's correctness even
   though 8,000 steps is nowhere near enough for a real trained model.

---

## 6. Architecture note: how the Dict observation is actually processed

Worth understanding before you tune network architecture further:
`MultiInputPolicy` uses SB3's default `CombinedExtractor`, which processes
each key of our observation `Dict` *independently* and concatenates the
results. For our 2D grid-shaped keys (`flyable_map`, `visited_map`,
`dynamic_obstacle_map`, `uav_position_map`), because they're 2D float32
arrays without a channel dimension SB3 recognizes as "image-like," they are
each **flattened into a flat vector** and passed through a small MLP —
*not* processed by a CNN, and *not* stacked together as multi-channel
"image" input the way a human would naturally think of 4 co-registered
grid layers. This works (confirmed by the verification run learning
*something*), but it throws away spatial structure a convolutional
extractor would exploit (nearby cells being processed with shared
receptive fields, translation invariance, etc.). This is a legitimate,
documented limitation — see §7.4 for the concrete upgrade path (a custom
`BaseFeaturesExtractor` that stacks the 4 map channels into one
multi-channel image and runs it through a small CNN before concatenating
with the flattened vector features).

---

## 7. Concrete improvement paths

### 7.1 Reward normalization (do this first — highest leverage per §5.3)
Wrap the training `VecEnv` with SB3's `VecNormalize`
(`stable_baselines3.common.vec_env.VecNormalize`), which maintains running
mean/std statistics and normalizes both observations and rewards on the
fly. This directly addresses the huge `value_loss`/near-zero
`explained_variance` finding in §5 — PPO's value function will have a much
easier time learning to predict *normalized* returns than raw returns in
the hundreds of thousands. **Important**: `VecNormalize` statistics must be
saved alongside the model (`env.save(path)`) and reloaded before evaluation
(`VecNormalize.load(path, eval_env)`) — evaluating with un-normalized
observations against a policy trained on normalized ones silently produces
garbage results. Not implemented in the current pipeline given the time
budget; this is the single most valuable next change to make before a real
training run.

### 7.2 Address the boundary-penalty dominance (carried over from Goal 3)
`REWARD_ENGINEERING.md` §5 already flagged this, and this run confirms it's
still the dominant reward term. Two independent fixes, not mutually
exclusive: (a) reduce `RewardConfig.outside_field_penalty` /
`mbr_boundary_penalty` magnitude directly, or (b) apply `VecNormalize`
(§7.1), which will partially compensate for scale imbalance automatically
by normalizing the combined reward signal — but (a) is more directly
interpretable and should be tried first, with the diagnostic method already
described in `REWARD_ENGINEERING.md` §5 (plot `reward_boundary` vs. the
other 7 categories over the first 10% of a real training run).

### 7.3 Run the actual 200k-step (or longer) training
The verification run (8,000 steps) proved the pipeline works; it did not
train a usable policy. At `~220 fps` (this machine, CPU-only, `n_envs=4`),
200,000 steps would take roughly 15 minutes; a more serious 1-2M timestep
run (more typical for a PPO result worth publishing) would take 1.5-2.5
hours. A GPU won't meaningfully speed up this specific environment (the
bottleneck is Python/Shapely environment stepping, not neural network
forward/backward passes, which are tiny here) — more `n_envs` (more
parallel environments, ideally via `SubprocVecEnv` on a multi-core machine,
§7.6) is the actual lever for wall-clock speedup.

### 7.4 Custom CNN feature extractor for the map observations
As described in §6, implement a `BaseFeaturesExtractor` subclass that
stacks `flyable_map`, `visited_map`, `dynamic_obstacle_map`,
`uav_position_map` into a single `(4, H, W)` tensor and processes it with 2-3
small conv layers before concatenating with the flattened `wind_vector` and
`battery_frac` features. Pass it via
`policy_kwargs=dict(features_extractor_class=YourExtractor)`. This is a
natural "if I had more time" upgrade to mention in the paper's Future Work
— cite that the default flattening approach was used for the reported
results, with a CNN extractor identified as a promising architectural
improvement.

### 7.5 Learning rate schedule
SB3 accepts a callable for `learning_rate` (a function of remaining
training progress, `1.0 → 0.0`), enabling linear or custom decay instead of
a constant rate. Common and often effective: linear decay to a small
fraction of the initial rate over training, giving large early exploratory
updates and small late fine-tuning updates.

### 7.6 SubprocVecEnv for more parallel environments
`build_train_vec_env()` in `ppo/train.py` uses `DummyVecEnv` (documented
rationale in `GOAL4_MODULE_GUIDE.md` §4). If you move to a multi-core
machine and want genuine multiprocess parallelism (more meaningful at
`n_envs` beyond, say, 8-16), swap `DummyVecEnv` for
`stable_baselines3.common.vec_env.SubprocVecEnv` — `make_env()` already
returns the correct callable shape, so this is a one-line change. Test it
first with a short run, since Shapely geometry objects inside `SwarmFarmEnv`
must be picklable for `SubprocVecEnv`'s worker processes (they are, but
verify after any custom geometry changes).

### 7.7 Systematic hyperparameter search
Once the reward-scale issue (§7.1) is fixed and a real training budget is
available, treat `learning_rate`, `clip_range`, `n_steps`, `batch_size`, and
`ent_coef` as a small grid/random search rather than hand-tuning — Optuna
integrates cleanly with SB3 (`sb3-contrib` or a simple custom loop calling
`train()` with different `PPOConfig`s and comparing final `EvalCallback`
mean reward). Given `ExperimentConfig` is already fully serializable, each
trial's exact config is automatically preserved.

---

## 8. Summary: what to actually do next

1. Read §5 above — don't launch a long training run before deciding whether
   to apply `VecNormalize` (§7.1) and/or rebalance the boundary penalty
   (§7.2). Both are cheap to try and directly address the diagnostics found.
2. Run the real training: `python -m ppo.train` (uses `PPOConfig`'s default
   200k timesteps) — or edit `PPOConfig.total_timesteps` for a longer run.
3. Watch `results/tensorboard/progress.csv` grow (or run TensorBoard live)
   — specifically watch `train/explained_variance` climb away from 0 (value
   function learning to predict returns) and `custom/coverage_fraction_mean`
   climb toward 1.0 as the real signal that the policy is actually learning
   the task, not just that training is running.
4. Once trained, `python -m ppo.evaluate --model results/models/best_model/best_model.zip
   --episodes 20` for a proper held-out evaluation (use the *best* checkpoint,
   not necessarily the *final* one — they can differ if late training
   degraded performance).
5. Keep the exact `results/ppo_experiment_config.json` from whatever run you
   report in the paper — that's your reproducibility record for the Methods section.
