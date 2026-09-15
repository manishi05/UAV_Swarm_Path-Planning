import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONITOR_DIR = (
    PROJECT_ROOT
    / "Research_Results"
    / "R4_1m_training_full"
    / "monitor"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Research_Results"
    / "R4_1m_training_full"
    / "analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load all PPO monitor files
# ---------------------------------------------------------

files = sorted(MONITOR_DIR.glob("train_env*.monitor.csv"))

if len(files) != 4:
    raise RuntimeError(
        f"Expected 4 PPO monitor files, found {len(files)}"
    )

frames = []

for file in files:

    df = pd.read_csv(file, comment="#")

    df["source_env"] = file.stem

    frames.append(df)


# ---------------------------------------------------------
# Combine all environments
# ---------------------------------------------------------

data = pd.concat(frames, ignore_index=True)

data = data.dropna(subset=["r", "l", "t"])


# ---------------------------------------------------------
# IMPORTANT:
# The four environments ran in parallel.
#
# We therefore sort completed episodes by their recorded
# elapsed time rather than concatenating env0/env1/env2/env3.
#
# This gives an approximate temporal ordering of episode
# completions.
# ---------------------------------------------------------

data = data.sort_values("t").reset_index(drop=True)

data["episode_global"] = range(1, len(data) + 1)

data["cumulative_steps"] = data["l"].cumsum()


# ---------------------------------------------------------
# Moving averages
# ---------------------------------------------------------

WINDOW = 50

data["reward_ma"] = (
    data["r"]
    .rolling(WINDOW, min_periods=1)
    .mean()
)

data["length_ma"] = (
    data["l"]
    .rolling(WINDOW, min_periods=1)
    .mean()
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n===== PPO TRAINING DATA SUMMARY =====")

print(f"Monitor files: {len(files)}")

print(
    f"Total completed episodes: "
    f"{len(data):,}"
)

print(
    f"Total recorded environment steps: "
    f"{data['l'].sum():,}"
)

print("\nReward:")

print(
    f"  Mean:  {data['r'].mean():.2f}"
)

print(
    f"  Std:   {data['r'].std():.2f}"
)

print(
    f"  Best:  {data['r'].max():.2f}"
)

print(
    f"  Worst: {data['r'].min():.2f}"
)


print("\nEpisode length:")

print(
    f"  Mean: {data['l'].mean():.2f}"
)

print(
    f"  Std:  {data['l'].std():.2f}"
)

print(
    f"  Min:  {data['l'].min():.0f}"
)

print(
    f"  Max:  {data['l'].max():.0f}"
)


# ---------------------------------------------------------
# Early vs late training
# ---------------------------------------------------------

n = len(data)

early_n = max(1, n // 10)
late_n = max(1, n // 10)

early = data.iloc[:early_n]
late = data.iloc[-late_n:]


print("\n===== EARLY vs LATE TRAINING =====")

print(
    f"\nFirst {early_n:,} completed episodes:"
)

print(
    f"  Mean reward: "
    f"{early['r'].mean():.2f}"
)

print(
    f"  Mean episode length: "
    f"{early['l'].mean():.2f}"
)


print(
    f"\nLast {late_n:,} completed episodes:"
)

print(
    f"  Mean reward: "
    f"{late['r'].mean():.2f}"
)

print(
    f"  Mean episode length: "
    f"{late['l'].mean():.2f}"
)


reward_change = (
    late["r"].mean()
    - early["r"].mean()
)

print(
    f"\nReward change "
    f"(late - early): "
    f"{reward_change:.2f}"
)


# ---------------------------------------------------------
# Figure 1: PPO reward curve
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    data["episode_global"],
    data["r"],
    alpha=0.20,
    linewidth=0.8,
    label="Episode reward"
)

plt.plot(
    data["episode_global"],
    data["reward_ma"],
    linewidth=2,
    label=f"{WINDOW}-episode moving average"
)

plt.xlabel("Completed Episodes")
plt.ylabel("Episode Reward")

plt.title(
    "PPO Training Reward Curve"
)

plt.legend()

plt.grid(alpha=0.25)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "ppo_training_reward_curve.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"\nSaved: {path}")


# ---------------------------------------------------------
# Figure 2: Episode length
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    data["episode_global"],
    data["l"],
    alpha=0.20,
    linewidth=0.8,
    label="Episode length"
)

plt.plot(
    data["episode_global"],
    data["length_ma"],
    linewidth=2,
    label=f"{WINDOW}-episode moving average"
)

plt.xlabel("Completed Episodes")

plt.ylabel("Steps per Episode")

plt.title(
    "PPO Training Episode Length"
)

plt.legend()

plt.grid(alpha=0.25)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "ppo_training_episode_length.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"Saved: {path}")


# ---------------------------------------------------------
# Figure 3: Reward vs cumulative steps
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    data["cumulative_steps"],
    data["reward_ma"],
    linewidth=2
)

plt.xlabel(
    "Cumulative Environment Steps"
)

plt.ylabel(
    f"{WINDOW}-Episode Moving Average Reward"
)

plt.title(
    "PPO Reward During Training"
)

plt.grid(alpha=0.25)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "ppo_reward_vs_steps.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"Saved: {path}")


# ---------------------------------------------------------
# Save processed dataset
# ---------------------------------------------------------

csv_path = (
    OUTPUT_DIR
    / "ppo_training_analysis.csv"
)

data.to_csv(
    csv_path,
    index=False
)

print(f"Saved: {csv_path}")


print(
    "\n===== ANALYSIS COMPLETE ====="
)