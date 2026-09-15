from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("Research_Results/R4_baseline_comparison")
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "PPO": BASE_DIR / "ppo_1m_eval_episodes_clean.csv",
    "Q-Learning": BASE_DIR / "qlearning_eval_10seeds.csv",
    "A*": BASE_DIR / "astar_eval_episodes.csv",
}


# ============================================================
# Load data
# ============================================================

data = {}

for method, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    data[method] = pd.read_csv(path)


print("\nLoaded datasets:")
for method, df in data.items():
    print(f"{method}: {len(df)} evaluation episodes")


# ============================================================
# Helper function
# ============================================================

def save_bar_plot(metric, ylabel, filename, title=None):
    methods = list(data.keys())

    means = [
        data[m][metric].mean()
        for m in methods
    ]

    stds = [
        data[m][metric].std()
        for m in methods
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        methods,
        means,
        yerr=stds,
        capsize=5
    )

    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    output = FIG_DIR / filename
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# Figure 1 — Coverage comparison
# ============================================================

save_bar_plot(
    metric="coverage_fraction",
    ylabel="Coverage Fraction",
    filename="baseline_coverage_comparison.png",
    title="Coverage Comparison Across Baselines",
)


# ============================================================
# Figure 2 — Total reward comparison
# ============================================================

save_bar_plot(
    metric="total_reward",
    ylabel="Total Reward",
    filename="baseline_reward_comparison.png",
    title="Total Reward Comparison Across Baselines",
)


# ============================================================
# Figure 3 — Valid action fraction
# ============================================================

save_bar_plot(
    metric="valid_action_fraction",
    ylabel="Valid Action Fraction",
    filename="baseline_valid_action_comparison.png",
    title="Valid Action Fraction Across Baselines",
)


# ============================================================
# Figure 4 — UAV-UAV collisions
# ============================================================

save_bar_plot(
    metric="collisions_uav_uav",
    ylabel="UAV-UAV Collisions",
    filename="baseline_uav_collision_comparison.png",
    title="UAV-UAV Collision Comparison",
)


# ============================================================
# Figure 5 — Boundary collisions
# ============================================================

save_bar_plot(
    metric="collisions_uav_boundary",
    ylabel="Boundary Collisions",
    filename="baseline_boundary_collision_comparison.png",
    title="Boundary Collision Comparison",
)


# ============================================================
# Figure 6 — Steps per episode
# ============================================================

save_bar_plot(
    metric="steps",
    ylabel="Steps per Episode",
    filename="baseline_steps_comparison.png",
    title="Episode Length Comparison",
)


# ============================================================
# Create a compact publication table
# ============================================================

summary_rows = []

for method, df in data.items():

    summary_rows.append({
        "Method": method,

        "Coverage Mean": df["coverage_fraction"].mean(),
        "Coverage Std": df["coverage_fraction"].std(),

        "Reward Mean": df["total_reward"].mean(),
        "Reward Std": df["total_reward"].std(),

        "Valid Action Mean": df["valid_action_fraction"].mean(),
        "Valid Action Std": df["valid_action_fraction"].std(),

        "Steps Mean": df["steps"].mean(),
        "Steps Std": df["steps"].std(),

        "UAV-UAV Collisions Mean":
            df["collisions_uav_uav"].mean(),

        "Static Collisions Mean":
            df["collisions_uav_static_obstacle"].mean(),

        "Dynamic Collisions Mean":
            df["collisions_uav_dynamic_obstacle"].mean(),

        "Boundary Collisions Mean":
            df["collisions_uav_boundary"].mean(),
    })


summary = pd.DataFrame(summary_rows)

summary_path = BASE_DIR / "baseline_comparison_publication_table.csv"

summary.to_csv(
    summary_path,
    index=False
)

print(f"\nSaved publication table: {summary_path}")

print("\n===== FINAL BASELINE SUMMARY =====\n")
print(summary.to_string(index=False))

print("\n===== FIGURES COMPLETE =====")
print(f"All figures saved to: {FIG_DIR}")