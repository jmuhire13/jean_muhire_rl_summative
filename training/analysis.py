"""
Builds every figure and table for the report from the training logs.

What it makes:
  1. Picks the best run of each algorithm (models/*/best_<algorithm>.pt)
  2. Hyperparameter tables, one per algorithm           -> logs/tables/
  3. Figures                                           -> assets/
     The numbers match the figure numbers in the report.
       fig2_rewards.png             reward during training, best run of each algorithm
       fig3_behaviour.png           which actions each best agent chooses
       fig4_training_stability.png  DQN loss (left) and policy entropy (right)
       fig5_generalization.png      best agents on unseen and harsher programs
       extra_sweeps.png             every hyperparameter run, per algorithm (not in the report)
     fig1_environment.png (the village picture) is a screenshot of the 3D view, not made here.
  4. Convergence and generalization tables              -> logs/tables/

Usage:
    uv run python -m training.analysis
"""

import json

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from environment.custom_env import ACTION_NAMES, AidImpactEnv
from training.baselines import BASELINES
from training.common import EVAL_SEEDS, LOGS_FOLDER, MODELS_FOLDER, evaluate, load_agent
from training.select_best import MODEL_FOLDERS, select_best

ASSETS = MODELS_FOLDER.parent / "assets"
TABLES = LOGS_FOLDER / "tables"

ALGORITHMS = ["dqn", "reinforce", "ppo", "a2c"]
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}

# One fixed colour per algorithm, used in every figure
COLORS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# One colour per action for the behaviour figure (fixed order)
# (assessment visit is brown so it cannot be confused with the green of graduation)
ACTION_COLORS = ["#b5b3ad", "#e34948", "#2a78d6", "#6250d6", "#e87ba4", "#eda100", "#8a5a2b", "#008300"]

# Test settings for generalization
UNSEEN_SEEDS = list(range(9000, 9050))
SCENARIOS = {
    "Exam programs (20)": {"seeds": EVAL_SEEDS},
    "Unseen programs (50)": {"seeds": UNSEEN_SEEDS},
    "Harsher shocks (50)": {"seeds": UNSEEN_SEEDS, "drought_chance": 0.04, "illness_chance": 0.03},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#b5b3ad")
    ax.tick_params(colors=MUTED)


def steps_in_thousands(ax):
    # Show environment steps as 20k, 40k, ... instead of 20000, 40000, ...
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))


def save_figure(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved assets/{name}")


def available_algorithms():
    return [a for a in ALGORITHMS if (LOGS_FOLDER / f"{a}_summary.csv").exists()]


def best_run(algorithm):
    summary = pd.read_csv(LOGS_FOLDER / f"{algorithm}_summary.csv")
    return summary.sort_values("eval_reward_mean", ascending=False).iloc[0]["run"]


def training_log(algorithm, run):
    return pd.read_csv(LOGS_FOLDER / algorithm / f"{run}.csv")


def eval_points(log):
    # rows where the greedy agent was tested
    return log.dropna(subset=["eval_reward"])


def baseline_scores():
    rng = np.random.default_rng(0)
    scores = {}
    for name in ["rule_based", "cash_only", "random"]:
        policy = BASELINES[name]
        scores[name] = evaluate(lambda obs, p=policy: p(obs, rng))["eval_reward_mean"]
    return scores


# ---------------------------------------------------------------------------
# 1. Best runs and hyperparameter tables
# ---------------------------------------------------------------------------
def make_tables(algorithms):
    TABLES.mkdir(parents=True, exist_ok=True)
    columns = ["run", "changes", "eval_reward_mean", "eval_reward_std", "graduations",
               "relapses", "in_crisis", "lasting_score", "minutes"]
    with open(TABLES / "hyperparameter_tables.md", "w") as md:
        for algorithm in algorithms:
            select_best(algorithm)
            table = pd.read_csv(LOGS_FOLDER / f"{algorithm}_summary.csv")[columns]
            table = table.sort_values("run").round(2)
            table.to_csv(TABLES / f"{algorithm}_sweep.csv", index=False)

            md.write(f"## {LABELS[algorithm]}\n\n")
            md.write("| Run | Change from baseline | Eval reward (mean ± std) | Graduations held | "
                     "Relapses | In crisis at end | Lasting score |\n|---|---|---|---|---|---|---|\n")
            for _, r in table.iterrows():
                md.write(f"| {r['run']} | {r['changes']} | {r['eval_reward_mean']:.1f} ± {r['eval_reward_std']:.1f} | "
                         f"{r['graduations']:.2f} | {r['relapses']:.2f} | {r['in_crisis']:.2f} | "
                         f"{r['lasting_score']:.2f} |\n")
            md.write("\n")
    print("saved logs/tables/hyperparameter_tables.md")


def make_convergence_table(algorithms):
    """How fast and how steadily each algorithm learned, for every run."""
    rows = []
    for algorithm in algorithms:
        for path in sorted((LOGS_FOLDER / algorithm).glob("*.csv")):
            points = eval_points(pd.read_csv(path))
            rewards = points["eval_reward"].to_numpy()
            frames = points["frames"].to_numpy()
            first, best = rewards[0], rewards.max()
            # converged = first time it gets 90% of the way from its first score to its best score
            target = first + 0.9 * (best - first)
            converged_at = int(frames[np.argmax(rewards >= target)])
            rows.append({
                "algorithm": LABELS[algorithm],
                "run": path.stem,
                "first_eval": round(first, 2),
                "best_eval": round(best, 2),
                "frames_to_90pct": converged_at,
                "last5_mean": round(rewards[-5:].mean(), 2),
                "last5_std": round(rewards[-5:].std(), 2),  # lower = more stable at the end
            })
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "convergence.csv", index=False)
    print("saved logs/tables/convergence.csv")
    return table


# ---------------------------------------------------------------------------
# 2. Figures
# ---------------------------------------------------------------------------
def figure_reward_comparison(algorithms, scores, convergence):
    """One panel per algorithm: the reward of each training program (thin line,
    smoothed) and the test reward on the 20 exam programs (thick line).
    A dotted line marks where the run converged (see make_convergence_table)."""
    names = {"rule_based": "Rule-based", "cash_only": "Cash-only", "random": "Random"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, algorithm in zip(axes.flat, algorithms):
        run = best_run(algorithm)
        log = training_log(algorithm, run)
        train = log.dropna(subset=["train_reward"])
        ax.plot(train["frames"], train["train_reward"].rolling(10, min_periods=1).mean(),
                color=COLORS[algorithm], alpha=0.35, linewidth=1.2,
                label="Training programs (moving average)")
        points = eval_points(log)
        ax.plot(points["frames"], points["eval_reward"], color=COLORS[algorithm], linewidth=2.2,
                marker="o", markersize=4, label="Test on 20 exam programs")
        # baseline officers as dashed lines, labelled on the left inside the panel
        for key, value in scores.items():
            ax.axhline(value, color=MUTED, linestyle="--", linewidth=0.9)
            ax.text(0.01, value, f" {names[key]} ({value:.0f})", transform=ax.get_yaxis_transform(),
                    va="bottom", fontsize=8, color=MUTED)
        # convergence point: first test that reached 90% of the way to the best score
        converged = convergence.loc[convergence["run"] == run, "frames_to_90pct"].iloc[0]
        ax.axvline(converged, color=COLORS[algorithm], linestyle=":", linewidth=1.5)
        ax.text(converged, 0.30, f" 90% reached \n {converged / 1000:.0f}k steps ", transform=ax.get_xaxis_transform(),
                fontsize=8, color=MUTED, va="top", ha="right" if converged > 120_000 else "left")
        style_axes(ax, f"{LABELS[algorithm]} ({run})", "Environment steps (frames)", "Total reward per program")
        steps_in_thousands(ax)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Cumulative reward per program during training (best run of each algorithm)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    save_figure(fig, "fig2_rewards.png")


def figure_training_stability(algorithms):
    """Left: the DQN loss of the best DQN run. Right: the policy entropy of the
    best REINFORCE, PPO and A2C runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    run = best_run("dqn")
    log = training_log("dqn", run).dropna(subset=["loss"])
    ax1.plot(log["frames"], log["loss"], color=COLORS["dqn"], alpha=0.3, linewidth=1, label="Per batch")
    ax1.plot(log["frames"], log["loss"].rolling(20, min_periods=1).mean(), color=COLORS["dqn"],
             linewidth=2, label="Moving average (20 batches)")
    # mark where epsilon reaches its final value
    explore_end = log.loc[log["epsilon"] <= log["epsilon"].min() + 1e-6, "frames"].min()
    ax1.axvline(explore_end, color=MUTED, linestyle=":", linewidth=1)
    ax1.text(explore_end, ax1.get_ylim()[1], " ε reaches 0.05", va="top", fontsize=8, color=MUTED)
    style_axes(ax1, f"DQN objective: Huber TD loss ({run})", "Environment steps (frames)",
               "Loss (on rewards x 0.1)")
    steps_in_thousands(ax1)
    ax1.legend(frameon=False, fontsize=8, loc="center left")

    for algorithm in [a for a in algorithms if a != "dqn"]:
        run = best_run(algorithm)
        log = training_log(algorithm, run)
        ax2.plot(log["frames"], log["entropy"].rolling(5, min_periods=1).mean(),
                 color=COLORS[algorithm], linewidth=2, label=f"{LABELS[algorithm]} ({run})")
    ax2.axhline(np.log(8), color=MUTED, linestyle="--", linewidth=1)
    ax2.text(0.01, np.log(8), " fully random (ln 8 = 2.08)", transform=ax2.get_yaxis_transform(),
             va="bottom", fontsize=8, color=MUTED)
    style_axes(ax2, "Policy entropy (best policy gradient runs)", "Environment steps (frames)",
               "Entropy (nats)")
    steps_in_thousands(ax2)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "fig4_training_stability.png")


def figure_sweeps(algorithms, scores):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, algorithm in zip(axes.flat, ALGORITHMS):
        if algorithm not in algorithms:
            ax.set_visible(False)
            continue
        best = best_run(algorithm)
        for path in sorted((LOGS_FOLDER / algorithm).glob("*.csv")):
            points = eval_points(pd.read_csv(path))
            if path.stem == best:
                continue
            is_baseline = path.stem.endswith("_01_baseline")
            ax.plot(points["frames"], points["eval_reward"],
                    color=TEXT if is_baseline else "#c9c7c1",
                    linewidth=1.5 if is_baseline else 1,
                    linestyle="--" if is_baseline else "-",
                    label="Baseline settings" if is_baseline else None)
        points = eval_points(training_log(algorithm, best))
        ax.plot(points["frames"], points["eval_reward"], color=COLORS[algorithm], linewidth=2.5,
                label=f"Best: {best}")
        ax.axhline(scores["rule_based"], color=MUTED, linestyle=":", linewidth=1)
        count = len(list((LOGS_FOLDER / algorithm).glob("*.csv")))
        style_axes(ax, f"{LABELS[algorithm]}: {count} hyperparameter runs", "Environment steps",
                   "Eval reward")
        steps_in_thousands(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Hyperparameter sweeps (grey = other runs, dotted = rule-based officer)",
                 x=0.01, ha="left", fontsize=11, color=MUTED)
    save_figure(fig, "extra_sweeps.png")


def make_generalization(algorithms):
    """Test the best agent of each algorithm (and the rule-based officer)
    on programs it never saw and on a harsher village."""
    agents = {}
    for algorithm in algorithms:
        folder = MODELS_FOLDER / MODEL_FOLDERS[algorithm]
        agents[LABELS[algorithm]], _ = load_agent(folder / f"best_{algorithm}.json")
    rng = np.random.default_rng(0)
    agents["Rule-based"] = lambda obs: BASELINES["rule_based"](obs, rng)

    rows = []
    for scenario, settings in SCENARIOS.items():
        settings = dict(settings)
        seeds = settings.pop("seeds")
        for name, choose_action in agents.items():
            result = evaluate(choose_action, seeds=seeds, **settings)
            rows.append({"scenario": scenario, "agent": name, **{k: round(v, 3) for k, v in result.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "generalization.csv", index=False)
    print("saved logs/tables/generalization.csv")

    # grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(agents)
    width = 0.8 / len(names)
    x = np.arange(len(SCENARIOS))
    colors = {LABELS[a]: COLORS[a] for a in algorithms}
    colors["Rule-based"] = "#b5b3ad"
    for i, name in enumerate(names):
        values = [table[(table.scenario == s) & (table.agent == name)]["eval_reward_mean"].iloc[0]
                  for s in SCENARIOS]
        errors = [table[(table.scenario == s) & (table.agent == name)]["eval_reward_std"].iloc[0]
                  for s in SCENARIOS]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, values, width * 0.9, yerr=errors,
               color=colors[name], label=name, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": MUTED})
    ax.axhline(0, color="#b5b3ad", linewidth=0.8)
    ax.set_xticks(x, list(SCENARIOS))
    style_axes(ax, "Generalization: best agents on unseen and harsher programs",
               "", "Mean total reward (± std)")
    ax.legend(frameon=False, fontsize=9, ncol=len(names), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, "fig5_generalization.png")
    return table


def figure_behaviour(algorithms):
    """Share of each action chosen by the best agents (on the exam programs)."""
    agents = {}
    for algorithm in algorithms:
        folder = MODELS_FOLDER / MODEL_FOLDERS[algorithm]
        agents[LABELS[algorithm]], _ = load_agent(folder / f"best_{algorithm}.json")
    rng = np.random.default_rng(0)
    agents["Rule-based"] = lambda obs: BASELINES["rule_based"](obs, rng)

    shares = {}
    env = AidImpactEnv()
    for name, choose_action in agents.items():
        counts = np.zeros(len(ACTION_NAMES))
        for seed in EVAL_SEEDS:
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                action = choose_action(obs)
                counts[action] += 1
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        shares[name] = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(shares) + 1.8))
    names = list(shares)[::-1]
    left = np.zeros(len(names))
    for a, action_name in enumerate(ACTION_NAMES):
        values = np.array([shares[n][a] for n in names])
        ax.barh(names, values, left=left, color=ACTION_COLORS[a], label=action_name,
                edgecolor="white", linewidth=1.5)
        # dark text on light colours, white text on dark colours
        label_color = TEXT if ACTION_COLORS[a] in ("#b5b3ad", "#eda100", "#e87ba4") else "white"
        for y, (v, l) in enumerate(zip(values, left)):
            if v >= 0.07:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8, color=label_color)
        left += values
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    style_axes(ax, "What each officer does: share of weekly decisions by action", "", "")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_figure(fig, "fig3_behaviour.png")
    with open(TABLES / "behaviour.json", "w") as f:
        json.dump({n: dict(zip(ACTION_NAMES, np.round(s, 3).tolist())) for n, s in shares.items()}, f, indent=2)


# ---------------------------------------------------------------------------
def main():
    algorithms = available_algorithms()
    if not algorithms:
        print("No training results found yet.")
        return
    print(f"Algorithms with results: {', '.join(LABELS[a] for a in algorithms)}")

    make_tables(algorithms)
    scores = baseline_scores()
    convergence = make_convergence_table(algorithms)
    figure_reward_comparison(algorithms, scores, convergence)
    if "dqn" in algorithms:
        figure_training_stability(algorithms)
    figure_sweeps(algorithms, scores)
    generalization = make_generalization(algorithms)
    figure_behaviour(algorithms)

    print("\nBest run per algorithm:")
    for algorithm in algorithms:
        print(f"  {LABELS[algorithm]:10s} {best_run(algorithm)}")
    print("\nGeneralization (mean total reward):")
    print(generalization.pivot(index="agent", columns="scenario", values="eval_reward_mean").round(1).to_string())


if __name__ == "__main__":
    main()
