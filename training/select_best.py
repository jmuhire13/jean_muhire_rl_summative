"""
Picks the best run of an algorithm (highest evaluation reward) and copies
its model to models/<folder>/best_<algorithm>.pt so main.py can load it.

Usage:
    uv run python -m training.select_best dqn
    uv run python -m training.select_best reinforce
"""

import shutil
import sys

import pandas as pd

from training.common import LOGS_FOLDER, MODELS_FOLDER

# where each algorithm keeps its models
MODEL_FOLDERS = {"dqn": "dqn", "reinforce": "pg", "ppo": "pg", "a2c": "pg"}


def select_best(algorithm):
    summary = pd.read_csv(LOGS_FOLDER / f"{algorithm}_summary.csv")
    best = summary.sort_values("eval_reward_mean", ascending=False).iloc[0]
    folder = MODELS_FOLDER / MODEL_FOLDERS[algorithm]

    shutil.copy(folder / f"{best['run']}.pt", folder / f"best_{algorithm}.pt")
    shutil.copy(folder / f"{best['run']}.json", folder / f"best_{algorithm}.json")
    print(f"Best {algorithm}: {best['run']} with eval reward {best['eval_reward_mean']:.2f}")


if __name__ == "__main__":
    select_best(sys.argv[1])
