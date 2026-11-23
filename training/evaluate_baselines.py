"""
Runs the hand-made baseline policies and saves their results.

Usage:
    uv run python -m training.evaluate_baselines
"""

import pandas as pd

from environment.custom_env import AidImpactEnv
from training.baselines import BASELINES, run_policy


def main():
    env = AidImpactEnv()
    rows = []
    for name, policy in BASELINES.items():
        results = run_policy(env, policy, episodes=50, seed=1000)
        for r in results:
            r["policy"] = name
        rows.extend(results)

    table = pd.DataFrame(rows)
    table.to_csv("logs/baseline_results.csv", index=False)

    summary = table.groupby("policy").mean(numeric_only=True).drop(columns="episode")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
