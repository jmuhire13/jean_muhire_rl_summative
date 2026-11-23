"""
Runs the hand-made baseline policies and saves their results.

This is a design check of the environment, run before any training: it uses
50 practice programs (seeds 1000 to 1049), not the 20 exam programs. The
baseline scores in the report and README come from the exam programs
(training/analysis.py), so the numbers here are a little different.

Usage:
    uv run python -m training.evaluate_baselines
"""

import pandas as pd

from environment.custom_env import AidImpactEnv
from training.baselines import BASELINES, run_policy
from training.common import LOGS_FOLDER


def main():
    env = AidImpactEnv()
    rows = []
    for name, policy in BASELINES.items():
        results = run_policy(env, policy, episodes=50, seed=1000)
        for r in results:
            r["policy"] = name
        rows.extend(results)

    table = pd.DataFrame(rows)
    table.to_csv(LOGS_FOLDER / "baseline_results.csv", index=False)

    summary = table.groupby("policy").mean(numeric_only=True).drop(columns="episode")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
