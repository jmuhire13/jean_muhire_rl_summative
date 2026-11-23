"""
Runs every hyperparameter experiment of all four algorithms, a few at a time.

Runs that already finished (they have a result file in logs/) are skipped,
so if the sweep is stopped, running it again continues where it left off.

Usage:
    uv run python -m training.run_sweep                  # all algorithms, 4 at a time
    uv run python -m training.run_sweep --workers 2
    uv run python -m training.run_sweep --algos dqn ppo
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training import dqn_training, pg_training

LOGS = Path("logs")


def list_jobs(algorithms):
    jobs = []
    for algorithm in algorithms:
        experiments = dqn_training.EXPERIMENTS if algorithm == "dqn" else pg_training.EXPERIMENTS[algorithm]
        for run in experiments:
            if (LOGS / algorithm / f"{run}_result.json").exists():
                continue  # already finished
            if algorithm == "dqn":
                command = [sys.executable, "-u", "-m", "training.dqn_training", "--run", run]
            else:
                command = [sys.executable, "-u", "-m", "training.pg_training", "--algo", algorithm, "--run", run]
            jobs.append((run, command))
    return jobs


def run_job(job):
    run, command = job
    output_file = LOGS / "sweep" / f"{run}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"started  {run}", flush=True)
    with open(output_file, "w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    last_line = output_file.read_text().strip().splitlines()[-1] if result.returncode == 0 else "FAILED, see log"
    print(f"finished {run}: {last_line}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run all hyperparameter experiments")
    parser.add_argument("--workers", type=int, default=4, help="experiments running at the same time")
    parser.add_argument("--algos", nargs="+", default=["dqn", "reinforce", "ppo", "a2c"])
    args = parser.parse_args()

    jobs = list_jobs(args.algos)
    print(f"{len(jobs)} experiments to run with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_job, jobs))
    print("All experiments finished.", flush=True)


if __name__ == "__main__":
    main()
