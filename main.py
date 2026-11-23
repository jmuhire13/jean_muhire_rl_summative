"""
Entry point for the Aid Impact Simulator.

    uv run main.py                     open the 3D village in the browser
    uv run main.py --policy cash_only  start with another policy selected
    uv run main.py --text              run one program in the terminal only

The 3D view runs in the browser (three.js). The terminal shows a verbose
log of every week. For now the available policies are the hand-made
baselines; trained agents will be added to this list after training.
"""

import argparse

import numpy as np

from environment.custom_env import AidImpactEnv
from environment.rendering import run_server, print_step
from training.baselines import BASELINES


def load_policies():
    """Return a dictionary of policy name -> function(observation) -> action."""
    rng = np.random.default_rng(0)
    policies = {}
    for name, policy in BASELINES.items():
        # each baseline needs a random generator, so we wrap it
        policies[name] = lambda obs, policy=policy: policy(obs, rng)
    return policies


def run_in_terminal(policy, seed):
    """Run one full program and print every week in the terminal."""
    env = AidImpactEnv(render_mode="ansi")
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        household = env.current
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        print_step(info["week"], household, action, reward, total_reward, info)

    print()
    print(env.render())
    print()
    print(f"Total reward: {total_reward:.2f}")
    print(f"Graduations that held: {info['successful_graduations']}")
    print(f"Relapses: {info['relapses']}")


def main():
    parser = argparse.ArgumentParser(description="Aid Impact Simulator")
    parser.add_argument("--policy", default="rule_based", help="policy to start with")
    parser.add_argument("--seed", type=int, default=0, help="random seed for --text mode")
    parser.add_argument("--port", type=int, default=8000, help="port for the web server")
    parser.add_argument("--text", action="store_true", help="terminal only, no 3D view")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    policies = load_policies()
    if args.policy not in policies:
        print(f"Unknown policy '{args.policy}'. Choose from: {', '.join(policies)}")
        return

    if args.text:
        run_in_terminal(policies[args.policy], args.seed)
    else:
        run_server(policies, args.policy, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
