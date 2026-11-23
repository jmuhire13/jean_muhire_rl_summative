"""
Shared helpers used by every algorithm (DQN, REINFORCE, PPO, A2C).

Using the same helpers keeps the comparison fair: every agent trains on the
same environment and is tested on the same evaluation programs.
"""

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, TransformedEnv

from environment.custom_env import AidImpactEnv

warnings.filterwarnings("ignore")  # hide library warnings in the training output

LOGS_FOLDER = Path("logs")
MODELS_FOLDER = Path("models")

# Fixed seeds used to test every agent on the same 20 programs
EVAL_SEEDS = list(range(5000, 5020))


def make_env(seed=0, reward_scale=1.0, **env_settings):
    """The Aid Impact environment wrapped for TorchRL.

    categorical_action_encoding=True means the action is a single number
    (0 to 7) instead of a list of 8 zeros and ones.
    RewardSum adds up the real reward of each episode so we can log it.
    RewardScaling then multiplies the reward the agent learns from by
    reward_scale (for example 0.1). Smaller numbers make training more stable,
    and the logged episode rewards stay in the normal units.
    """
    env = GymWrapper(AidImpactEnv(**env_settings), categorical_action_encoding=True)
    env = TransformedEnv(env, Compose(RewardSum(), RewardScaling(loc=0.0, scale=reward_scale)))
    env.set_seed(seed)
    return env


def evaluate(choose_action, seeds=EVAL_SEEDS, **env_settings):
    """Run full programs with a policy and return the average results.

    choose_action: function(observation) -> action number
    The agent always picks its best action here (no random exploration).
    """
    env = AidImpactEnv(**env_settings)
    rewards, graduations, relapses, crisis, lasting = [], [], [], [], []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        total = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(choose_action(obs))
            total += reward
            done = terminated or truncated
        rewards.append(total)
        graduations.append(info["successful_graduations"])
        relapses.append(info["relapses"])
        crisis.append(info["households_in_crisis"])
        lasting.append(info["mean_lasting_score"])
    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "graduations": float(np.mean(graduations)),
        "relapses": float(np.mean(relapses)),
        "in_crisis": float(np.mean(crisis)),
        "lasting_score": float(np.mean(lasting)),
    }


def greedy_from_network(network):
    """Turn a network that outputs one score per action into choose_action()."""
    def choose_action(obs):
        with torch.no_grad():
            scores = network(torch.as_tensor(obs, dtype=torch.float32))
        return int(scores.argmax())
    return choose_action


def load_agent(config_path):
    """Load a saved agent from its .json settings file and .pt weights file.

    Every agent is a small network: 16 observation numbers in, 8 numbers out
    (one per action). Returns choose_action(observation) -> action number.
    """
    from torchrl.modules import MLP

    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    activation = torch.nn.Tanh if config["activation"] == "tanh" else torch.nn.ReLU
    network = MLP(in_features=16, out_features=8,
                  num_cells=config["hidden_layers"], activation_class=activation)
    network.load_state_dict(torch.load(config_path.with_suffix(".pt")))
    network.eval()
    return greedy_from_network(network), config


def save_log(rows, path):
    """Save a list of dictionaries as a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_run_result(algorithm, result):
    """Save the final result of one run, then rebuild the summary table.

    Each run writes its own small file, so several runs can safely train at
    the same time. The summary CSV is rebuilt from all of those files.
    """
    folder = LOGS_FOLDER / algorithm
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{result['run']}_result.json", "w") as file:
        json.dump(result, file, indent=2)

    results = [json.loads(p.read_text()) for p in sorted(folder.glob("*_result.json"))]
    save_log(results, LOGS_FOLDER / f"{algorithm}_summary.csv")


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(config, file, indent=2)
