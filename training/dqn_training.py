"""
DQN (Deep Q-Network) training with TorchRL on the Aid Impact environment.

How DQN learns, in short:
  - A neural network gives a score (Q-value) to each of the 8 actions.
    The score means "how much total reward I expect if I take this action now".
  - The officer usually picks the action with the highest score, but with
    probability epsilon it tries a random action to explore.
  - Every experience (observation, action, reward, next observation) is stored
    in a replay buffer. The network learns from random batches of old
    experiences, which makes learning more stable.
  - A second copy of the network (the target network) is updated slowly and
    is used to compute the learning targets, which also adds stability.

Usage:
    uv run python -m training.dqn_training --run dqn_01_baseline
    uv run python -m training.dqn_training --all
    uv run python -m training.dqn_training --list
"""

import argparse
import json
import time

import torch
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from tensordict.nn import TensorDictSequential

from training.common import (
    LOGS_FOLDER, MODELS_FOLDER, evaluate, greedy_from_network, make_env,
    save_config, save_log, save_run_result,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# The baseline settings. Every experiment below changes only one or two of
# them, so we can see the effect of each change.
BASELINE = {
    "learning_rate": 1e-3,
    "gamma": 0.99,               # how much future rewards count
    "batch_size": 128,           # experiences used in one learning step
    "buffer_size": 50_000,       # how many experiences the replay buffer keeps
    "eps_start": 1.0,            # start fully random
    "eps_end": 0.05,             # end with 5% random actions
    "eps_decay_fraction": 0.5,   # epsilon reaches eps_end after 50% of training
    "target_update": "hard",     # "hard" = copy every N steps, "soft" = blend slowly
    "target_update_interval": 500,
    "target_tau": 0.01,          # only used with soft updates
    "double_dqn": False,
    "loss_function": "smooth_l1",  # Huber loss: gentler on big errors (l2 = squared error, tested in dqn_16)
    "reward_scale": 1.0,         # rewards are multiplied by this during training only
    "hidden_layers": [128, 128],
    "total_frames": 150_000,     # environment steps (about 1,440 programs)
    "frames_per_batch": 520,     # steps collected before each learning phase (5 programs)
    "updates_per_batch": 64,     # learning steps after each collection
    "random_start_frames": 2_000,
}

# Each experiment = baseline + changes
EXPERIMENTS = {
    "dqn_01_baseline":       {},
    "dqn_02_lr_low":         {"learning_rate": 1e-4},
    "dqn_03_lr_high":        {"learning_rate": 5e-3},
    "dqn_04_gamma_090":      {"gamma": 0.90},
    "dqn_05_gamma_0999":     {"gamma": 0.999},
    "dqn_06_batch_32":       {"batch_size": 32},
    "dqn_07_small_buffer":   {"buffer_size": 5_000},
    "dqn_08_eps_end_001":    {"eps_end": 0.01},
    "dqn_09_eps_end_020":    {"eps_end": 0.20},
    "dqn_10_fast_decay":     {"eps_decay_fraction": 0.1},
    "dqn_11_slow_decay":     {"eps_decay_fraction": 0.9},
    "dqn_12_target_freq_50": {"target_update_interval": 50},
    "dqn_13_soft_target":    {"target_update": "soft", "target_tau": 0.01},
    "dqn_14_double_dqn":     {"double_dqn": True},
    "dqn_15_small_network":  {"hidden_layers": [64]},
    "dqn_16_l2_loss":        {"loss_function": "l2"},
    "dqn_17_reward_scale":   {"reward_scale": 0.1},
}

EVAL_EVERY_FRAMES = 10_400   # test the greedy agent every 100 programs


def train(run_name, changes, seed=0, quiet=False):
    config = {**BASELINE, **changes, "run": run_name, "seed": seed, "algorithm": "dqn", "activation": "relu"}
    torch.manual_seed(seed)
    torch.set_num_threads(2)  # small network: 2 CPU threads is enough, lets runs share the CPU

    # --- Environment ---
    env = make_env(seed=seed, reward_scale=config["reward_scale"])
    num_actions = env.action_spec.space.n
    num_inputs = env.observation_spec["observation"].shape[-1]

    # --- Q-network: 16 observation numbers in, 8 action scores out ---
    q_network = MLP(in_features=num_inputs, out_features=num_actions,
                    num_cells=config["hidden_layers"], activation_class=torch.nn.ReLU)
    actor = QValueActor(q_network, in_keys=["observation"], spec=env.action_spec)

    # --- Exploration: epsilon-greedy ---
    decay_steps = int(config["eps_decay_fraction"] * config["total_frames"])
    exploration = EGreedyModule(env.action_spec, eps_init=config["eps_start"],
                                eps_end=config["eps_end"], annealing_num_steps=decay_steps)
    explore_policy = TensorDictSequential(actor, exploration)

    # --- Data collector: runs the environment with the exploring policy ---
    collector = Collector(env, explore_policy,
                          frames_per_batch=config["frames_per_batch"],
                          total_frames=config["total_frames"],
                          init_random_frames=config["random_start_frames"])

    # --- Replay buffer ---
    buffer = ReplayBuffer(storage=LazyTensorStorage(config["buffer_size"]),
                          batch_size=config["batch_size"])

    # --- Loss, target network and optimizer ---
    loss_module = DQNLoss(actor, action_space=env.action_spec, delay_value=True,
                          double_dqn=config["double_dqn"], loss_function=config["loss_function"])
    loss_module.make_value_estimator(gamma=config["gamma"])
    if config["target_update"] == "hard":
        target_updater = HardUpdate(loss_module, value_network_update_interval=config["target_update_interval"])
    else:
        target_updater = SoftUpdate(loss_module, tau=config["target_tau"])
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config["learning_rate"])

    # --- Training loop ---
    log_rows = []
    frames = 0
    next_eval = EVAL_EVERY_FRAMES
    best_eval = -float("inf")
    start_time = time.time()
    choose_action = greedy_from_network(q_network)

    for data in collector:
        frames += data.numel()
        buffer.extend(data.reshape(-1))

        # learn from random batches of past experience
        losses = []
        if frames >= config["random_start_frames"]:
            for _ in range(config["updates_per_batch"]):
                batch = buffer.sample()
                loss = loss_module(batch)["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
                optimizer.step()
                target_updater.step()
                losses.append(loss.item())

        exploration.step(data.numel())  # lower epsilon a little

        # rewards of the programs that finished in this batch
        finished = data["next", "done"].squeeze(-1)
        episode_rewards = data["next", "episode_reward"][finished]

        row = {
            "frames": frames,
            "epsilon": round(float(exploration.eps), 4),
            "loss": round(sum(losses) / len(losses), 5) if losses else "",
            "train_reward": round(float(episode_rewards.mean()), 3) if len(episode_rewards) else "",
            "eval_reward": "",
        }

        # test the greedy agent (no exploration) on the fixed evaluation programs
        if frames >= next_eval or frames >= config["total_frames"]:
            next_eval += EVAL_EVERY_FRAMES
            results = evaluate(choose_action)
            row["eval_reward"] = round(results["eval_reward_mean"], 3)
            if results["eval_reward_mean"] > best_eval:
                best_eval = results["eval_reward_mean"]
                torch.save(q_network.state_dict(), MODELS_FOLDER / "dqn" / f"{run_name}.pt")
            if not quiet:
                print(f"[{run_name}] frames {frames:7d} | epsilon {row['epsilon']:.3f} | "
                      f"loss {row['loss']} | train reward {row['train_reward']} | "
                      f"eval reward {row['eval_reward']:.2f}")

        log_rows.append(row)

    collector.shutdown()
    minutes = (time.time() - start_time) / 60

    # --- Final test of the best saved model ---
    q_network.load_state_dict(torch.load(MODELS_FOLDER / "dqn" / f"{run_name}.pt"))
    final = evaluate(choose_action)

    save_log(log_rows, LOGS_FOLDER / "dqn" / f"{run_name}.csv")
    save_config(config, MODELS_FOLDER / "dqn" / f"{run_name}.json")
    summary = {"run": run_name, **changes_to_text(changes), **final,
               "best_eval_during_training": round(best_eval, 3), "minutes": round(minutes, 1)}
    save_run_result("dqn", summary)

    print(f"[{run_name}] done in {minutes:.1f} min | eval reward {final['eval_reward_mean']:.2f} "
          f"| graduations {final['graduations']:.2f} | relapses {final['relapses']:.2f} "
          f"| in crisis {final['in_crisis']:.2f}")
    return final


def changes_to_text(changes):
    text = ", ".join(f"{k}={v}" for k, v in changes.items()) or "baseline"
    return {"changes": text}


def main():
    parser = argparse.ArgumentParser(description="Train DQN on the Aid Impact environment")
    parser.add_argument("--run", nargs="+", help="name of one or more experiments to run")
    parser.add_argument("--all", action="store_true", help="run every experiment")
    parser.add_argument("--list", action="store_true", help="show the experiments")
    parser.add_argument("--frames", type=int, help="override total frames (for quick tests)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override settings for a quick test, e.g. --set learning_rate=0.0005")
    parser.add_argument("--name", help="custom run name for a quick test")
    args = parser.parse_args()

    if args.list or not (args.run or args.all):
        for name, changes in EXPERIMENTS.items():
            print(f"{name:24s} {changes_to_text(changes)['changes']}")
        return

    names = list(EXPERIMENTS) if args.all else args.run
    for name in names:
        changes = dict(EXPERIMENTS[name])
        if args.frames:
            changes["total_frames"] = args.frames
        for item in args.set:
            key, value = item.split("=")
            changes[key] = json.loads(value) if value[0] in "0123456789-[tf" else value
        train(args.name or name, changes)


if __name__ == "__main__":
    main()
