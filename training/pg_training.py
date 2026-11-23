"""
Policy gradient training with TorchRL: REINFORCE, PPO and A2C.

The difference from DQN, in short:
  DQN learns a score for each action and picks the highest.
  Policy gradient methods learn the policy directly: a network that gives the
  probability of each action. Actions that led to good results become more
  likely, actions that led to bad results become less likely.

  REINFORCE: waits until a program (episode) ends, then uses the real total
             future reward of each step. Simple, but noisy.
  A2C:       an actor (the policy) and a critic (estimates how good a
             situation is). Learns from short pieces of experience, without
             waiting for the program to end.
  PPO:       like A2C, but limits how much the policy may change in one
             update (clipping), which keeps learning stable.

Usage:
    uv run python -m training.pg_training --algo ppo --run ppo_01_baseline
    uv run python -m training.pg_training --algo a2c --all
    uv run python -m training.pg_training --algo reinforce --list
"""

import argparse
import json
import time

import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from training.common import (
    LOGS_FOLDER, MODELS_FOLDER, evaluate, greedy_from_network, make_env,
    save_config, save_log, save_run_result,
)

# ---------------------------------------------------------------------------
# Hyperparameters: baseline settings and experiments for each algorithm
# ---------------------------------------------------------------------------
SHARED = {
    "hidden_layers": [128, 128],
    "total_frames": 150_000,
    "reward_scale": 0.1,
    "max_grad_norm": 1.0,
}

BASELINES = {
    "reinforce": {
        **SHARED,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "episodes_per_batch": 5,     # full programs collected before each update
        "use_baseline": True,        # subtract a learned value estimate (less noise)
        "normalize_returns": True,
        "entropy_coeff": 0.01,       # bonus for keeping the policy random enough to explore
    },
    "ppo": {
        **SHARED,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,          # balance between short and long term advantage estimates
        "clip_epsilon": 0.2,         # how far the policy may move in one update
        "epochs": 4,                 # passes over each rollout
        "minibatch_size": 256,
        "rollout_steps": 1040,       # steps collected before each update (10 programs)
        "entropy_coeff": 0.01,
        "critic_coeff": 0.5,
        "normalize_advantage": True,
    },
    "a2c": {
        **SHARED,
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "gae_lambda": 1.0,           # 1.0 = classic n-step returns
        "rollout_steps": 32,         # A2C updates after every 32 steps
        "entropy_coeff": 0.01,
        "critic_coeff": 0.5,
    },
}

EXPERIMENTS = {
    "reinforce": {
        "reinforce_01_baseline":      {},
        "reinforce_02_lr_low":        {"learning_rate": 1e-4},
        "reinforce_03_lr_high":       {"learning_rate": 5e-3},
        "reinforce_04_gamma_090":     {"gamma": 0.90},
        "reinforce_05_gamma_0999":    {"gamma": 0.999},
        "reinforce_06_no_baseline":   {"use_baseline": False},
        "reinforce_07_no_normalize":  {"normalize_returns": False},
        "reinforce_08_no_entropy":    {"entropy_coeff": 0.0},
        "reinforce_09_high_entropy":  {"entropy_coeff": 0.05},
        "reinforce_10_batch_1":       {"episodes_per_batch": 1},
        "reinforce_11_batch_20":      {"episodes_per_batch": 20},
        "reinforce_12_small_network": {"hidden_layers": [64]},
    },
    "ppo": {
        "ppo_01_baseline":       {},
        "ppo_02_lr_low":         {"learning_rate": 1e-4},
        "ppo_03_lr_high":        {"learning_rate": 1e-3},
        "ppo_04_clip_010":       {"clip_epsilon": 0.1},
        "ppo_05_clip_030":       {"clip_epsilon": 0.3},
        "ppo_06_epochs_10":      {"epochs": 10},
        "ppo_07_rollout_short":  {"rollout_steps": 520},
        "ppo_08_rollout_long":   {"rollout_steps": 2080},
        "ppo_09_lambda_080":     {"gae_lambda": 0.80},
        "ppo_10_lambda_100":     {"gae_lambda": 1.0},
        "ppo_11_no_entropy":     {"entropy_coeff": 0.0},
        "ppo_12_high_entropy":   {"entropy_coeff": 0.05},
        "ppo_13_gamma_090":      {"gamma": 0.90},
    },
    "a2c": {
        "a2c_01_baseline":       {},
        "a2c_02_lr_low":         {"learning_rate": 1e-4},
        "a2c_03_lr_high":        {"learning_rate": 3e-3},
        "a2c_04_steps_8":        {"rollout_steps": 8},
        "a2c_05_steps_128":      {"rollout_steps": 128},
        "a2c_06_lambda_095":     {"gae_lambda": 0.95},
        "a2c_07_no_entropy":     {"entropy_coeff": 0.0},
        "a2c_08_high_entropy":   {"entropy_coeff": 0.05},
        "a2c_09_critic_10":      {"critic_coeff": 1.0},
        "a2c_10_gamma_090":      {"gamma": 0.90},
        "a2c_11_small_network":  {"hidden_layers": [64]},
    },
}

EVAL_EVERY_FRAMES = 10_400   # test the greedy agent every 100 programs


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def build_actor(env, hidden_layers):
    """Policy network: 16 observation numbers in, 8 action scores (logits) out.
    The scores are turned into probabilities, and an action is sampled."""
    network = MLP(in_features=16, out_features=8, num_cells=hidden_layers,
                  activation_class=torch.nn.Tanh)
    module = TensorDictModule(network, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(module, in_keys=["logits"], out_keys=["action"],
                               spec=env.action_spec, distribution_class=Categorical,
                               return_log_prob=True)
    return actor, network


def build_critic(hidden_layers):
    """Value network (critic): 16 observation numbers in, 1 number out:
    how much future reward it expects from this situation."""
    network = MLP(in_features=16, out_features=1, num_cells=hidden_layers,
                  activation_class=torch.nn.Tanh)
    return ValueOperator(network, in_keys=["observation"])


# ---------------------------------------------------------------------------
# REINFORCE update
# ---------------------------------------------------------------------------
def discounted_returns(rewards, dones, gamma):
    """Total future reward from each step until the end of its program.

    We walk backwards through the batch. When a program ends (done), the
    running total starts again from zero.
    Steps of a program that did not finish inside the batch are marked
    invalid, because their full return is unknown.
    """
    returns = torch.zeros_like(rewards)
    valid = torch.zeros_like(rewards, dtype=torch.bool)
    running = 0.0
    seen_end = False
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running = 0.0
            seen_end = True
        running = rewards[t] + gamma * running
        returns[t] = running
        valid[t] = seen_end
    return returns, valid


def reinforce_update(data, actor_net, critic, optimizer, config):
    rewards = data["next", "reward"].squeeze(-1)
    dones = data["next", "done"].squeeze(-1)
    returns, valid = discounted_returns(rewards, dones, config["gamma"])

    observations = data["observation"][valid]
    actions = data["action"][valid]
    returns = returns[valid]

    # Probability the policy now gives to the actions it took
    distribution = Categorical(logits=actor_net(observations))
    log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()

    # Advantage: how much better than expected each step turned out
    critic_loss = torch.tensor(0.0)
    if config["use_baseline"]:
        values = critic(data.select("observation")[valid])["state_value"].squeeze(-1)
        advantages = returns - values.detach()
        critic_loss = 0.5 * ((returns - values) ** 2).mean()
    else:
        advantages = returns
    if config["normalize_returns"]:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Make good actions more likely, bad actions less likely
    policy_loss = -(log_probs * advantages).mean()
    loss = policy_loss - config["entropy_coeff"] * entropy + critic_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], config["max_grad_norm"])
    optimizer.step()
    return {"policy_loss": policy_loss.item(), "critic_loss": critic_loss.item(),
            "entropy": entropy.item()}


# ---------------------------------------------------------------------------
# A2C and PPO update (both use TorchRL loss modules and GAE advantages)
# ---------------------------------------------------------------------------
def actor_critic_update(data, loss_module, advantage_module, optimizer, config, algorithm):
    with torch.no_grad():
        advantage_module(data)  # adds "advantage" and "value_target" to the data

    if algorithm == "a2c":
        epochs, minibatch = 1, len(data)  # A2C: one update on the whole short rollout
    else:
        epochs, minibatch = config["epochs"], config["minibatch_size"]

    logs = {"policy_loss": [], "critic_loss": [], "entropy": []}
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for start in range(0, len(data), minibatch):
            batch = data[order[start:start + minibatch]]
            out = loss_module(batch)
            loss = out["loss_objective"] + out["loss_critic"]
            # When entropy_coeff is 0, TorchRL's A2C loss leaves out the entropy terms
            if "loss_entropy" in out.keys():
                loss = loss + out["loss_entropy"]
                entropy = out["entropy"].item()
            else:
                # still measure the entropy so it can be plotted
                with torch.no_grad():
                    entropy = loss_module.actor_network.get_dist(batch).entropy().mean().item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config["max_grad_norm"])
            optimizer.step()
            logs["policy_loss"].append(out["loss_objective"].item())
            logs["critic_loss"].append(out["loss_critic"].item())
            logs["entropy"].append(entropy)
    return {key: sum(values) / len(values) for key, values in logs.items()}


# ---------------------------------------------------------------------------
# Training loop (same for all three algorithms)
# ---------------------------------------------------------------------------
def train(algorithm, run_name, changes, seed=0):
    config = {**BASELINES[algorithm], **changes, "run": run_name, "seed": seed,
              "algorithm": algorithm, "activation": "tanh"}
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    env = make_env(seed=seed, reward_scale=config["reward_scale"])
    actor, actor_net = build_actor(env, config["hidden_layers"])
    critic = build_critic(config["hidden_layers"])

    if algorithm == "reinforce":
        steps_per_batch = config["episodes_per_batch"] * 104
        parameters = list(actor_net.parameters()) + list(critic.parameters())
    else:
        steps_per_batch = config["rollout_steps"]
        advantage_module = GAE(gamma=config["gamma"], lmbda=config["gae_lambda"],
                               value_network=critic, average_gae=False)
        if algorithm == "ppo":
            loss_module = ClipPPOLoss(actor, critic, clip_epsilon=config["clip_epsilon"],
                                      entropy_coeff=config["entropy_coeff"],
                                      critic_coeff=config["critic_coeff"],
                                      normalize_advantage=config["normalize_advantage"])
        else:
            loss_module = A2CLoss(actor, critic, entropy_coeff=config["entropy_coeff"],
                                  critic_coeff=config["critic_coeff"])
        parameters = loss_module.parameters()

    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    collector = Collector(env, actor, frames_per_batch=steps_per_batch,
                          total_frames=config["total_frames"])

    log_rows = []
    frames = 0
    next_eval = EVAL_EVERY_FRAMES
    best_eval = -float("inf")
    start_time = time.time()
    choose_action = greedy_from_network(actor_net)
    recent_rewards = []  # finished programs since the last log line

    for data in collector:
        frames += data.numel()

        if algorithm == "reinforce":
            stats = reinforce_update(data, actor_net, critic, optimizer, config)
        else:
            stats = actor_critic_update(data, loss_module, advantage_module, optimizer,
                                        config, algorithm)

        finished = data["next", "done"].squeeze(-1)
        recent_rewards += data["next", "episode_reward"][finished].flatten().tolist()

        # A2C updates very often, so we only write a log line about every 1,040 steps
        if frames % 1040 >= steps_per_batch and frames < config["total_frames"] and frames < next_eval:
            continue

        row = {
            "frames": frames,
            "policy_loss": round(stats["policy_loss"], 5),
            "critic_loss": round(stats["critic_loss"], 5),
            "entropy": round(stats["entropy"], 4),
            "train_reward": round(sum(recent_rewards) / len(recent_rewards), 3) if recent_rewards else "",
            "eval_reward": "",
        }
        recent_rewards = []

        if frames >= next_eval or frames >= config["total_frames"]:
            next_eval += EVAL_EVERY_FRAMES
            results = evaluate(choose_action)
            row["eval_reward"] = round(results["eval_reward_mean"], 3)
            if results["eval_reward_mean"] > best_eval:
                best_eval = results["eval_reward_mean"]
                torch.save(actor_net.state_dict(), MODELS_FOLDER / "pg" / f"{run_name}.pt")
            print(f"[{run_name}] frames {frames:7d} | entropy {row['entropy']:.3f} | "
                  f"train reward {row['train_reward']} | eval reward {row['eval_reward']:.2f}")

        log_rows.append(row)

    collector.shutdown()
    minutes = (time.time() - start_time) / 60

    # Final test of the best saved model
    actor_net.load_state_dict(torch.load(MODELS_FOLDER / "pg" / f"{run_name}.pt"))
    final = evaluate(choose_action)

    save_log(log_rows, LOGS_FOLDER / algorithm / f"{run_name}.csv")
    save_config(config, MODELS_FOLDER / "pg" / f"{run_name}.json")
    changes_text = ", ".join(f"{k}={v}" for k, v in changes.items()) or "baseline"
    save_run_result(algorithm, {"run": run_name, "changes": changes_text, **final,
                                "best_eval_during_training": round(best_eval, 3),
                                "minutes": round(minutes, 1)})

    print(f"[{run_name}] done in {minutes:.1f} min | eval reward {final['eval_reward_mean']:.2f} "
          f"| graduations {final['graduations']:.2f} | relapses {final['relapses']:.2f} "
          f"| in crisis {final['in_crisis']:.2f}")
    return final


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE, PPO or A2C")
    parser.add_argument("--algo", required=True, choices=["reinforce", "ppo", "a2c"])
    parser.add_argument("--run", nargs="+", help="one or more experiment names")
    parser.add_argument("--all", action="store_true", help="run every experiment")
    parser.add_argument("--list", action="store_true", help="show the experiments")
    parser.add_argument("--frames", type=int, help="override total frames (for quick tests)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override settings for a quick test")
    parser.add_argument("--name", help="custom run name for a quick test")
    args = parser.parse_args()

    experiments = EXPERIMENTS[args.algo]
    if args.list or not (args.run or args.all):
        for name, changes in experiments.items():
            print(f"{name:28s} {changes or 'baseline'}")
        return

    names = list(experiments) if args.all else args.run
    for name in names:
        changes = dict(experiments[name])
        if args.frames:
            changes["total_frames"] = args.frames
        for item in args.set:
            key, value = item.split("=")
            changes[key] = json.loads(value) if value[0] in "0123456789-[tf" else value
        train(args.algo, args.name or name, changes)


if __name__ == "__main__":
    main()
