"""Basic checks that the environment follows the rules we expect."""

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import environment  # registers AidImpact-v0
from environment.custom_env import AidImpactEnv, CASH_TRANSFER, GIVE_ASSET, GRADUATE, PROGRAM_WEEKS
from training.baselines import BASELINES, run_policy


def test_gymnasium_api():
    # Gymnasium's own checker for spaces, reset and step
    check_env(AidImpactEnv(), skip_render_check=True)


def test_registered_name_works():
    env = gym.make("AidImpact-v0")
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)


def test_same_seed_gives_same_episode():
    rewards = []
    for _ in range(2):
        env = AidImpactEnv()
        env.reset(seed=42)
        total = 0.0
        for week in range(50):
            _, reward, terminated, truncated, _ = env.step(week % 8)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    assert rewards[0] == rewards[1]


def test_budget_never_goes_below_zero():
    env = AidImpactEnv()
    env.reset(seed=3)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(GIVE_ASSET)
        assert info["budget"] >= 0
        done = terminated or truncated


def test_episode_ends_by_program_end():
    env = AidImpactEnv()
    env.reset(seed=5)
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        done = terminated or truncated
    assert steps <= PROGRAM_WEEKS


def test_cash_transfer_raises_food_and_dependency():
    env = AidImpactEnv()
    env.reset(seed=7)
    h = env.current
    food_before = env.food[h]
    env._apply_action(h, CASH_TRANSFER)
    assert env.food[h] > food_before
    assert env.dependency[h] > 0


def test_graduating_everyone_does_not_end_the_program_early():
    # Graduating households that are not ready must not be a shortcut:
    # life goes on, they relapse and come back, and the program continues.
    env = AidImpactEnv()
    env.reset(seed=11)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(GRADUATE)
        done = terminated or truncated
    assert info["week"] == PROGRAM_WEEKS
    assert info["relapses"] > 0


def test_graduating_everyone_scores_worse_than_rule_based():
    env = AidImpactEnv()
    always_graduate = run_policy(env, lambda obs, rng: GRADUATE, episodes=10, seed=200)
    rule_based = run_policy(env, BASELINES["rule_based"], episodes=10, seed=200)
    assert np.mean([r["total_reward"] for r in always_graduate]) < np.mean([r["total_reward"] for r in rule_based])


def test_rule_based_beats_cash_only_and_random():
    # The reward should favour lasting support over handouts
    env = AidImpactEnv()
    scores = {}
    for name, policy in BASELINES.items():
        results = run_policy(env, policy, episodes=20, seed=100)
        scores[name] = np.mean([r["total_reward"] for r in results])
    assert scores["rule_based"] > scores["cash_only"]
    assert scores["rule_based"] > scores["random"]
