"""
Simple hand-made policies used as a comparison for the trained agents.

- random_policy:  picks any action at random
- cash_only_policy: always gives a cash transfer (short term relief only)
- rule_based_policy: a sensible checklist an NGO officer might follow

Each policy takes an observation and returns an action number.
"""

import numpy as np

from environment.custom_env import (
    MONITOR, CASH_TRANSFER, GIVE_ASSET, SKILLS_TRAINING,
    HEALTH_SUPPORT, SAVINGS_GROUP, ASSESSMENT_VISIT, GRADUATE,
    ACTION_COSTS, START_BUDGET,
)


def random_policy(obs, rng):
    return int(rng.integers(0, 8))


def cash_only_policy(obs, rng):
    return CASH_TRANSFER


def rule_based_policy(obs, rng):
    food, income, savings, health = obs[0], obs[1], obs[2], obs[3]
    has_asset, has_training, in_group, has_cover = obs[6], obs[7], obs[8], obs[9]
    weeks_since_visit = obs[10]
    budget_left = obs[12] * START_BUDGET

    # Keep a small reserve so assessment visits are always possible
    reserve = 5.0

    def can_pay(action):
        if action == ASSESSMENT_VISIT:
            return ACTION_COSTS[action] <= budget_left
        return ACTION_COSTS[action] <= budget_left - reserve

    # Graduate when the household looks strong enough (free action)
    ready = food > 0.45 and income > 0.40 and savings > 0.15 and health > 0.50
    if ready:
        return GRADUATE
    # Old record: go and check before deciding
    if weeks_since_visit > 0.9 and can_pay(ASSESSMENT_VISIT):
        return ASSESSMENT_VISIT
    # Emergency first
    if food < 0.30 and can_pay(CASH_TRANSFER):
        return CASH_TRANSFER
    # Then build lasting support step by step
    if not has_asset and can_pay(GIVE_ASSET):
        return GIVE_ASSET
    if not in_group and can_pay(SAVINGS_GROUP):
        return SAVINGS_GROUP
    if not has_training and can_pay(SKILLS_TRAINING):
        return SKILLS_TRAINING
    if not has_cover and can_pay(HEALTH_SUPPORT):
        return HEALTH_SUPPORT
    return MONITOR


BASELINES = {
    "random": random_policy,
    "cash_only": cash_only_policy,
    "rule_based": rule_based_policy,
}


def run_policy(env, policy, episodes=20, seed=0):
    """Run a policy for several episodes and return a list of results."""
    rng = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = False
        while not done:
            action = policy(obs, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append({
            "episode": episode,
            "total_reward": total_reward,
            "weeks": info["week"],
            "final_lasting_score": info["mean_lasting_score"],
            "successful_graduations": info["successful_graduations"],
            "relapses": info["relapses"],
            "households_in_crisis": info["households_in_crisis"],
            "budget_left": info["budget"],
        })
    return results
