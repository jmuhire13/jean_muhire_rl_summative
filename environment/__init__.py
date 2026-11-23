"""Aid Impact Simulator environment package."""

from gymnasium.envs.registration import register

from environment.custom_env import AidImpactEnv, ACTION_NAMES, OBSERVATION_NAMES

# Register the environment so it can be created with gym.make("AidImpact-v0")
register(
    id="AidImpact-v0",
    entry_point="environment.custom_env:AidImpactEnv",
)

__all__ = ["AidImpactEnv", "ACTION_NAMES", "OBSERVATION_NAMES"]
