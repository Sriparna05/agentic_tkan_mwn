"""
Baseline and Agentic Policies for Mobile Wireless Network Resource Allocation.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class BasePolicy(ABC):
    """Abstract base class for resource allocation policies."""

    @abstractmethod
    def select_action(self, obs: Dict[str, np.ndarray], env) -> np.ndarray:
        """Returns array of BS associations for each user."""
        pass


class RandomPolicy(BasePolicy):
    """Assigns users to base stations randomly."""

    def select_action(self, obs: Dict[str, np.ndarray], env) -> np.ndarray:
        return env.rng.integers(0, env.n_base_stations, size=env.n_users)


class RoundRobinPolicy(BasePolicy):
    """Assigns users across base stations cyclically."""

    def __init__(self):
        self.step = 0

    def select_action(self, obs: Dict[str, np.ndarray], env) -> np.ndarray:
        actions = np.array([(self.step + i) % env.n_base_stations for i in range(env.n_users)])
        self.step += 1
        return actions


class MaxSNRPolicy(BasePolicy):
    """Assigns each user to the base station with the strongest channel gain."""

    def select_action(self, obs: Dict[str, np.ndarray], env) -> np.ndarray:
        gains = obs["channel_gains"]  # shape (n_bs, n_users)
        best_bs = np.argmax(gains, axis=0)
        return best_bs


class ProportionalFairPolicy(BasePolicy):
    """Proportional fair heuristic considering demand and channel gain."""

    def select_action(self, obs: Dict[str, np.ndarray], env) -> np.ndarray:
        gains = obs["channel_gains"]
        demands = obs["user_demands"]
        # Weight channel gain by normalized user demand
        weighted_gains = gains * (demands[None, :] / (np.mean(demands) + 1e-6))
        return np.argmax(weighted_gains, axis=0)


def get_policy(name: str) -> BasePolicy:
    registry = {
        "random": RandomPolicy,
        "round_robin": RoundRobinPolicy,
        "max_snr": MaxSNRPolicy,
        "proportional_fair": ProportionalFairPolicy,
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(f"Unknown policy '{name}'. Available policies: {list(registry.keys())}")
    return registry[key]()
