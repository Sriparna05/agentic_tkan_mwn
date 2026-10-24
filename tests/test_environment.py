"""
Tests for MobileWirelessEnv and baseline policies.
"""

import sys
from pathlib import Path
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

# Ensure src is importable
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from mwnsim.environment import MobileWirelessEnv
from mwnsim.policies import get_policy


def test_environment_initialization():
    env = MobileWirelessEnv(n_base_stations=3, n_users=10, max_steps=50)
    assert env.n_base_stations == 3
    assert env.n_users == 10
    assert env.current_step == 0


def test_environment_reset():
    env = MobileWirelessEnv(n_base_stations=2, n_users=5)
    obs = env.reset()
    assert "channel_gains" in obs
    assert "user_demands" in obs
    assert obs["channel_gains"].shape == (2, 5)
    assert obs["user_demands"].shape == (5,)


def test_environment_step():
    env = MobileWirelessEnv(n_base_stations=3, n_users=6, max_steps=10)
    obs = env.reset()
    action = np.zeros(6, dtype=int)
    next_obs, reward, done, info = env.step(action)

    assert isinstance(reward, float)
    assert not done
    assert "total_throughput_mbps" in info
    assert env.current_step == 1


def test_policies_execution():
    env = MobileWirelessEnv(n_base_stations=3, n_users=6, max_steps=5)
    obs = env.reset()

    for policy_name in ["random", "round_robin", "max_snr", "proportional_fair"]:
        policy = get_policy(policy_name)
        action = policy.select_action(obs, env)
        assert action.shape == (6,)
        assert np.all(action >= 0) and np.all(action < 3)


if __name__ == "__main__":
    test_environment_initialization()
    test_environment_reset()
    test_environment_step()
    test_policies_execution()
    print("All tests passed successfully!")

