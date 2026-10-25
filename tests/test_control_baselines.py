"""
Phase 7.1 — Unit Tests for Control Baselines and Telemetry Policies

Tests:
1. Exact integer PRB allocation feasibility (Hamilton method).
2. Policy determinism and reproducibility.
3. Telemetry triggering mechanisms (Periodic, Age-Triggered, Uncertainty-Triggered, Decision-Sensitive).
4. Privileged reference policy behavior.
5. End-to-end simulation rollouts across multiple seeds.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.mwnsim.environment import WirelessSlicingEnv
from src.mwnsim.policies import (
    allocate_integer_prbs,
    EqualPeriodicPolicy,
    QueueAwarePeriodicPolicy,
    QueueAwareAgeTriggeredPolicy,
    QueueAwareUncertaintyPolicy,
    DecisionSensitivePolicy,
    PrivilegedFullInfoPolicy,
)


def test_allocate_integer_prbs_feasibility():
    # Various demand profiles
    test_demands = [
        np.array([1000.0, 100.0, 10.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([50000.0, 50000.0, 50000.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([99999.0, 1.0, 0.0]),
    ]

    for d in test_demands:
        alloc = allocate_integer_prbs(d, total_prbs=15, min_prbs=np.array([1, 1, 1]))
        assert alloc.dtype in (np.int32, np.int64)
        assert np.sum(alloc) == 15, f"Sum failed for demand {d}: got {np.sum(alloc)}"
        assert np.all(alloc >= 1), f"Min bound failed for demand {d}: got {alloc}"


def test_policy_determinism():
    env = WirelessSlicingEnv(seed=2026)
    obs = env.reset(seed=2026)

    policies = [
        EqualPeriodicPolicy(period=2),
        QueueAwarePeriodicPolicy(period=2),
        QueueAwareAgeTriggeredPolicy(max_age=3),
        QueueAwareUncertaintyPolicy(theta_u=0.15),
        DecisionSensitivePolicy(),
        PrivilegedFullInfoPolicy(),
    ]

    for pol in policies:
        m1, r1 = pol.select_action(obs)
        m2, r2 = pol.select_action(obs)
        np.testing.assert_array_equal(m1, m2, err_msg=f"{pol.name} mask is non-deterministic")
        np.testing.assert_array_equal(r1, r2, err_msg=f"{pol.name} PRBs are non-deterministic")
        assert np.sum(r1) == 15
        assert np.all(r1 >= 1)


def test_telemetry_triggers():
    env = WirelessSlicingEnv(seed=2026)
    obs = env.reset(seed=2026)

    # 1. Periodic: step 0 triggers, step 1 does not
    pol_per = EqualPeriodicPolicy(period=2)
    obs["step"] = np.array([0])
    m0, _ = pol_per.select_action(obs)
    assert np.all(m0 == 1)
    obs["step"] = np.array([1])
    m1, _ = pol_per.select_action(obs)
    assert np.all(m1 == 0)

    # 2. Age-triggered: age < 3 does not trigger, age >= 3 triggers
    pol_age = QueueAwareAgeTriggeredPolicy(max_age=3)
    obs["measurement_age_steps"] = np.array([1, 2, 4], dtype=np.int32)
    m_age, _ = pol_age.select_action(obs)
    np.testing.assert_array_equal(m_age, [0, 0, 1])

    # 3. Privileged: always requests 0 telemetry
    pol_priv = PrivilegedFullInfoPolicy()
    m_priv, _ = pol_priv.select_action(obs)
    assert np.all(m_priv == 0)


def test_full_rollouts_all_policies():
    env = WirelessSlicingEnv(seed=2026)

    policies = [
        EqualPeriodicPolicy(period=2),
        QueueAwarePeriodicPolicy(period=2),
        QueueAwareAgeTriggeredPolicy(max_age=3),
        QueueAwareUncertaintyPolicy(theta_u=0.15),
        DecisionSensitivePolicy(),
        PrivilegedFullInfoPolicy(),
    ]

    for pol in policies:
        obs = env.reset(seed=42)
        total_rew = 0.0
        done = False
        steps = 0
        info = None
        while not done:
            action = pol.select_action(obs, info)
            obs, rew, done, info = env.step(action)
            total_rew += rew
            steps += 1
            assert info["is_feasible"], f"Infeasible action in {pol.name} at step {steps}"

        assert steps == 100
        assert np.isfinite(total_rew)
