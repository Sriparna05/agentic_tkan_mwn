"""
Phase 6.1 — Comprehensive Unit Tests for WirelessSlicingEnv

Validates all 10 simulator requirements:
1. Exact PRB feasibility.
2. Queue bounds and finite capacity.
3. Reset behavior and state cleanliness.
4. Deterministic seed reproducibility.
5. Telemetry delay modeling.
6. Report loss modeling.
7. Stale measurement handling.
8. No future/hidden information leakage into controller observations.
9. Reward decomposition and unit consistency.
10. Correct termination behavior.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is importable
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.mwnsim.environment import WirelessSlicingEnv


def test_exact_prb_feasibility():
    env = WirelessSlicingEnv(seed=2026)
    
    # Valid action: sum == 15, >= [1, 1, 1]
    valid_alloc = np.array([7, 5, 3], dtype=np.int32)
    is_feas, msg = env.is_feasible_action(valid_alloc)
    assert is_feas, f"Expected feasible, got: {msg}"
    
    # Invalid actions:
    # 1. Sum != 15
    bad_sum = np.array([5, 5, 4], dtype=np.int32)
    is_feas, msg = env.is_feasible_action(bad_sum)
    assert not is_feas and "does not equal budget" in msg
    
    # 2. Below minimum PRB (min is [1, 1, 1])
    below_min = np.array([15, 0, 0], dtype=np.int32)
    is_feas, msg = env.is_feasible_action(below_min)
    assert not is_feas and "violates minimum PRB" in msg
    
    # 3. Negative PRBs
    negative_prb = np.array([17, -1, -1], dtype=np.int32)
    is_feas, msg = env.is_feasible_action(negative_prb)
    assert not is_feas and "Negative PRB" in msg

    # Test projection onto feasible action space
    projected = env.project_feasible_action(np.array([20, 0, 0]))
    assert np.sum(projected) == 15
    assert np.all(projected >= env.min_prbs)


def test_queue_bounds_and_overflow():
    env = WirelessSlicingEnv(seed=2026)
    env.reset(seed=2026)
    
    # Run 20 steps with 0 PRB for slice 0 to force queue growth and overflow
    for _ in range(20):
        action = (np.array([0, 0, 0]), np.array([1, 7, 7]))
        obs, reward, done, info = env.step(action)
        # Verify queue never exceeds finite capacity
        assert np.all(info["true_queue_bytes"] <= env.queue_capacities)
        assert np.all(info["true_queue_bytes"] >= 0.0)
        assert np.all(info["overflow_bytes"] >= 0.0)


def test_reset_behavior():
    env = WirelessSlicingEnv(seed=2026)
    obs1 = env.reset(seed=2026)
    
    # Execute a few steps
    action = (np.array([1, 1, 1]), np.array([5, 5, 5]))
    env.step(action)
    env.step(action)
    assert env.current_step == 2
    
    # Reset
    obs_reset = env.reset(seed=2026)
    assert env.current_step == 0
    assert len(env.pending_telemetry) == 0
    assert np.all(obs_reset["measurement_age_steps"] == 0)
    assert np.allclose(obs1["observed_queue_bytes"], obs_reset["observed_queue_bytes"])


def test_seed_reproducibility():
    env1 = WirelessSlicingEnv(seed=42)
    env2 = WirelessSlicingEnv(seed=42)
    
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1["observed_queue_bytes"], obs2["observed_queue_bytes"])
    
    action = (np.array([1, 0, 1]), np.array([6, 5, 4]))
    for _ in range(10):
        o1, r1, d1, i1 = env1.step(action)
        o2, r2, d2, i2 = env2.step(action)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)
        np.testing.assert_array_equal(o1["observed_queue_bytes"], o2["observed_queue_bytes"])
        np.testing.assert_array_equal(i1["true_queue_bytes"], i2["true_queue_bytes"])


def test_telemetry_delay():
    env = WirelessSlicingEnv(seed=2026)
    env.reset(seed=2026)
    
    initial_observed = env.latest_measurements.copy()
    
    # Request telemetry on slice 0 only (configured delay: 1 step)
    # Action taken at step 0: delivery scheduled for step 0 + 1 + 1 = step 2!
    action_req = (np.array([1, 0, 0]), np.array([5, 5, 5]))
    obs1, _, _, info1 = env.step(action_req)
    
    # At step 1, report should NOT be delivered yet (delay is 1 step, delivery is at step 2)
    assert obs1["measurement_age_steps"][0] == 1
    assert np.isclose(obs1["observed_queue_bytes"][0], initial_observed[0], rtol=1e-5)
    assert len(env.pending_telemetry) >= 1
    
    # Step 2: no new request, scheduled report should be processed
    action_idle = (np.array([0, 0, 0]), np.array([5, 5, 5]))
    obs2, _, _, info2 = env.step(action_idle)
    # Delivered at step 2 (unless dropped by random loss)
    assert len(env.pending_telemetry) == 0


def test_report_loss_and_stale_handling():
    # Force 100% loss rate for testing
    env = WirelessSlicingEnv(seed=2026)
    env.telemetry_loss_rates = np.array([1.0, 1.0, 1.0])
    env.reset(seed=2026)
    
    initial_obs_val = env.latest_measurements[0]
    
    # Request telemetry on slice 0 with 100% loss
    action_req = (np.array([1, 0, 0]), np.array([5, 5, 5]))
    env.step(action_req)
    env.step(action_req)
    obs3, _, _, _ = env.step((np.array([0, 0, 0]), np.array([5, 5, 5])))
    
    # Report was dropped: observed value must be preserved as stale
    assert np.isclose(obs3["observed_queue_bytes"][0], initial_obs_val, rtol=1e-5)
    # Age must have incremented each step
    assert obs3["measurement_age_steps"][0] == 3



def test_no_future_information_leakage():
    env = WirelessSlicingEnv(seed=2026)
    obs = env.reset(seed=2026)
    
    # Verify controller observation dictionary keys
    allowed_keys = {
        "observed_queue_bytes",
        "measurement_age_steps",
        "previous_prb_allocation",
        "previous_telemetry_mask",
        "step"
    }
    assert set(obs.keys()) == allowed_keys
    
    # Ensure hidden state is NOT in obs
    forbidden_keys = {"true_queue_bytes", "arrivals_bytes", "service_efficiencies", "overflow_bytes", "dwell_time_steps"}
    for k in forbidden_keys:
        assert k not in obs


def test_reward_decomposition():
    env = WirelessSlicingEnv(seed=2026)
    env.reset(seed=2026)
    
    action = (np.array([1, 0, 1]), np.array([6, 5, 4]))
    obs, total_reward, done, info = env.step(action)
    
    components = info["reward_components"]
    expected_components = [
        "queue_cost", "overflow_cost", "sla_violation_cost",
        "telemetry_cost", "switch_cost", "infeasibility_penalty"
    ]
    for c in expected_components:
        assert c in components
        assert isinstance(components[c], float)
        assert components[c] <= 0.0, f"Cost term {c} should be negative penalty"
        
    # Verify sum equals total_reward
    comp_sum = sum(components.values())
    assert abs(comp_sum - total_reward) < 1e-6


def test_termination_behavior():
    env = WirelessSlicingEnv(seed=2026)
    env.max_steps = 10
    env.reset(seed=2026)
    
    action = (np.array([0, 0, 0]), np.array([5, 5, 5]))
    done = False
    step_count = 0
    while not done:
        _, _, done, _ = env.step(action)
        step_count += 1
        
    assert done
    assert step_count == 10
    assert env.current_step == 10
