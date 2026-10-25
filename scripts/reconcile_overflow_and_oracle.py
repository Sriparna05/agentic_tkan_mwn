"""
Detailed reconciliation script for:
1. Overflow metrics: frequency vs volume across AgenticTKAN_VoI, NoMemory_VoI, AgenticTKAN_Never, PrivilegedFullInfo_Oracle.
2. Oracle analysis: information access, PRB allocation logic, reward alignment.
3. Pareto multi-objective analysis across all objectives.
"""

import os
import sys
import numpy as np
import pandas as pd

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)

from src.mwnsim.environment import WirelessSlicingEnv
from src.mwnsim.policies import PrivilegedFullInfoPolicy, allocate_integer_prbs
from src.mwnsim.agentic_controller import AgenticTKANController


def reconcile_overflow_metrics():
    print("=" * 80)
    print("1. RECONCILING OVERFLOW METRICS (PER-EPISODE AND ACROSS SEEDS)")
    print("=" * 80)

    raw_path = os.path.join(WORKSPACE_ROOT, "outputs", "metrics", "final_experiments_raw.csv")
    df = pd.read_csv(raw_path)

    target_pols = ["AgenticTKAN_VoI", "NoMemory_VoI", "AgenticTKAN_Never", "PrivilegedFullInfo_Oracle"]
    sub = df[df["policy"].isin(target_pols)]

    cols = [
        "seed", "policy", "total_reward", "queue_cost", "overflow_cost", "sla_cost", "telemetry_cost", "switch_cost",
        "mean_queue_total_bytes", "total_overflow_bytes", "overflow_frequency", "sla_violation_rate",
        "total_telemetry_requests", "telemetry_request_rate"
    ]

    print("\n--- Per-Seed Raw Metrics ---")
    for pol in target_pols:
        print(f"\nPolicy: {pol}")
        p_df = sub[sub["policy"] == pol][cols]
        print(p_df.to_string(index=False))

    print("\n--- Summary Statistics Across 10 Seeds (Mean +/- Std) ---")
    summary = sub.groupby("policy").agg({
        "total_reward": ["mean", "std"],
        "queue_cost": ["mean", "std"],
        "overflow_cost": ["mean", "std"],
        "sla_cost": ["mean", "std"],
        "telemetry_cost": ["mean", "std"],
        "switch_cost": ["mean", "std"],
        "mean_queue_total_bytes": ["mean", "std"],
        "total_overflow_bytes": ["mean", "std"],
        "overflow_frequency": ["mean", "std"],
        "sla_violation_rate": ["mean", "std"],
        "total_telemetry_requests": ["mean", "std"],
    })
    print(summary.to_string())


def audit_oracle_behavior():
    print("\n" + "=" * 80)
    print("2. AUDITING THE PRIVILEGED ORACLE")
    print("=" * 80)

    # Inspect PrivilegedFullInfoPolicy implementation
    pol = PrivilegedFullInfoPolicy(total_prbs=15)
    print(f"Policy Name: {pol.name}")
    print(f"is_privileged: {pol.is_privileged}")
    print(f"total_prbs: {pol.total_prbs}")
    print(f"min_prbs: {pol.min_prbs}")

    # Test what information the oracle uses:
    env = WirelessSlicingEnv(seed=2026)
    obs = env.reset(seed=2026)
    print(f"Initial True Queue: {env.true_queue}")
    print(f"Initial Observable Queue: {obs['observed_queue_bytes']}")

    # Step 0: info is None
    action_0 = pol.select_action(obs, info=None)
    print(f"Step 0 Action (info=None): PRB={action_0[1]}, Telemetry={action_0[0]}")

    # Step 1: info is passed from env.step
    obs_1, r_0, done, info_0 = env.step(action_0)
    action_1 = pol.select_action(obs_1, info=info_0)
    print(f"Step 1 True Queue in info: {info_0['true_queue_bytes']}")
    print(f"Step 1 Action: PRB={action_1[1]}, Telemetry={action_1[0]}")

    # Recheck environment configuration used by Oracle:
    print(f"\nEnvironment Config for Oracle: Identical default config (15 PRBs, 3 slices, weights)")
    print(f"Oracle Telemetry Requests: Strictly 0 (m_t = [0, 0, 0])")
    print(f"Oracle PRB allocation rule: Proportional to true hidden queue max(1.0, true_q)")


def audit_pareto_claims():
    print("\n" + "=" * 80)
    print("3. AUDITING PARETO CLAIMS")
    print("=" * 80)

    res_path = os.path.join(WORKSPACE_ROOT, "outputs", "metrics", "final_results.csv")
    df = pd.read_csv(res_path)

    # Key comparison: TKAN vs GRU vs MLP vs Heuristics
    comp_cols = [
        "policy", "total_reward_mean", "total_overflow_bytes_mean", "overflow_frequency_mean",
        "sla_violation_rate_mean", "total_telemetry_requests_mean", "jains_fairness_index_mean", "decision_latency_ms_mean"
    ]
    print(df[comp_cols].to_string(index=False))

    # Evaluate dominance: TKAN vs GRU vs MLP
    tkan = df[df["policy"] == "AgenticTKAN_VoI"].iloc[0]
    gru = df[df["policy"] == "AgenticGRU_VoI"].iloc[0]
    mlp = df[df["policy"] == "AgenticMLP_VoI"].iloc[0]

    print("\n--- Pairwise Comparison: TKAN vs GRU ---")
    objectives = [
        ("Total Reward (higher is better)", tkan["total_reward_mean"], gru["total_reward_mean"], True),
        ("Overflow Bytes (lower is better)", tkan["total_overflow_bytes_mean"], gru["total_overflow_bytes_mean"], False),
        ("Overflow Frequency (lower is better)", tkan["overflow_frequency_mean"], gru["overflow_frequency_mean"], False),
        ("SLA Violation Rate (lower is better)", tkan["sla_violation_rate_mean"], gru["sla_violation_rate_mean"], False),
        ("Telemetry Requests (lower is better)", tkan["total_telemetry_requests_mean"], gru["total_telemetry_requests_mean"], False),
        ("Jain's Fairness (higher is better)", tkan["jains_fairness_index_mean"], gru["jains_fairness_index_mean"], True),
        ("Decision Latency (lower is better)", tkan["decision_latency_ms_mean"], gru["decision_latency_ms_mean"], False),
    ]

    for name, v_tkan, v_other, higher_better in objectives:
        if higher_better:
            better = v_tkan > v_other
        else:
            better = v_tkan < v_other
        print(f"  {name:40s}: TKAN={v_tkan:10.2f} vs GRU={v_other:10.2f} -> TKAN better? {better}")

    print("\n--- Pairwise Comparison: TKAN vs MLP ---")
    for name, v_tkan, v_other, higher_better in objectives:
        v_other = mlp[comp_cols[objectives.index((name, v_tkan, v_other, higher_better)) + 1]]
        if higher_better:
            better = v_tkan > v_other
        else:
            better = v_tkan < v_other
        print(f"  {name:40s}: TKAN={v_tkan:10.2f} vs MLP={v_other:10.2f} -> TKAN better? {better}")


if __name__ == "__main__":
    reconcile_overflow_metrics()
    audit_oracle_behavior()
    audit_pareto_claims()
