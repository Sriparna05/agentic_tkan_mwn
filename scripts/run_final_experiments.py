"""
Phase 9.1 — Final Controlled Experiments and Statistical Analysis Runner

Executes rigorous benchmarking across:
1. Proposed Controller:
   - AgenticTKAN_VoI
2. Model Ablations (Matched VoI Acquisition):
   - AgenticGRU_VoI
   - AgenticMLP_VoI
   - NoMemory_VoI
3. Telemetry Mode Ablations (TKAN Predictor Fixed):
   - AgenticTKAN_PeriodicP2
   - AgenticTKAN_PeriodicP4
   - AgenticTKAN_AgeT3
   - AgenticTKAN_UncertaintyTh0.15
   - AgenticTKAN_Always
   - AgenticTKAN_Never
4. Approved Control Baselines:
   - EqualAllocation (Open-loop static equal PRBs)
   - EqualPeriodic_P2 (Static equal PRBs, periodic telemetry)
   - QueueAwarePeriodic_P2 (Queue-proportional PRBs, periodic telemetry)
   - QueueAwareAge_T3 (Queue-proportional PRBs, age-triggered telemetry)
   - QueueAwareUncertainty_Th0.15 (Queue-proportional PRBs, uncertainty-triggered telemetry)
   - DecisionSensitive (Allocation sensitivity & overflow proxy telemetry)
   - PrivilegedFullInfo_Oracle (Privileged reference benchmark)

Statistical Rigor:
- 10 held-out evaluation seeds: [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
- Separate tuning seeds: [101, 102, 103, 104, 105]
- Mean, standard deviation, and Student's t 95% confidence intervals.
- Throughput and Jain's fairness index.
- Batch-one decision latency.
- Comprehensive sensitivity sweeps across delay, loss rate, traffic bursts, queue capacity, and VoI threshold.
"""

import os
import sys
import time
import copy
import json
import subprocess
import platform
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mwnsim.environment import WirelessSlicingEnv
from src.mwnsim.policies import (
    EqualAllocationPolicy,
    EqualPeriodicPolicy,
    QueueAwarePeriodicPolicy,
    QueueAwareAgeTriggeredPolicy,
    QueueAwareUncertaintyPolicy,
    DecisionSensitivePolicy,
    PrivilegedFullInfoPolicy,
)
from src.mwnsim.agentic_controller import AgenticTKANController

EVAL_SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_ci95(data: np.ndarray) -> float:
    """Computes two-sided 95% confidence interval half-width using Student's t distribution."""
    n = len(data)
    if n < 2:
        return 0.0
    sem = st.sem(data, ddof=1)
    if sem == 0 or np.isnan(sem):
        return 0.0
    return float(st.t.ppf(0.975, df=n - 1) * sem)


def compute_jains_fairness(throughputs: np.ndarray) -> float:
    """Computes Jain's Fairness Index: (sum(T_k))^2 / (K * sum(T_k^2))."""
    k = len(throughputs)
    sum_t = np.sum(throughputs)
    sum_sq = np.sum(throughputs ** 2)
    if sum_sq <= 1e-9:
        return 1.0 / k
    return float((sum_t ** 2) / (k * sum_sq))


def run_episode(env: WirelessSlicingEnv, policy, seed: int):
    """Executes a single episode and returns granular trajectory and summary metrics."""
    obs = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    reward_components_sum = {
        "queue_cost": 0.0,
        "overflow_cost": 0.0,
        "sla_violation_cost": 0.0,
        "telemetry_cost": 0.0,
        "switch_cost": 0.0,
        "infeasibility_penalty": 0.0,
    }

    queue_history = []
    overflow_history = []
    throughput_history = []
    prb_history = []
    switch_history = []
    decision_times = []

    telemetry_requested_count = 0
    delivered_reports_count = 0
    sla_violation_steps = 0
    overflow_steps = 0

    done = False
    info = None

    while not done:
        t0 = time.perf_counter()
        action = policy.select_action(obs, info)
        t_decision = (time.perf_counter() - t0) * 1000.0
        decision_times.append(t_decision)

        obs, reward, done, info = env.step(action)

        total_reward += reward
        for k, v in info["reward_components"].items():
            reward_components_sum[k] += v

        queue_history.append(info["true_queue_bytes"].copy())
        overflow_history.append(info["overflow_bytes"].copy())
        throughput_history.append(info["served_bytes"].copy())
        prb_history.append(info["action_prb_allocated"].copy())
        switch_history.append(np.sum(np.abs(info["action_prb_allocated"] - env.prev_allocation)))

        telemetry_requested_count += int(np.sum(info["action_telemetry_requested"]))
        if "delivered_reports" in info:
            delivered_reports_count += int(np.sum(info["delivered_reports"]))

        if np.any(info["overflow_bytes"] > 0):
            overflow_steps += 1
        if np.any(info["latency_violations"] > 0):
            sla_violation_steps += 1

    steps = env.current_step
    queue_arr = np.array(queue_history)
    overflow_arr = np.array(overflow_history)
    throughput_arr = np.array(throughput_history)
    prb_arr = np.array(prb_history)

    slice_throughputs = np.sum(throughput_arr, axis=0)
    total_throughput = float(np.sum(slice_throughputs))
    jains_fairness = compute_jains_fairness(slice_throughputs)

    total_possible_telemetry = steps * env.num_slices
    telemetry_request_rate = telemetry_requested_count / total_possible_telemetry

    mean_prbs_per_slice = np.mean(prb_arr, axis=0)

    return {
        "policy": policy.name,
        "is_privileged": policy.is_privileged,
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward_per_step": total_reward / steps,
        "mean_queue_total_bytes": float(np.mean(np.sum(queue_arr, axis=1))),
        "mean_queue_slice_0_bytes": float(np.mean(queue_arr[:, 0])),
        "mean_queue_slice_1_bytes": float(np.mean(queue_arr[:, 1])),
        "mean_queue_slice_2_bytes": float(np.mean(queue_arr[:, 2])),
        "total_overflow_bytes": float(np.sum(overflow_arr)),
        "overflow_frequency": float(overflow_steps / steps),
        "total_throughput_bytes": total_throughput,
        "slice_0_throughput_bytes": float(slice_throughputs[0]),
        "slice_1_throughput_bytes": float(slice_throughputs[1]),
        "slice_2_throughput_bytes": float(slice_throughputs[2]),
        "jains_fairness_index": jains_fairness,
        "mean_prb_slice_0": float(mean_prbs_per_slice[0]),
        "mean_prb_slice_1": float(mean_prbs_per_slice[1]),
        "mean_prb_slice_2": float(mean_prbs_per_slice[2]),
        "total_prb_switching": float(np.sum(switch_history)),
        "total_telemetry_requests": telemetry_requested_count,
        "telemetry_request_rate": float(telemetry_request_rate),
        "total_delivered_reports": delivered_reports_count,
        "sla_violation_steps": sla_violation_steps,
        "sla_violation_rate": float(sla_violation_steps / steps),
        "decision_latency_ms": float(np.mean(decision_times)),
        "max_decision_latency_ms": float(np.max(decision_times)),
        "queue_cost": reward_components_sum["queue_cost"],
        "overflow_cost": reward_components_sum["overflow_cost"],
        "sla_cost": reward_components_sum["sla_violation_cost"],
        "telemetry_cost": reward_components_sum["telemetry_cost"],
        "switch_cost": reward_components_sum["switch_cost"],
    }


def build_policy_roster():
    """Builds dictionary of all approved policies and ablation variants."""
    policies = {}

    # 1. Complete Proposed Controller
    policies["AgenticTKAN_VoI"] = AgenticTKANController(
        name="AgenticTKAN_VoI", model_type="tkan", telemetry_mode="voi", voi_threshold=1.0, device=DEVICE
    )

    # 2. Model Ablations (Matched VoI Acquisition)
    policies["AgenticGRU_VoI"] = AgenticTKANController(
        name="AgenticGRU_VoI", model_type="gru", telemetry_mode="voi", voi_threshold=1.0, device=DEVICE
    )
    policies["AgenticMLP_VoI"] = AgenticTKANController(
        name="AgenticMLP_VoI", model_type="mlp", telemetry_mode="voi", voi_threshold=1.0, device=DEVICE
    )
    policies["NoMemory_VoI"] = AgenticTKANController(
        name="NoMemory_VoI", model_type="none", telemetry_mode="voi", voi_threshold=1.0, device=DEVICE
    )

    # 3. Telemetry Mode Ablations (TKAN Predictor Fixed)
    policies["AgenticTKAN_PeriodicP2"] = AgenticTKANController(
        name="AgenticTKAN_PeriodicP2", model_type="tkan", telemetry_mode="periodic", periodic_interval=2, device=DEVICE
    )
    policies["AgenticTKAN_PeriodicP4"] = AgenticTKANController(
        name="AgenticTKAN_PeriodicP4", model_type="tkan", telemetry_mode="periodic", periodic_interval=4, device=DEVICE
    )
    policies["AgenticTKAN_AgeT3"] = AgenticTKANController(
        name="AgenticTKAN_AgeT3", model_type="tkan", telemetry_mode="age", age_threshold=3, device=DEVICE
    )
    policies["AgenticTKAN_UncertaintyTh0.15"] = AgenticTKANController(
        name="AgenticTKAN_UncertaintyTh0.15", model_type="tkan", telemetry_mode="uncertainty", uncertainty_threshold=0.15, device=DEVICE
    )
    policies["AgenticTKAN_Always"] = AgenticTKANController(
        name="AgenticTKAN_Always", model_type="tkan", telemetry_mode="always", device=DEVICE
    )
    policies["AgenticTKAN_Never"] = AgenticTKANController(
        name="AgenticTKAN_Never", model_type="tkan", telemetry_mode="never", device=DEVICE
    )

    # 4. Approved Control Baselines
    policies["EqualAllocation"] = EqualAllocationPolicy(total_prbs=15)
    policies["EqualPeriodic_P2"] = EqualPeriodicPolicy(period=2, total_prbs=15)
    policies["QueueAwarePeriodic_P2"] = QueueAwarePeriodicPolicy(period=2, total_prbs=15)
    policies["QueueAwareAge_T3"] = QueueAwareAgeTriggeredPolicy(max_age=3, total_prbs=15)
    policies["QueueAwareUncertainty_Th0.15"] = QueueAwareUncertaintyPolicy(theta_u=0.15, total_prbs=15)
    policies["DecisionSensitive"] = DecisionSensitivePolicy(total_prbs=15)
    policies["PrivilegedFullInfo_Oracle"] = PrivilegedFullInfoPolicy(total_prbs=15)

    return policies


def aggregate_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Computes mean, standard deviation, and 95% confidence intervals per policy."""
    summary_rows = []
    numeric_cols = [
        "total_reward", "mean_reward_per_step", "mean_queue_total_bytes",
        "total_overflow_bytes", "overflow_frequency", "total_throughput_bytes",
        "jains_fairness_index", "mean_prb_slice_0", "mean_prb_slice_1", "mean_prb_slice_2",
        "total_prb_switching", "total_telemetry_requests", "telemetry_request_rate",
        "sla_violation_steps", "sla_violation_rate", "decision_latency_ms",
        "queue_cost", "overflow_cost", "sla_cost", "telemetry_cost", "switch_cost"
    ]

    for policy_name, group in df_raw.groupby("policy"):
        is_priv = bool(group["is_privileged"].iloc[0])
        n_seeds = len(group)
        row = {
            "policy": policy_name,
            "is_privileged": is_priv,
            "n_seeds": n_seeds,
        }

        for col in numeric_cols:
            vals = group[col].to_numpy(dtype=np.float64)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0
            ci95_val = compute_ci95(vals)
            row[f"{col}_mean"] = mean_val
            row[f"{col}_std"] = std_val
            row[f"{col}_ci95"] = ci95_val

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    # Sort with proposed first, then baselines, then oracle
    return df_summary.sort_values(by="total_reward_mean", ascending=False).reset_index(drop=True)


def run_sensitivity_sweeps(seeds):
    """Executes sensitivity sweeps across delay, loss rate, traffic bursts, queue capacity, and VoI threshold."""
    print("\n--- Executing Parameter Sensitivity Sweeps ---")
    sweep_policies = {
        "AgenticTKAN_VoI": AgenticTKANController(name="AgenticTKAN_VoI", model_type="tkan", telemetry_mode="voi", voi_threshold=1.0, device=DEVICE),
        "AgenticGRU_VoI": AgenticTKANController(name="AgenticGRU_VoI", model_type="gru", telemetry_mode="voi", voi_threshold=1.0, device=DEVICE),
        "DecisionSensitive": DecisionSensitivePolicy(total_prbs=15),
        "PrivilegedFullInfo_Oracle": PrivilegedFullInfoPolicy(total_prbs=15),
    }

    env_base = WirelessSlicingEnv()
    base_cfg = copy.deepcopy(env_base.cfg)

    sensitivity_rows = []

    # 1. Telemetry Delay Sweep
    for delay in [1, 2, 4]:
        cfg = copy.deepcopy(base_cfg)
        for s in cfg["slices"]:
            s["telemetry_delay_steps"] = delay
        for pol_name, pol in sweep_policies.items():
            for seed in seeds[:5]:  # 5 seeds for sweeps for efficiency
                env = WirelessSlicingEnv(config=cfg, seed=seed)
                res = run_episode(env, pol, seed)
                res.update({"parameter": "telemetry_delay", "parameter_value": delay})
                sensitivity_rows.append(res)

    # 2. Telemetry Loss Rate Sweep
    for loss_rate in [0.0, 0.10, 0.20, 0.30]:
        cfg = copy.deepcopy(base_cfg)
        for s in cfg["slices"]:
            s["telemetry_loss_rate"] = loss_rate
        for pol_name, pol in sweep_policies.items():
            for seed in seeds[:5]:
                env = WirelessSlicingEnv(config=cfg, seed=seed)
                res = run_episode(env, pol, seed)
                res.update({"parameter": "telemetry_loss_rate", "parameter_value": loss_rate})
                sensitivity_rows.append(res)

    # 3. Traffic Burst Multiplier Sweep
    for burst_mult in [0.8, 1.0, 1.5, 2.0]:
        cfg = copy.deepcopy(base_cfg)
        for s in cfg["slices"]:
            s["arrival_mean_bytes"] *= burst_mult
            s["arrival_std_bytes"] *= burst_mult
        for pol_name, pol in sweep_policies.items():
            for seed in seeds[:5]:
                env = WirelessSlicingEnv(config=cfg, seed=seed)
                res = run_episode(env, pol, seed)
                res.update({"parameter": "burst_multiplier", "parameter_value": burst_mult})
                sensitivity_rows.append(res)

    # 4. Queue Capacity Sweep
    for cap_scale in [0.5, 1.0, 2.0]:
        cfg = copy.deepcopy(base_cfg)
        for s in cfg["slices"]:
            s["queue_capacity_bytes"] *= cap_scale
        for pol_name, pol in sweep_policies.items():
            for seed in seeds[:5]:
                env = WirelessSlicingEnv(config=cfg, seed=seed)
                res = run_episode(env, pol, seed)
                res.update({"parameter": "queue_capacity_scale", "parameter_value": cap_scale})
                sensitivity_rows.append(res)

    # 5. VoI Threshold Sweep (AgenticTKAN_VoI only)
    for gamma in [0.2, 0.5, 1.0, 2.0]:
        pol = AgenticTKANController(name=f"AgenticTKAN_VoI_th{gamma}", model_type="tkan", telemetry_mode="voi", voi_threshold=gamma, device=DEVICE)
        for seed in seeds[:5]:
            env = WirelessSlicingEnv(config=base_cfg, seed=seed)
            res = run_episode(env, pol, seed)
            res.update({"parameter": "gamma_voi", "parameter_value": gamma})
            sensitivity_rows.append(res)

    df_sens = pd.DataFrame(sensitivity_rows)
    df_sens.to_csv("outputs/metrics/sensitivity_results.csv", index=False)
    print("Saved sensitivity sweep results to outputs/metrics/sensitivity_results.csv")
    return df_sens


def capture_outage_recovery_trace(seed: int = 2026):
    """Logs step-by-step queue backlogs and telemetry to demonstrate recovery during an outage."""
    print("\n--- Capturing Step-by-Step Outage Recovery Trajectory ---")
    env_base = WirelessSlicingEnv()
    cfg = copy.deepcopy(env_base.cfg)
    # Configure an outage between steps 30 and 45: loss rate = 1.0, delay = 10
    cfg_outage = copy.deepcopy(cfg)

    trace_policies = {
        "AgenticTKAN_VoI": AgenticTKANController(name="AgenticTKAN_VoI", model_type="tkan", telemetry_mode="voi", voi_threshold=1.0, device=DEVICE),
        "AgenticTKAN_PeriodicP2": AgenticTKANController(name="AgenticTKAN_PeriodicP2", model_type="tkan", telemetry_mode="periodic", periodic_interval=2, device=DEVICE),
        "DecisionSensitive": DecisionSensitivePolicy(total_prbs=15),
        "AgenticTKAN_Never": AgenticTKANController(name="AgenticTKAN_Never", model_type="tkan", telemetry_mode="never", device=DEVICE),
    }

    trace_records = []
    for pol_name, pol in trace_policies.items():
        env = WirelessSlicingEnv(config=cfg_outage, seed=seed)
        obs = env.reset(seed=seed)
        pol.reset()

        for step in range(100):
            # Induce outage window: steps 30 to 45
            in_outage = 30 <= step <= 45
            if in_outage:
                env.telemetry_loss_rates[:] = 1.0
            else:
                env.telemetry_loss_rates[:] = [0.05, 0.02, 0.10]

            action = pol.select_action(obs)
            obs, reward, done, info = env.step(action)

            trace_records.append({
                "policy": pol_name,
                "step": step,
                "in_outage": in_outage,
                "true_queue_slice_0": float(info["true_queue_bytes"][0]),
                "true_queue_slice_1": float(info["true_queue_bytes"][1]),
                "true_queue_slice_2": float(info["true_queue_bytes"][2]),
                "total_true_queue": float(np.sum(info["true_queue_bytes"])),
                "telemetry_requested": int(np.sum(info["action_telemetry_requested"])),
                "overflow_bytes": float(np.sum(info["overflow_bytes"])),
                "prb_slice_0": int(info["action_prb_allocated"][0]),
                "prb_slice_1": int(info["action_prb_allocated"][1]),
                "prb_slice_2": int(info["action_prb_allocated"][2]),
                "reward": float(reward),
            })

    df_trace = pd.DataFrame(trace_records)
    df_trace.to_csv("outputs/metrics/outage_recovery_trace.csv", index=False)
    print("Saved outage recovery trace to outputs/metrics/outage_recovery_trace.csv")


def save_reproducibility_manifest():
    """Generates execution and environment reproducibility manifest."""
    manifest_path = "outputs/manifests/final_experiments_manifest.json"

    # Git hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_hash = "unknown"

    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git_commit_hash": git_hash,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "cpu_processor": platform.processor(),
        "evaluation_seeds": EVAL_SEEDS,
        "command": "python scripts/run_final_experiments.py",
        "models": {
            "TKAN": {"parameters": 4409, "checkpoint": "outputs/checkpoints/tkan_best.pt"},
            "GRU": {"parameters": 3681, "checkpoint": "outputs/checkpoints/singlelayergru_best.pt"},
            "MLP": {"parameters": 2113, "checkpoint": "outputs/checkpoints/smallmlp_best.pt"},
        },
        "sla_definitions": {
            "slice_0_eMBB": {"capacity_bytes": 200000.0, "latency_bound_steps": 4.0, "arrival_mean": 45000.0},
            "slice_1_URLLC": {"capacity_bytes": 20000.0, "latency_bound_steps": 1.5, "arrival_mean": 3000.0},
            "slice_2_MIoT": {"capacity_bytes": 10000.0, "latency_bound_steps": 8.0, "arrival_mean": 800.0},
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved reproducibility manifest to {manifest_path}")


def main():
    print("================================================================================")
    print("   PHASE 9.1: FINAL EXPERIMENTS & STATISTICAL ANALYSIS")
    print(f"   Seeds ({len(EVAL_SEEDS)}): {EVAL_SEEDS}")
    print(f"   Hardware Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("================================================================================")

    t_start = time.time()
    policies = build_policy_roster()
    print(f"Loaded {len(policies)} policies for benchmarking.")

    raw_path = "outputs/metrics/final_experiments_raw.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        if len(df_raw) >= len(policies) * len(EVAL_SEEDS):
            print(f"Reusing existing {len(df_raw)} completed episodes from {raw_path}")
        else:
            df_raw = None
    else:
        df_raw = None

    if df_raw is None:
        raw_records = []
        env = WirelessSlicingEnv()
        for idx, (pol_name, policy) in enumerate(policies.items(), 1):
            print(f"[{idx}/{len(policies)}] Evaluating policy: {pol_name} ...", end="", flush=True)
            t_pol_start = time.time()
            for seed in EVAL_SEEDS:
                rec = run_episode(env, policy, seed)
                raw_records.append(rec)
            elapsed = time.time() - t_pol_start
            print(f" Done ({elapsed:.1f}s)")

        df_raw = pd.DataFrame(raw_records)
        df_raw.to_csv(raw_path, index=False)
        print(f"\nSaved raw episode metrics to {raw_path}")

    # Aggregate Statistics
    df_summary = aggregate_results(df_raw)
    df_summary.to_csv("outputs/metrics/final_results.csv", index=False)
    print("Saved final results with 95% CIs to outputs/metrics/final_results.csv")

    # Separate Ablation Subset
    ablation_policies = [
        "AgenticTKAN_VoI", "AgenticGRU_VoI", "AgenticMLP_VoI", "NoMemory_VoI",
        "AgenticTKAN_PeriodicP2", "AgenticTKAN_PeriodicP4", "AgenticTKAN_AgeT3",
        "AgenticTKAN_UncertaintyTh0.15", "AgenticTKAN_Always", "AgenticTKAN_Never"
    ]
    df_ablations = df_summary[df_summary["policy"].isin(ablation_policies)].copy()
    df_ablations.to_csv("outputs/metrics/ablation_results.csv", index=False)
    print("Saved ablation metrics to outputs/metrics/ablation_results.csv")

    # Run Sensitivity Sweeps
    run_sensitivity_sweeps(EVAL_SEEDS)

    # Capture Outage Recovery Trace
    capture_outage_recovery_trace(seed=2026)

    # Save Manifest
    save_reproducibility_manifest()

    total_time = time.time() - t_start
    print(f"\n================================================================================")
    print(f"   Completed Phase 9.1 Experiments in {total_time:.1f} seconds.")
    print("================================================================================")


if __name__ == "__main__":
    main()
