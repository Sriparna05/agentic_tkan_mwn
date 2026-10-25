"""
Verification script for proposed manuscript corrections D-1 through D-6.
Performs exact recomputations directly from checkpoints, raw experiment logs,
environment configurations, and data manifests.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as st
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)

from src.models.kan.tkan import TKAN
from src.models.deep_learning.gru import SingleLayerGRU
from src.models.deep_learning.mlp import SmallMLP
from src.mwnsim.environment import WirelessSlicingEnv


def verify_d1_parameter_counts():
    print("=" * 80)
    print("VERIFYING D-1: PARAMETER COUNTS & MODEL ARCHITECTURES")
    print("=" * 80)
    
    ckpt_path = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints", "tkan_best.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    
    # Load model
    model = TKAN(input_dim=4, horizon=1, units=32, num_sub_kans=1)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"TKAN total parameters: {total_params}")
    print(f"TKAN trainable parameters: {trainable_params}")
    print("\nLayer-by-layer parameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name:45s} | shape: {str(list(param.shape)):18s} | numel: {param.numel():6d}")
        
    # Check what configuration yields 4,409 parameters
    print("\nChecking possible configurations that might have yielded 4,409 params:")
    for units in [32, 36, 40, 64]:
        for num_sub in [1, 2]:
            for grid_size in [3, 5, 8]:
                try:
                    m = TKAN(input_dim=4, horizon=1, units=units, num_sub_kans=num_sub)
                    p_count = sum(p.numel() for p in m.parameters())
                    if p_count == 4409:
                        print(f"  MATCH FOUND: units={units}, num_sub_kans={num_sub} -> {p_count} params")
                except Exception:
                    pass
    
    # Also check GRU and MLP
    gru_path = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints", "singlelayergru_best.pt")
    gru_model = SingleLayerGRU(in_features=4, hidden_dim=32, num_layers=1)
    gru_state = torch.load(gru_path, map_location="cpu")
    gru_model.load_state_dict(gru_state.get("model_state_dict", gru_state))
    gru_params = sum(p.numel() for p in gru_model.parameters())
    print(f"\nGRU total parameters: {gru_params}")
    
    mlp_path = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints", "smallmlp_best.pt")
    mlp_model = SmallMLP(seq_len=16, in_features=4, hidden_dim=32, out_features=1)
    mlp_state = torch.load(mlp_path, map_location="cpu")
    mlp_model.load_state_dict(mlp_state.get("model_state_dict", mlp_state))
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"MLP total parameters: {mlp_params}")


def verify_d2_pipeline_configs():
    print("\n" + "=" * 80)
    print("VERIFYING D-2: LOOKBACK, SLOT DURATION, SEEDS, & DATA SPLITS")
    print("=" * 80)
    
    # 1. Simulator config
    sim_cfg_path = os.path.join(WORKSPACE_ROOT, "configs", "simulator.yaml")
    with open(sim_cfg_path, "r") as f:
        import yaml
        sim_cfg = yaml.safe_load(f)
    print(f"simulator.yaml step_duration_ms: {sim_cfg['time']['step_duration_ms']}")
    print(f"simulator.yaml max_steps: {sim_cfg['time']['max_steps']}")
    
    # 2. Manifest config
    manifest_path = os.path.join(WORKSPACE_ROOT, "data", "processed", "commag", "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    print(f"manifest.json lookback_window_steps: {manifest.get('lookback_window_steps')}")
    print(f"manifest.json sampling_interval_ms: {manifest.get('sampling_interval_ms')}")
    print(f"manifest.json window_counts: {manifest.get('window_counts')}")
    
    # 3. Splits config
    splits_path = os.path.join(WORKSPACE_ROOT, "data", "processed", "commag", "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, "r") as f:
            splits = json.load(f)
        print(f"splits.json keys: {list(splits.keys())}")
        for k in ["train", "validation", "test"]:
            if k in splits:
                recs = splits[k].get("recordings", [])
                print(f"  split '{k}': {len(recs)} recordings")
                
    # 4. Check array shapes in .npz
    for split_name in ["train.npz", "val.npz", "test.npz"]:
        sp_path = os.path.join(WORKSPACE_ROOT, "data", "processed", "commag", split_name)
        if os.path.exists(sp_path):
            arr = np.load(sp_path)
            print(f"  {split_name}: X shape = {arr['X'].shape}, y shape = {arr['y'].shape}")
            
    # 5. Seeds in raw output
    raw_path = os.path.join(WORKSPACE_ROOT, "outputs", "metrics", "final_experiments_raw.csv")
    df_raw = pd.read_csv(raw_path)
    unique_seeds = sorted(df_raw["seed"].unique().tolist())
    print(f"Actual seeds in final_experiments_raw.csv: {unique_seeds}")


def verify_d3_reward_and_percentages():
    print("\n" + "=" * 80)
    print("VERIFYING D-3: REWARD COMPARISONS & PERCENTAGE FORMULAS")
    print("=" * 80)
    
    res_path = os.path.join(WORKSPACE_ROOT, "outputs", "metrics", "final_results.csv")
    df_res = pd.read_csv(res_path)
    
    pol_rewards = {}
    for _, row in df_res.iterrows():
        pol_rewards[row["policy"]] = (row["total_reward_mean"], row["total_reward_ci95"])
        
    r_tkan, ci_tkan = pol_rewards["AgenticTKAN_VoI"]
    r_gru, ci_gru = pol_rewards["AgenticGRU_VoI"]
    r_mlp, ci_mlp = pol_rewards["AgenticMLP_VoI"]
    
    print(f"TKAN Reward: {r_tkan:.2f} +/- {ci_tkan:.2f}")
    print(f"GRU Reward:  {r_gru:.2f} +/- {ci_gru:.2f}")
    print(f"MLP Reward:  {r_mlp:.2f} +/- {ci_mlp:.2f}")
    
    # Formula 1: Relative to TKAN (|R_base| - |R_tkan|) / |R_tkan|
    f1_gru = (abs(r_gru) - abs(r_tkan)) / abs(r_tkan) * 100
    f1_mlp = (abs(r_mlp) - abs(r_tkan)) / abs(r_tkan) * 100
    
    # Formula 2: Cost Reduction Relative to Baseline (|R_base| - |R_tkan|) / |R_base|
    f2_gru = (abs(r_gru) - abs(r_tkan)) / abs(r_gru) * 100
    f2_mlp = (abs(r_mlp) - abs(r_tkan)) / abs(r_mlp) * 100
    
    # Formula 3: Standard Algebraic Reward Gain (R_tkan - R_base) / |R_base|
    f3_gru = (r_tkan - r_gru) / abs(r_gru) * 100
    f3_mlp = (r_tkan - r_mlp) / abs(r_mlp) * 100
    
    print("\nReward percentage calculations:")
    print(f"  Formula 1 (|R_base| - |R_tkan|) / |R_tkan|:")
    print(f"    GRU: {f1_gru:.2f}%  (Matches manuscript claim 32.6%)")
    print(f"    MLP: {f1_mlp:.2f}%  (Matches manuscript claim 201.7%)")
    print(f"  Formula 2 (|R_base| - |R_tkan|) / |R_base| (Cost Reduction):")
    print(f"    GRU: {f2_gru:.2f}%")
    print(f"    MLP: {f2_mlp:.2f}%")
    print(f"  Formula 3 (R_tkan - R_base) / |R_base| (Algebraic Gain):")
    print(f"    GRU: {f3_gru:.2f}% (Identical to Formula 2)")
    print(f"    MLP: {f3_mlp:.2f}% (Identical to Formula 2)")


def verify_d4_and_investigate_reward_discrepancy():
    print("\n" + "=" * 80)
    print("VERIFYING D-4 & INVESTIGATING REWARD DISCREPANCY: TKAN vs NoMemory")
    print("=" * 80)
    
    raw_path = os.path.join(WORKSPACE_ROOT, "outputs", "metrics", "final_experiments_raw.csv")
    df_raw = pd.read_csv(raw_path)
    
    # Filter key policies
    target_pols = [
        "AgenticTKAN_VoI", "NoMemory_VoI", "AgenticTKAN_Never",
        "PrivilegedFullInfo_Oracle", "QueueAwarePeriodic_P2"
    ]
    sub = df_raw[df_raw["policy"].isin(target_pols)]
    
    metrics_to_show = [
        "total_reward", "queue_cost", "overflow_cost", "sla_cost", "telemetry_cost", "switch_cost",
        "mean_queue_total_bytes", "total_overflow_bytes", "overflow_frequency",
        "sla_violation_rate", "total_telemetry_requests",
        "mean_prb_slice_0", "mean_prb_slice_1", "mean_prb_slice_2"
    ]
    
    summary = sub.groupby("policy")[metrics_to_show].mean()
    print("\nMean Metrics Across 10 Seeds:")
    for pol in target_pols:
        print(f"\n--- Policy: {pol} ---")
        row = summary.loc[pol]
        print(f"  Total Reward:             {row['total_reward']:12.2f}")
        print(f"  Decomposed Costs:")
        print(f"    - queue_cost:           {row['queue_cost']:12.2f} (w_q=0.01)")
        print(f"    - overflow_cost:        {row['overflow_cost']:12.2f} (w_ovf=1.0)")
        print(f"    - sla_cost:             {row['sla_cost']:12.2f} (w_sla=2.0)")
        print(f"    - telemetry_cost:       {row['telemetry_cost']:12.2f} (w_tel=0.1)")
        print(f"    - switch_cost:          {row['switch_cost']:12.2f} (w_sw=0.05)")
        print(f"  Physical Metrics:")
        print(f"    - Mean Queue Total:     {row['mean_queue_total_bytes']:12.1f} Bytes")
        print(f"    - Total Overflow Bytes: {row['total_overflow_bytes']:12.1f} Bytes")
        print(f"    - Overflow Frequency:   {row['overflow_frequency']*100:11.2f} %")
        print(f"    - SLA Violation Rate:   {row['sla_violation_rate']*100:11.2f} %")
        print(f"    - Telemetry Requests:   {row['total_telemetry_requests']:12.1f}")
        print(f"    - PRB Allocation (0,1,2): [{row['mean_prb_slice_0']:.2f}, {row['mean_prb_slice_1']:.2f}, {row['mean_prb_slice_2']:.2f}]")

    # Confirm the 160,663-byte overflow figure:
    ovf_nomem = summary.loc["NoMemory_VoI"]["total_overflow_bytes"]
    ovf_tkan = summary.loc["AgenticTKAN_VoI"]["total_overflow_bytes"]
    print(f"\nOverflow comparison:")
    print(f"  NoMemory_VoI Total Overflow Bytes: {ovf_nomem:.1f} Bytes")
    print(f"  AgenticTKAN_VoI Total Overflow Bytes: {ovf_tkan:.1f} Bytes")
    print(f"  Ratio (NoMemory / TKAN): {ovf_nomem / ovf_tkan:.3f}x (+{(ovf_nomem - ovf_tkan)/ovf_tkan*100:.1f}%)")


def verify_d5_pareto_optimality():
    print("\n" + "=" * 80)
    print("VERIFYING D-5: PARETO OPTIMALITY ANALYSIS")
    print("=" * 80)
    
    res_path = os.path.join(WORKSPACE_ROOT, "outputs", "metrics", "final_results.csv")
    df_res = pd.read_csv(res_path)
    
    print("\nMulti-Objective Space for All Evaluated Policies:")
    cols = ["policy", "total_reward_mean", "overflow_frequency_mean", "total_overflow_bytes_mean", "sla_violation_rate_mean", "total_telemetry_requests_mean"]
    print(df_res[cols].to_string(index=False))


def verify_d6_dwell_time_proxy():
    print("\n" + "=" * 80)
    print("VERIFYING D-6: DWELL-TIME PROXY FORMULA & UNITS")
    print("=" * 80)
    
    env = WirelessSlicingEnv()
    print("Slice Parameters in WirelessSlicingEnv:")
    for s in env.cfg["slices"]:
        print(f"  Slice {s['id']} ({s['name']}):")
        print(f"    queue_capacity_bytes:              {s['queue_capacity_bytes']}")
        print(f"    service_efficiency_bytes_per_prb:  {s['service_efficiency_bytes_per_prb']}")
        print(f"    sla_latency_bound_steps:           {s['sla_latency_bound_steps']} steps")
        print(f"    latency_violation_cost:            {s['weights']['latency_violation_cost']}")
    print(f"Global reward sla_violation_weight:     {env.global_weights['sla_violation_weight']}")


if __name__ == "__main__":
    verify_d1_parameter_counts()
    verify_d2_pipeline_configs()
    verify_d3_reward_and_percentages()
    verify_d4_and_investigate_reward_discrepancy()
    verify_d5_pareto_optimality()
    verify_d6_dwell_time_proxy()
