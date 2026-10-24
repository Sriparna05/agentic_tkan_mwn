#!/usr/bin/env python3
"""
Run baseline heuristic and agentic policies on the Mobile Wireless Network simulation.
"""

import argparse
import json
from pathlib import Path
import yaml
import numpy as np

# Support running directly or as module
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from mwnsim.environment import MobileWirelessEnv
from mwnsim.policies import get_policy


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_baselines(config: dict):
    seed = config.get("seed", 42)
    np.random.seed(seed)

    env = MobileWirelessEnv(
        n_base_stations=config["simulation"]["n_base_stations"],
        n_users=config["simulation"]["n_users"],
        max_steps=config["simulation"]["time_steps"],
        bandwidth_mhz=config["simulation"]["bandwidth_mhz"],
        seed=seed,
    )

    baseline_names = config["policies"]["baselines"]
    metrics_summary = {}

    print(f"Running baseline policies: {baseline_names}")

    for name in baseline_names:
        policy = get_policy(name)
        obs = env.reset()
        done = False
        total_reward = 0.0
        total_throughput = 0.0

        while not done:
            action = policy.select_action(obs, env)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            total_throughput += info.get("total_throughput_mbps", 0.0)

        metrics_summary[name] = {
            "total_reward": round(float(total_reward), 4),
            "mean_reward_per_step": round(float(total_reward / env.max_steps), 4),
            "total_throughput_mbps": round(float(total_throughput), 4),
        }
        print(f"Policy '{name}': Total Reward = {total_reward:.2f}, Throughput = {total_throughput:.2f} Mbps")

    out_dir = Path(config["outputs"]["metric_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved baseline evaluation results to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Run simulation baselines")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_baselines(config)


if __name__ == "__main__":
    main()
