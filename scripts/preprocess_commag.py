#!/usr/bin/env python3
"""
Preprocess raw CommMag traces and prepare sequential/tabular features
for Mobile Wireless Network simulation and TKAN modeling.
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess(config: dict, sample_scenarios: list = None, max_csvs_per_scenario: int = 50):
    raw_dir = Path(config["data"]["raw_dir"])
    interim_dir = Path(config["data"]["interim_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw CommMag data from: {raw_dir}")

    # Look for Colosseum metric CSVs (scenarios -> tr* -> exp* -> bs* -> slices_bs* -> *_metrics.csv)
    target_base = raw_dir / "slice_traffic" if (raw_dir / "slice_traffic").exists() else raw_dir
    metric_files = list(target_base.glob("rome_*/*/*/*/slices_bs*/*_metrics.csv"))

    if metric_files:
        print(f"Found {len(metric_files)} slice metric CSV files across scenarios.")
        # Subsample for preprocessing demonstration / pipeline if needed
        selected_files = metric_files[:max_csvs_per_scenario] if max_csvs_per_scenario else metric_files
        print(f"Aggregating {len(selected_files)} metric files into interim representation...")

        records = []
        for f in selected_files:
            try:
                parts = f.parts
                # Identify scenario and experiment
                sc_name = [p for p in parts if p.startswith("rome_")][0]
                tr_name = [p for p in parts if p.startswith("tr")][0]
                exp_name = [p for p in parts if p.startswith("exp")][0]
                bs_name = [p for p in parts if p.startswith("bs")][0]

                df = pd.read_csv(f, usecols=lambda col: not col.startswith("Unnamed"))
                df.columns = [c.strip() for c in df.columns]

                # Standardize column naming
                df["scenario"] = sc_name
                df["training_config"] = tr_name
                df["experiment"] = exp_name
                df["base_station"] = bs_name

                records.append(df)
            except Exception as e:
                continue

        if records:
            combined_df = pd.concat(records, ignore_index=True)
            interim_path = interim_dir / "commag_raw_aggregated.parquet"
            combined_df.to_parquet(interim_path, index=False)
            print(f"Saved aggregated interim dataset to: {interim_path} (Shape: {combined_df.shape})")

            # Clean and normalize relevant features for modeling
            cols_to_keep = [
                "Timestamp", "scenario", "training_config", "experiment", "base_station",
                "IMSI", "slice_id", "slice_prb", "scheduling_policy",
                "dl_cqi", "ul_sinr", "tx_brate downlink [Mbps]", "rx_brate uplink [Mbps]",
                "sum_requested_prbs", "sum_granted_prbs"
            ]
            available_cols = [c for c in cols_to_keep if c in combined_df.columns]
            processed_df = combined_df[available_cols].copy()

            # Numeric cleaning & fillna
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)

            out_csv = processed_dir / "commag_processed.csv"
            processed_df.to_csv(out_csv, index=False)
            print(f"Saved processed dataset to: {out_csv} (Shape: {processed_df.shape})")
            return



def main():
    parser = argparse.ArgumentParser(description="Preprocess CommMag dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    preprocess(config)


if __name__ == "__main__":
    main()
