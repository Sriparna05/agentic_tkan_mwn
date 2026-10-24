#!/usr/bin/env python3
"""
Comprehensive Colosseum O-RAN COMMAG dataset inventory and schema audit script.
Produces:
  1. reports/dataset_audit/inventory.csv
  2. Detailed metrics for reports/dataset_audit/schema_report.md
"""

import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np


def run_audit():
    root_raw = Path("data/raw/commag").resolve()
    reports_dir = Path("reports/dataset_audit").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CommMag Dataset Audit Starting ===")
    print(f"Dataset root: {root_raw}")

    # 1. Catalog scenarios
    scenarios = {}
    dataset_dirs = [root_raw / "slice_traffic", root_raw / "slice_mixed"]

    inventory_records = []
    scenario_summary = {}

    for d_dir in dataset_dirs:
        if not d_dir.exists():
            continue
        d_type = d_dir.name
        for sc in sorted(d_dir.iterdir()):
            if not sc.is_dir():
                continue
            sc_name = sc.name
            key = f"{d_type}/{sc_name}"
            tr_dirs = sorted([t for t in sc.iterdir() if t.is_dir() and t.name.startswith("tr")])
            scenario_summary[key] = {
                "dataset_type": d_type,
                "scenario": sc_name,
                "num_tr_configs": len(tr_dirs),
                "tr_configs": [t.name for t in tr_dirs],
                "total_files": 0,
                "total_bytes": 0,
                "num_experiments": 0,
                "num_slice_csvs": 0,
                "num_bs_csvs": 0,
                "num_ue_csvs": 0
            }

            for tr in tr_dirs:
                exp_dirs = sorted([e for e in tr.iterdir() if e.is_dir() and e.name.startswith("exp")])
                scenario_summary[key]["num_experiments"] += len(exp_dirs)

                for exp in exp_dirs:
                    bs_dirs = sorted([b for b in exp.iterdir() if b.is_dir() and b.name.startswith("bs")])

                    for bs in bs_dirs:
                        bs_csv = bs / f"{bs.name}.csv"
                        ue_csvs = sorted(list(bs.glob("ue*.csv")))
                        slice_dir = bs / f"slices_{bs.name}"
                        slice_csvs = sorted(list(slice_dir.glob("*_metrics.csv"))) if slice_dir.exists() else []

                        # Record counts
                        scenario_summary[key]["num_bs_csvs"] += 1 if bs_csv.exists() else 0
                        scenario_summary[key]["num_ue_csvs"] += len(ue_csvs)
                        scenario_summary[key]["num_slice_csvs"] += len(slice_csvs)

                        # Add record for inventory.csv
                        total_size_bs = (
                            (bs_csv.stat().st_size if bs_csv.exists() else 0)
                            + sum(f.stat().st_size for f in ue_csvs)
                            + sum(f.stat().st_size for f in slice_csvs)
                        )
                        scenario_summary[key]["total_bytes"] += total_size_bs
                        scenario_summary[key]["total_files"] += (1 if bs_csv.exists() else 0) + len(ue_csvs) + len(slice_csvs)

                        inventory_records.append({
                            "dataset_type": d_type,
                            "scenario": sc_name,
                            "training_config": tr.name,
                            "experiment": exp.name,
                            "base_station": bs.name,
                            "num_ue_csvs": len(ue_csvs),
                            "num_slice_metric_csvs": len(slice_csvs),
                            "has_bs_aggregate_csv": bs_csv.exists(),
                            "total_bytes": total_size_bs,
                            "path": str(bs.relative_to(root_raw))
                        })

    inv_df = pd.DataFrame(inventory_records)
    inv_csv_path = reports_dir / "inventory.csv"
    inv_df.to_csv(inv_csv_path, index=False)
    print(f"Saved inventory CSV: {inv_csv_path} ({len(inv_df)} BS experiment records)")

    # 2. Detailed Profiling of Representative Files
    # Profile representative files from every single scenario
    audit_results = {}
    rep_targets = [
        ("slice_traffic", "rome_static_close", "tr0", "exp1", "bs1"),
        ("slice_traffic", "rome_slow_close", "tr0", "exp1", "bs1"),
        ("slice_traffic", "rome_static_medium", "tr0", "exp1", "bs1"),
        ("slice_traffic", "rome_static_far", "tr0", "exp1", "bs1"),
        ("slice_mixed", "rome_static_close", "tr0", "exp1", "bs1"),
        ("slice_mixed", "rome_slow_close", "tr0", "exp1", "bs1"),
    ]

    for d_type, sc, tr, exp, bs in rep_targets:
        bs_dir = root_raw / d_type / sc / tr / exp / bs
        if not bs_dir.exists():
            continue

        key = f"{d_type}/{sc}/{tr}/{exp}/{bs}"
        audit_results[key] = {}

        # 1. BS aggregate CSV
        bs_file = bs_dir / f"{bs}.csv"
        if bs_file.exists():
            df_bs = pd.read_csv(bs_file)
            time_diffs = df_bs["time"].diff().dropna() if "time" in df_bs.columns else pd.Series()
            audit_results[key]["bs_aggregate"] = {
                "file_name": bs_file.name,
                "file_size": bs_file.stat().st_size,
                "rows": len(df_bs),
                "cols": len(df_bs.columns),
                "columns": list(df_bs.columns),
                "dtypes": {c: str(df_bs[c].dtype) for c in df_bs.columns},
                "nulls": df_bs.isnull().sum().to_dict(),
                "duplicates": int(df_bs.duplicated().sum()),
                "timestamp_col": "time",
                "ts_min": int(df_bs["time"].min()),
                "ts_max": int(df_bs["time"].max()),
                "ts_step_mode": float(time_diffs.mode()[0]) if len(time_diffs) > 0 else None,
                "duration_sec": (int(df_bs["time"].max()) - int(df_bs["time"].min())) / 1000.0
            }

        # 2. Representative UE PHY CSV
        ue_files = sorted(list(bs_dir.glob("ue*.csv")))
        if ue_files:
            sample_ue = ue_files[0]
            df_ue = pd.read_csv(sample_ue)
            time_diffs_ue = df_ue["time"].diff().dropna() if "time" in df_ue.columns else pd.Series()
            audit_results[key]["ue_phy"] = {
                "file_name": sample_ue.name,
                "file_size": sample_ue.stat().st_size,
                "rows": len(df_ue),
                "cols": len(df_ue.columns),
                "columns": list(df_ue.columns),
                "dtypes": {c: str(df_ue[c].dtype) for c in df_ue.columns},
                "nulls": df_ue.isnull().sum().to_dict(),
                "duplicates": int(df_ue.duplicated().sum()),
                "timestamp_col": "time",
                "ts_min": int(df_ue["time"].min()),
                "ts_max": int(df_ue["time"].max()),
                "ts_step_mode": float(time_diffs_ue.mode()[0]) if len(time_diffs_ue) > 0 else None,
            }

        # 3. Representative Slice Metric CSV
        slice_dir = bs_dir / f"slices_{bs}"
        if slice_dir.exists():
            slice_files = sorted(list(slice_dir.glob("*_metrics.csv")))
            if slice_files:
                sample_slice = slice_files[0]
                df_sl = pd.read_csv(sample_slice)
                ts_diffs = df_sl["Timestamp"].diff().dropna() if "Timestamp" in df_sl.columns else pd.Series()
                clean_cols = [c for c in df_sl.columns if not c.startswith("Unnamed")]

                # Check unique slice IDs across all slice files in this BS
                all_slices_in_bs = set()
                for sf in slice_files:
                    try:
                        sdf = pd.read_csv(sf, usecols=["slice_id"], nrows=10)
                        all_slices_in_bs.update(sdf["slice_id"].unique())
                    except Exception:
                        pass

                audit_results[key]["slice_metrics"] = {
                    "file_name": sample_slice.name,
                    "file_size": sample_slice.stat().st_size,
                    "num_files_in_bs": len(slice_files),
                    "slices_in_bs": sorted([int(s) for s in all_slices_in_bs]),
                    "rows": len(df_sl),
                    "cols": len(df_sl.columns),
                    "clean_columns": clean_cols,
                    "all_columns": list(df_sl.columns),
                    "dtypes": {c: str(df_sl[c].dtype) for c in clean_cols},
                    "nulls": {c: int(df_sl[c].isnull().sum()) for c in clean_cols},
                    "duplicates": int(df_sl.duplicated().sum()),
                    "timestamp_col": "Timestamp",
                    "ts_min": int(df_sl["Timestamp"].min()),
                    "ts_max": int(df_sl["Timestamp"].max()),
                    "ts_step_mode": float(ts_diffs.mode()[0]) if len(ts_diffs) > 0 else None,
                    "duration_sec": (int(df_sl["Timestamp"].max()) - int(df_sl["Timestamp"].min())) / 1000.0,
                    # Key 5 starter fields
                    "slice_id_vals": sorted([int(x) for x in df_sl["slice_id"].unique()]),
                    "dl_buffer_min": float(df_sl["dl_buffer [bytes]"].min()),
                    "dl_buffer_max": float(df_sl["dl_buffer [bytes]"].max()),
                    "dl_buffer_mean": float(df_sl["dl_buffer [bytes]"].mean()),
                    "tx_brate_min": float(df_sl["tx_brate downlink [Mbps]"].min()),
                    "tx_brate_max": float(df_sl["tx_brate downlink [Mbps]"].max()),
                    "tx_brate_mean": float(df_sl["tx_brate downlink [Mbps]"].mean()),
                    "req_prbs_min": int(df_sl["sum_requested_prbs"].min()),
                    "req_prbs_max": int(df_sl["sum_requested_prbs"].max()),
                    "req_prbs_mean": float(df_sl["sum_requested_prbs"].mean()),
                    "granted_prbs_min": int(df_sl["sum_granted_prbs"].min()),
                    "granted_prbs_max": int(df_sl["sum_granted_prbs"].max()),
                    "granted_prbs_mean": float(df_sl["sum_granted_prbs"].mean()),
                    # PRB quota
                    "slice_prb_vals": sorted([int(x) for x in df_sl["slice_prb"].unique()]),
                    # Candidate QoS columns
                    "cqi_mean": float(df_sl["dl_cqi"].mean()) if "dl_cqi" in df_sl else None,
                    "sinr_mean": float(df_sl["ul_sinr"].mean()) if "ul_sinr" in df_sl else None,
                    "bler_mean": float(df_sl["tx_errors downlink (%)"].mean()) if "tx_errors downlink (%)" in df_sl else None,
                }

    print("\n=== Scenario Overview ===")
    for k, v in scenario_summary.items():
        print(f"{k}: {v['num_tr_configs']} tr configs, {v['num_experiments']} exps, {v['num_slice_csvs']} slice CSVs, {v['num_ue_csvs']} UE CSVs, {v['num_bs_csvs']} BS CSVs (Total: {v['total_bytes'] / (1024**3):.2f} GB)")

    # Save summary and audit results as JSON for report writing
    import json
    with open(reports_dir / "audit_cache.json", "w", encoding="utf-8") as f:
        json.dump({"scenarios": scenario_summary, "profiles": audit_results}, f, indent=2)

    print(f"Audit completed. Inventory and cache written to {reports_dir}")


if __name__ == "__main__":
    run_audit()
