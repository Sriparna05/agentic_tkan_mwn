#!/usr/bin/env python3
"""
Phase 4.1: Reproducible Colosseum O-RAN COMMAG Preprocessing Pipeline.
Processes the approved pilot benchmark runs according to configs/data_protocol.yaml.
Outputs:
  - data/processed/commag/train.npz
  - data/processed/commag/val.npz
  - data/processed/commag/test.npz
  - data/processed/commag/ood_test.npz
  - data/processed/commag/splits.json
  - data/processed/commag/manifest.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = [
    "Timestamp",
    "slice_id",
    "IMSI",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "sum_requested_prbs",
    "sum_granted_prbs",
]

FEATURE_COLUMNS = [
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "sum_requested_prbs",
    "sum_granted_prbs",
]

TARGET_COLUMN = "dl_buffer [bytes]"


def get_git_revision(cwd: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class CommMagPreprocessor:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.project_root = self.config_path.resolve().parent.parent
        self.raw_root = self.project_root / self.cfg["dataset"]["source_root"]
        self.output_dir = self.project_root / "data" / "processed" / "commag"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lookback = self.cfg["forecasting_task"]["parameters"]["lookback_window_steps"] # 16
        self.horizon = self.cfg["forecasting_task"]["parameters"]["forecast_horizon_steps"]  # 1
        self.sampling_interval_ms = self.cfg["sampling"]["sampling_interval_ms"]            # 250.0
        self.interval_tolerance_ms = self.cfg["sampling"]["interval_tolerance_ms"]          # 5.0

        # Scaler parameters fit strictly on train
        self.scaler_mean = None
        self.scaler_std = None
        self.scaler_min = None
        self.scaler_max = None

        # Tracking audits
        self.audit_log = {
            "duplicate_rows_dropped": 0,
            "duplicate_timestamps_dropped": 0,
            "segments_split_on_gaps": 0,
            "excluded_short_segments": 0,
            "total_raw_rows_read": 0,
            "total_windows_generated": {}
        }

    def _load_and_validate_csv(self, file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {file_path}")

        # Validate header before loading all rows
        df_header = pd.read_csv(file_path, nrows=0)
        clean_header = [c.strip() for c in df_header.columns]
        for col in REQUIRED_COLUMNS:
            if col not in clean_header:
                raise ValueError(f"Schema violation in {file_path}: missing required column '{col}'. Available: {clean_header}")

        df = pd.read_csv(file_path, usecols=REQUIRED_COLUMNS)
        self.audit_log["total_raw_rows_read"] += len(df)

        # 1. Deduplicate exact duplicate rows
        dup_rows = df.duplicated().sum()
        if dup_rows > 0:
            df = df.drop_duplicates().copy()
            self.audit_log["duplicate_rows_dropped"] += dup_rows

        # 2. Check for missing values in required fields
        null_counts = df.isnull().sum()
        if null_counts.any():
            # Drop NaN rows if any (Colosseum clean data has 0, but safeguard)
            df = df.dropna().copy()

        # 3. Sort strictly by timestamp
        df = df.sort_values(by="Timestamp").reset_index(drop=True)

        # 4. Check for duplicate timestamps
        dup_ts = df["Timestamp"].duplicated().sum()
        if dup_ts > 0:
            df = df.drop_duplicates(subset=["Timestamp"], keep="first").copy()
            self.audit_log["duplicate_timestamps_dropped"] += dup_ts

        return df

    def _extract_segments(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits a single UE dataframe into contiguous segments if timestamp gaps
        exceed sampling_interval_ms + interval_tolerance_ms (e.g. > 255 ms).
        Never interpolates across missing intervals.
        """
        if len(df) < (self.lookback + self.horizon):
            self.audit_log["excluded_short_segments"] += 1
            return []

        ts = df["Timestamp"].values
        diffs = np.diff(ts)
        max_allowed_gap = self.sampling_interval_ms * 2.0  # > 500 ms gap triggers split

        gap_indices = np.where(diffs > max_allowed_gap)[0]
        if len(gap_indices) == 0:
            return [df]

        self.audit_log["segments_split_on_gaps"] += len(gap_indices)
        segments = []
        start_idx = 0
        for gap_idx in gap_indices:
            seg = df.iloc[start_idx : gap_idx + 1].copy()
            if len(seg) >= (self.lookback + self.horizon):
                segments.append(seg)
            else:
                self.audit_log["excluded_short_segments"] += 1
            start_idx = gap_idx + 1

        # Last segment
        if start_idx < len(df):
            seg = df.iloc[start_idx:].copy()
            if len(seg) >= (self.lookback + self.horizon):
                segments.append(seg)
            else:
                self.audit_log["excluded_short_segments"] += 1

        return segments

    def _create_windows_from_segment(
        self, seg: pd.DataFrame, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates lookback-16, horizon-1 sliding windows.
        Strictly causal:
          X[i] = features at t-15 .. t
          y[i] = dl_buffer[bytes] at t+1
        """
        feature_vals = seg[FEATURE_COLUMNS].values.astype(np.float64)
        target_vals = seg[TARGET_COLUMN].values.astype(np.float64)
        ts_vals = seg["Timestamp"].values.astype(np.int64)
        slice_vals = seg["slice_id"].values.astype(np.int32)
        imsi_vals = seg["IMSI"].values.astype(np.int64)

        L = len(seg)
        num_windows = L - self.lookback - self.horizon + 1
        if num_windows <= 0:
            return (
                np.empty((0, self.lookback, len(FEATURE_COLUMNS))),
                np.empty((0,)),
                np.empty((0,)),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )

        X = np.lib.stride_tricks.sliding_window_view(feature_vals[: L - self.horizon], window_shape=self.lookback, axis=0)
        # sliding_window_view returns shape (num_windows, len(FEATURE_COLUMNS), lookback), swap axes to (num_windows, lookback, len(FEATURE_COLUMNS))
        X = np.swapaxes(X, 1, 2)

        # Target is at t + horizon (t + 1)
        y = target_vals[self.lookback :]
        y_log1p = np.log1p(np.maximum(y, 0.0))

        # Timestamps of window end (t) and target (t + 1)
        ts_t = ts_vals[self.lookback - 1 : L - self.horizon]
        ts_target = ts_vals[self.lookback :]

        slice_id = slice_vals[self.lookback - 1 : L - self.horizon]
        imsi = imsi_vals[self.lookback - 1 : L - self.horizon]

        return X, y, y_log1p, slice_id, imsi, ts_t, ts_target

    def process_split_runs(self, run_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes a group of independent experiment runs.
        """
        X_list = []
        y_list = []
        y_log1p_list = []
        slice_list = []
        imsi_list = []
        ts_t_list = []
        ts_target_list = []
        recording_manifest = []

        for run in run_dicts:
            run_path = self.raw_root / run["path"]
            parts = Path(run["path"]).parts
            sc_name = parts[1]
            tr_name = parts[2]
            exp_name = parts[3]

            metric_files = sorted(list(run_path.glob("bs*/slices_bs*/*_metrics.csv")))
            run_windows = 0
            run_rows = 0

            for mf in metric_files:
                df = self._load_and_validate_csv(mf)
                bs_name = mf.parents[1].name
                imsi_val = int(df["IMSI"].iloc[0]) if len(df) > 0 else 0
                slice_val = int(df["slice_id"].iloc[0]) if len(df) > 0 else 0

                stable_rec_id = f"{sc_name}__{tr_name}__{exp_name}__{bs_name}__imsi_{imsi_val}__slice_{slice_val}"
                segments = self._extract_segments(df)

                meta = {
                    "recording_id": stable_rec_id,
                    "scenario": sc_name,
                    "training_config": tr_name,
                    "experiment": exp_name,
                    "base_station": bs_name,
                    "imsi": imsi_val,
                    "slice_id": slice_val,
                    "source_file": str(mf.relative_to(self.raw_root)),
                }

                for seg in segments:
                    X_seg, y_seg, y_log_seg, sl_seg, im_seg, t_t_seg, t_tar_seg = self._create_windows_from_segment(seg, meta)
                    if len(y_seg) > 0:
                        X_list.append(X_seg)
                        y_list.append(y_seg)
                        y_log1p_list.append(y_log_seg)
                        slice_list.append(sl_seg)
                        imsi_list.append(im_seg)
                        ts_t_list.append(t_t_seg)
                        ts_target_list.append(t_tar_seg)
                        run_windows += len(y_seg)
                        run_rows += len(seg)

            recording_manifest.append({
                "run_id": run["id"],
                "path": run["path"],
                "metric_files_count": len(metric_files),
                "total_rows_processed": run_rows,
                "total_windows_formed": run_windows,
            })

        if not X_list:
            raise RuntimeError(f"No windows could be formed for runs: {[r['id'] for r in run_dicts]}")

        X_all = np.concatenate(X_list, axis=0).astype(np.float32)
        y_all = np.concatenate(y_list, axis=0).astype(np.float32)
        y_log1p_all = np.concatenate(y_log1p_list, axis=0).astype(np.float32)
        slice_all = np.concatenate(slice_list, axis=0).astype(np.int32)
        imsi_all = np.concatenate(imsi_list, axis=0).astype(np.int64)
        ts_t_all = np.concatenate(ts_t_list, axis=0).astype(np.int64)
        ts_target_all = np.concatenate(ts_target_list, axis=0).astype(np.int64)

        return {
            "X": X_all,
            "y": y_all,
            "y_log1p": y_log1p_all,
            "slice_id": slice_all,
            "imsi": imsi_all,
            "timestamps_t": ts_t_all,
            "timestamps_target": ts_target_all,
            "runs_manifest": recording_manifest,
        }

    def fit_scaler(self, X_train: np.ndarray):
        """
        Fits StandardScaler strictly on the training set features across all time steps.
        X_train shape: (N_train, lookback, num_features).
        """
        N, W, F = X_train.shape
        flat_X = X_train.reshape(-1, F)
        self.scaler_mean = np.mean(flat_X, axis=0)
        self.scaler_std = np.std(flat_X, axis=0)
        # Prevent division by zero
        self.scaler_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)

        self.scaler_min = np.min(flat_X, axis=0)
        self.scaler_max = np.max(flat_X, axis=0)

        print("\n=== Fitted Scaler Parameters on Training Data Only ===")
        for i, col in enumerate(FEATURE_COLUMNS):
            print(f"  [{i}] {col:<26}: Mean={self.scaler_mean[i]:.4f}, Std={self.scaler_std[i]:.4f}, Min={self.scaler_min[i]:.4f}, Max={self.scaler_max[i]:.4f}")

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Scaler must be fitted on training data before transforming.")
        return ((X - self.scaler_mean) / self.scaler_std).astype(np.float32)

    def run(self):
        print("=" * 80)
        print("PHASE 4.1 REPRODUCIBLE COMMAG PREPROCESSING PIPELINE")
        print("=" * 80)

        splits_cfg = self.cfg["splits"]
        train_runs = splits_cfg["train"]["runs"]
        val_runs = splits_cfg["validation"]["runs"]
        test_runs = splits_cfg["test"]["runs"]
        ood_runs = splits_cfg["out_of_distribution_test"]["runs"]

        # 1. Process Train Runs
        print(f"\n[1/4] Processing Training Set ({len(train_runs)} independent runs)...")
        train_data = self.process_split_runs(train_runs)
        print(f"  -> Generated {len(train_data['y']):,} training windows.")

        # 2. Fit Scaler STRICTLY on Train
        self.fit_scaler(train_data["X"])
        train_X_scaled = self.transform_X(train_data["X"])

        # 3. Process Validation Runs
        print(f"\n[2/4] Processing Validation Set ({len(val_runs)} independent runs)...")
        val_data = self.process_split_runs(val_runs)
        val_X_scaled = self.transform_X(val_data["X"])
        print(f"  -> Generated {len(val_data['y']):,} validation windows.")

        # 4. Process In-Distribution Test Runs
        print(f"\n[3/4] Processing In-Distribution Test Set ({len(test_runs)} independent runs)...")
        test_data = self.process_split_runs(test_runs)
        test_X_scaled = self.transform_X(test_data["X"])
        print(f"  -> Generated {len(test_data['y']):,} test windows.")

        # 5. Process Out-of-Distribution Test Runs
        print(f"\n[4/4] Processing OOD Mobility Test Set ({len(ood_runs)} independent run)...")
        ood_data = self.process_split_runs(ood_runs)
        ood_X_scaled = self.transform_X(ood_data["X"])
        print(f"  -> Generated {len(ood_data['y']):,} OOD test windows.")

        # 6. Save Arrays to Compressed NPZ
        print("\nSaving compressed tensor files to:", self.output_dir)

        np.savez_compressed(
            self.output_dir / "train.npz",
            X=train_X_scaled,
            X_raw=train_data["X"],
            y=train_data["y"],
            y_log1p=train_data["y_log1p"],
            slice_id=train_data["slice_id"],
            imsi=train_data["imsi"],
            timestamps_t=train_data["timestamps_t"],
            timestamps_target=train_data["timestamps_target"],
        )
        print(f"  -> Saved {self.output_dir / 'train.npz'} ({len(train_data['y']):,} samples)")

        np.savez_compressed(
            self.output_dir / "val.npz",
            X=val_X_scaled,
            X_raw=val_data["X"],
            y=val_data["y"],
            y_log1p=val_data["y_log1p"],
            slice_id=val_data["slice_id"],
            imsi=val_data["imsi"],
            timestamps_t=val_data["timestamps_t"],
            timestamps_target=val_data["timestamps_target"],
        )
        print(f"  -> Saved {self.output_dir / 'val.npz'} ({len(val_data['y']):,} samples)")

        np.savez_compressed(
            self.output_dir / "test.npz",
            X=test_X_scaled,
            X_raw=test_data["X"],
            y=test_data["y"],
            y_log1p=test_data["y_log1p"],
            slice_id=test_data["slice_id"],
            imsi=test_data["imsi"],
            timestamps_t=test_data["timestamps_t"],
            timestamps_target=test_data["timestamps_target"],
        )
        print(f"  -> Saved {self.output_dir / 'test.npz'} ({len(test_data['y']):,} samples)")

        np.savez_compressed(
            self.output_dir / "ood_test.npz",
            X=ood_X_scaled,
            X_raw=ood_data["X"],
            y=ood_data["y"],
            y_log1p=ood_data["y_log1p"],
            slice_id=ood_data["slice_id"],
            imsi=ood_data["imsi"],
            timestamps_t=ood_data["timestamps_t"],
            timestamps_target=ood_data["timestamps_target"],
        )
        print(f"  -> Saved {self.output_dir / 'ood_test.npz'} ({len(ood_data['y']):,} samples)")

        # 7. Write splits.json
        splits_manifest = {
            "random_seed": self.cfg["random_seed"],
            "protocol_version": self.cfg["protocol_version"],
            "splits": {
                "train": {
                    "num_runs": len(train_runs),
                    "total_windows": int(len(train_data["y"])),
                    "runs": train_data["runs_manifest"],
                },
                "validation": {
                    "num_runs": len(val_runs),
                    "total_windows": int(len(val_data["y"])),
                    "runs": val_data["runs_manifest"],
                },
                "test": {
                    "num_runs": len(test_runs),
                    "total_windows": int(len(test_data["y"])),
                    "runs": test_data["runs_manifest"],
                },
                "out_of_distribution_test": {
                    "num_runs": len(ood_runs),
                    "total_windows": int(len(ood_data["y"])),
                    "runs": ood_data["runs_manifest"],
                },
            },
        }

        with open(self.output_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump(splits_manifest, f, indent=2)
        print(f"  -> Saved {self.output_dir / 'splits.json'}")

        # 8. Write manifest.json
        git_rev = get_git_revision(self.project_root)
        manifest = {
            "protocol_version": self.cfg["protocol_version"],
            "git_revision": git_rev,
            "created_at_utc": str(pd.Timestamp.now(tz="UTC")),
            "dataset_name": self.cfg["dataset"]["name"],
            "lookback_window_steps": self.lookback,
            "forecast_horizon_steps": self.horizon,
            "sampling_interval_ms": self.sampling_interval_ms,
            "features": {
                "feature_names": FEATURE_COLUMNS,
                "target_name": TARGET_COLUMN,
                "grouping_variable": "slice_id",
            },
            "scaler_parameters": {
                "scaler_type": "StandardScaler",
                "fit_partition": "train_only",
                "feature_names": FEATURE_COLUMNS,
                "mean": [float(m) for m in self.scaler_mean],
                "scale_std": [float(s) for s in self.scaler_std],
                "min": [float(m) for m in self.scaler_min],
                "max": [float(m) for m in self.scaler_max],
            },
            "target_transform": {
                "transform": "log1p",
                "formula": "log(1 + y)",
                "inverse_formula": "exp(y_pred) - 1",
            },
            "window_counts": {
                "train": int(len(train_data["y"])),
                "validation": int(len(val_data["y"])),
                "test": int(len(test_data["y"])),
                "out_of_distribution_test": int(len(ood_data["y"])),
                "total": int(len(train_data["y"]) + len(val_data["y"]) + len(test_data["y"]) + len(ood_data["y"])),
            },
            "audit_log": self.audit_log,
        }

        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"  -> Saved {self.output_dir / 'manifest.json'}")

        print("\nPreprocessing pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Colosseum COMMAG Pilot Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_protocol.yaml",
        help="Path to YAML protocol configuration file",
    )
    args = parser.parse_args()
    preprocessor = CommMagPreprocessor(args.config)
    preprocessor.run()


if __name__ == "__main__":
    main()
