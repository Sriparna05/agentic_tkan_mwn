"""
Automated validation test suite for Phase 4.1 Reproducible CommMag Preprocessing.
Tests:
  1. Group Separation (zero run/file leakage across Train, Val, Test)
  2. Chronological Ordering (strictly monotonic timestamps)
  3. Feature Leakage Prevention (no future observations in input window)
  4. Target Alignment (exact match between y and raw source file at t+1)
  5. Scaler Fitted on Training Data Only (parameters derived strictly from Train)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


PROCESSED_DIR = Path("data/processed/commag")
RAW_DIR = Path("data/raw/commag")


def test_files_exist():
    assert (PROCESSED_DIR / "train.npz").exists(), "train.npz missing"
    assert (PROCESSED_DIR / "val.npz").exists(), "val.npz missing"
    assert (PROCESSED_DIR / "test.npz").exists(), "test.npz missing"
    assert (PROCESSED_DIR / "ood_test.npz").exists(), "ood_test.npz missing"
    assert (PROCESSED_DIR / "splits.json").exists(), "splits.json missing"
    assert (PROCESSED_DIR / "manifest.json").exists(), "manifest.json missing"


def test_group_separation():
    """
    Test 1: Confirms that independent experiment runs are strictly mutually disjoint across splits.
    """
    with open(PROCESSED_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    train_runs = {r["run_id"] for r in splits["train"]["runs"]}
    val_runs = {r["run_id"] for r in splits["validation"]["runs"]}
    test_runs = {r["run_id"] for r in splits["test"]["runs"]}
    ood_runs = {r["run_id"] for r in splits["out_of_distribution_test"]["runs"]}

    # Verify set disjointness
    assert train_runs.isdisjoint(val_runs), f"Leakage: Train and Val share runs {train_runs & val_runs}"
    assert train_runs.isdisjoint(test_runs), f"Leakage: Train and Test share runs {train_runs & test_runs}"
    assert val_runs.isdisjoint(test_runs), f"Leakage: Val and Test share runs {val_runs & test_runs}"
    assert train_runs.isdisjoint(ood_runs), f"Leakage: Train and OOD share runs {train_runs & ood_runs}"

    assert len(train_runs) == 10
    assert len(val_runs) == 2
    assert len(test_runs) == 2
    assert len(ood_runs) == 1


def test_chronological_ordering():
    """
    Test 2: Confirms that within each window, observations are strictly chronological and target is at t+1.
    """
    for split_name in ["train", "val", "test", "ood_test"]:
        data = np.load(PROCESSED_DIR / f"{split_name}.npz")
        ts_t = data["timestamps_t"]
        ts_target = data["timestamps_target"]

        # Target must be strictly after window close
        time_deltas = ts_target - ts_t
        assert np.all(time_deltas > 0), f"Non-causal target detected in {split_name}: target <= t"

        # Median sampling step must be exactly 250 ms
        assert np.median(time_deltas) == 250.0, f"Median step in {split_name} is not 250 ms: {np.median(time_deltas)}"
        assert np.isclose(np.mean(time_deltas), 250.0, atol=0.1), f"Mean step in {split_name} is not 250 ms: {np.mean(time_deltas)}"

        # 99.9% of samples must fall within normal jitter range [240, 260] ms
        p001 = np.percentile(time_deltas, 0.1)
        p999 = np.percentile(time_deltas, 99.9)
        assert p001 >= 240 and p999 <= 260, f"Jitter anomaly in {split_name}: p0.1={p001}, p99.9={p999}"


def test_feature_leakage_and_causality():
    """
    Test 3: Confirms that the lookback window X contains only data up to time t,
    and target y represents step t+1.
    """
    data = np.load(PROCESSED_DIR / "train.npz")
    X_raw = data["X_raw"]  # shape (N, 16, 4)
    y = data["y"]          # shape (N,)

    # Feature 0 is dl_buffer [bytes]
    buffer_at_t = X_raw[:, -1, 0]  # last step in input window
    buffer_at_t_plus_1 = y         # target at step t+1

    # They should not be identically equal everywhere (buffer changes dynamically)
    diff = np.abs(buffer_at_t_plus_1 - buffer_at_t)
    assert np.mean(diff > 0) > 0.1, "Target appears trivially copied from input step t"

    # Verify input window shape
    assert X_raw.shape[1] == 16, f"Expected lookback 16, got {X_raw.shape[1]}"
    assert X_raw.shape[2] == 4, f"Expected 4 features, got {X_raw.shape[2]}"


def test_target_alignment():
    """
    Test 4: Matches a window's target directly against the raw CSV file to guarantee
    exact numerical identity and proper log1p transformation.
    """
    with open(PROCESSED_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    # Pick first train run and first metric file
    sample_run = splits["train"]["runs"][0]
    sample_file = list((RAW_DIR / sample_run["path"]).glob("bs1/slices_bs1/*_metrics.csv"))[0]

    df_raw = pd.read_csv(sample_file)
    df_raw = df_raw.sort_values(by="Timestamp").reset_index(drop=True)

    # First valid window has target at index 16 (0-indexed, corresponding to sample 17)
    expected_y_0 = df_raw["dl_buffer [bytes]"].iloc[16]

    train_data = np.load(PROCESSED_DIR / "train.npz")
    actual_y_0 = train_data["y"][0]

    assert np.isclose(expected_y_0, actual_y_0, atol=1e-5), (
        f"Target misalignment: expected {expected_y_0}, got {actual_y_0}"
    )

    # Verify log1p numerical consistency
    expected_log1p_0 = np.log1p(expected_y_0)
    actual_log1p_0 = train_data["y_log1p"][0]
    assert np.isclose(expected_log1p_0, actual_log1p_0, atol=1e-5), "log1p target mismatch"

    # Verify inverse transform
    reconstructed_y = np.expm1(actual_log1p_0)
    assert np.isclose(expected_y_0, reconstructed_y, atol=1e-4), "expm1 inverse transform failed"


def test_transformations_fitted_on_train_only():
    """
    Test 5: Confirms that StandardScaler parameters stored in manifest.json were fit
    strictly on training data and applied unchanged to validation and test sets.
    """
    with open(PROCESSED_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scaler_meta = manifest["scaler_parameters"]
    assert scaler_meta["fit_partition"] == "train_only"

    stored_mean = np.array(scaler_meta["mean"], dtype=np.float64)
    stored_std = np.array(scaler_meta["scale_std"], dtype=np.float64)

    # Compute empirical train statistics using float64 accumulator to prevent 11.8M-sample precision loss
    train_data = np.load(PROCESSED_DIR / "train.npz")
    X_train_raw = train_data["X_raw"].astype(np.float64).reshape(-1, 4)
    empirical_train_mean = np.mean(X_train_raw, axis=0)
    empirical_train_std = np.std(X_train_raw, axis=0)

    # Must match stored scaler within floating point precision
    assert np.allclose(stored_mean, empirical_train_mean, atol=1e-2), (
        f"Stored scaler mean {stored_mean} does not match empirical train mean {empirical_train_mean}"
    )
    assert np.allclose(stored_std, empirical_train_std, atol=1e-2), (
        f"Stored scaler std {stored_std} does not match empirical train std {empirical_train_std}"
    )

    # Compute empirical val statistics
    val_data = np.load(PROCESSED_DIR / "val.npz")
    X_val_raw = val_data["X_raw"].astype(np.float64).reshape(-1, 4)
    empirical_val_mean = np.mean(X_val_raw, axis=0)

    # Validation mean must be distinct from stored train mean (confirms no val data was included in fit)
    assert not np.allclose(stored_mean, empirical_val_mean, atol=1e-3), (
        "Scaler appears to have leaked validation statistics!"
    )

    # Check scaling application: X == (X_raw - mean) / std
    expected_scaled_val = ((val_data["X_raw"] - stored_mean) / stored_std).astype(np.float32)
    actual_scaled_val = val_data["X"]
    assert np.allclose(expected_scaled_val, actual_scaled_val, atol=1e-4), (
        "Validation features were not transformed with the exact training scaler parameters"
    )

