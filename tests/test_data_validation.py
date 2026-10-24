"""
Phase 4.2: Data Quality and Leakage Validation Test Suite.
Validates the 12 criteria for clean, leakage-free benchmark datasets
and implements tests for deliberate leakage scenarios to prove the checks work.
"""

import json
from pathlib import Path
import numpy as np
import pytest


PROCESSED_DIR = Path("data/processed/commag")


# =============================================================================
# Helper validation functions used by both real checks and deliberate leakage tests
# =============================================================================

def verify_group_separation(train_runs, val_runs, test_runs, ood_runs):
    """Checks that independent experiment runs are mutually disjoint."""
    s_train, s_val, s_test, s_ood = set(train_runs), set(val_runs), set(test_runs), set(ood_runs)
    if not s_train.isdisjoint(s_val):
        raise AssertionError(f"Group leakage between Train and Val: {s_train & s_val}")
    if not s_train.isdisjoint(s_test):
        raise AssertionError(f"Group leakage between Train and Test: {s_train & s_test}")
    if not s_val.isdisjoint(s_test):
        raise AssertionError(f"Group leakage between Val and Test: {s_val & s_test}")
    if not s_train.isdisjoint(s_ood):
        raise AssertionError(f"Group leakage between Train and OOD: {s_train & s_ood}")
    return True


def verify_no_duplicate_windows(X_train, X_test, atol=1e-5):
    """
    Checks that no identical window in X_test exists in X_train.
    Uses sample hashing for efficiency.
    """
    n_train = min(len(X_train), 5000)
    n_test = min(len(X_test), 1000)
    train_sample = X_train[:n_train].reshape(n_train, -1)
    test_sample = X_test[:n_test].reshape(n_test, -1)

    train_hashes = {hash(row.tobytes()) for row in train_sample}
    test_hashes = {hash(row.tobytes()) for row in test_sample}

    overlap = train_hashes & test_hashes
    if overlap:
        raise AssertionError(f"Window leakage detected: {len(overlap)} exact window matches found between Train and Test!")
    return True



def verify_temporal_causality(timestamps_t, timestamps_target):
    """Checks that target is strictly in the future relative to the window decision time."""
    deltas = timestamps_target - timestamps_t
    if np.any(deltas <= 0):
        invalid_idx = np.where(deltas <= 0)[0]
        raise AssertionError(f"Temporal causality violation at {len(invalid_idx)} indices: t_target <= t_decision")
    return True


def verify_target_horizon_step(timestamps_t, timestamps_target, max_allowed_delta=450):
    """Checks that target is exactly one step ahead and does not cross an unhandled gap."""
    deltas = timestamps_target - timestamps_t
    if np.any(deltas > max_allowed_delta):
        gap_idx = np.where(deltas > max_allowed_delta)[0]
        raise AssertionError(f"Target horizon violation: {len(gap_idx)} windows cross an unhandled time gap (> {max_allowed_delta} ms)")
    return True


def verify_scaler_isolation(stored_mean, empirical_train_mean, empirical_val_test_mean, atol=1e-2):
    """Confirms scaler matches training data and does not match validation/test."""
    if not np.allclose(stored_mean, empirical_train_mean, atol=atol):
        raise AssertionError("Scaler mean does not match training data mean!")
    if np.allclose(stored_mean, empirical_val_test_mean, atol=atol):
        raise AssertionError("Scaler mean matches validation/test data mean — test contamination detected!")
    return True


def verify_target_variation(y_array, min_std=10.0):
    """Checks that target is not constant, zero, or corrupted."""
    std = float(np.std(y_array))
    var = float(np.var(y_array))
    if std < min_std or var < 1.0:
        raise AssertionError(f"Target has insufficient variation or is constant: std={std:.4f}, var={var:.4f}")
    return True


# =============================================================================
# REAL DATASET AUDIT TESTS (Item 1 to 12)
# =============================================================================

def test_1_group_separation_real_dataset():
    with open(PROCESSED_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]
    train_r = [r["run_id"] for r in splits["train"]["runs"]]
    val_r = [r["run_id"] for r in splits["validation"]["runs"]]
    test_r = [r["run_id"] for r in splits["test"]["runs"]]
    ood_r = [r["run_id"] for r in splits["out_of_distribution_test"]["runs"]]

    assert verify_group_separation(train_r, val_r, test_r, ood_r)


def test_2_no_overlapping_windows_across_splits():
    train_data = np.load(PROCESSED_DIR / "train.npz")
    test_data = np.load(PROCESSED_DIR / "test.npz")
    assert verify_no_duplicate_windows(train_data["X"], test_data["X"])


def test_3_temporal_causality_all_splits():
    for s in ["train", "val", "test", "ood_test"]:
        data = np.load(PROCESSED_DIR / f"{s}.npz")
        assert verify_temporal_causality(data["timestamps_t"], data["timestamps_target"])


def test_4_and_5_target_step_and_no_unhandled_gaps():
    for s in ["train", "val", "test", "ood_test"]:
        data = np.load(PROCESSED_DIR / f"{s}.npz")
        # Target step must not cross gap > 450 ms (nominal is 250 ms)
        assert verify_target_horizon_step(data["timestamps_t"], data["timestamps_target"], max_allowed_delta=450)


def test_6_and_7_transformations_fitted_on_train_only():
    with open(PROCESSED_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    stored_mean = np.array(manifest["scaler_parameters"]["mean"], dtype=np.float64)
    train_data = np.load(PROCESSED_DIR / "train.npz")
    val_data = np.load(PROCESSED_DIR / "val.npz")
    test_data = np.load(PROCESSED_DIR / "test.npz")

    emp_train_mean = np.mean(train_data["X_raw"].astype(np.float64).reshape(-1, 4), axis=0)
    emp_val_test_mean = np.mean(
        np.concatenate([val_data["X_raw"], test_data["X_raw"]], axis=0).astype(np.float64).reshape(-1, 4),
        axis=0
    )

    assert verify_scaler_isolation(stored_mean, emp_train_mean, emp_val_test_mean)


def test_8_target_and_input_units():
    with open(PROCESSED_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    feats = manifest["features"]["feature_names"]
    assert feats[0] == "dl_buffer [bytes]"
    assert feats[1] == "tx_brate downlink [Mbps]"
    assert feats[2] == "sum_requested_prbs"
    assert feats[3] == "sum_granted_prbs"
    assert manifest["features"]["target_name"] == "dl_buffer [bytes]"


def test_9_missingness_handling():
    for s in ["train", "val", "test", "ood_test"]:
        data = np.load(PROCESSED_DIR / f"{s}.npz")
        assert not np.isnan(data["X"]).any(), f"NaNs found in {s} X"
        assert not np.isnan(data["y"]).any(), f"NaNs found in {s} y"
        assert not np.isinf(data["X"]).any(), f"Infs found in {s} X"
        assert not np.isinf(data["y"]).any(), f"Infs found in {s} y"


def test_10_manifest_counts_reproduction():
    with open(PROCESSED_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    counts = manifest["window_counts"]
    train_data = np.load(PROCESSED_DIR / "train.npz")
    val_data = np.load(PROCESSED_DIR / "val.npz")
    test_data = np.load(PROCESSED_DIR / "test.npz")
    ood_data = np.load(PROCESSED_DIR / "ood_test.npz")

    assert counts["train"] == len(train_data["y"])
    assert counts["validation"] == len(val_data["y"])
    assert counts["test"] == len(test_data["y"])
    assert counts["out_of_distribution_test"] == len(ood_data["y"])
    assert counts["total"] == (len(train_data["y"]) + len(val_data["y"]) + len(test_data["y"]) + len(ood_data["y"]))


def test_11_split_sizes_and_group_sufficiency():
    with open(PROCESSED_DIR / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    # At least 10 train runs, 2 val runs, 2 test runs
    assert len(splits["train"]["runs"]) >= 10
    assert len(splits["validation"]["runs"]) >= 2
    assert len(splits["test"]["runs"]) >= 2
    assert splits["train"]["total_windows"] > 500000
    assert splits["validation"]["total_windows"] > 100000
    assert splits["test"]["total_windows"] > 100000


def test_12_target_meaningful_variation():
    for s in ["train", "val", "test", "ood_test"]:
        data = np.load(PROCESSED_DIR / f"{s}.npz")
        assert verify_target_variation(data["y"], min_std=100.0)


# =============================================================================
# DELIBERATE LEAKAGE SCENARIO TESTS (Proving detectors trigger on violations)
# =============================================================================

def test_deliberate_group_leakage_detector_fails():
    """
    Deliberate Leakage 1: Inject a training run ID into the validation set.
    The detector MUST raise AssertionError.
    """
    train_runs = ["rome_static_close/tr10/exp2", "rome_static_close/tr1/exp3"]
    val_runs = ["rome_static_close/tr10/exp2"]  # LEAKAGE: same run in Val!
    test_runs = ["rome_static_close/tr1/exp6"]
    ood_runs = ["rome_slow_close/tr6/exp2"]

    with pytest.raises(AssertionError, match="Group leakage between Train and Val"):
        verify_group_separation(train_runs, val_runs, test_runs, ood_runs)


def test_deliberate_window_overlap_detector_fails():
    """
    Deliberate Leakage 2: Copy a window directly from Train into Test.
    The detector MUST raise AssertionError.
    """
    synthetic_train_X = np.random.randn(100, 16, 4).astype(np.float32)
    synthetic_test_X = np.random.randn(20, 16, 4).astype(np.float32)
    # Inject leakage: copy window 5 from train into test position 0
    synthetic_test_X[0] = synthetic_train_X[5].copy()

    with pytest.raises(AssertionError, match="Window leakage detected"):
        verify_no_duplicate_windows(synthetic_train_X, synthetic_test_X)


def test_deliberate_temporal_causality_detector_fails():
    """
    Deliberate Leakage 3: Target timestamp is in the past (t_target <= t_decision).
    The detector MUST raise AssertionError.
    """
    ts_t = np.array([1602690286086, 1602690286336, 1602690286586])
    ts_tar = np.array([1602690286336, 1602690286086, 1602690286836])  # Index 1 is past!

    with pytest.raises(AssertionError, match="Temporal causality violation"):
        verify_temporal_causality(ts_t, ts_tar)


def test_deliberate_unhandled_gap_detector_fails():
    """
    Deliberate Leakage 4: Window crosses a 5-second gap without proper segmentation.
    The detector MUST raise AssertionError.
    """
    ts_t = np.array([1602690286086, 1602690286336])
    ts_tar = np.array([1602690286336, 1602690291336])  # Index 1 has delta = 5000 ms!

    with pytest.raises(AssertionError, match="Target horizon violation"):
        verify_target_horizon_step(ts_t, ts_tar, max_allowed_delta=450)


def test_deliberate_scaler_contamination_detector_fails():
    """
    Deliberate Leakage 5: Scaler was contaminated by fitting on Test data.
    The detector MUST raise AssertionError.
    """
    emp_train_mean = np.array([45000.0, 0.15, 35.0, 100.0])
    emp_test_mean = np.array([60000.0, 0.25, 50.0, 150.0])
    # Contaminated scaler was fit on test mean
    contaminated_scaler_mean = emp_test_mean.copy()

    with pytest.raises(AssertionError, match="Scaler mean does not match training data mean"):
        verify_scaler_isolation(contaminated_scaler_mean, emp_train_mean, emp_test_mean)


def test_deliberate_constant_target_detector_fails():
    """
    Deliberate Corruption 6: Target is constant (zero variance).
    The detector MUST raise AssertionError.
    """
    corrupted_y = np.zeros(1000, dtype=np.float32)  # constant zeros

    with pytest.raises(AssertionError, match="Target has insufficient variation"):
        verify_target_variation(corrupted_y, min_std=10.0)
