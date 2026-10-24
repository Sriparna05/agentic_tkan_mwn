# Reproducible Preprocessing Report: Colosseum O-RAN COMMAG Benchmark

**Target Track**: IEEE ICC 2027 — Mobile and Wireless Networks  
**Research Topic**: Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing  
**Protocol Specification**: [`configs/data_protocol.yaml`](../../configs/data_protocol.yaml)  
**Execution Script**: [`scripts/preprocess_commag.py`](../../scripts/preprocess_commag.py)  
**Git Revision**: `6df3cc067ef7a75373603dd22740360b49399c34`  
**Execution Date**: October 2026  

---

## 1. Executive Summary

Phase 4.1 implemented a small-memory, fully reproducible preprocessing pipeline for the Colosseum O-RAN COMMAG dataset. The pipeline processed the **15 approved independent pilot runs** (14 in-distribution runs from `rome_static_close` + 1 out-of-distribution mobility run from `rome_slow_close`), preserving the original dataset files completely unchanged.

The pipeline generated **$1,088,061$ causal sliding windows** across four strictly disjoint splits, stored in compressed `.npz` format under [`data/processed/commag/`](../../data/processed/commag):
- **Training Set**: $737,270$ windows ($71.4\%$ of in-distribution pool)
- **Validation Set**: $146,977$ windows ($14.3\%$)
- **In-Distribution Test Set**: $146,971$ windows ($14.3\%$)
- **Out-of-Distribution (OOD) Test Set**: $56,843$ windows (mobility stress-test)

All feature normalization parameters were fit **strictly on training observations**, and temporal causal boundaries were enforced so that no lookahead or future information is accessible during inference.

---

## 2. Source Data Integrity & Quality Audit

### 2.1 File Ingestion and Verification
- **Total Raw Telemetry Rows Ingested**: $1,097,693$ rows across $600$ individual slice metric CSV files.
- **Exact Schema Enforcement**: Every source CSV was pre-screened to verify the presence and data types of:
  `['Timestamp', 'slice_id', 'IMSI', 'dl_buffer [bytes]', 'tx_brate downlink [Mbps]', 'sum_requested_prbs', 'sum_granted_prbs']`.
- **Zero In-File Duplicates**: $0$ duplicate rows and $0$ duplicate timestamps were detected in the approved runs.
- **Gap Detection & Handling**: Consecutive timestamps exceeding $500.0\text{ ms}$ ($>2\times$ sampling interval $\Delta t = 250\text{ ms}$) triggered an immediate sequence split. Exactly $2$ gaps were detected across the $1.097\text{M}$ observations, creating independent contiguous sub-segments. No missing intervals were interpolated or resampled.
- **Short Sequence Rejection**: Any sub-segment with length $< 17$ ($< W + H$) was rejected to prevent truncated windows. In the selected pilot runs, zero sub-segments required rejection because all segments had full length $\approx 1,830$ samples.

---

## 3. Causal Feature Engineering & Target Formulation

### 3.1 Target Formulation
- **Target Variable**: $y(t+1) = \text{dl\_buffer}[t+1]$
- **Physical Meaning**: Downlink RLC/MAC transmission queue buffer backlog in bytes at the subsequent sampling step ($t + 250\text{ ms}$).
- **Non-Negative Target & `log1p` Transform**:
  - Raw `dl_buffer [bytes]` is bimodal: $56.6\%$ of observations are exactly $0\text{ B}$ (drained queue), while backlogged bursts reach $184,665\text{ B}$.
  - To stabilize gradient propagation in neural architectures (e.g., Temporal KAN), the target is transformed via:
    $$\tilde{y} = \log(1 + y)$$
    mapping $[0, 184665]$ monotonically to $[0, 12.126]$.
  - For benchmark evaluation, predictions $\hat{\tilde{y}}$ are inverse-transformed via $\hat{y} = \exp(\hat{\tilde{y}}) - 1$ before reporting MAE and RMSE in original physical bytes. Both raw $y$ and $\tilde{y}$ are saved in the `.npz` archives.

### 3.2 Input Feature Matrix
For each sample $i$, the input tensor $\mathbf{X}_i \in \mathbb{R}^{16 \times 4}$ represents a lookback of $16\text{ steps}$ ($4.0\text{ seconds}$):
$$\mathbf{X}_i = \begin{bmatrix}
\text{dl\_buffer}[t-15] & \text{tx\_brate}[t-15] & \text{req\_prbs}[t-15] & \text{granted\_prbs}[t-15] \\
\vdots & \vdots & \vdots & \vdots \\
\text{dl\_buffer}[t] & \text{tx\_brate}[t] & \text{req\_prbs}[t] & \text{granted\_prbs}[t]
\end{bmatrix}$$
- Strictly causal: All values are observed at or before decision step $t$.
- Grouping variable: `slice_id` is tracked for every window ($0$=eMBB, $1$=MTC, $2$=URLLC).

---

## 4. Leakage-Free Splitting & Transformation Parameters

### 4.1 Independent Group Splitting
Splits were constructed strictly at the **entire experimental run boundary** (`scenario/tr_config/exp_id`) using deterministic seed `2026`:

| Split | Run ID | Source Scenario / Config | Total Files | Sample SHA-256 Hash |
| :--- | :--- | :--- | :---: | :--- |
| **Train** | `rome_static_close/tr10/exp2` | `slice_traffic/rome_static_close/tr10/exp2` | 40 | `c96f0e535dafe30f5962cc16fcd6b4207d86834a59d442f3d14ed75ba136c8dd` |
| **Train** | `rome_static_close/tr1/exp3` | `slice_traffic/rome_static_close/tr1/exp3` | 40 | `e81d2cb93c2c128650c5d08d304fe0f26fcc3c06ee2d0a996026b7907ab947af` |
| **Train** | `rome_static_close/tr10/exp6` | `slice_traffic/rome_static_close/tr10/exp6` | 40 | `369fee5daf01204dc2b136f00c8ae19853cd6d9d2abd2d70385732456d04322d` |
| **Train** | `rome_static_close/tr11/exp2` | `slice_traffic/rome_static_close/tr11/exp2` | 40 | `f77a1f368e00da9c41443b0d0b541f174f3c58d9d854d5fb73d5b7e7e83d6f74` |
| **Train** | `rome_static_close/tr11/exp5` | `slice_traffic/rome_static_close/tr11/exp5` | 40 | `d97c7ebaeda6b99b5796eb399d8f4760c29c0d6088d9379b33bd0c7ba85bf704` |
| **Train** | `rome_static_close/tr1/exp1` | `slice_traffic/rome_static_close/tr1/exp1` | 40 | `2b29f1f0b09039cbf1d413685210757a5102015cb4b7377df719be2e1e4437d6` |
| **Train** | `rome_static_close/tr1/exp4` | `slice_traffic/rome_static_close/tr1/exp4` | 40 | `82397a809b3b802c9e674c4456ef7bc747c7236309466653be2a497aeef497f1` |
| **Train** | `rome_static_close/tr11/exp4` | `slice_traffic/rome_static_close/tr11/exp4` | 40 | `8c1ab4ab67343a664574f84237a647b16734f29375cdbe32fb8f5ba731daddb9` |
| **Train** | `rome_static_close/tr1/exp5` | `slice_traffic/rome_static_close/tr1/exp5` | 40 | `d95028d2c4fb2642e2db4073aeb8461b0b6eae17f21dd06fe7b65370aa942111` |
| **Train** | `rome_static_close/tr10/exp1` | `slice_traffic/rome_static_close/tr10/exp1` | 40 | `7ea053f355c0bc15086a51d6cfaa63a000c5d9059b67292004c8192632a0f6f0` |
| **Validation** | `rome_static_close/tr11/exp6` | `slice_traffic/rome_static_close/tr11/exp6` | 40 | `3b588c83ff32f061629961e572203073a33a1091e7ba54a2d088ba2ec3b394b1` |
| **Validation** | `rome_static_close/tr10/exp4` | `slice_traffic/rome_static_close/tr10/exp4` | 40 | `0fe654518d5c353202ee4c43a18ce326eb196792f5f8505483bcd36c9ba2cbfc` |
| **Test** | `rome_static_close/tr1/exp6` | `slice_traffic/rome_static_close/tr1/exp6` | 40 | `3d67a2febd7105eb8315c848cfd991e11903504337dfecf9b0a8fafb946cd4c4` |
| **Test** | `rome_static_close/tr11/exp3` | `slice_traffic/rome_static_close/tr11/exp3` | 40 | `64dfe55d0e85a9110eb802e3f2d91bc284fc5ccd804415a8bd9fa6538ad983f5` |
| **OOD Test** | `rome_slow_close/tr6/exp2` | `slice_traffic/rome_slow_close/tr6/exp2` | 40 | `11071b22d926d445c4bb566384a60e128492f4086df8b9c01aeb07d4be1ec29e` |

### 4.2 Fitted Transformation Parameters (Train Only)
Scaler statistics were fit across all $737,270 \times 16 = 11,796,320$ training observation points:

| Feature Index | Feature Name | Mean ($\mu$) | Standard Deviation ($\sigma$) | Minimum | Maximum |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | `dl_buffer [bytes]` | $45,562.430$ | $78,750.250$ | $0.0$ | $184,665.0$ |
| `1` | `tx_brate downlink [Mbps]` | $0.1477$ | $0.2625$ | $0.0$ | $3.6222$ |
| `2` | `sum_requested_prbs` | $34.834$ | $163.887$ | $-292.0$ | $2,128.0$ |
| `3` | `sum_granted_prbs` | $103.426$ | $138.702$ | $0.0$ | $1,440.0$ |

These parameters are recorded in [`data/processed/commag/manifest.json`](../../data/processed/commag/manifest.json) and applied unchanged to transform validation, test, and OOD test arrays.

---

## 5. Artifact Manifest & Output Summary

The following artifacts have been created in [`data/processed/commag/`](../../data/processed/commag):

```text
data/processed/commag/
├── train.npz       # 737,270 windows (X, X_raw, y, y_log1p, slice_id, imsi, timestamps)
├── val.npz         # 146,977 windows
├── test.npz        # 146,971 windows
├── ood_test.npz    # 56,843 windows
├── splits.json     # Detailed run mapping, file paths, and window counts per split
└── manifest.json   # Full provenance, git revision, scaler parameters, and audit metrics
```

---

## 6. Automated Validation Suite Results

An automated test suite was constructed in [`tests/test_commag_preprocessing.py`](../../tests/test_commag_preprocessing.py) and executed via `pytest`:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Research-kan\agentic_tkan_mwn
collected 10 items

tests/test_commag_preprocessing.py::test_files_exist PASSED              [ 10%]
tests/test_commag_preprocessing.py::test_group_separation PASSED         [ 20%]
tests/test_commag_preprocessing.py::test_chronological_ordering PASSED   [ 30%]
tests/test_commag_preprocessing.py::test_feature_leakage_and_causality PASSED [ 40%]
tests/test_commag_preprocessing.py::test_target_alignment PASSED         [ 50%]
tests/test_commag_preprocessing.py::test_transformations_fitted_on_train_only PASSED [ 60%]
tests/test_environment.py::test_environment_initialization PASSED        [ 70%]
tests/test_environment.py::test_environment_reset PASSED                 [ 80%]
tests/test_environment.py::test_environment_step PASSED                  [ 90%]
tests/test_environment.py::test_policies_execution PASSED                [100%]

============================= 10 passed in 6.57s ==============================
```

### Verified Properties
1. **Group Separation**: Confirmed zero run ID or source file overlap between Train, Validation, Test, and OOD sets ($Train \cap Val \cap Test = \emptyset$).
2. **Chronological Monotonicity**: Confirmed $t_{\text{target}} > t_{\text{window}}$ for all $1.088\text{M}$ samples, with median sampling interval $\Delta t = 250.0\text{ ms}$.
3. **Feature Leakage Prevention**: Confirmed that input windows $\mathbf{X}_i$ are strictly causal ($t-15 \dots t$), while target $y_i$ is at $t+1$.
4. **Target Alignment**: Confirmed exact floating-point identity between sampled processed targets and original source CSV values at $t+1$, as well as exact reversibility of $\text{expm1}(\text{log1p}(y))$.
5. **Train-Only Transformation**: Confirmed that scaler parameters in `manifest.json` match empirical training data while strictly diverging from validation and test empirical distributions.

The dataset is fully preprocessed, verified, and ready for baseline model training in Phase 4.2.
