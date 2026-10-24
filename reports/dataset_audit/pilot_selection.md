# Pilot Dataset Selection and Target Definition Report

**Target Track**: IEEE ICC 2027 — Mobile and Wireless Networks  
**Research Topic**: Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing  
**Protocol Version**: 1.0.0  
**Audit & Protocol Date**: October 2026  
**Deterministic Random Seed**: `2026`  
**Configuration File**: [`configs/data_protocol.yaml`](../../configs/data_protocol.yaml)

---

## 1. Executive Summary & Objective

This report formalizes Phase 3.2: the **pilot dataset selection, target formulation, causal feature availability rules, and leakage-free splitting protocol** for benchmarking the proposed Agentic Temporal Kolmogorov-Arnold Network (TKAN).

The core objective is to formulate a **near-real-time downlink buffer backlog forecasting benchmark** that directly empowers an autonomous RAN slicing controller. By accurately predicting queue backlog surges before they manifest as packet drops or latency violations, the agentic controller can proactively reallocate Physical Resource Blocks (PRBs) and adjust slice scheduling weights at the sub-second timescale.

---

## 2. Primary Target Definition & Verification

### 2.1 Target Variable
- **Target Field**: `dl_buffer [bytes]`
- **Measurement Type**: Direct measurement (RLC/MAC queue buffer occupancy).
- **Physical Unit**: Bytes ($B$).
- **Verification of Nature**:
  - The Phase 3.1 audit confirmed that `dl_buffer [bytes]` is a **direct physical queue measurement** reported by the srsLTE protocol stack every $250\text{ ms}$.
  - **Not a packet-delay metric**: It does not measure milliseconds or transmission delay; it records the discrete volume of unacknowledged/untransmitted payload bytes buffered in the base station for that UE.
  - In the original Colosseum benchmark, the authors scaled this metric by $10^5$ to compute state representations for DRL slicing agents.

### 2.2 Forecasting Horizon & Lookback Parameters
- **Forecasting Formulation**:
  $$y(t+1) = \text{dl\_buffer}[t+1]$$
  predicting the buffer state at the immediately following reporting interval for the same slice, UE, base station, and continuous experimental recording.
- **Lookback Window**: $W = 16\text{ samples}$.
  - At the native sampling interval ($\Delta t = 250.0\text{ ms}$), $16\text{ samples} = 4.0\text{ seconds}$ of historical context.
- **Forecasting Horizon**: $H = 1\text{ sample}$.
  - At the native sampling interval, $1\text{ sample} = 0.25\text{ seconds}$ lookahead.
  - This $250\text{ ms}$ lookahead aligns with near-RT RIC control loop execution intervals ($10\text{ ms} - 1\text{ s}$).

---

## 3. Input Features & Causal Availability Model

### 3.1 Candidate Input Features
At time step $t$, the input feature vector $\mathbf{x}_t \in \mathbb{R}^4$ is constructed from four core telemetry metrics:

| Feature Name | Dtype | Unit | Causal Availability & Verification |
| :--- | :---: | :---: | :--- |
| **`dl_buffer [bytes]`** | `float64` | Bytes | Instantaneous queue backlog measured at the close of time window $t$. Available at $t$. |
| **`tx_brate downlink [Mbps]`** | `float64` | Mbps | Total downlink physical bitrate transmitted during the preceding $250\text{ ms}$ window ending at $t$. |
| **`sum_requested_prbs`** | `float64` | Count | Cumulative PRBs requested across all subframes in the window ending at $t$. |
| **`sum_granted_prbs`** | `float64` | Count | Cumulative PRBs scheduled/granted by the MAC scheduler in the window ending at $t$. |

### 3.2 Temporal Causality Guarantee
- **Strictly Causal**: All input values in the lookback window $\mathbf{X}_{t-15:t} = [\mathbf{x}_{t-15}, \dots, \mathbf{x}_t]$ are restricted to information that has already been generated and reported by the base station up to time $t$.
- **No Future Leakage**: The target $y(t+1)$ is strictly at step $t+1$. No features from step $t+1$ (such as granted PRBs at $t+1$) are accessible to the model at inference time $t$.

### 3.3 Slice Grouping & Conditioning
- **Field**: `slice_id` (`int64` taking values $\{0, 1, 2\}$).
- **Physical Meaning**:
  - `0`: eMBB (Constant Bitrate streaming, $1\text{ Mbps}$).
  - `1`: MTC (Periodic small packets, $30\text{ pkt/s}$ of $125\text{ B}$).
  - `2`: URLLC (Bursty Poisson traffic, $10\text{ pkt/s}$ of $125\text{ B}$).
- **Handling Rule**: `slice_id` must **never** be treated as a continuous numeric feature (to avoid implying that Slice 2 is "twice" Slice 1).
  - Primary strategy: Train separate temporal models per slice, or use one-hot categorical conditioning / slice-specific agentic heads.

---

## 4. Pilot Dataset Selection

### 4.1 Selection Rationale
To ensure maximum benchmark integrity without premature computational bloat, candidate experimental recordings were audited against three strict quality gates:
1. **Full-Length Continuity**: Replicates must run for the full planned duration ($\approx 456\text{ s}$, $\ge 1,700\text{ samples}$) without premature software radio crashes.
2. **Complete Cell Coverage**: All 4 base stations (`bs1` to `bs4`) and all 40 UEs must be present with uninterrupted connectivity ($40\text{ slice metric CSVs}$).
3. **Pristine Temporal Stability**: No missing timestamps or transmission gaps $>500\text{ ms}$.

From the 39 pristine recordings identified in Phase 3.1, a **14-run representative in-distribution pilot pool** was selected from `slice_traffic/rome_static_close`, plus **1 dedicated out-of-distribution (OOD) mobility recording** from `slice_traffic/rome_slow_close`.

### 4.2 Configuration Diversity in Selected Pool
The selected 14 runs span three distinct baseline scheduling regimes:
- **`tr1`** (5 runs): Slice 0 ($WF$), Slice 1 ($RR$), Slice 2 ($RR$); Initial RBG quotas ($1, 4, 2$).
- **`tr10`** (4 runs): Slice 0 ($RR$), Slice 1 ($RR$), Slice 2 ($PF$); Initial RBG quotas ($3, 1, 3$).
- **`tr11`** (5 runs): Slice 0 ($RR$), Slice 1 ($PF$), Slice 2 ($RR$); Initial RBG quotas ($1, 3, 3$).

---

## 5. Leakage-Free Grouped Splitting Protocol

### 5.1 Partitioning Level
The split is executed strictly at the **independent experimental recording level** (`scenario/tr_config/exp_id`).
- UEs within the same experiment share the same physical time, base stations, and inter-cell interference environment.
- Therefore, overlapping sliding windows must **never** be randomly split across train and test. Entire 456-second experiments are placed in either Train, Validation, or Test.

### 5.2 Deterministic Split Assignment (Seed = 2026)
Applying a pseudo-random permutation with seed `2026` yielded the following division:

| Partition | Number of Runs | Share of Pool | Total Slice CSVs | Total Telemetry Rows |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | **10** | **71.4%** | $400$ | $\approx 742,000$ |
| **Validation Set** | **2** | **14.3%** | $80$ | $\approx 148,000$ |
| **Test Set** | **2** | **14.3%** | $80$ | $\approx 149,000$ |
| **OOD Stress Test** | **1** | — | $40$ | $\approx 58,000$ |
| **Total Pilot Corpus**| **15** | **100.0%** | **600** | **$\approx 1,097,000$** |

### 5.3 Detailed Split Roster & File Hashes

#### Training Set ($71.4\%$, 10 Independent Runs)
1. `rome_static_close/tr10/exp2` — 40 CSVs  
   Sample SHA-256: `c96f0e535dafe30f5962cc16fcd6b4207d86834a59d442f3d14ed75ba136c8dd`
2. `rome_static_close/tr1/exp3` — 40 CSVs  
   Sample SHA-256: `e81d2cb93c2c128650c5d08d304fe0f26fcc3c06ee2d0a996026b7907ab947af`
3. `rome_static_close/tr10/exp6` — 40 CSVs  
   Sample SHA-256: `369fee5daf01204dc2b136f00c8ae19853cd6d9d2abd2d70385732456d04322d`
4. `rome_static_close/tr11/exp2` — 40 CSVs  
   Sample SHA-256: `f77a1f368e00da9c41443b0d0b541f174f3c58d9d854d5fb73d5b7e7e83d6f74`
5. `rome_static_close/tr11/exp5` — 40 CSVs  
   Sample SHA-256: `d97c7ebaeda6b99b5796eb399d8f4760c29c0d6088d9379b33bd0c7ba85bf704`
6. `rome_static_close/tr1/exp1` — 40 CSVs  
   Sample SHA-256: `2b29f1f0b09039cbf1d413685210757a5102015cb4b7377df719be2e1e4437d6`
7. `rome_static_close/tr1/exp4` — 40 CSVs  
   Sample SHA-256: `82397a809b3b802c9e674c4456ef7bc747c7236309466653be2a497aeef497f1`
8. `rome_static_close/tr11/exp4` — 40 CSVs  
   Sample SHA-256: `8c1ab4ab67343a664574f84237a647b16734f29375cdbe32fb8f5ba731daddb9`
9. `rome_static_close/tr1/exp5` — 40 CSVs  
   Sample SHA-256: `d95028d2c4fb2642e2db4073aeb8461b0b6eae17f21dd06fe7b65370aa942111`
10. `rome_static_close/tr10/exp1` — 40 CSVs  
    Sample SHA-256: `7ea053f355c0bc15086a51d6cfaa63a000c5d9059b67292004c8192632a0f6f0`

#### Validation Set ($14.3\%$, 2 Independent Runs)
1. `rome_static_close/tr11/exp6` — 40 CSVs  
   Sample SHA-256: `3b588c83ff32f061629961e572203073a33a1091e7ba54a2d088ba2ec3b394b1`
2. `rome_static_close/tr10/exp4` — 40 CSVs  
   Sample SHA-256: `0fe654518d5c353202ee4c43a18ce326eb196792f5f8505483bcd36c9ba2cbfc`

#### In-Distribution Test Set ($14.3\%$, 2 Independent Runs)
1. `rome_static_close/tr1/exp6` — 40 CSVs  
   Sample SHA-256: `3d67a2febd7105eb8315c848cfd991e11903504337dfecf9b0a8fafb946cd4c4`
2. `rome_static_close/tr11/exp3` — 40 CSVs  
   Sample SHA-256: `64dfe55d0e85a9110eb802e3f2d91bc284fc5ccd804415a8bd9fa6538ad983f5`

#### Out-of-Distribution (OOD) Test Set (1 Mobility Run)
1. `rome_slow_close/tr6/exp2` — 40 CSVs ($3\text{ m/s}$ pedestrian speed)  
   Sample SHA-256: `11071b22d926d445c4bb566384a60e128492f4086df8b9c01aeb07d4be1ec29e`

---

## 6. Preprocessing Constraints & Leakage Safeguards

1. **Scaler Fitting**:
   - Any normalization (e.g. `StandardScaler` or `MinMaxScaler`) must be fit strictly on the **Training Set**.
   - The learned $\mu, \sigma$ or $\min, \max$ parameters are saved and applied to transform Validation and Test data without re-fitting.
2. **Window Construction**:
   - Temporal sliding windows ($16\text{ lookback} \to 1\text{ horizon}$) are generated **only within contiguous UE time-series boundaries**.
   - No window is allowed to cross between different UEs, different base stations, or different experiment runs.
3. **Imputation & Missingness Rules**:
   - If consecutive timestamp gaps $>500\text{ ms}$ occur within a UE sequence, the sequence must be split into independent contiguous segments rather than forward-filling missing steps.

---

## 7. Rejection Rationale for Other Candidate Recordings

During the Phase 3.1 audit, multiple candidate recordings were evaluated and rejected:
1. **Premature Termination**:
   - e.g., `rome_static_close/tr2/exp6` ran for only $4\text{ seconds}$ ($17\text{ rows}$) before aborting.
   - e.g., `rome_static_close/tr3/exp1` ran for only $61\text{ seconds}$ ($245\text{ rows}$).
   - **Reason for Rejection**: Too short to support temporal windowing and model training.
2. **Incomplete Cell Clusters**:
   - e.g., `rome_static_close/tr0/exp1` recorded only 39 UEs (1 UE failed to attach).
   - **Reason for Rejection**: Distorts multi-cell aggregate resource competition and slice loading.
3. **High Mobility Dropouts in Early `rome_slow_close` Runs**:
   - In several `rome_slow_close` runs, UEs dropped off within $5$ samples due to channel fades.
   - **Reason for Rejection**: Kept outside the primary in-distribution benchmark, but isolated as a dedicated out-of-distribution stress test.

---

## 8. Summary Table of Deliverables

- **YAML Protocol Specification**: [`configs/data_protocol.yaml`](../../configs/data_protocol.yaml)
- **Selection & Audit Documentation**: [`reports/dataset_audit/pilot_selection.md`](pilot_selection.md)
- **Status**: Ready for review. Preprocessing implementation will proceed upon approval.
