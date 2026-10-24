# Dataset Audit and Schema Report: Colosseum O-RAN COMMAG Dataset

**Target Track**: IEEE ICC 2027 — Mobile and Wireless Networks  
**Research Direction**: Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing  
**Audit Date**: October 2026  
**Auditor**: Antigravity Assistant  
**Dataset Reference**: L. Bonati, S. D'Oro, M. Polese, S. Basagni, T. Melodia, *"Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks,"* IEEE Communications Magazine, vol. 59, no. 10, pp. 21–27, October 2021.

---

## Executive Summary

A comprehensive inventory and structural schema audit of the Colosseum O-RAN COMMAG dataset was executed. The dataset resides under `data/raw/commag/` and comprises **6 distinct scenario corpora** across **2,144 base-station experimental runs**, containing over **45,000 individual CSV files** totaling **~6.72 GB** on disk.

All 5 starter preprocessing fields (`slice_id`, `dl_buffer [bytes]`, `tx_brate downlink [Mbps]`, `sum_requested_prbs`, `sum_granted_prbs`) exist with **exact matching names, strictly numeric data types, and 0 missing values**. Timestamps are real Unix epoch milliseconds sampled at a strict **250 ms period (4.0 Hz)**. 

---

## 1. Workspace Inventory & Authoritative Location

### Directory Resolution
A workspace search confirmed that **exactly one authoritative instance** of the cloned repository exists on the system:
- **Authoritative Path**: `d:\Research-kan\agentic_tkan_mwn\data\raw\commag\`
- **Dataset Structure**:
  - `slice_traffic/`: Primary traffic-differentiated slice dataset (4 scenarios).
  - `slice_mixed/`: Mixed-traffic slice dataset (2 scenarios).
  - `ml_models/`: Pre-trained DRL benchmark agents (`embb_policy`, `mtc_policy`, `urllc_policy`, `encoder.h5`).
  - `COLOSSEUM_README.md`: Official experimental setup and protocol documentation.
  - `test_agent_release.py`: Authoritative reference data loader and inference harness.
- **Access Junctions**: NTFS junctions were created directly under `data/raw/commag/` for `rome_static_close`, `rome_slow_close`, `rome_static_medium`, and `rome_static_far`, enabling direct access without replicating 4.88 GB of data.

---

## 2. Global Dataset Inventory

| Dataset Sub-corpus | Scenario Name | Tr Configs | Total Exps | Slice CSVs | UE PHY CSVs | BS Aggregate CSVs | On-disk Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `slice_traffic` | `rome_static_close` | 18 (`tr0`–`tr17`) | 106 | 4,235 | 4,239 | 421 | 1.46 GB |
| `slice_traffic` | `rome_slow_close` | 18 (`tr0`–`tr17`) | 90 | 3,701 | 3,599 | 342 | 0.96 GB |
| `slice_traffic` | `rome_static_medium`| 18 (`tr0`–`tr17`) | 90 | 3,578 | 3,600 | 352 | 1.22 GB |
| `slice_traffic` | `rome_static_far` | 18 (`tr0`–`tr17`) | 90 | 3,638 | 3,599 | 353 | 1.12 GB |
| `slice_mixed` | `rome_static_close` | 16 (`tr0`–`tr15`) | 80 | 3,185 | 3,200 | 313 | 1.10 GB |
| `slice_mixed` | `rome_slow_close` | 16 (`tr0`–`tr15`) | 80 | 3,275 | 3,199 | 308 | 0.86 GB |
| **Total** | **6 Scenarios** | **104** | **536** | **21,612** | **21,436** | **2,089** | **6.72 GB** |

Detailed per-experiment metadata is cataloged in [`reports/dataset_audit/inventory.csv`](inventory.csv) (2,144 base-station rows).

---

## 3. Experimental Hierarchy & CSV Taxonomy

Each experiment follows a 5-tier directory hierarchy:
```text
data/raw/commag/
└── {dataset_type}/               # slice_traffic or slice_mixed
    └── {scenario}/               # rome_static_close, rome_slow_close, etc.
        └── {tr_config}/          # tr0 .. tr17 (Scheduling policies & RBG quotas)
            └── {experiment}/     # exp1 .. exp6 (Replicate runs)
                └── {base_station}/ # bs1 .. bs4 (Cell sites)
                    ├── bs{N}.csv # File Type A: Base Station Aggregates
                    ├── ue{M}.csv # File Type B: UE Physical Layer Telemetry
                    └── slices_bs{N}/
                        └── {IMSI}_metrics.csv # File Type C: Per-UE Slice Telemetry
```

### Taxonomy Justification: What Each CSV Represents
1. **Slice Metric CSVs (`slices_bs{N}/{IMSI}_metrics.csv`)**:
   - **Entity**: A continuous time series for **one individual User Equipment (UE)** identified by unique `IMSI` and temporary `RNTI` attached to base station `bs{N}`.
   - **Scope**: Reports slice-level MAC scheduler telemetry, buffer backlogs, CQI, MCS, and granted PRBs.
   - **Note**: It is **not** an aggregate summary of the entire slice; it is UE-granular.

2. **Base Station Aggregate CSVs (`bs{N}.csv`)**:
   - **Entity**: Cell-wide time series tracking the entire base station.
   - **Scope**: Active connected UE count (`nof_ue`), aggregate downlink bitrate (`dl_brate`), and aggregate uplink bitrate (`ul_brate`).

3. **UE Modem Physical CSVs (`ue{M}.csv`)**:
   - **Entity**: Physical layer receiver telemetry from the srsUE modem software radio stack.
   - **Scope**: Carrier configuration (`cc`, `pci`, `earfcn`), radio channel metrics (`rsrp`, path loss `pl`, `cfo`, `dl_snr`), Turbo decoder metrics (`dl_turbo`), and transport block BLER (`dl_bler`, `ul_bler`).

---

## 4. Multi-File Synchronization vs. Independence Analysis

- **Multi-Cell Synchronized Views**:
  - Within a single `{scenario}/{tr_config}/{experiment}/` directory, all base stations (`bs1` to `bs4`) and their respective UE metric files are **concurrent, synchronized observations of the same multi-cell network**.
  - All slice metric CSVs and BS aggregate CSVs record identical Unix epoch timestamps starting at the exact same millisecond (e.g., `1602690286086`), advancing in lockstep at $250\text{ ms}$ intervals.
  - This enables multi-agent and federated RAN modeling where BS agents coordinate under shared time frames.
- **Statistical Replicates**:
  - Different `exp` directories (e.g., `exp1` vs. `exp2` under `tr0`) are **independent, distinct experimental executions** under identical channel and traffic profiles, suitable for train/validation/test splitting.
- **Policy Variations**:
  - Different `tr` directories (`tr0` to `tr17`) correspond to different baseline slicing configurations (different initial combinations of Round-Robin, Waterfilling, and Proportional Fair schedulers and RBG quotas).

---

## 5. Telemetry Schema & Statistical Audit

Representative files across all scenarios were profiled. The schema is identical across scenarios.

### 5.1 Slice Telemetry Columns (`{IMSI}_metrics.csv`)
Total columns in file: **36** (31 active telemetry fields + 5 empty delimiter artifacts `Unnamed: 4, 10, 18, 28, 31` from srsLTE `,,` formatting).

| Index | Column Name | Data Type | Null Count | Value Range | Physical / Logical Meaning |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `Timestamp` | `int64` | 0 | $\approx 1.602\times 10^{12}$ | Unix Epoch timestamp in milliseconds |
| 2 | `num_ues` | `int64` | 0 | $1$ to $10$ | Active UEs associated with this base station |
| 3 | `IMSI` | `int64` | 0 | $1010123456001\dots$ | International Mobile Subscriber Identity |
| 4 | `RNTI` | `int64` | 0 | $60$ to $80$ | Radio Network Temporary Identifier |
| 6 | `slicing_enabled` | `int64` | 0 | $0$ or $1$ | Flag indicating whether slicing is active |
| 7 | **`slice_id`** | `int64` | 0 | $0, 1, 2$ | **Slice identifier**: 0=eMBB, 1=MTC, 2=URLLC |
| 8 | `slice_prb` | `int64` | 0 | $1$ to $8$ | RBG quota assigned to this slice (out of 15 PRBs total) |
| 9 | `power_multiplier`| `int64` | 0 | $1$ | Base station transmit power scaling factor |
| 10 | `scheduling_policy`| `int64` | 0 | $0, 1, 2$ | Slice scheduler: 0=Round-Robin, 1=Waterfilling, 2=Prop-Fair |
| 12 | `dl_mcs` | `float64` | 0 | $0.0$ to $28.0$ | Downlink Modulation and Coding Scheme index |
| 13 | `dl_n_samples` | `int64` | 0 | $0$ to $100$ | Number of subframe transmission samples in window |
| 14 | **`dl_buffer [bytes]`**| `int64` | 0 | $0$ to $>10^5$ | **Downlink RLC/MAC transmission queue buffer backlog** |
| 15 | **`tx_brate downlink [Mbps]`**| `float64`| 0 | $0.0$ to $3.5$ | **Physical transmitted downlink throughput in Mbps** |
| 16 | `tx_pkts downlink`| `int64` | 0 | $0$ to $>200$ | Count of downlink transport blocks / packets sent |
| 17 | `tx_errors downlink (%)`| `float64`| 0 | $0.0$ to $100.0$ | Downlink Block Error Rate (BLER) percentage |
| 18 | `dl_cqi` | `float64` | 0 | $0.0$ to $15.0$ | Downlink Channel Quality Indicator (0–15) |
| 20 | `ul_mcs` | `float64` | 0 | $0.0$ to $28.0$ | Uplink Modulation and Coding Scheme index |
| 21 | `ul_n_samples` | `int64` | 0 | $0$ to $100$ | Number of uplink transmission samples in window |
| 22 | `ul_buffer [bytes]`| `int64` | 0 | $0$ to $>10^5$ | Uplink buffer backlog reported via BSR |
| 23 | `rx_brate uplink [Mbps]`| `float64` | 0 | $0.0$ to $2.0$ | Physical received uplink throughput in Mbps |
| 24 | `rx_pkts uplink` | `int64` | 0 | $0$ to $>100$ | Count of uplink packets received |
| 25 | `rx_errors uplink (%)`| `float64` | 0 | $0.0$ to $100.0$ | Uplink Block Error Rate (BLER) percentage |
| 26 | `ul_rssi` | `float64` | 0 | $-100.0$ to $0.0$| Received Signal Strength Indicator in dBm |
| 27 | `ul_sinr` | `float64` | 0 | $-10.0$ to $35.0$| Uplink Signal-to-Interference-plus-Noise Ratio in dB |
| 28 | `phr` | `int64` | 0 | $0$ to $63$ | Power Headroom Report (dB relative to max power) |
| 30 | **`sum_requested_prbs`**| `int64` | 0 | $0$ to $>600$ | **Cumulative PRBs requested across subframes in window** |
| 31 | **`sum_granted_prbs`**| `int64` | 0 | $0$ to $>600$ | **Cumulative PRBs scheduled across subframes in window** |
| 33 | `dl_pmi` | `int64` | 0 | $0$ to $3$ | Precoding Matrix Indicator |
| 34 | `dl_ri` | `int64` | 0 | $1$ to $2$ | Rank Indicator (MIMO layers) |
| 35 | `ul_n` | `int64` | 0 | $0$ to $10$ | Uplink transmission parameter |
| 36 | `ul_turbo_iters` | `float64` | 0 | $1.0$ to $5.0$ | Average Turbo decoding iterations per code block |

### 5.2 Missing Values and Duplicate Rows
- **Missing Values**: **0 null values** across all active columns in `{IMSI}_metrics.csv`.
- **Duplicate Rows**: **0 duplicate rows** found in any audited scenario file.

### 5.3 Timestamp Units and Sampling Rate
- **Field**: `Timestamp`
- **Unit**: Unix Epoch Milliseconds (`ms`).
- **Sampling Interval ($\Delta t$)**: **$250.0\text{ ms}$** ($4.0\text{ Hz}$).
- **Interval Consistency**: $99.67\%$ of samples have exact $\Delta t = 250\text{ ms}$; remaining $<0.33\%$ show $\pm 1\text{ ms}$ jitter ($249$ or $251\text{ ms}$).

### 5.4 PRB Semantics (Counts vs. Rates vs. Aggregates)
- **`slice_prb`**: Static/quasi-static PRB quota assigned to the slice. For a $3\text{ MHz}$ LTE carrier ($15$ PRBs total), typical quotas are $1, 2, 4,$ or $8$ PRBs.
- **`sum_requested_prbs`**: Discrete cumulative count of PRBs requested across all $1\text{ ms}$ subframes within the $250\text{ ms}$ window (up to $250 \times 15 = 3750$ theoretical max).
- **`sum_granted_prbs`**: Discrete cumulative count of PRBs actually scheduled and allocated across all subframes in the window.
- **Satisfaction Ratio**: As defined in the reference literature, the ratio:
  $$\rho = \min\left(1.0, \frac{\text{sum\_granted\_prbs}}{\text{sum\_requested\_prbs} + \epsilon}\right)$$
  quantifies scheduling fulfillment and congestion.

---

## 6. Official Documented Meanings of Starter Fields

According to the official dataset documentation (`COLOSSEUM_README.md`) and original paper:
1. **`slice_id`**: Network slice index based on 3GPP traffic class:
   - `0` = **eMBB** (Enhanced Mobile Broadband): High constant bitrate streaming ($1\text{ Mbps}$ CBR per UE).
   - `1` = **MTC** (Machine-Type Communications): Periodic small-packet sensor traffic ($30\text{ pkt/s}$ of $125\text{ bytes}$).
   - `2` = **URLLC** (Ultra-Reliable Low-Latency Communications): Event-driven bursty traffic ($10\text{ pkt/s}$ of $125\text{ bytes}$).
2. **`dl_buffer [bytes]`**: Downlink queue backlog in bytes waiting in the base station RLC buffer for this UE. In the authors' DRL reward pipeline, this is scaled by $10^5$.
3. **`tx_brate downlink [Mbps]`**: Downlink physical data rate transmitted over the radio link in Megabits per second, measured by srsLTE.
4. **`sum_requested_prbs`**: Aggregate number of Physical Resource Blocks requested by the UE for buffer draining during the $250\text{ ms}$ window.
5. **`sum_granted_prbs`**: Aggregate number of Physical Resource Blocks allocated to the UE by the slice scheduler during the $250\text{ ms}$ window.

---

## 7. Audit of QoS Measurements (Recorded vs. Absent)

For an IEEE ICC paper focusing on QoS-constrained RAN slicing, identifying available vs. missing QoS parameters is vital:

### 7.1 Available QoS Measurements
- **Reliability & Decoding Stress**:
  - `tx_errors downlink (%)`: Transport Block Error Rate (BLER) at the MAC layer.
  - `rx_errors uplink (%)`: Uplink BLER.
  - `ul_turbo_iters`: Average Turbo decoder iterations (reflects FEC stress near decoding threshold).
- **Buffer Backlog & Congestion**:
  - `dl_buffer [bytes]` and `ul_buffer [bytes]`: Queue backlog states.
- **Throughput Fulfillment**:
  - `tx_brate downlink [Mbps]` vs. traffic generation demand ($1\text{ Mbps}$ for eMBB).
- **Radio Channel State**:
  - `dl_cqi`, `dl_mcs`, `ul_sinr`, `ul_rssi`.

### 7.2 Explicitly Absent QoS Measurements
1. **Direct Packet Delay / One-Way Latency**:
   - **Status**: **ABSENT**. The srsLTE logging pipeline does not record packet header timestamps or end-to-end packet transmission latency.
   - **Author Justification & Solution**: The original authors explicitly addressed this limitation in `COLOSSEUM_README.md`:
     > *"URLLC slice: Minimize latency. This is done by setting the reward equal to `ratio_granted_req`, which represents the ratio between the number of PRBs allocated by the scheduler and those requested by the UEs. The higher the value, the faster requests are satisfied and traffic experience low latency."*
2. **Packet Jitter / Delay Variation**:
   - **Status**: **ABSENT**. Individual packet arrival and departure inter-times are not recorded.
3. **Packet-Level Arrival / Departure Timestamps**:
   - **Status**: **ABSENT**. Packets are only reported as cumulative counts (`tx_pkts downlink`, `rx_pkts uplink`) every $250\text{ ms}$.
4. **HARQ Retransmission Protocol States**:
   - **Status**: **ABSENT**. Individual ACK/NACK sequences and Redundancy Version (RV) indices are not exposed; only aggregate BLER percentage is recorded.

---

## 8. Pilot Subset Recommendation

Because processing all 21,612 slice metric CSVs (~6.72 GB) upfront is computationally wasteful during model development, a **focused, statistically defensible pilot subset** is recommended:

### Recommended Pilot Subset Definition
- **Scenarios**:
  1. `slice_traffic/rome_static_close`: Provides clean, static RF propagation (no mobility confounding factors) to validate baseline KAN convergence.
  2. `slice_traffic/rome_slow_close`: Incorporates $3\text{ m/s}$ pedestrian mobility (Doppler shifts, fast fading) to test Temporal KAN memory and recurrent dynamics.
- **Training Configurations**:
  - `tr0`: Asymmetric initial policy ($PF, RR, PF$) and asymmetric RBG allocation ($1, 2, 4$).
  - `tr10`: Symmetric round-robin baseline ($RR, RR, PF$) with balanced quotas ($3, 1, 3$).
- **Experiment Runs**:
  - `exp1`: Multi-cell run across all 4 base stations (`bs1`, `bs2`, `bs3`, `bs4`).
- **Scale**:
  - **Files**: $\approx 40\text{ to }80$ slice metric CSVs.
  - **Timesteps**: $480\text{ s}$ duration $\times 4\text{ samples/s} \approx 1,920$ timesteps per UE.
  - **Total Rows**: $\approx 75,000\text{ to }150,000$ telemetry rows.
  - **Memory Footprint**: $\approx 25\text{ MB}$, allowing sub-second data loading and rapid iterative experimentation.
