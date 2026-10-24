# Comprehensive Audit and Reuse Inventory of the Original Agentic-KAN Repository

**Project Title:** Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing  
**Target Venue:** IEEE ICC 2027 — Mobile and Wireless Networks Track  
**Original Repository (Read-Only Source):** `D:\DSAA`  
**Current Repository (New Wireless Research):** `d:\Research-kan\agentic_tkan_mwn`  
**Dataset Location:** `d:\Research-kan\colosseum-oran-commag-dataset` (Colosseum O-RAN COMMAG dataset, processed into `data/processed/commag/`)  
**Audit Date:** October 2026  
**Auditor:** Antigravity AI Research Assistant

---

## 1. Executive Summary & Audit Mandate

This document serves as the formal model audit and reuse inventory conducted under strict safety constraints:
1. **Strict Read-Only Enforcement:** The original repository (`D:\DSAA`) was examined exclusively through read-only inspection commands. No files were written, renamed, modified, deleted, or executed within `D:\DSAA`.
2. **Selective Reuse (No Indiscriminate Copying):** Only model implementations directly relevant to **Phase 5 (Forecasting Baselines)** and foundational references for **Phase 8 (Agentic Telemetry & Resource Controller)** were copied. Bulk copying was avoided.
3. **Initial Content Preservation:** Copied source files were transferred without initial modification to establish verified cryptographic parity against their sources via SHA-256 hashes.
4. **Separation of Concerns:** All domain adaptations, package import fixes, mathematical corrections, and O-RAN wireless integrations are reserved for execution exclusively within `agentic_tkan_mwn`.

---

## 2. Complete Model Inventory Table

The following table summarizes all model implementations discovered in `D:\DSAA`, their technical properties, limitations, relevance to wireless telemetry forecasting, and reuse status:

| Model Name | Original Source Path in `D:\DSAA` | Supporting Files / Imports | Primary Dependencies | Intended Functionality | Known Limitations & Suspicious Code | Relevance to Wireless Research | Recommended Phase | Copy Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla KAN** | `models/kan/vanilla_kan.py` | None | `torch`, `torch.nn` | Multi-layer B-spline Kolmogorov-Arnold Network. | Fixed uniform knot grid on $[-1, 1]$. Inputs with large unscaled magnitudes saturate or zero out outer spline basis functions. | Core spline basis layer required by TKAN and Agentic TKAN. | Phase 5 | **Copied** (`src/models/kan/vanilla_kan.py`) |
| **EfficientKAN** | `models/kan/efficient_kan.py` | None | `torch`, `torch.nn` | Memory-efficient B-spline KAN reformulating spline operations into matrix multiplications. | Fixed knot grid; grid update logic present but requires manual activation. Linear projection overhead if grid size is large. | Fast spline baseline for buffer forecasting comparison. | Phase 5 | **Copied** (`src/models/kan/efficient_kan.py`) |
| **TKAN (Temporal KAN)** | `models/kan/tkan.py` | `models.kan.vanilla_kan.KANLayer` | `torch`, `torch.nn` | Recurrent KAN cell integrating B-spline nonlinear transforms into memory update gating. | Hardcoded import `from models.kan.vanilla_kan import KANLayer`. High parameter count if `units` is unconstrained. | Primary temporal KAN benchmark for next-step buffer forecasting ($y(t+1)$). | Phase 5 | **Copied** (`src/models/kan/tkan.py`) |
| **Standard LSTM** | `models/deep_learning/lstm.py` | None | `torch`, `torch.nn` | Standard Multi-layer Long Short-Term Memory network. | Projection layer assumes static lookback and horizon dimensions defined in constructor. | Canonical non-KAN recurrent baseline for O-RAN telemetry prediction. | Phase 5 | **Copied** (`src/models/deep_learning/lstm.py`) |
| **Standard GRU** | `models/deep_learning/gru.py` | None | `torch`, `torch.nn` | Standard Multi-layer Gated Recurrent Unit network. | Linear output projection expects fixed sequence length. | Canonical low-overhead recurrent baseline. | Phase 5 | **Copied** (`src/models/deep_learning/gru.py`) |
| **Agentic TKAN** | `models/agentic_kan/tkan_agentic.py` | `models.kan.vanilla_kan.KANLayer` | `torch`, `torch.nn` | Recurrent KAN cell with internal "actor" and "critic" gating for feature weighting. | **Not a closed-loop controller.** Trained purely with supervised MSE/Huber forecasting loss. Hardcoded import `models.kan.vanilla_kan`. | Architectural reference for gating mechanisms; cannot be used as RL controller as-is. | Phase 8 (Refactored) | **Copied** (`src/models/agentic_kan/tkan_agentic.py`) |
| **Standard BiLSTM** | `models/deep_learning/bilstm.py` | None | `torch`, `torch.nn` | Bidirectional LSTM. | Backward pass uses future context; strictly causes temporal leakage if used in online real-time slicing. | Incompatible with causal online forecasting at O-RAN near-RT RIC. | Excluded | Deliberately Not Copied |
| **Jacobi / Padé RKAN** | `models/kan/rkan.py` | None | `torch`, `torch.nn` | Rational / Jacobi orthogonal polynomial KAN. | **Fatal Bug:** Code contains placeholder comments `"# Truncated for brevity here, but paste YOUR exact math logic provided in the prompt"` and raises exceptions for degree $> 3$. | Unstable and mathematically incomplete in source repo. | Excluded | Deliberately Not Copied |
| **Agentic Padé-RKAN** | `models/agentic_kan/Agentic_pad_rkan_.py` | `models.kan.rkan.PadeRKAN` | `torch`, `torch.nn` | Rational polynomial KAN with actor-critic gating. | Inherits fatal placeholder bugs from `rkan.py`. Strictly supervised forecasting; no RL reward loop. | Broken upstream dependencies; unsuited for wireless control. | Excluded | Deliberately Not Copied |
| **Fourier KAN** | `models/kan/fourier_kan.py` | None | `torch`, `torch.nn` | KAN with Fourier harmonic basis functions. | Phase drift on non-stationary bursty traffic; unbounded harmonic expansion. | Secondary ablation candidate; not required for Phase 5 pilot. | Phase 9 (Ablation) | Deliberately Not Copied |
| **Wavelet KAN** | `models/kan/wavelet_kan.py` | None | `torch`, `torch.nn` | Mexican hat / Morlet wavelet KAN. | Wavelet parameters can suffer gradient instability without custom learning rate decay. | Secondary ablation candidate. | Phase 9 (Ablation) | Deliberately Not Copied |
| **ChebyKAN** | `models/kan/cheby_kan.py` | None | `torch`, `torch.nn` | Chebyshev polynomial basis KAN. | Polynomial values diverge rapidly when inputs leave $[-1, 1]$. | Highly sensitive to wireless burst dynamic ranges. | Phase 9 (Ablation) | Deliberately Not Copied |
| **DropKAN** | `models/kan/drop_kan.py` | `models.kan.vanilla_kan.KANLayer` | `torch`, `torch.nn` | Spline regularization via dropout. | Custom dropout mask logic complicates determinism and PyTorch JIT tracing. | Regularization ablation only. | Phase 9 (Ablation) | Deliberately Not Copied |
| **SplineKAN** | `models/kan/spline_kan.py` | None | `torch`, `torch.nn` | Alternative B-spline formulation. | Redundant with `efficient_kan.py` and slower in runtime benchmarks. | Redundant. | Excluded | Deliberately Not Copied |
| **Transformers (Informer / PatchTST)** | `models/transformers/*` | Transformer layers | `torch`, `einops` | Long-sequence attention-based forecasting models. | Quadratic/linear attention overhead exceeds sub-10ms near-RT RIC control loop constraints. | Incompatible with O-RAN near-RT RIC latency constraints (<10 ms). | Excluded | Deliberately Not Copied |

---

## 3. Cryptographic Verification & Manifest of Copied Source Files

The selected model files were copied into `agentic_tkan_mwn` preserving initial contents exactly. SHA-256 hashes were computed for both source and destination copies to guarantee byte-for-byte integrity:

| Original Source Relative Path (`D:\DSAA`) | Destination Relative Path (`agentic_tkan_mwn`) | SHA-256 Checksum | Integrity Match |
| :--- | :--- | :--- | :--- |
| `models\kan\vanilla_kan.py` | `src/models/kan/vanilla_kan.py` | `4c0e3639f1d9a6ee108eb295ec08d819c4e0f33ec32d3a8bfd3ccb7d2337d1aa` | **Verified Match** |
| `models\kan\tkan.py` | `src/models/kan/tkan.py` | `0035dde8c9387ad7d91c81b539b2e066c9ddab0d8e3f748dcf15f63612291b6e` | **Verified Match** |
| `models\kan\efficient_kan.py` | `src/models/kan/efficient_kan.py` | `8c3b0004a037db44b76c401a212a69ec7c3995c715585ca1c3629c0bf7d016ee` | **Verified Match** |
| `models\deep_learning\lstm.py` | `src/models/deep_learning/lstm.py` | `ac21977ab6dc1afe79185a7133ae089b153b2d2d91c35a1509a76428735fd552` | **Verified Match** |
| `models\deep_learning\gru.py` | `src/models/deep_learning/gru.py` | `3667436c37b215e961307ed3085f7a170d4f52db658932b39ca0a7aece224912` | **Verified Match** |
| `models\agentic_kan\tkan_agentic.py` | `src/models/agentic_kan/tkan_agentic.py` | `d0c2364e5344b4f2460be985ddaec1b03e15f4366d21c0967529136f0a0e61aa` | **Verified Match** |

*The machine-readable manifest is also saved at [copied_models_manifest.json](file:///d:/Research-kan/agentic_tkan_mwn/data/processed/copied_models_manifest.json).*

---

## 4. Deep Architectural and Code Analysis

### 4.1 Vanilla KAN (`src/models/kan/vanilla_kan.py`)
- **Architecture:** Implements `KANLayer` and multi-layer `VanillaKAN`. Uses B-spline basis expansion of degree $k$ over $G$ grid intervals ($G+k$ basis functions per input).
- **Forward Pass:** Computes base activation $\phi(x) = \text{SiLU}(x)$ combined with linear weighting of B-spline curves:
  $$y_j = \sum_{i} \left( w_{b, ij} \text{SiLU}(x_i) + \sum_{m} c_{ijm} B_m(x_i) \right)$$
- **Tensor Shapes:** 
  - Input: `[batch_size, in_features]`
  - Output: `[batch_size, out_features]`
- **Required Dependency:** PyTorch (`torch`, `torch.nn`).

### 4.2 EfficientKAN (`src/models/kan/efficient_kan.py`)
- **Architecture:** Reformulates spline evaluation by computing the B-splines over inputs first and performing a single batched linear transform against spline coefficients, avoiding high memory footprints.
- **Forward Pass:** Employs SiLU base transform with vectorized B-spline matrix multiply:
  $$\text{Splines}(X) \in \mathbb{R}^{B \times I \times (G+k)} \to \text{Contracted with } W \in \mathbb{R}^{O \times I \times (G+k)}$$
- **Tensor Shapes:** `[batch_size, in_features]` $\to$ `[batch_size, out_features]`.

### 4.3 TKAN (`src/models/kan/tkan.py`)
- **Architecture:** Recurrent architecture (`TKANCell` and `TKAN`). Employs gated recurrence where candidate hidden states are transformed using `KANLayer` instead of affine transformations.
- **Forward Pass:**
  - Input gate $i_t$, forget gate $f_t$, and output gate $o_t$ are evaluated using KAN layers.
  - Recurrent hidden state update:
    $$h_t = o_t \odot \tanh(c_t), \quad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
    where $\tilde{c}_t = \text{KANLayer}(x_t, h_{t-1})$.
- **Tensor Shapes:**
  - Input: `[batch_size, seq_len, in_features]` (Lookback = 16)
  - Output: `[batch_size, horizon, out_features]` or `[batch_size, out_features]` (Horizon = 1)

### 4.4 Standard LSTM & GRU (`src/models/deep_learning/lstm.py`, `gru.py`)
- **Architecture:** Native `nn.LSTM` / `nn.GRU` layers followed by linear projection layer.
- **Tensor Shapes:**
  - Input: `[batch_size, seq_len, in_features]`
  - Output: `[batch_size, horizon, out_features]`

### 4.5 Agentic TKAN (`src/models/agentic_kan/tkan_agentic.py`)
- **Architecture:** Contains `AgenticTKANCell` and `AgenticTKAN`. Augments the standard TKAN recurrence with internal "actor" and "critic" sub-networks:
  - An "actor" sub-layer produces candidate state proposals $a_{\text{cand}}$.
  - A "critic" sub-layer computes a scalar confidence gate $c_t = \sigma(\text{Linear}(h_{t-1}, x_t))$.
  - The gated state proposal is formed as $a_t = a_{\text{cand}} \odot c_t$.
- **Training Reality:**
  - **Supervised Loss Only:** In `D:\DSAA\scripts\28_agentic_kan_models.py`, `AgenticTKAN` is trained strictly via supervised loss:
    $$\mathcal{L} = \text{MSELoss}(y_{\text{pred}}, y_{\text{true}}) \quad \text{or} \quad \text{HuberLoss}(y_{\text{pred}}, y_{\text{true}})$$
  - **No Reinforcement Learning:** There is **no MDP (Markov Decision Process)**, no policy gradient (PPO/DDPG), no environment state transition, no QoS SLA penalty reward, and no wireless action space.
  - **Terminology Discrepancy:** The "actor" and "critic" labels in `D:\DSAA` refer strictly to internal gating modules inside a recurrent cell, not an autonomous agent interacting with a wireless environment.

---

## 5. Critical Audit Findings, Bugs & Limitations in `D:\DSAA`

1. **Domain and Dataset Discrepancy:**
   - `D:\DSAA` was constructed for electrical load and meteorological benchmarking (ETTh1, ETTh2, ETTm1, ETTm2, ERA5, NASA POWER).
   - It contains **zero O-RAN code, zero 3GPP telemetry interfaces, zero PRB slicing models, and zero wireless simulation logic**.
2. **Import Path Coupling:**
   - Both `models/kan/tkan.py` and `models/agentic_kan/tkan_agentic.py` hardcode:
     ```python
     from models.kan.vanilla_kan import KANLayer
     ```
   - In `agentic_tkan_mwn`, the modular path is `src.models.kan.vanilla_kan`. To keep copied files unmodified at copy time, this coupling must be addressed via a thin adapter or relative package import during Phase 5 refactoring.
3. **Broken Implementations in Original Repository:**
   - `models/kan/rkan.py` and `models/agentic_kan/Agentic_pad_rkan_.py` contain placeholder strings (`"# Truncated for brevity here, but paste YOUR exact math logic provided in the prompt"`) and throw `NotImplementedError` for polynomial orders $>3$. These were deliberately rejected from copying.
4. **B-Spline Grid Range Vulnerability:**
   - In `vanilla_kan.py`, B-spline knot grids are initialized over $[-1, 1]$. In wireless networks, unscaled buffer values (e.g. `dl_buffer [bytes]` reaching $10^5-10^6$ bytes) would evaluate to exact zero spline activations if fed without z-score standardization or min-max normalization.
   - Fortunately, our Phase 4 preprocessing pipeline fits a train-only Robust/Standard scaler saving normalized windows into `train.npz`, ensuring inputs fall within normal active ranges.
5. **Static Sequence Horizon in Deep Learning Models:**
   - `StandardLSTM` and `StandardGRU` expect static sequence lengths defined at initialization. Dynamic sequence lengths require adapting forward passes to slice the last recurrent state `h[:, -1, :]`.

---

## 6. Research Phase Mapping & Recommendations

### 6.1 Phase 5: Forecasting Baselines (Immediate Next Phase)
- **Approved Models for Phase 5:**
  1. `StandardLSTM` (`src/models/deep_learning/lstm.py`) — Deep learning recurrent baseline.
  2. `StandardGRU` (`src/models/deep_learning/gru.py`) — Lightweight recurrent baseline.
  3. `VanillaKAN` (`src/models/kan/vanilla_kan.py`) — Foundational spline KAN.
  4. `EfficientKAN` (`src/models/kan/efficient_kan.py`) — Scalable B-spline KAN.
  5. `TKAN` (`src/models/kan/tkan.py`) — Temporal KAN for next-step buffer forecasting.
- **Target Task:**
  - Predicting next-sample buffer occupancy $y(t+1) = \text{dl\_buffer}[t+1]$ using lookback $L=16$ on 4 input features (`dl_buffer [bytes]`, `tx_brate downlink [Mbps]`, `sum_requested_prbs`, `sum_granted_prbs`).
  - Evaluated on verified leakage-free splits (`train.npz`, `val.npz`, `test.npz`, `ood_test.npz`).

### 6.2 Phase 8: Agentic Telemetry Acquisition and Constrained Slicing
- **Status of `AgenticTKAN`:**
  - Copied into `src/models/agentic_kan/tkan_agentic.py` purely as an **architectural reference** for dual-branch KAN gating.
  - **Not Ready for Control:** To function as an actual agent in Phase 8, it must be embedded within a true closed-loop RL framework (e.g., actor outputting PRB allocation fractions $a_t \in [0, 1]^K$ and telemetry sampling intervals $\tau_t \in \{\tau_{\min}, \tau_{\max}\}$, trained against a reward function penalizing buffer overruns, QoS SLA violations, and telemetry reporting overhead).

---

## 7. Audit Checklist & Verification Summary

- [x] Original repository `D:\DSAA` preserved completely unchanged (verified read-only).
- [x] All candidate models inspected across `models/kan/`, `models/agentic_kan/`, `models/deep_learning/`, and `models/transformers/`.
- [x] Only approved, relevant models copied to `src/models/` in `agentic_tkan_mwn`.
- [x] Incomplete/broken models (`rkan.py`, `Agentic_pad_rkan_.py`) identified and excluded.
- [x] SHA-256 hashes generated, verified, and recorded in `copied_models_manifest.json`.
- [x] Pytest configuration configured (`pytest.ini`) and test suite verified passing (26/26 tests passed).
