# Reproducibility Appendix: Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing

This reproducibility appendix details the experimental environment, dataset provenance, model configurations, hyperparameter settings, and step-by-step instructions to replicate all empirical results reported in the IEEE ICC 2027 Mobile and Wireless Networks (MWN) submission: *"Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing"*.

---

## 1. Conference Call and Scope Verification

* **Venue:** IEEE International Conference on Communications (ICC 2027)
* **Symposium:** Mobile and Wireless Networks (MWN)
* **Venue Location & Dates:** Washington, DC, USA (May 30 – June 3, 2027)
* **Submission Guidelines:**
  * Maximum 6 pages for initial peer review (IEEE standard double-column conference format, 10pt font, US Letter).
  * Up to 7 pages allowed for camera-ready upon payment of an over-length page charge.
* **Submission Window Status:**
  * Technical paper submission deadline: October 2, 2026.
  * Acceptance notification: January 18, 2027.
  * Camera-ready deadline: February 15, 2027.
  * *Verification Note:* At the execution timestamp (October 25, 2026), the initial submission window for IEEE ICC 2027 is closed. The manuscript is prepared according to strict formatting and reproducibility standards for post-review archival and symposium presentation.

---

## 2. Experimental Environment & Hardware Specifications

All benchmark simulations, predictor training, and closed-loop control evaluations were executed on the following validated environment:

* **Operating System:** Microsoft Windows 11 Enterprise (64-bit)
* **Host Processor:** 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70 GHz (6 Cores, 12 Threads)
* **Host Memory:** 16.0 GB DDR4 RAM
* **Graphics Processor:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB GDDR6 VRAM, CUDA Driver 12.x)
* **Software Toolchain:**
  * Python 3.10.11
  * PyTorch 2.1.2+cu121
  * NumPy 1.24.3, SciPy 1.11.1
  * Pandas 2.0.3, Matplotlib 3.7.2, Seaborn 0.12.2

---

## 3. Dataset Provenance and Preprocessing

### 3.1 COMMAG 5G Traces
* **Origin:** Open real-world 5G base station measurements from IEEE Communications Magazine (COMMAG) open trace initiative.
* **Slice Types:**
  1. **Slice 1 (eMBB):** Video streaming / mobile broadband with bursty arrivals.
  2. **Slice 2 (URLLC):** Low-latency industrial control with strict buffer constraints.
  3. **Slice 3 (mMTC):** Massive machine-type telemetry with periodic aggregate transmissions.
* **Data Volume:** 50,000 temporal frames across 3 slices, recording instantaneous buffer occupancy ($Q_k(t)$), arrival volumes ($A_k(t)$), and PRB allocations ($R_k(t)$).
* **Data Nature & Counterfactual Limitation:**
  * *Crucial Scientific Principle:* The COMMAG traces record actual operational telemetry under historically executed base-station scheduling policies. They do **not** reveal counterfactual outcomes for actions not taken.
  * *Evaluation Setup:* The traces serve to parameterize the arrival distributions, channel spectral efficiencies ($\mu_k$), and realistic temporal correlation dynamics within our calibrated discrete-time closed-loop simulator.

### 3.2 Feature Preprocessing & Leakage Elimination
* Input feature vector for each slice $k$ at step $t$:
  $$\mathbf{x}_k(t) = \left[ \widehat{Q}_k(t), \Delta_k(t), R_k(t-1), \bar{\mu}_k(t) \right]^\top$$
* **Window Size:** Temporal sequence length $W = 10$ steps (corresponding to $100\text{ ms}$ at $10\text{ ms}$ TTI).
* **Normalization:** Standard zero-mean, unit-variance scaling computed strictly on the training partition ($t \in [0, 30\,000]$) and applied out-of-sample to validation ($[30\,000, 40\,000]$) and held-out test sets ($[40\,000, 50\,000]$). No future labels or cross-partition statistics leak into the feature pipeline.

---

## 4. Model Architectures and Hyperparameters

### 4.1 Temporal Kolmogorov-Arnold Network (TKAN)
* **Basis Functions:** Cubic B-splines ($k=3$).
* **Grid Intervals:** $G = 5$ equidistant points spanning normalized feature range $[-2.0, +2.0]$.
* **Network Dimensions:** Input $d_{\text{in}} = 4$, Hidden $d_h = [64, 32]$, Output $d_{\text{out}} = 1$ (predicted queue demand $\widetilde{Q}_k(t+1)$).
* **Recurrent Gating:** Gated recurrent cell with spline-transformed candidate states and forget/update gates.
* **Loss Function:** Huber Loss ($\delta = 1.0$) combined with L1 spline regularization ($\lambda_{\text{reg}} = 10^{-4}$).
* **Optimizer:** Adam with learning rate $\eta = 10^{-3}$, batch size $B = 32$, trained for 50 epochs with early stopping (patience $= 10$).

### 4.2 Matched Baselines
To ensure rigorous and fair comparison, all baseline architectures match the parameter order of magnitude ($\sim 18\text{k} - 25\text{k}$ parameters):
* **Agentic GRU:** 2-layer Gated Recurrent Unit, hidden dimension $d_h = 64$, followed by a linear projection layer.
* **Agentic MLP:** 3-layer Multi-Layer Perceptron ($d_{\text{in}} \to 64 \to 32 \to 1$), LeakyReLU activations ($\alpha=0.01$).

### 4.3 Controller & Value-of-Information (VoI) Parameters
* **Total PRB Budget:** $C = 15$ PRBs per decision epoch.
* **VoI Loss Function Weights:**
  * Delay / Queue penalty: $\alpha = 1.0$
  * Overflow penalty: $\beta = 10.0$
  * Reconfiguration penalty: $\gamma = 5.0$
  * Measurement acquisition cost: $c_{\text{req}} = 1.0$
* **Non-Ideal E2 Channel Settings:**
  * Signaling delay: $d_k \sim \text{Poisson}(1.5)$ decision intervals.
  * Report drop rate: $p_{\text{loss}, k} = 5\%$.
* **Fairness & Feasibility Guarantee:**
  * Largest-Remainder Hamilton integer apportionment ensuring $\sum_{k=1}^3 R_k(t) = 15$ and $R_k(t) \ge R_{\min} = 1, \forall k$.

---

## 5. Seed Allocation Protocol

To guarantee no post-hoc tuning on evaluation sets, random seeds were segregated into two strict cohorts prior to running experiments:
* **Tuning Cohort (Validation):** Seeds 100, 102, 104, 106, 108.
* **Held-Out Evaluation Cohort (Final Reporting):** 10 independent random seeds:
  `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`
* All tables and figures report the sample mean $\pm$ 95% Student-$t$ confidence intervals across these 10 held-out seeds.

---

## 6. Reproduction Step-by-Step Instructions

From the workspace root directory `d:\Research-kan\agentic_tkan_mwn`, run the following commands in order:

### Step 1: Environment Setup
```powershell
# Activate your Python virtual environment
.\.venv\Scripts\Activate.ps1

# Verify required packages
python -c "import torch, numpy, scipy, pandas, matplotlib; print('Environment dependencies verified successfully.')"
```

### Step 2: Validate Raw COMMAG Data Integrity
```powershell
python scripts/inspect_data.py
```
*Expected Output:* Confirms 50,000 records, validates absence of NaNs, and checks schema definitions.

### Step 3: Run Closed-Loop Control Evaluation Suite
```powershell
python -m src.experiments.run_closed_loop_evaluation --seeds 42 101 202 303 404 505 606 707 808 909 --budget 15 --episodes 300
```
*Output File:* `outputs/metrics/final_results.csv` and `outputs/manifests/final_experiments_manifest.json`.

### Step 4: Run Architectural and Telemetry Ablation Sweeps
```powershell
python -m src.experiments.run_ablation_study --seeds 42 101 202 303 404 505 606 707 808 909
python -m src.experiments.run_sensitivity_analysis --param loss_rate --values 0.01 0.05 0.10 0.20
python -m src.experiments.run_sensitivity_analysis --param delay_mean --values 0.5 1.0 1.5 2.5
```
*Output Files:* `outputs/metrics/ablation_results.csv`, `outputs/metrics/sensitivity_results.csv`.

### Step 5: Generate Publication-Grade Figures
```powershell
python scripts/generate_paper_figures.py
```
*Generated Assets:*
* `outputs/figures/fig1_pareto_constraint_vs_telemetry.png`
* `outputs/figures/fig2_architecture_ablation.png`
* `outputs/figures/fig3_telemetry_ablation.png`
* `outputs/figures/fig4_sensitivity_sweeps.png`
* `outputs/figures/fig5_recovery_outage.png`

### Step 6: Validate LaTeX Manuscript Structure and Bibliography
```powershell
python scripts/validate_latex.py
```
*Expected Output:*
```text
======================================================================
                 IEEE LATEX MANUSCRIPT VALIDATION
======================================================================
[*] Checking main LaTeX source: paper/main.tex
    Total lines: 266
    Total characters: 26038
[+] Brace matching passed: 424 open and 424 close braces.
[+] AST environment nesting passed (all 20 begin/end pairs match in LIFO order).
[+] All 21 citation keys resolved in paper/references.bib.
[+] All 5 figure graphic files verified on disk.
[+] All 13 mandatory IEEE ICC conference sections verified.
======================================================================
SUCCESS: All LaTeX structural, citation, and asset integrity checks passed 100%!
======================================================================
```

---

## 7. Artifact Manifest Checksums

The integrity of key output artifacts is tracked in `outputs/manifests/final_experiments_manifest.json`:
* `outputs/metrics/final_results.csv`: 90 completed evaluations (9 policies $\times$ 10 held-out seeds).
* `outputs/metrics/ablation_results.csv`: 40 evaluations (4 architectures $\times$ 10 held-out seeds).
* `outputs/metrics/sensitivity_results.csv`: 80 evaluations (8 channel conditions $\times$ 10 held-out seeds).
