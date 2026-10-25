# Scientific Authenticity and Reproducibility Audit Report

**Project:** Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing  
**Target Venue:** IEEE International Conference on Communications (ICC 2027) — Mobile and Wireless Networks (MWN) Symposium  
**Audit Date:** October 25, 2026  
**Auditor:** Autonomous Verification Subsystem  
**Scope:** Complete research codebase, dataset pipelines, mathematical formulations, simulation logs, checkpoints, manuscript (`paper/main.tex`), and reproducibility appendix.

---

## Executive Summary

This comprehensive audit evaluated the factual accuracy, reproducibility, mathematical consistency, and traceability of all empirical claims in the research manuscript.

### Overall Status: **VALID WITH IDENTIFIED REVISIONS**
* **Traceability & Numerical Fidelity:** **100% Verified.** All 17 policies, 170 raw evaluation episodes, and summary metrics in Table 1 match the raw experiment output logs (`outputs/metrics/final_experiments_raw.csv` and `outputs/metrics/final_results.csv`) with zero fabrication.
* **Mathematical Soundness:** **Verified.** The Hamilton integer PRB allocation algorithm guarantees feasibility ($0$ violations across $5,000$ randomized tests), queue dynamics conform to mass-conservation laws, and non-zero telemetry latency/loss barriers are rigorously enforced.
* **Key Inconsistencies & Required Corrections:**
  1. **Parameter Count Mismatch:** Manuscript text states TKAN has 4,409 parameters; the actual loaded checkpoint (`tkan_best.pt`) has **3,929 parameters**.
  2. **Reproducibility Appendix Discrepancy:** `paper/reproducibility_appendix.md` lists window size $W=10$ ($100\text{ ms}$ at $10\text{ ms}$ TTI) and evaluation seeds `[42, 101, ...]`, whereas the actual code and data pipeline use $H=16$ ($4000\text{ ms}$ at $250\text{ ms}$ TTI) and evaluation seeds `[2026, 2027, ..., 2035]`.
  3. **Relative Percentage Gain Formula:** The claimed 32.6% reward improvement over GRU defines gain relative to TKAN's penalty ($|R_{\text{GRU}} - R_{\text{TKAN}}| / |R_{\text{TKAN}}|$). If defined relative to the baseline GRU, the improvement is **24.6%**. The manuscript must explicitly provide this formula.
  4. **Heuristic Baseline Dynamics:** Myopic heuristics (`NoMemory_VoI` at $-547.2$ and `AgenticTKAN_Never` at $-425.4$) achieve less negative rewards than `AgenticTKAN_VoI` ($-6940.7$) because they allocate slightly more PRBs to the highly-penalized URLLC slice. However, `AgenticTKAN_VoI` achieves **35.3% lower overflow volume** and requires **67.2% fewer telemetry requests**. This trade-off must be explicitly clarified.
  5. **Overstated "Strict SLA Compliance":** With an SLA breach rate of $42.5\%$, claims of "strict SLA compliance" must be tempered to "Pareto-efficient SLA mitigation under sparse telemetry".

---

## 1. Traceability and Evidence Audit

Every metric reported in the manuscript was traced back to its origin:

| Metric in Manuscript | Manuscript Value | Raw Artifact Source | Recomputed Value | Discrepancy / Status |
| :--- | :--- | :--- | :--- | :--- |
| **AgenticTKAN_VoI Reward** | $-6940.7 \pm 2118.5$ | `final_experiments_raw.csv` | $-6940.65 \pm 2118.49$ | **0.00% (Verified)** |
| **AgenticTKAN_VoI Overflow**| $13.1\% \pm 7.0\%$ | `final_experiments_raw.csv` | $13.10\% \pm 6.96\%$ | **0.00% (Verified)** |
| **AgenticTKAN_VoI SLA Breach**| $42.5\% \pm 12.8\%$ | `final_experiments_raw.csv` | $42.50\% \pm 12.77\%$ | **0.00% (Verified)** |
| **AgenticTKAN_VoI Requests** | $6.0 \pm 0.0$ ($2.0\%$) | `final_experiments_raw.csv` | $6.00 \pm 0.00$ ($2.00\%$) | **0.00% (Verified)** |
| **AgenticTKAN_VoI Latency** | $168.90\text{ ms}$ | `final_experiments_raw.csv` | $168.90\text{ ms}$ | **0.00% (Verified)** |
| **AgenticGRU_VoI Reward** | $-9201.4 \pm 5498.8$ | `final_experiments_raw.csv` | $-9201.43 \pm 5498.81$ | **0.00% (Verified)** |
| **AgenticMLP_VoI Reward** | $-20942.3 \pm 7400.9$| `final_experiments_raw.csv` | $-20942.26 \pm 7400.90$| **0.00% (Verified)** |
| **Privileged Oracle Reward**| $-1062.7 \pm 1131.9$ | `final_experiments_raw.csv` | $-1062.67 \pm 1131.87$ | **0.00% (Verified)** |

* **Episode Counts:** Exactly 170 episodes (17 policies $\times$ 10 held-out seeds).
* **Seeds:** `[2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]`.
* **Confidence Interval Formula:** Two-sided Student's $t$ with $N-1 = 9$ degrees of freedom ($t_{\text{crit}} = 2.262157$). Verified across all 17 policies.

---

## 2. Mathematical Model Audit

### 2.1 Queue Dynamics and Overflow Conservation
Equations (71)–(74) in `paper/main.tex`:
$$\tilde{Q}_{k, t} = \max(0, Q_{k, t} - S_{k, t}) + A_{k, t}$$
$$Q_{k, t+1} = \min(C_k, \tilde{Q}_{k, t})$$
$$O_{k, t} = \max(0, \tilde{Q}_{k, t} - C_k)$$
* **Code Verification (`src/mwnsim/environment.py` lines 228–245):** Verified exact mathematical identity. Overflow is strictly bounded, and mass conservation holds: $\tilde{Q}_{k, t} = Q_{k, t+1} + O_{k, t}$.
* **Severity:** **None (Pass)**.

### 2.2 Delay SLA Dwell-Time Proxy vs. Packet Delay
* **Equation (95):** Formulates delay cost as $C_{L, t} = \sum_k w_{L, k} \max\left(0, \frac{Q_{k, t}}{R_{k, t} \bar{\mu}_k} - D_{\max, k}\right)$.
* **Unit Verification:**
  * $Q_{k, t}$ is in Bytes.
  * $R_{k, t} \bar{\mu}_k$ is service capacity in Bytes/slot.
  * Therefore, $\frac{Q_{k, t}}{R_{k, t} \bar{\mu}_k}$ has units of **slots (steps)**, representing queue drain dwell-time.
  * $D_{\max, k}$ is correctly configured in steps ($4.0, 1.5, 8.0$ steps).
* **Potential Risk:** Describing this metric simply as "packet delay" is scientifically inaccurate because individual packet timestamps and queuing delay distributions are not tracked.
* **Finding:** The metric is a **queue drain dwell-time proxy**. The manuscript must explicitly use this term.
* **Severity:** **Medium**.

### 2.3 Value of Information (VoI) Uncertainty Formulation
* **Equation (120):** $\sigma_{k, t} = \sqrt{\Delta_{k, t} + 1} \cdot \sigma_{A, k}$.
* **Assumptions:** This formula holds under the assumption that traffic arrivals $A_{k, t}$ are independent increments with variance $\sigma_{A, k}^2$. Under burst autocorrelation (e.g., Markov-modulated arrivals), variance grows super-linearly.
* **Finding:** Valid under the white-noise assumption; the manuscript should note that this represents a lower bound on unobserved variance for burst-correlated traffic.
* **Severity:** **Low**.

### 2.4 PRB Allocation Simplex
* **Constraint:** $\sum_{k=0}^2 R_{k, t} = 15, \quad R_{k, t} \ge 1$.
* **Algorithm:** Hamilton Largest-Remainder apportionment (`src/mwnsim/policies.py` lines 18–60).
* **Stress Test:** $5,000$ randomized demand vectors evaluated; **0 violations** of budget or minimum bounds.
* **Severity:** **None (Pass)**.

---

## 3. Dataset and Preprocessing Integrity

* **Source Data:** Colosseum O-RAN COMMAG dataset.
* **Data Volume:** $737,270$ training windows, $146,977$ validation windows, and $146,971$ test windows.
* **Anti-Leakage Boundaries:**
  * Scaler parameters ($\mu, \sigma$) were fitted purely on the training partition ($t \in [0, 30\,000]$).
  * Feature columns: `dl_buffer [bytes]`, `tx_brate downlink [Mbps]`, `sum_requested_prbs`, `sum_granted_prbs`.
  * Target: `dl_buffer [bytes]`, transformed via $\log(1 + y)$.
* **Counterfactual Limitation:** The COMMAG traces record physical base-station states under historical schedules. The simulator uses COMMAG parameters for arrival rates and channel efficiencies, but closed-loop trajectories are evaluated via the discrete-time environment.
* **Severity:** **None (Properly documented in Section 5.1)**.

---

## 4. Discrepancies and Inconsistencies Table

| ID | Component | Location | Documented Claim | Ground Truth in Code/Artifacts | Severity | Recommended Fix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D-1** | Parameter Count | `paper/main.tex` line 48, `manifest.json` | TKAN has 4,409 parameters | `tkan_best.pt` has 3,929 parameters | Low | Update parameter count to 3,929. |
| **D-2** | Window Size & Seeds | `paper/reproducibility_appendix.md` | $W=10$ ($100\text{ ms}$), seeds `[42, 101, ...]` | $H=16$ ($4000\text{ ms}$), seeds `[2026..2035]` | High | Update appendix to match actual pipeline. |
| **D-3** | Percentage Formula | `paper/main.tex` Abstract & Section 6 | 32.6% reward gain over GRU | Defined as $\frac{\|R_{\text{GRU}}\| - \|R_{\text{TKAN}}\|}{\|R_{\text{TKAN}}\|}$ (vs 24.6% relative to GRU) | Medium | State the mathematical percentage formula explicitly. |
| **D-4** | Heuristic Baselines | `paper/main.tex` Table 1 | Heuristic rewards higher than TKAN | Heuristics allocate $\sim 2.0$ PRBs to URLLC; TKAN cuts overflows by 35% | High | Add dedicated discussion explaining trade-off. |
| **D-5** | SLA Claim Phrasing | `paper/main.tex` Abstract | "strict SLA compliance" | SLA breach rate is $42.5\%$ under $2\%$ telemetry | Medium | Soften to "SLA mitigation under sparse telemetry". |

---

## 5. Prioritized Action Plan Before Paper Submission

1. **Priority 1 (Correct Appendix Discrepancies):** Align `paper/reproducibility_appendix.md` with actual $H=16$ lookback and `[2026..2035]` seeds.
2. **Priority 2 (Clarify Heuristic Baseline Dynamics):** Provide a clear, scientifically defensible explanation in Section 6.1 of why myopic/open-loop baselines achieve less negative total reward while suffering 54% higher buffer overflow volume.
3. **Priority 3 (Explicit Improvement Formulas):** Add explicit mathematical definitions for relative reward gain in Section 6.2.
4. **Priority 4 (Update Parameter Count):** Change TKAN parameter count from 4,409 to 3,929 in `main.tex`.
5. **Priority 5 (Terminology Refinement):** Replace loose mentions of "packet delay" with "queue dwell-time proxy".
