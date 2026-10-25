# Reproducibility Checklist for Independent Verification

**Project:** Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing  
**Target Venue:** IEEE ICC 2027 (Mobile and Wireless Networks Track)  
**Repository Branch:** `research`  

This checklist provides an independent auditor or researcher with the exact sequence of commands and verification milestones required to replicate all empirical results reported in the manuscript.

---

## 1. Environment & Prerequisite Verification

- [ ] **Operating System:** Windows 10/11 (or Linux with equivalent PyTorch installation).
- [ ] **Python Version:** 3.10+ (tested on Python 3.13.5).
- [ ] **Hardware Requirements:** Minimum 8 GB Host RAM; NVIDIA GPU with $\ge 4\text{ GB}$ VRAM recommended (CPU fallback supported).
- [ ] **Virtual Environment Setup:**
  ```powershell
  # Activate workspace virtual environment
  .\.venv\Scripts\Activate.ps1
  # Verify PyTorch and dependency installations
  python -c "import torch, numpy, scipy, pandas, matplotlib, yaml; print('Environment dependencies verified.')"
  ```

---

## 2. Unit and Integration Test Verification

Run the full pytest suite covering data integrity, anti-leakage guards, mathematical formulations, and baseline policies:

```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

- [ ] **Milestone 1:** All **48 tests** must pass without errors or warnings.
  - `test_environment.py` passes 9 tests (PRB feasibility, queue bounds, report loss, no-leakage, reward decomposition).
  - `test_agentic_controller.py` passes 5 tests (VoI logic, rolling memory, rollouts).
  - `test_data_validation.py` passes 15 tests (anti-leakage guards, group separation, temporal causality).
  - `test_commag_preprocessing.py` passes 6 tests.
  - `test_control_baselines.py` passes 4 tests.
  - `test_agentic_tkan.py` and `test_forecasting_baselines.py` pass 9 tests.

---

## 3. Checkpoint & Architecture Verification

Verify that trained neural network checkpoints exist and have valid parameter counts:

```powershell
python scripts/audit_scientific_authenticity.py
```

- [ ] **Milestone 2:** Checkpoint hashes and parameter counts verified:
  - `outputs/checkpoints/tkan_best.pt`: **3,929 parameters** (SHA-256 starts with `ab79187a1bf3`).
  - `outputs/checkpoints/singlelayergru_best.pt`: **3,681 parameters** (SHA-256 starts with `1c63c71898ee`).
  - `outputs/checkpoints/smallmlp_best.pt`: **2,113 parameters** (SHA-256 starts with `301c62c8e788`).

---

## 4. Re-Running Controlled Experiments (If Regenerating from Scratch)

*Note: Pre-computed raw episode logs exist at `outputs/metrics/final_experiments_raw.csv` (170 episodes). To perform a complete independent re-run:*

```powershell
# Run the complete Phase 9.1 evaluation suite across all 10 held-out seeds
python scripts/run_final_experiments.py
```

- [ ] **Milestone 3:** `outputs/metrics/final_experiments_raw.csv` generated (170 rows).
- [ ] **Milestone 4:** `outputs/metrics/final_results.csv` generated (17 rows with means, stds, and 95% CIs).
- [ ] **Milestone 5:** `outputs/metrics/ablation_results.csv` and `sensitivity_results.csv` generated.
- [ ] **Milestone 6:** `outputs/metrics/outage_recovery_trace.csv` generated.

---

## 5. Statistical Aggregation Verification

Verify that the manuscript Table 1 numbers match the raw generated logs exactly:

```powershell
python -c "
import pandas as pd
raw = pd.read_csv('outputs/metrics/final_experiments_raw.csv')
tkan = raw[raw['policy'] == 'AgenticTKAN_VoI']
print(f'TKAN Mean Reward: {tkan[\"total_reward\"].mean():.1f}')
print(f'TKAN Overflow Freq: {tkan[\"overflow_frequency\"].mean()*100:.1f}%')
print(f'TKAN SLA Breach: {tkan[\"sla_violation_rate\"].mean()*100:.1f}%')
print(f'TKAN Tel Requests: {tkan[\"total_telemetry_requests\"].mean():.1f}')
"
```

- [ ] **Milestone 7:** Output matches:
  - Mean Reward: `-6940.7`
  - Overflow Freq: `13.1%`
  - SLA Breach: `42.5%`
  - Telemetry Requests: `6.0`

---

## 6. Figure Generation & Visual Verification

Regenerate all publication-grade figures:

```powershell
python scripts/generate_figures.py
```

- [ ] **Milestone 8:** Check generated PNG figures in `outputs/figures/`:
  - `fig1_pareto_constraint_vs_telemetry.png`: Confirms Pareto curve without text overlap.
  - `fig2_architecture_ablation.png`: Confirms non-overlapping bracket labels and clean bar plots.
  - `fig3_telemetry_ablation.png`: Confirms VoI vs Periodic vs Always telemetry rates.
  - `fig4_sensitivity_sweeps.png`: Confirms 4-panel sweeps (delay, loss, burst, threshold).
  - `fig5_recovery_outage.png`: Confirms blackout recovery trajectory at steps 30–45.

---

## 7. LaTeX Manuscript & Citation Validation

Validate LaTeX compilation integrity without requiring local `pdflatex`:

```powershell
python scripts/validate_latex.py
```

- [ ] **Milestone 9:** Structural AST check confirms 100% environment nesting balance.
- [ ] **Milestone 10:** Curly brace check confirms 424 open and 424 close braces.
- [ ] **Milestone 11:** Bibliography check confirms all 21 citations resolve to `paper/references.bib`.
- [ ] **Milestone 12:** Figure file resolver confirms all 5 graphic files exist on disk.
