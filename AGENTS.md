# Agent Guidelines for agentic_tkan_mwn

## Project Overview
`agentic_tkan_mwn` is a research framework investigating Agentic Temporal Kolmogorov-Arnold Networks (TKAN) for Mobile Wireless Network (MWN) simulation, resource allocation, and predictive optimization.

## Workflow Rules
1. **Modularity**:
   - Simulation components belong in `src/mwnsim/`.
   - Data preprocessing and analysis belong in `scripts/`.
   - Experiment configurations belong in `configs/`.
2. **Configuration**:
   - Avoid hardcoding simulation or hyperparameter values; reference `configs/default.yaml`.
3. **Reproducibility**:
   - Seed random generators (`numpy`, `random`, `torch`) for all simulation experiments.
   - Save experiment artifacts (checkpoints, metrics, plots) under `outputs/`.
4. **Testing**:
   - Write unit and integration tests under `tests/`.
   - Ensure `pytest tests/` passes before committing changes.
