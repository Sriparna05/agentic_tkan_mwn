# Agentic TKAN for Mobile Wireless Networks (`agentic_tkan_mwn`)

A modular simulation and research framework for Agentic Temporal Kolmogorov-Arnold Networks (TKAN) in Mobile Wireless Networks (MWN).

## Directory Structure

```text
agentic_tkan_mwn/
├── AGENTS.md                 # Agent instructions and developer guidelines
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── configs/
│   └── default.yaml          # Default configuration parameters
├── data/
│   ├── raw/
│   │   └── commag/           # Raw CommMag datasets / traces
│   ├── interim/              # Cleaned intermediate data
│   └── processed/            # Final training/evaluation datasets
├── scripts/
│   ├── inspect_data.py       # Dataset summary and inspection script
│   ├── preprocess_commag.py  # Data cleaning and pipeline processing
│   └── run_sim_baselines.py  # Benchmark baseline simulation policies
├── src/
│   └── mwnsim/               # Mobile Wireless Network simulation engine
│       ├── __init__.py
│       ├── environment.py    # Gym-style MWN environment definition
│       └── policies.py       # Baseline heuristic and agentic policies
├── tests/
│   └── test_environment.py   # Test suite for simulation environment
└── outputs/
    ├── checkpoints/          # Model weights and checkpoints
    ├── metrics/              # CSV/JSON performance evaluations
    └── figures/              # Simulation and baseline comparison plots
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Inspect Dataset
```bash
python scripts/inspect_data.py --data-dir data/raw/commag
```

### 3. Run Preprocessing
```bash
python scripts/preprocess_commag.py --config configs/default.yaml
```

### 4. Run Baseline Simulations
```bash
python scripts/run_sim_baselines.py --config configs/default.yaml
```

### 5. Running Tests
```bash
pytest tests/
```
