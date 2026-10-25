"""
Phase 8.1 — Agentic Temporal KAN Closed-Loop Controller

Implements the hybrid closed-loop architecture:
1. Temporal Memory: Maintains rolling history buffer of actually received observations.
2. Temporal Prediction: Forecasts buffer outcomes using trained TKAN (or GRU/MLP/None ablations).
3. Value-of-Information Telemetry Acquisition: Evaluates reduction in downstream decision loss vs. reporting cost.
4. Constrained Allocation: Integer PRB allocation via Largest-Remainder method, bounded by exact budget.
"""

from typing import Dict, Tuple, Optional, Any, List
import os
import math
import numpy as np
import torch

from src.mwnsim.policies import BaseControlPolicy, allocate_integer_prbs
from src.models.kan.tkan import TKAN
from src.models.deep_learning.gru import SingleLayerGRU
from src.models.deep_learning.mlp import SmallMLP

# Standard scaler parameters from Phase 4 manifest
SCALER_MEAN = np.array([45562.43, 0.14774, 34.834, 103.426], dtype=np.float32)
SCALER_STD = np.array([78750.25, 0.26254, 163.887, 138.702], dtype=np.float32)


class AgenticTKANController(BaseControlPolicy):
    """
    Closed-loop O-RAN slicing controller integrating:
    - Temporal memory & KAN prediction
    - Explicit Value of Information (VoI) telemetry acquisition
    - Integer-constrained PRB allocation
    """

    def __init__(
        self,
        name: str = "AgenticTKAN_VoI",
        model_type: str = "tkan",
        telemetry_mode: str = "voi",
        voi_threshold: float = 1.0,
        periodic_interval: int = 2,
        age_threshold: int = 3,
        uncertainty_threshold: float = 0.15,
        total_prbs: int = 15,
        num_slices: int = 3,
        min_prbs: Optional[np.ndarray] = None,
        lookback: int = 16,
        hidden_dim: int = 32,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(name, is_privileged=False)
        self.model_type = model_type.lower()
        self.telemetry_mode = telemetry_mode.lower()
        self.voi_threshold = voi_threshold
        self.periodic_interval = max(1, periodic_interval)
        self.age_threshold = age_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.total_prbs = total_prbs
        self.num_slices = num_slices
        self.min_prbs = min_prbs if min_prbs is not None else np.ones(num_slices, dtype=np.int32)
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.device = device if device is not None else torch.device("cpu")

        # Physical constants for VoI calculation (from simulator config)
        self.queue_capacities = np.array([200000.0, 20000.0, 10000.0], dtype=np.float64)
        self.arrival_means = np.array([45000.0, 3000.0, 800.0], dtype=np.float64)
        self.arrival_stds = np.array([15000.0, 1000.0, 300.0], dtype=np.float64)
        self.nominal_efficiencies = np.array([4000.0, 2500.0, 1500.0], dtype=np.float64)
        self.latency_bounds = np.array([4.0, 1.5, 8.0], dtype=np.float64)  # dwell time in steps
        self.overflow_weights = np.array([10.0, 50.0, 15.0], dtype=np.float64)
        self.latency_weights = np.array([5.0, 25.0, 2.0], dtype=np.float64)
        self.telemetry_costs = np.array([0.5, 0.5, 0.2], dtype=np.float64) * 0.1  # normalized cost

        # Load Neural Forecasting Model
        self.model = None
        self._load_forecasting_model(checkpoint_path)

        # Temporal Memory History: rolling buffer of shape (num_slices, lookback, 4)
        self.history = np.zeros((num_slices, lookback, 4), dtype=np.float32)
        self.last_predicted_queue = np.zeros(num_slices, dtype=np.float64)

    def _load_forecasting_model(self, checkpoint_path: Optional[str]):
        """Loads model weights according to model_type ablation."""
        if self.model_type == "none":
            self.model = None
            return

        default_ckpts = {
            "tkan": "outputs/checkpoints/tkan_best.pt",
            "gru": "outputs/checkpoints/singlelayergru_best.pt",
            "mlp": "outputs/checkpoints/smallmlp_best.pt",
        }
        ckpt_to_use = checkpoint_path or default_ckpts.get(self.model_type)

        if self.model_type == "tkan":
            self.model = TKAN(input_dim=4, horizon=1, units=self.hidden_dim, num_sub_kans=1)
        elif self.model_type == "gru":
            self.model = SingleLayerGRU(in_features=4, hidden_dim=self.hidden_dim, num_layers=1)
        elif self.model_type == "mlp":
            self.model = SmallMLP(seq_len=self.lookback, in_features=4, hidden_dim=self.hidden_dim, out_features=1)
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'")

        if ckpt_to_use and os.path.exists(ckpt_to_use):
            checkpoint = torch.load(ckpt_to_use, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"Loaded {self.model_type.upper()} model weights from {ckpt_to_use}")
        else:
            print(f"Warning: Checkpoint {ckpt_to_use} not found. Running {self.model_type.upper()} with initialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def reset(self):
        """Resets temporal memory buffer at the start of an episode."""
        self.history.fill(0.0)
        self.last_predicted_queue.fill(0.0)

    def _update_temporal_memory(
        self, obs: Dict[str, np.ndarray], prev_r: np.ndarray, prev_m: np.ndarray
    ):
        """
        Rolls forward temporal memory for each slice using only received information.
        Features per slice: [dl_buffer, tx_brate, sum_requested_prbs, sum_granted_prbs]
        """
        obs_q = obs["observed_queue_bytes"]
        for s in range(self.num_slices):
            # Estimate features from observable signals
            # Buffer: latest observed measurement
            f_buf = float(obs_q[s])
            # Service rate proxy: granted PRBs * nominal efficiency in Mbps
            f_brate = float((prev_r[s] * self.nominal_efficiencies[s] * 8) / (self.step_duration_ms_proxy * 1000.0))
            # Requested PRBs proxy: backlog / nominal capacity
            f_req = float(obs_q[s] / max(1.0, self.nominal_efficiencies[s]))
            # Granted PRBs: previous allocation
            f_grant = float(prev_r[s])

            new_vec = np.array([f_buf, f_brate, f_req, f_grant], dtype=np.float32)
            # Roll buffer: push new entry to the end
            self.history[s, :-1, :] = self.history[s, 1:, :]
            self.history[s, -1, :] = new_vec

    @property
    def step_duration_ms_proxy(self) -> float:
        return 250.0

    def predict_next_queues(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predicts next-step queue backlog \tilde{Q}_{k, t+1} for each slice.
        """
        obs_q = obs["observed_queue_bytes"].astype(np.float64)
        if self.model is None or self.model_type == "none":
            # No-temporal-memory ablation: predict persistence of latest observed measurement
            return obs_q.copy()

        predicted = np.zeros(self.num_slices, dtype=np.float64)
        with torch.no_grad():
            for s in range(self.num_slices):
                # Standardize input features using train-fitted scaler
                norm_hist = (self.history[s] - SCALER_MEAN) / SCALER_STD
                x_tensor = torch.from_numpy(norm_hist).unsqueeze(0).to(self.device)  # (1, 16, 4)
                pred_log = self.model(x_tensor).view(-1).item()
                # Invert log1p target: max(0, expm1(pred))
                pred_bytes = max(0.0, math.expm1(pred_log))
                predicted[s] = pred_bytes

        self.last_predicted_queue = predicted.copy()
        return predicted

    def compute_value_of_information(
        self, obs: Dict[str, np.ndarray], predicted_queues: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Value of Information (VoI_k) for each slice:
        VoI_k represents the expected downstream decision loss reduction if telemetry is refreshed,
        compared to making allocation decisions on stale/uncertain data.
        """
        ages = obs["measurement_age_steps"].astype(np.float64)
        prev_r = obs["previous_prb_allocation"].astype(np.float64)

        # Predictive uncertainty spread grows with unobserved duration: sqrt(age + 1) * sigma_arrival
        uncertainty_bytes = np.sqrt(ages + 1.0) * self.arrival_stds

        # Potential unobserved upper queue state: Q_upper = predicted + uncertainty
        q_upper = predicted_queues + uncertainty_bytes

        # Downstream loss risk 1: Buffer overflow risk
        overflow_risk = np.maximum(0.0, q_upper - self.queue_capacities) / self.queue_capacities

        # Downstream loss risk 2: Latency SLA dwell time breach risk
        service_cap = np.maximum(1.0, prev_r * self.nominal_efficiencies)
        dwell_time_steps = q_upper / service_cap
        latency_risk = np.maximum(0.0, dwell_time_steps - self.latency_bounds)

        # Expected Decision Loss (VoI in penalty units)
        expected_decision_loss = (
            self.overflow_weights * overflow_risk + self.latency_weights * latency_risk
        )

        # Marginal Cost of Telemetry Report
        cost = self.telemetry_costs

        # Value-of-Information Ratio
        voi_ratio = expected_decision_loss / cost
        return expected_decision_loss, voi_ratio

    def decide_telemetry(
        self, obs: Dict[str, np.ndarray], predicted_queues: np.ndarray
    ) -> np.ndarray:
        """
        Decides telemetry acquisition mask m_t according to configured telemetry_mode.
        """
        step = int(obs["step"][0])
        mode = self.telemetry_mode

        if mode == "always":
            return np.ones(self.num_slices, dtype=np.int32)
        elif mode == "never":
            return np.zeros(self.num_slices, dtype=np.int32)
        elif mode == "periodic":
            m = np.zeros(self.num_slices, dtype=np.int32)
            if step % self.periodic_interval == 0:
                m[:] = 1
            return m
        elif mode == "age":
            ages = obs["measurement_age_steps"]
            return (ages >= self.age_threshold).astype(np.int32)
        elif mode == "uncertainty":
            ages = obs["measurement_age_steps"].astype(np.float64)
            uncertainty_norm = (np.sqrt(ages + 1.0) * self.arrival_stds) / self.queue_capacities
            return (uncertainty_norm >= self.uncertainty_threshold).astype(np.int32)
        elif mode == "voi":
            # Value of Information Decision Policy
            _, voi_ratio = self.compute_value_of_information(obs, predicted_queues)
            return (voi_ratio >= self.voi_threshold).astype(np.int32)
        else:
            raise ValueError(f"Unknown telemetry_mode: '{mode}'")

    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Closed-loop control step:
        1. Update temporal memory from actually received observation.
        2. Forecast queue state using temporal model.
        3. Evaluate Value of Information to decide telemetry mask.
        4. Solve constrained PRB allocation based on forecast + risk margin.
        """
        prev_r = obs["previous_prb_allocation"]
        prev_m = obs["previous_telemetry_mask"]

        # 1. Update temporal memory
        self._update_temporal_memory(obs, prev_r, prev_m)

        # 2. Forecast next queue state
        predicted_queues = self.predict_next_queues(obs)

        # 3. Decide telemetry request mask
        telemetry_mask = self.decide_telemetry(obs, predicted_queues)

        # 4. Constrained PRB Allocation
        # Demand incorporates predicted backlog plus uncertainty safety margin
        ages = obs["measurement_age_steps"].astype(np.float64)
        uncertainty_bytes = np.sqrt(ages + 1.0) * self.arrival_stds
        effective_demand = predicted_queues + 0.3 * uncertainty_bytes

        prb_alloc = allocate_integer_prbs(
            np.maximum(1.0, effective_demand), self.total_prbs, self.min_prbs
        )

        return telemetry_mask, prb_alloc
