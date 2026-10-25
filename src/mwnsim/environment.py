"""
Phase 6.1 — Closed-Loop Wireless Slicing Simulator

Deterministic-by-seed discrete-time simulation environment for 3 O-RAN slices:
- Slice 0: eMBB (Enhanced Mobile Broadband)
- Slice 1: URLLC (Ultra-Reliable Low-Latency Communication)
- Slice 2: MIoT (Massive IoT)

Maintains strict separation between:
1. True physical state (hidden queue occupancy, channel efficiency, arrivals, overflows).
2. Controller observable state (delayed/stale measurements, measurement ages, allocation history).
3. Telemetry subsystem (delays, loss rates, delivery queue).
4. Decomposed reward terms (queue cost, SLA violations, telemetry overhead, switching penalties).
"""

from typing import Dict, Any, Tuple, List, Optional
import os
import yaml
import copy
import numpy as np


class WirelessSlicingEnv:
    """
    Closed-loop O-RAN slicing simulator for joint telemetry acquisition and PRB allocation.
    Transparent, deterministic-by-seed, and strictly leakage-safe.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if config is not None:
            self.cfg = copy.deepcopy(config)
        else:
            # Default fallback config if no path provided
            default_config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "configs", "simulator.yaml"
            )
            path_to_use = config_path if config_path is not None else default_config_path

            if os.path.exists(path_to_use):
                with open(path_to_use, "r") as f:
                    self.cfg = yaml.safe_load(f)
            else:
                # Inline fallback defaults
                self.cfg = {
                "time": {"step_duration_ms": 250.0, "max_steps": 100, "random_seed": 2026},
                "prb_budget": {"total_prbs": 15, "num_slices": 3, "min_prbs_per_slice": [1, 1, 1]},
                "slices": [
                    {
                        "id": 0, "name": "eMBB", "queue_capacity_bytes": 200000.0,
                        "arrival_mean_bytes": 45000.0, "arrival_std_bytes": 15000.0,
                        "service_efficiency_bytes_per_prb": 4000.0, "telemetry_delay_steps": 1,
                        "telemetry_loss_rate": 0.05, "sla_latency_bound_steps": 4.0,
                        "sla_min_throughput_bytes": 20000.0,
                        "weights": {"queue_cost": 1.0, "overflow_cost": 10.0, "latency_violation_cost": 5.0, "telemetry_cost": 0.5}
                    },
                    {
                        "id": 1, "name": "URLLC", "queue_capacity_bytes": 20000.0,
                        "arrival_mean_bytes": 3000.0, "arrival_std_bytes": 1000.0,
                        "service_efficiency_bytes_per_prb": 2500.0, "telemetry_delay_steps": 1,
                        "telemetry_loss_rate": 0.02, "sla_latency_bound_steps": 1.5,
                        "sla_min_throughput_bytes": 2500.0,
                        "weights": {"queue_cost": 2.0, "overflow_cost": 50.0, "latency_violation_cost": 25.0, "telemetry_cost": 0.5}
                    },
                    {
                        "id": 2, "name": "MIoT", "queue_capacity_bytes": 10000.0,
                        "arrival_mean_bytes": 800.0, "arrival_std_bytes": 300.0,
                        "service_efficiency_bytes_per_prb": 1500.0, "telemetry_delay_steps": 2,
                        "telemetry_loss_rate": 0.10, "sla_latency_bound_steps": 8.0,
                        "sla_min_throughput_bytes": 600.0,
                        "weights": {"queue_cost": 0.5, "overflow_cost": 15.0, "latency_violation_cost": 2.0, "telemetry_cost": 0.2}
                    }
                ],
                "reward_coefficients": {
                    "queue_weight": 0.01, "overflow_weight": 1.0, "sla_violation_weight": 2.0,
                    "telemetry_weight": 0.1, "switch_weight": 0.05
                }
            }

        self.num_slices = int(self.cfg["prb_budget"]["num_slices"])
        self.total_prbs = int(self.cfg["prb_budget"]["total_prbs"])
        self.min_prbs = np.array(self.cfg["prb_budget"].get("min_prbs_per_slice", [0] * self.num_slices), dtype=np.int32)
        self.max_steps = int(self.cfg["time"]["max_steps"])
        self.step_duration_ms = float(self.cfg["time"]["step_duration_ms"])

        # Slice-specific physical parameters
        self.queue_capacities = np.array([s["queue_capacity_bytes"] for s in self.cfg["slices"]], dtype=np.float64)
        self.arrival_means = np.array([s["arrival_mean_bytes"] for s in self.cfg["slices"]], dtype=np.float64)
        self.arrival_stds = np.array([s["arrival_std_bytes"] for s in self.cfg["slices"]], dtype=np.float64)
        self.nominal_efficiencies = np.array([s["service_efficiency_bytes_per_prb"] for s in self.cfg["slices"]], dtype=np.float64)
        self.telemetry_delays = np.array([s["telemetry_delay_steps"] for s in self.cfg["slices"]], dtype=np.int32)
        self.telemetry_loss_rates = np.array([s["telemetry_loss_rate"] for s in self.cfg["slices"]], dtype=np.float64)
        self.latency_bounds = np.array([s["sla_latency_bound_steps"] for s in self.cfg["slices"]], dtype=np.float64)
        self.min_throughputs = np.array([s["sla_min_throughput_bytes"] for s in self.cfg["slices"]], dtype=np.float64)

        # Weights
        self.slice_weights = [s["weights"] for s in self.cfg["slices"]]
        self.global_weights = self.cfg.get("reward_coefficients", {
            "queue_weight": 0.01, "overflow_weight": 1.0, "sla_violation_weight": 2.0,
            "telemetry_weight": 0.1, "switch_weight": 0.05
        })

        # Master Seed & RNG
        self.initial_seed = seed if seed is not None else int(self.cfg["time"].get("random_seed", 2026))
        self.rng = np.random.default_rng(self.initial_seed)

        # State Variables
        self.current_step = 0
        self.true_queue = np.zeros(self.num_slices, dtype=np.float64)
        self.service_efficiencies = np.zeros(self.num_slices, dtype=np.float64)
        self.prev_allocation = np.zeros(self.num_slices, dtype=np.int32)
        self.prev_telemetry_mask = np.zeros(self.num_slices, dtype=np.int32)

        # Observable Telemetry State
        self.latest_measurements = np.zeros(self.num_slices, dtype=np.float64)
        self.measurement_ages = np.zeros(self.num_slices, dtype=np.int32)

        # Pending Telemetry In-Flight Queue: list of dicts
        self.pending_telemetry: List[Dict[str, Any]] = []

        self.reset(seed=self.initial_seed)

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Resets simulator state deterministically using seed.
        Returns the initial controller observation.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.pending_telemetry = []

        # Initial true queues: set to nominal initial fraction of capacity
        self.true_queue = self.rng.uniform(0.1, 0.3, size=self.num_slices) * self.queue_capacities

        # Initial channel efficiencies with small log-normal jitter
        jitter = np.exp(self.rng.normal(0.0, 0.1, size=self.num_slices))
        self.service_efficiencies = self.nominal_efficiencies * jitter

        # Initial allocation: fair distribution satisfying min_prbs and total_prbs
        self.prev_allocation = self._get_default_allocation()
        self.prev_telemetry_mask = np.zeros(self.num_slices, dtype=np.int32)

        # Initial telemetry observation: perfectly synchronized at step 0
        self.latest_measurements = self.true_queue.copy()
        self.measurement_ages = np.zeros(self.num_slices, dtype=np.int32)

        return self._get_obs()

    def _get_default_allocation(self) -> np.ndarray:
        """Computes a valid default allocation satisfying sum(R) == total_prbs and R >= min_prbs."""
        alloc = self.min_prbs.copy()
        remaining = self.total_prbs - int(np.sum(alloc))
        for i in range(remaining):
            alloc[i % self.num_slices] += 1
        return alloc

    def is_feasible_action(self, prb_allocation: np.ndarray) -> Tuple[bool, str]:
        """Validates action feasibility (non-negative integer, sum == 15, >= min_prbs)."""
        alloc = np.array(prb_allocation, dtype=np.int64)
        if len(alloc) != self.num_slices:
            return False, f"Allocation length {len(alloc)} does not match num_slices {self.num_slices}"
        if np.any(alloc < 0):
            return False, f"Negative PRB allocation detected: {alloc}"
        if np.any(alloc < self.min_prbs):
            return False, f"Allocation {alloc} violates minimum PRB constraints {self.min_prbs}"
        if np.sum(alloc) != self.total_prbs:
            return False, f"PRB sum {np.sum(alloc)} does not equal budget {self.total_prbs}"
        return True, "Feasible"

    def project_feasible_action(self, prb_allocation: np.ndarray) -> np.ndarray:
        """Projects any allocation onto the nearest valid integer simplex satisfying all constraints."""
        alloc = np.maximum(self.min_prbs, np.array(prb_allocation, dtype=np.int32))
        curr_sum = int(np.sum(alloc))
        diff = self.total_prbs - curr_sum

        if diff > 0:
            # Distribute excess PRBs
            for i in range(diff):
                alloc[i % self.num_slices] += 1
        elif diff < 0:
            # Decrement above min_prbs
            diff = abs(diff)
            idx = 0
            while diff > 0:
                s_idx = idx % self.num_slices
                if alloc[s_idx] > self.min_prbs[s_idx]:
                    alloc[s_idx] -= 1
                    diff -= 1
                idx += 1
                if idx > 1000:  # Safety break
                    break
        return alloc

    def step(
        self, action: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Executes one discrete simulation step (250 ms).

        action: (telemetry_request_mask, prb_allocation)
            - telemetry_request_mask: array of shape (3,), binary 0/1 for requesting measurement.
            - prb_allocation: array of shape (3,), non-negative integers summing to 15.
        """
        telemetry_mask = np.array(action[0], dtype=np.int32)
        raw_prb_alloc = np.array(action[1], dtype=np.int32)

        # Validate action
        is_feasible, reason = self.is_feasible_action(raw_prb_alloc)
        if not is_feasible:
            # Fall back to projection and apply feasibility penalty
            prb_alloc = self.project_feasible_action(raw_prb_alloc)
            infeasibility_penalty = 10.0
        else:
            prb_alloc = raw_prb_alloc
            infeasibility_penalty = 0.0

        # 1. Allocation Switching Penalty
        switch_cost = float(np.sum(np.abs(prb_alloc - self.prev_allocation)))

        # 2. Service & Queue Depletion
        # Channel capacity = allocated PRBs * service efficiency (bytes/PRB)
        service_capacity_bytes = prb_alloc.astype(np.float64) * self.service_efficiencies
        served_bytes = np.minimum(self.true_queue, service_capacity_bytes)
        queue_after_service = self.true_queue - served_bytes

        # Latency Dwell Time Proxy (in steps): queue at start of step / service capacity
        # Represents how many steps would be needed to drain the current backlog
        dwell_time_steps = self.true_queue / np.maximum(1.0, service_capacity_bytes)
        latency_violations = np.maximum(0.0, dwell_time_steps - self.latency_bounds)

        # Throughput starvation violations
        starvation_deficit_bytes = np.maximum(0.0, self.min_throughputs - served_bytes)

        # 3. Stochastic Traffic Arrivals & Overflow
        arrivals = np.maximum(0.0, self.rng.normal(self.arrival_means, self.arrival_stds))
        raw_new_queue = queue_after_service + arrivals
        overflow_bytes = np.maximum(0.0, raw_new_queue - self.queue_capacities)
        next_true_queue = np.minimum(self.queue_capacities, raw_new_queue)

        # 4. Telemetry Subsystem Processing
        # Step A: Check pending telemetry in-flight arriving at current_step + 1
        target_delivery_step = self.current_step + 1
        delivered_slices = set()

        remaining_pending = []
        for report in self.pending_telemetry:
            if report["delivery_step"] == target_delivery_step:
                if not report["dropped"]:
                    s_id = report["slice_id"]
                    self.latest_measurements[s_id] = report["measured_value"]
                    self.measurement_ages[s_id] = target_delivery_step - report["generation_step"]
                    delivered_slices.add(s_id)
            elif report["delivery_step"] > target_delivery_step:
                remaining_pending.append(report)
        self.pending_telemetry = remaining_pending

        # Step B: Age measurements for slices that received no delivery
        for s in range(self.num_slices):
            if s not in delivered_slices:
                self.measurement_ages[s] += 1

        # Step C: Schedule new telemetry requests made in this action
        telemetry_cost = 0.0
        for s in range(self.num_slices):
            if telemetry_mask[s] == 1:
                telemetry_cost += self.slice_weights[s]["telemetry_cost"]
                is_dropped = bool(self.rng.random() < self.telemetry_loss_rates[s])
                delivery_step = self.current_step + 1 + int(self.telemetry_delays[s])
                # Measurement snapshots the true queue at decision time
                self.pending_telemetry.append({
                    "slice_id": s,
                    "generation_step": self.current_step,
                    "delivery_step": delivery_step,
                    "measured_value": float(self.true_queue[s]),
                    "dropped": is_dropped
                })

        # 5. Channel State Evolution for Next Step (Gauss-Markov / AR(1) fading)
        rho = 0.9  # Temporal channel correlation
        channel_innov = self.rng.normal(0.0, 0.1, size=self.num_slices)
        self.service_efficiencies = np.maximum(
            0.2 * self.nominal_efficiencies,
            rho * self.service_efficiencies + (1 - rho) * self.nominal_efficiencies + self.nominal_efficiencies * channel_innov
        )

        # 6. Decomposed Cost & Reward Calculation
        # Normalized Queue Cost
        q_cost = float(np.sum([
            self.slice_weights[s]["queue_cost"] * (self.true_queue[s] / self.queue_capacities[s])
            for s in range(self.num_slices)
        ]))

        # Normalized Overflow Cost
        ovf_cost = float(np.sum([
            self.slice_weights[s]["overflow_cost"] * (overflow_bytes[s] / self.queue_capacities[s])
            for s in range(self.num_slices)
        ]))

        # SLA Delay & Starvation Penalties
        sla_cost = float(np.sum([
            self.slice_weights[s]["latency_violation_cost"] * latency_violations[s] +
            (starvation_deficit_bytes[s] / self.min_throughputs[s])
            for s in range(self.num_slices)
        ]))

        # Reward terms (negative costs)
        w_q = self.global_weights["queue_weight"]
        w_ovf = self.global_weights["overflow_weight"]
        w_sla = self.global_weights["sla_violation_weight"]
        w_tel = self.global_weights["telemetry_weight"]
        w_sw = self.global_weights["switch_weight"]

        reward_components = {
            "queue_cost": -float(w_q * q_cost),
            "overflow_cost": -float(w_ovf * ovf_cost),
            "sla_violation_cost": -float(w_sla * sla_cost),
            "telemetry_cost": -float(w_tel * telemetry_cost),
            "switch_cost": -float(w_sw * switch_cost),
            "infeasibility_penalty": -float(infeasibility_penalty)
        }

        total_reward = float(sum(reward_components.values()))

        # Update persistent step and state
        self.current_step += 1
        self.true_queue = next_true_queue
        self.prev_allocation = prb_alloc.copy()
        self.prev_telemetry_mask = telemetry_mask.copy()

        done = self.current_step >= self.max_steps

        # Strictly Leakage-Safe Controller Observation
        obs = self._get_obs()

        # Detailed logging info (contains true environment state for analysis, NOT for controller input)
        info = {
            "step": self.current_step,
            "true_queue_bytes": next_true_queue.copy(),
            "arrivals_bytes": arrivals.copy(),
            "served_bytes": served_bytes.copy(),
            "overflow_bytes": overflow_bytes.copy(),
            "dwell_time_steps": dwell_time_steps.copy(),
            "latency_violations": latency_violations.copy(),
            "service_efficiencies": self.service_efficiencies.copy(),
            "reward_components": reward_components,
            "total_reward": total_reward,
            "is_feasible": is_feasible,
            "action_prb_allocated": prb_alloc.copy(),
            "action_telemetry_requested": telemetry_mask.copy(),
            "pending_telemetry_count": len(self.pending_telemetry)
        }

        return obs, total_reward, done, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Constructs the controller's observable view.
        Strictly excludes true queue, unreleased telemetry, and future arrivals.
        """
        return {
            "observed_queue_bytes": self.latest_measurements.copy().astype(np.float32),
            "measurement_age_steps": self.measurement_ages.copy().astype(np.int32),
            "previous_prb_allocation": self.prev_allocation.copy().astype(np.int32),
            "previous_telemetry_mask": self.prev_telemetry_mask.copy().astype(np.int32),
            "step": np.array([self.current_step], dtype=np.int32)
        }
