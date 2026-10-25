"""
Phase 7.1 — Wireless Slice Control Baselines and Telemetry Policies

Implements 6 distinct control policies for O-RAN slicing:
1. EqualPeriodicPolicy: Static equal PRB allocation with periodic telemetry.
2. QueueAwarePeriodicPolicy: Backlog-proportional PRB allocation with periodic telemetry.
3. QueueAwareAgeTriggeredPolicy: Queue-aware allocation with age-thresholded telemetry.
4. QueueAwareUncertaintyPolicy: Queue-aware allocation with variance/uncertainty-triggered telemetry.
5. DecisionSensitivePolicy: Telemetry triggered by allocation sensitivity and overflow risk proxies.
6. PrivilegedFullInfoPolicy: Privileged reference accessing true hidden state (upper bound benchmark).
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np


def allocate_integer_prbs(
    demands: np.ndarray, total_prbs: int = 15, min_prbs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Deterministic Largest-Remainder (Hamilton) integer apportionment method.
    Guarantees sum(R) == total_prbs exactly and R >= min_prbs.
    """
    K = len(demands)
    if min_prbs is None:
        min_prbs = np.ones(K, dtype=np.int32)
    else:
        min_prbs = np.array(min_prbs, dtype=np.int32)

    remaining_budget = total_prbs - int(np.sum(min_prbs))
    if remaining_budget <= 0:
        return min_prbs.copy()

    # Avoid divide-by-zero
    demand_sum = float(np.sum(demands))
    if demand_sum <= 1e-6:
        # Uniform distribution of remaining PRBs
        extra = np.zeros(K, dtype=np.int32)
        for i in range(remaining_budget):
            extra[i % K] += 1
        return min_prbs + extra

    # Exact quota
    quotas = (demands / demand_sum) * remaining_budget
    int_alloc = np.floor(quotas).astype(np.int32)
    remainder = quotas - int_alloc

    leftover = remaining_budget - int(np.sum(int_alloc))
    if leftover > 0:
        # Rank by largest remainder; tie-break deterministically by index
        rank = np.lexsort((np.arange(K), -remainder))
        for i in range(leftover):
            int_alloc[rank[i]] += 1

    final_alloc = min_prbs + int_alloc
    assert np.sum(final_alloc) == total_prbs
    assert np.all(final_alloc >= min_prbs)
    return final_alloc


class BaseControlPolicy(ABC):
    """Abstract base class for joint telemetry-acquisition and PRB-slicing policies."""

    def __init__(self, name: str, is_privileged: bool = False):
        self.name = name
        self.is_privileged = is_privileged

    @abstractmethod
    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            telemetry_mask: np.ndarray of shape (num_slices,), binary {0, 1}
            prb_allocation: np.ndarray of shape (num_slices,), non-negative integers summing to total_prbs
        """
        pass

    def reset(self):
        """Reset policy internal state between episodes."""
        pass


class EqualPeriodicPolicy(BaseControlPolicy):
    """
    Policy 1: Static equal PRB allocation ([5, 5, 5]) with periodic telemetry polling.
    """

    def __init__(self, period: int = 2, total_prbs: int = 15, num_slices: int = 3):
        super().__init__(f"EqualPeriodic_P{period}")
        self.period = max(1, period)
        self.total_prbs = total_prbs
        self.num_slices = num_slices
        self.static_alloc = np.full(num_slices, total_prbs // num_slices, dtype=np.int32)
        # Remainder adjustment
        rem = total_prbs % num_slices
        for i in range(rem):
            self.static_alloc[i] += 1

    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        step = int(obs["step"][0])
        telemetry_mask = np.zeros(self.num_slices, dtype=np.int32)
        if step % self.period == 0:
            telemetry_mask[:] = 1
        return telemetry_mask, self.static_alloc.copy()


class QueueAwarePeriodicPolicy(BaseControlPolicy):
    """
    Policy 2: Queue-aware PRB allocation based on latest observed buffer backlog,
    with periodic telemetry polling.
    """

    def __init__(
        self,
        period: int = 2,
        total_prbs: int = 15,
        num_slices: int = 3,
        min_prbs: Optional[np.ndarray] = None,
    ):
        super().__init__(f"QueueAwarePeriodic_P{period}")
        self.period = max(1, period)
        self.total_prbs = total_prbs
        self.num_slices = num_slices
        self.min_prbs = min_prbs if min_prbs is not None else np.ones(num_slices, dtype=np.int32)

    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        step = int(obs["step"][0])
        telemetry_mask = np.zeros(self.num_slices, dtype=np.int32)
        if step % self.period == 0:
            telemetry_mask[:] = 1

        obs_q = obs["observed_queue_bytes"]
        # Demand proxy is observed backlog; minimum 1.0 to avoid zero division
        demands = np.maximum(1.0, obs_q)
        prb_alloc = allocate_integer_prbs(demands, self.total_prbs, self.min_prbs)
        return telemetry_mask, prb_alloc


class QueueAwareAgeTriggeredPolicy(BaseControlPolicy):
    """
    Policy 3: Queue-aware allocation with age-thresholded telemetry.
    Requests telemetry only when measurement staleness Delta_k >= max_age.
    """

    def __init__(
        self,
        max_age: int = 3,
        total_prbs: int = 15,
        num_slices: int = 3,
        min_prbs: Optional[np.ndarray] = None,
    ):
        super().__init__(f"QueueAwareAge_T{max_age}")
        self.max_age = max_age
        self.total_prbs = total_prbs
        self.num_slices = num_slices
        self.min_prbs = min_prbs if min_prbs is not None else np.ones(num_slices, dtype=np.int32)

    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        ages = obs["measurement_age_steps"]
        telemetry_mask = (ages >= self.max_age).astype(np.int32)

        obs_q = obs["observed_queue_bytes"]
        demands = np.maximum(1.0, obs_q)
        prb_alloc = allocate_integer_prbs(demands, self.total_prbs, self.min_prbs)
        return telemetry_mask, prb_alloc


class QueueAwareUncertaintyPolicy(BaseControlPolicy):
    """
    Policy 4: Uncertainty-triggered telemetry.
    Queue uncertainty standard deviation grows as sqrt(Delta_k) * sigma_arrival.
    Telemetry is requested when normalized uncertainty exceeds threshold theta_u.
    PRB allocation guards against backlog variance by adding an uncertainty margin.
    """

    def __init__(
        self,
        theta_u: float = 0.15,
        arrival_stds: Optional[np.ndarray] = None,
        queue_capacities: Optional[np.ndarray] = None,
        total_prbs: int = 15,
        num_slices: int = 3,
        min_prbs: Optional[np.ndarray] = None,
    ):
        super().__init__(f"QueueAwareUncertainty_Th{theta_u:.2f}")
        self.theta_u = theta_u
        self.arrival_stds = (
            arrival_stds if arrival_stds is not None else np.array([15000.0, 1000.0, 300.0])
        )
        self.queue_capacities = (
            queue_capacities if queue_capacities is not None else np.array([200000.0, 20000.0, 10000.0])
        )
        self.total_prbs = total_prbs
        self.num_slices = num_slices
        self.min_prbs = min_prbs if min_prbs is not None else np.ones(num_slices, dtype=np.int32)

    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        ages = obs["measurement_age_steps"]
        # Cumulative arrival standard deviation over unobserved window Delta_k
        uncertainty_bytes = np.sqrt(np.maximum(1, ages)) * self.arrival_stds
        normalized_uncertainty = uncertainty_bytes / self.queue_capacities

        # Request telemetry if uncertainty breaches tolerance threshold
        telemetry_mask = (normalized_uncertainty >= self.theta_u).astype(np.int32)

        # Conservative effective demand: observed buffer + 1 standard deviation uncertainty margin
        obs_q = obs["observed_queue_bytes"]
        effective_backlog = obs_q + 0.5 * uncertainty_bytes
        demands = np.maximum(1.0, effective_backlog)
        prb_alloc = allocate_integer_prbs(demands, self.total_prbs, self.min_prbs)
        return telemetry_mask, prb_alloc


class DecisionSensitivePolicy(BaseControlPolicy):
    """
    Policy 5: Decision-sensitive telemetry.
    Polls telemetry only when:
    (1) Projected queue drift indicates PRB reallocation would change by >= 1 PRB.
    (2) High risk of buffer overflow (>70% capacity) or SLA delay breach.
    """

    def __init__(
        self,
        nominal_arrivals: Optional[np.ndarray] = None,
        nominal_efficiencies: Optional[np.ndarray] = None,
        queue_capacities: Optional[np.ndarray] = None,
        total_prbs: int = 15,
        num_slices: int = 3,
        min_prbs: Optional[np.ndarray] = None,
    ):
        super().__init__("DecisionSensitive")
        self.arrival_means = (
            nominal_arrivals if nominal_arrivals is not None else np.array([45000.0, 3000.0, 800.0])
        )
        self.service_efficiencies = (
            nominal_efficiencies
            if nominal_efficiencies is not None
            else np.array([4000.0, 2500.0, 1500.0])
        )
        self.queue_capacities = (
            queue_capacities if queue_capacities is not None else np.array([200000.0, 20000.0, 10000.0])
        )
        self.total_prbs = total_prbs
        self.num_slices = num_slices
        self.min_prbs = min_prbs if min_prbs is not None else np.ones(num_slices, dtype=np.int32)

    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs_q = obs["observed_queue_bytes"]
        ages = obs["measurement_age_steps"]
        prev_r = obs["previous_prb_allocation"]

        # 1. Compute baseline allocation based on current stale observation
        base_alloc = allocate_integer_prbs(np.maximum(1.0, obs_q), self.total_prbs, self.min_prbs)

        # 2. Project queue drift: Q_proj = obs_q + Delta * (arrival - served)
        net_drift = self.arrival_means - prev_r * self.service_efficiencies
        projected_q = np.maximum(0.0, obs_q + ages * net_drift)
        drift_alloc = allocate_integer_prbs(
            np.maximum(1.0, projected_q), self.total_prbs, self.min_prbs
        )

        # 3. Decision Sensitivity & Risk Trigger
        # Condition A: Projected mismatch >= 1 PRB
        allocation_mismatch = np.abs(drift_alloc - base_alloc) >= 1
        # Condition B: High overflow risk (>70% capacity)
        overflow_risk = (projected_q / self.queue_capacities) > 0.70
        # Condition C: Staleness bound (max 5 steps)
        max_staleness = ages >= 5

        telemetry_mask = (allocation_mismatch | overflow_risk | max_staleness).astype(np.int32)
        return telemetry_mask, drift_alloc


class PrivilegedFullInfoPolicy(BaseControlPolicy):
    """
    Policy 6: Privileged Full-Information Reference (Upper Bound).
    Directly accesses true instantaneous queue backlog Q_{k, t} and channel efficiency mu_{k, t}
    from the simulator's true state.
    Requires 0 telemetry polling (m_t = [0, 0, 0]).
    STRICT DISCLAIMER: Privileged oracle benchmark for comparison only; not fair for deployment.
    """

    def __init__(
        self,
        total_prbs: int = 15,
        num_slices: int = 3,
        min_prbs: Optional[np.ndarray] = None,
    ):
        super().__init__("PrivilegedFullInfo_Oracle", is_privileged=True)
        self.total_prbs = total_prbs
        self.num_slices = num_slices
        self.min_prbs = min_prbs if min_prbs is not None else np.ones(num_slices, dtype=np.int32)

    def select_action(
        self, obs: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Privileged access: reads true queue from info if available, else falls back to obs
        if info is not None and "true_queue_bytes" in info:
            true_q = info["true_queue_bytes"]
        else:
            true_q = obs["observed_queue_bytes"]

        # No telemetry requested: oracle already knows truth
        telemetry_mask = np.zeros(self.num_slices, dtype=np.int32)

        # Allocate PRBs strictly proportional to true instantaneous backlog
        demands = np.maximum(1.0, true_q)
        prb_alloc = allocate_integer_prbs(demands, self.total_prbs, self.min_prbs)
        return telemetry_mask, prb_alloc
