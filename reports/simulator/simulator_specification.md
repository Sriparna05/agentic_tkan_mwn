# Phase 6.1 — Closed-Loop Wireless Slicing Simulator Specification

**Project Title:** Agentic Temporal KAN for Joint Telemetry Acquisition and QoS-Constrained RAN Slicing  
**Target Venue:** IEEE ICC 2027 — Mobile and Wireless Networks Track  
**Component:** Discrete-Time Closed-Loop Simulator (`mwnsim`)  
**Specification Date:** October 2026  
**Implementation Source:** [src/mwnsim/environment.py](file:///d:/Research-kan/agentic_tkan_mwn/src/mwnsim/environment.py)  
**Configuration File:** [configs/simulator.yaml](file:///d:/Research-kan/agentic_tkan_mwn/configs/simulator.yaml)  
**Test Suite:** [tests/test_environment.py](file:///d:/Research-kan/agentic_tkan_mwn/tests/test_environment.py) (All 9 unit tests passing, 39/39 repository tests passing)

---

## 1. System Scope & Disclaimer

This simulator is designed specifically for **controlled research experiments** evaluating joint telemetry acquisition and PRB allocation policies.
- **Scope:** A transparent, discrete-time, deterministic-by-seed simulation environment modeling 3 distinct 3GPP/O-RAN network slices.
- **Scientific Disclaimer:** This environment is **not** a full PHY/MAC simulator (such as ns-3 or OpenAirInterface) and does not claim to be a field-validated O-RAN deployment. All arrival distributions, nominal channel efficiencies, and queue capacities are configurable research abstractions aligned with the dynamic ranges observed in the Colosseum O-RAN COMMAG dataset.

---

## 2. State Space Architecture

The simulator enforces a strict **information barrier** between physical hidden ground truth and the observable controller state.

```mermaid
graph TD
    subgraph True Physical State (Hidden)
        TQ[True Queue Backlog Q_t]
        ARR[Traffic Arrivals A_t]
        EFF[Channel Efficiency mu_t]
        OVF[Buffer Overflows O_t]
    end

    subgraph Telemetry Subsystem
        REQ[Controller Request Mask m_t] --> DELAY[Configured Delay d_k]
        DELAY --> LOSS{Report Loss p_loss?}
        LOSS -- No --> IN_FLIGHT[Delivery at t + 1 + d_k]
        LOSS -- Yes --> DROPPED[Dropped / Never Delivered]
    end

    subgraph Controller Observable State (Leakage-Free)
        OBS_Q[Observed Queue Hat_Q_t]
        AGE[Measurement Age Delta_t]
        PREV_R[Previous PRB Allocation R_t-1]
        PREV_M[Previous Telemetry Mask m_t-1]
        STEP[Time Step t]
    end

    IN_FLIGHT --> OBS_Q
    TQ --> DELAY
```

### 2.1 Hidden Physical State
1. **True Queue Occupancy ($Q_{k, t} \in \mathbb{R}_{\ge 0}$):** Buffer occupancy in bytes for each slice $k \in \{0, 1, 2\}$, bounded by finite capacity $C_k$.
2. **Traffic Arrivals ($A_{k, t} \in \mathbb{R}_{\ge 0}$):** Stochastic byte arrivals generated per step from slice-specific distributions matching COMMAG burst characteristics.
3. **Channel Service Efficiency ($\mu_{k, t} \in \mathbb{R}_{> 0}$):** Effective transmission rate per allocated PRB in bytes/PRB per 250 ms step, evolved via Gauss-Markov AR(1) fading:
   $$\mu_{k, t+1} = \max\left(0.2 \mu_{0, k}, \, \rho \mu_{k, t} + (1 - \rho)\mu_{0, k} + \mu_{0, k} \cdot \epsilon_{k, t}\right)$$
4. **Buffer Overflows ($O_{k, t} \in \mathbb{R}_{\ge 0}$):** Packets arriving beyond the finite buffer capacity $C_k$ that are irrevocably dropped.

### 2.2 Observable Controller State
The controller observation dictionary `obs` contains **only**:
- `observed_queue_bytes`: Latest successfully delivered telemetry measurement $\hat{Q}_{k, t}$.
- `measurement_age_steps`: Elapsed timesteps $\Delta_{k, t} \in \mathbb{Z}_{\ge 0}$ since the last measurement was generated and delivered.
- `previous_prb_allocation`: $R_{k, t-1} \in \mathbb{Z}_{\ge 0}^3$.
- `previous_telemetry_mask`: $m_{k, t-1} \in \{0, 1\}^3$.
- `step`: Current time step index $t \in [0, T_{\max}]$.

---

## 3. Action Space and Feasibility Constraints

At each decision step $t$, the controller selects a composite action $a_t = (m_t, R_t)$:
1. **Telemetry Request Mask ($m_t \in \{0, 1\}^3$):** Binary decision vector indicating whether to trigger a telemetry report for each slice.
2. **Integer PRB Allocation ($R_t = [R_0, R_1, R_2] \in \mathbb{Z}_{\ge 0}^3$):** Physical Resource Block assignments across the 3 slices.

### 3.1 Mathematical Feasibility Conditions
$$\sum_{k=0}^{2} R_{k, t} = B = 15 \quad (\text{Exact total budget constraint})$$
$$R_{k, t} \ge R_{\min, k} = 1 \quad \forall k \in \{0, 1, 2\} \quad (\text{Minimum slice reservation})$$
$$R_{k, t} \in \mathbb{Z}_{\ge 0} \quad (\text{Discrete resource block integrity})$$

### 3.2 Action Space Partitioning
With $B=15$ total PRBs and minimum reservation $R_{\min} = [1, 1, 1]$, the number of unallocated free PRBs is $15 - 3 = 12$. The total number of valid integer PRB partitions is given by stars-and-bars:
$$\binom{12 + 3 - 1}{3 - 1} = \binom{14}{2} = 91 \text{ valid PRB allocation vectors}$$
Combined with the $2^3 = 8$ telemetry acquisition masks, the discrete joint action space contains:
$$|\mathcal{A}| = 8 \times 91 = 728 \text{ feasible joint actions}$$

---

## 4. Telemetry Subsystem Mechanics

When a telemetry measurement is requested on slice $k$ ($m_{k, t} = 1$):
1. **Timestamping:** The true queue backlog $Q_{k, t}$ is snapshotted at generation time $t_{\text{gen}} = t$.
2. **Delay Scheduling:** Delivery is scheduled for future timestep:
   $$t_{\text{delivery}} = t + 1 + d_k$$
   where $d_k$ is the configured transmission delay in steps ($d_0 = 1, d_1 = 1, d_2 = 2$).
3. **Report Loss:** With probability $p_{\text{loss}, k}$, the telemetry report is dropped in the radio link or fronthaul, simulating E2 interface packet loss.
4. **Delivery & Staleness Handling:**
   - When $t_{\text{curr}} = t_{\text{delivery}}$, if the report was not dropped, $\hat{Q}_{k, t} \leftarrow Q_{k, t_{\text{gen}}}$ and age is reset to $\Delta_{k, t} = t_{\text{curr}} - t_{\text{gen}}$.
   - If no report arrives at step $t$ (due to no request, ongoing transit delay, or report loss), the observation remains **frozen at the last valid measurement**:
     $$\hat{Q}_{k, t} = \hat{Q}_{k, t-1}, \quad \Delta_{k, t} = \Delta_{k, t-1} + 1$$

---

## 5. Queue Dynamics & State Evolution

The discrete-time queue transition operates in exact sequential phases within each 250 ms step:

1. **Service Depletion:**
   $$C_{\text{serv}, k, t} = R_{k, t} \cdot \mu_{k, t} \quad (\text{Bytes of transmission capacity})$$
   $$S_{k, t} = \min(Q_{k, t}, \, C_{\text{serv}, k, t}) \quad (\text{Actual served bytes})$$
   $$Q_{k, t}^{\text{post-serv}} = Q_{k, t} - S_{k, t}$$

2. **Traffic Inflow & Finite Buffer Overflow:**
   $$A_{k, t} \sim \max(0, \, \mathcal{N}(\mu_{A, k}, \sigma_{A, k}^2)) \quad (\text{Stochastic arrival in bytes})$$
   $$Q_{k, t+1} = \min(C_k, \, Q_{k, t}^{\text{post-serv}} + A_{k, t})$$
   $$O_{k, t} = \max(0.0, \, Q_{k, t}^{\text{post-serv}} + A_{k, t} - C_k) \quad (\text{Dropped overflow bytes})$$

---

## 6. Decomposed Reward & Objective Formulation

The simulator returns both a scalar total reward $r_t$ and a transparent dictionary `info["reward_components"]`. Every term is explicitly defined with documented units:

| Component Key | Mathematical Formulation | Physical Interpretation & Units |
| :--- | :--- | :--- |
| **`queue_cost`** | $- w_q \sum_k w_{Q, k} \frac{Q_{k, t}}{C_k}$ | Normalized buffer occupancy penalty (dimensionless). |
| **`overflow_cost`** | $- w_{\text{ovf}} \sum_k w_{O, k} \frac{O_{k, t}}{C_k}$ | Packet drop penalty when queue exceeds capacity (dimensionless). |
| **`sla_violation_cost`** | $- w_{\text{sla}} \sum_k \left[ w_{L, k} (T_{\text{dwell}, k} - D_{\max, k})^+ + \left(1 - \frac{S_{k, t}}{S_{\min, k}}\right)^+ \right]$ | Dwell time delay bound breach and throughput starvation penalties. |
| **`telemetry_cost`** | $- w_{\text{tel}} \sum_k w_{T, k} \cdot m_{k, t}$ | Signaling overhead for E2 telemetry polling (dimensionless). |
| **`switch_cost`** | $- w_{\text{sw}} \sum_k |R_{k, t} - R_{k, t-1}|$ | Reconfiguration and signaling overhead of changing PRB allocations. |
| **`infeasibility_penalty`** | $- 10.0 \cdot \mathbf{1}(\text{action is infeasible})$ | Barrier penalty for violating PRB budget or bounds. |

---

## 7. Verification Test Suite Summary

The unit test suite in [tests/test_environment.py](file:///d:/Research-kan/agentic_tkan_mwn/tests/test_environment.py) validates all 10 simulator specifications:

1. **Exact PRB Feasibility:** Confirmed valid allocations pass, while sum violations, below-minimum PRBs, and negative PRBs are rejected and projected.
2. **Queue Bounds:** Confirmed $0 \le Q_{k, t} \le C_k$ and overflow $O_{k, t} \ge 0$ under forced saturation.
3. **Reset Behavior:** Confirmed step resets to 0, pending queues clear, and initial states restore cleanly.
4. **Seed Reproducibility:** Confirmed identical seeds produce bit-level identical observations, true states, and rewards across 10 simulation steps.
5. **Telemetry Delay:** Verified that requests made at step 0 are not delivered at step 1 and arrive at scheduled step 2.
6. **Report Loss:** Verified that with 100% loss, dropped reports are never revealed and measurement age increases monotonically.
7. **Stale Measurement Handling:** Verified that previous measurements are preserved without corruption when no new report arrives.
8. **No Information Leakage:** Confirmed that `obs` contains only observable keys and completely excludes `true_queue`, `arrivals`, or `service_efficiency`.
9. **Reward Decomposition:** Confirmed all 6 cost terms are negative, physically consistent, and sum exactly to total reward.
10. **Termination Behavior:** Confirmed `done == True` when `current_step == max_steps`.
