"""
Phase 9.1 — Publication-Quality Figure Generator

Generates high-resolution IEEE-style figures (300 DPI) from saved experimental outputs:
1. fig1_pareto_constraint_vs_telemetry.png: SLA violation rate vs Telemetry request rate.
2. fig2_architecture_ablation.png: Model ablation (TKAN vs GRU vs MLP vs NoMemory).
3. fig3_telemetry_ablation.png: Telemetry acquisition strategy comparison under TKAN.
4. fig4_sensitivity_sweeps.png: 4-panel sensitivity analysis (Delay, Loss, Bursts, Capacity).
5. fig5_recovery_outage.png: Transient backlog trajectory during telemetry outage.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Style settings for IEEE Transactions / Conferences
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_fig1_pareto():
    """Figure 1: SLA Violation Rate vs. Telemetry Request Rate (Pareto Trade-off)."""
    df = pd.read_csv("outputs/metrics/final_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.8))

    # Color palette
    colors = {
        "AgenticTKAN_VoI": "#1f77b4",
        "AgenticGRU_VoI": "#ff7f0e",
        "AgenticMLP_VoI": "#d62728",
        "NoMemory_VoI": "#2ca02c",
        "DecisionSensitive": "#9467bd",
        "PrivilegedFullInfo_Oracle": "#8c564b",
    }

    for _, row in df.iterrows():
        name = row["policy"]
        x = row["telemetry_request_rate_mean"] * 100
        y = row["sla_violation_rate_mean"] * 100
        x_err = row["telemetry_request_rate_ci95"] * 100
        y_err = row["sla_violation_rate_ci95"] * 100

        color = colors.get(name, "#7f7f7f")
        marker = "*" if "VoI" in name else ("D" if "Oracle" in name else "o")
        size = 120 if "VoI" in name or "Oracle" in name else 60

        ax.errorbar(
            x, y, xerr=x_err, yerr=y_err, fmt=marker, color=color,
            markersize=9 if "VoI" in name else 6, capsize=3, elinewidth=1.2,
            label=name if name in colors or "Periodic" in name or "Always" in name else None
        )

        # Label key points
        if name in ["AgenticTKAN_VoI", "AgenticGRU_VoI", "AgenticMLP_VoI", "NoMemory_VoI", "DecisionSensitive", "PrivilegedFullInfo_Oracle"]:
            offset_y = 2.0 if "MLP" not in name else -3.0
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, offset_y), fontsize=8, fontweight="bold" if "TKAN" in name else "normal")

    ax.set_xlabel("Telemetry Request Rate (%)")
    ax.set_ylabel("SLA Violation Rate (%)")
    ax.set_title("Pareto Trade-Off: QoS Compliance vs. Telemetry Overhead")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Invert desired direction: ideal is bottom-left (low telemetry, low SLA violations)
    ax.annotate("Ideal Direction\n(Efficient & Compliant)", xy=(5, 25), xytext=(20, 15),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5), fontsize=8, color="green")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig1_pareto_constraint_vs_telemetry.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Generated {fig_path}")


def plot_fig2_architecture_ablation():
    """Figure 2: Architecture Ablation (TKAN vs GRU vs MLP vs NoMemory under matched VoI)."""
    df = pd.read_csv("outputs/metrics/ablation_results.csv")
    models = ["AgenticTKAN_VoI", "AgenticGRU_VoI", "AgenticMLP_VoI", "NoMemory_VoI"]
    df_sub = df[df["policy"].isin(models)].set_index("policy").reindex(models)

    labels = ["TKAN\n(Proposed)", "GRU\n(Recurrent)", "MLP\n(Feedforward)", "No-Memory\n(Persistence)"]
    x = np.arange(len(labels))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.2))

    # Left: Total Mean Reward
    rewards = df_sub["total_reward_mean"].to_numpy()
    rewards_err = df_sub["total_reward_ci95"].to_numpy()
    bars1 = ax1.bar(x, rewards, width=0.5, yerr=rewards_err, capsize=4, color=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Total Mean Reward")
    ax1.set_title("(a) Cumulative Control Reward (Higher is Better)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Add values on bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval - 1200, f"{yval:.0f}", ha="center", va="top", fontsize=8, fontweight="bold")

    # Right: SLA Violation & Overflow Frequency
    sla_rates = df_sub["sla_violation_rate_mean"].to_numpy() * 100
    sla_err = df_sub["sla_violation_rate_ci95"].to_numpy() * 100
    ovf_rates = df_sub["overflow_frequency_mean"].to_numpy() * 100
    ovf_err = df_sub["overflow_frequency_ci95"].to_numpy() * 100

    ax2.bar(x - width/2, sla_rates, width, yerr=sla_err, label="SLA Violation Rate (%)", color="#e377c2", capsize=3, alpha=0.85)
    ax2.bar(x + width/2, ovf_rates, width, yerr=ovf_err, label="Overflow Frequency (%)", color="#bcbd22", capsize=3, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Frequency (%)")
    ax2.set_title("(b) Constraint Breach Frequencies")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig2_architecture_ablation.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Generated {fig_path}")


def plot_fig3_telemetry_ablation():
    """Figure 3: Telemetry Strategy Ablation under TKAN predictor."""
    df = pd.read_csv("outputs/metrics/ablation_results.csv")
    policies = [
        "AgenticTKAN_VoI", "AgenticTKAN_PeriodicP2", "AgenticTKAN_PeriodicP4",
        "AgenticTKAN_AgeT3", "AgenticTKAN_UncertaintyTh0.15", "AgenticTKAN_Always", "AgenticTKAN_Never"
    ]
    df_sub = df[df["policy"].isin(policies)].set_index("policy").reindex(policies)

    labels = ["VoI (Prop.)", "Periodic P=2", "Periodic P=4", "Age T=3", "Uncertainty", "Always (100%)", "Never (0%)"]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(8.5, 4.4))

    # Bar: Telemetry Requests
    tel_req = df_sub["total_telemetry_requests_mean"].to_numpy()
    color_req = "#3498db"
    ax1.bar(x - 0.2, tel_req, width=0.38, color=color_req, alpha=0.85, label="Telemetry Requests (Count)")
    ax1.set_ylabel("Mean Telemetry Requests per Episode", color=color_req)
    ax1.tick_params(axis="y", labelcolor=color_req)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")

    # Secondary Axis: SLA Violation Rate
    ax2 = ax1.twinx()
    sla_rate = df_sub["sla_violation_rate_mean"].to_numpy() * 100
    sla_err = df_sub["sla_violation_rate_ci95"].to_numpy() * 100
    color_sla = "#e74c3c"
    ax2.errorbar(x + 0.2, sla_rate, yerr=sla_err, fmt="o-", color=color_sla, lw=2, capsize=4, label="SLA Violation Rate (%)")
    ax2.set_ylabel("SLA Violation Rate (%)", color=color_sla)
    ax2.tick_params(axis="y", labelcolor=color_sla)

    plt.title("Telemetry Acquisition Ablation: Overhead vs. Downstream SLA Violations")
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig3_telemetry_ablation.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Generated {fig_path}")


def plot_fig4_sensitivity_sweeps():
    """Figure 4: 4-Panel Sensitivity Sweeps."""
    df_sens = pd.read_csv("outputs/metrics/sensitivity_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    policies = ["AgenticTKAN_VoI", "AgenticGRU_VoI", "DecisionSensitive", "PrivilegedFullInfo_Oracle"]
    colors = {"AgenticTKAN_VoI": "#1f77b4", "AgenticGRU_VoI": "#ff7f0e", "DecisionSensitive": "#9467bd", "PrivilegedFullInfo_Oracle": "#8c564b"}
    markers = {"AgenticTKAN_VoI": "s-", "AgenticGRU_VoI": "^-", "DecisionSensitive": "d-", "PrivilegedFullInfo_Oracle": "o--"}

    # Subplot 1: Telemetry Delay
    ax = axes[0, 0]
    sub1 = df_sens[df_sens["parameter"] == "telemetry_delay"]
    for pol in policies:
        data = sub1[sub1["policy"] == pol].groupby("parameter_value")["total_reward"].mean()
        ax.plot(data.index, data.values, markers[pol], color=colors[pol], label=pol if pol == "AgenticTKAN_VoI" else pol.split("_")[0])
    ax.set_xlabel("Telemetry Delivery Delay (steps)")
    ax.set_ylabel("Total Mean Reward")
    ax.set_title("(a) Sensitivity to Telemetry Delay")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Report Loss Rate
    ax = axes[0, 1]
    sub2 = df_sens[df_sens["parameter"] == "telemetry_loss_rate"]
    for pol in policies:
        data = sub2[sub2["policy"] == pol].groupby("parameter_value")["total_reward"].mean()
        ax.plot(data.index, data.values, markers[pol], color=colors[pol])
    ax.set_xlabel("Report Loss Probability ($p_{\\mathrm{loss}}$)")
    ax.set_ylabel("Total Mean Reward")
    ax.set_title("(b) Sensitivity to Telemetry Packet Loss")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Traffic Burst Multiplier
    ax = axes[1, 0]
    sub3 = df_sens[df_sens["parameter"] == "burst_multiplier"]
    for pol in policies:
        data = sub3[sub3["policy"] == pol].groupby("parameter_value")["total_reward"].mean()
        ax.plot(data.index, data.values, markers[pol], color=colors[pol])
    ax.set_xlabel("Traffic Burst Multiplier (Arrivals)")
    ax.set_ylabel("Total Mean Reward")
    ax.set_title("(c) Robustness to Traffic Bursts")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: VoI Threshold Sensitivity
    ax = axes[1, 1]
    sub4 = df_sens[df_sens["parameter"] == "gamma_voi"]
    gamma_rew = sub4.groupby("parameter_value")["total_reward"].mean()
    gamma_tel = sub4.groupby("parameter_value")["total_telemetry_requests"].mean()
    ax.plot(gamma_rew.index, gamma_rew.values, "o-", color="#1f77b4", label="Total Reward")
    ax.set_xlabel("VoI Decision Threshold ($\\gamma_{\\mathrm{VoI}}$)")
    ax.set_ylabel("Total Mean Reward", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(gamma_tel.index, gamma_tel.values, "s--", color="#e74c3c", label="Telemetry Requests")
    ax2.set_ylabel("Telemetry Requests", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax.set_title("(d) VoI Threshold Tuning Trade-off")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Global legend
    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig4_sensitivity_sweeps.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Generated {fig_path}")


def plot_fig5_recovery_outage():
    """Figure 5: Time-series recovery during reporting outage (Steps 30-45)."""
    df_trace = pd.read_csv("outputs/metrics/outage_recovery_trace.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True)

    policies = ["AgenticTKAN_VoI", "DecisionSensitive", "AgenticTKAN_PeriodicP2", "AgenticTKAN_Never"]
    colors = {
        "AgenticTKAN_VoI": "#1f77b4",
        "DecisionSensitive": "#9467bd",
        "AgenticTKAN_PeriodicP2": "#ff7f0e",
        "AgenticTKAN_Never": "#7f7f7f",
    }

    # Top: Slice 0 Queue Backlog (eMBB)
    for pol in policies:
        sub = df_trace[df_trace["policy"] == pol]
        ax1.plot(sub["step"], sub["true_queue_slice_0"] / 1000.0, lw=1.8, color=colors[pol], label=pol)

    # Highlight outage window
    ax1.axvspan(30, 45, color="#e74c3c", alpha=0.15, label="Outage Window (100% Loss)")
    ax1.set_ylabel("Slice 0 Backlog (KB)")
    ax1.set_title("Dynamic Queue Recovery under Reporting Outage (Steps 30-45)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bottom: Telemetry Requests
    for pol in policies:
        sub = df_trace[df_trace["policy"] == pol]
        ax2.step(sub["step"], sub["telemetry_requested"], where="mid", lw=1.5, color=colors[pol], alpha=0.8)

    ax2.axvspan(30, 45, color="#e74c3c", alpha=0.15)
    ax2.set_xlabel("Simulation Step (0.25s per step)")
    ax2.set_ylabel("Telemetry Requests (Slices)")
    ax2.set_ylim(-0.2, 3.2)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig5_recovery_outage.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Generated {fig_path}")


def main():
    print("Generating publication-quality figures from experimental outputs...")
    plot_fig1_pareto()
    plot_fig2_architecture_ablation()
    plot_fig3_telemetry_ablation()
    plot_fig4_sensitivity_sweeps()
    plot_fig5_recovery_outage()
    print("All figures successfully created in outputs/figures/")


if __name__ == "__main__":
    main()
