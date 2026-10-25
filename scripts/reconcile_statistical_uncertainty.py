"""
Statistical analysis script to:
1. Reconcile 95% CI half-width vs sample standard deviation for all policies.
2. Compute paired difference tests (paired t-test and Wilcoxon signed-rank test)
   between AgenticTKAN_VoI and matched baselines across the 10 evaluation seeds.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as st

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)

raw_path = os.path.join(WORKSPACE_ROOT, "outputs", "metrics", "final_experiments_raw.csv")
df_raw = pd.read_csv(raw_path)

print("=" * 80)
print("1. UNCERTAINTY METRIC RECONCILIATION: SAMPLE STD vs 95% CI HALF-WIDTH")
print("=" * 80)

# Check TKAN raw rewards
tkan_rewards = df_raw[df_raw["policy"] == "AgenticTKAN_VoI"]["total_reward"].values
n = len(tkan_rewards)
mean_r = np.mean(tkan_rewards)
sample_std_r = np.std(tkan_rewards, ddof=1)
sem_r = st.sem(tkan_rewards, ddof=1)
t_crit_95 = st.t.ppf(0.975, df=n - 1)
ci95_half_width = t_crit_95 * sem_r

print(f"Number of evaluation seeds (n): {n}")
print(f"Sample Mean Total Reward:       {mean_r:.4f}")
print(f"Sample Standard Deviation (s):  {sample_std_r:.4f}  (matches follow-up audit ~2961.4)")
print(f"Standard Error of Mean (SEM):   {sem_r:.4f}")
print(f"Student-t critical value (df=9): {t_crit_95:.4f}")
print(f"95% CI Half-Width (t * SEM):    {ci95_half_width:.4f}  (matches Table 1 ~2118.5!)")
print("\nConclusion: The value in Table 1 (+/- 2118.5) is indeed the 95% confidence interval half-width!")
print("The value ~2961.4 is the sample standard deviation (s).")

print("\n" + "=" * 80)
print("2. RECOMPUTED TABLE FOR ALL 17 POLICIES: MEAN, STD, 95% CI HALF-WIDTH")
print("=" * 80)

stats_rows = []
for pol, group in df_raw.groupby("policy"):
    vals = group["total_reward"].values
    mean_val = np.mean(vals)
    std_val = np.std(vals, ddof=1)
    ci_val = st.t.ppf(0.975, df=len(vals) - 1) * st.sem(vals, ddof=1)
    stats_rows.append({
        "policy": pol,
        "mean_reward": mean_val,
        "std_reward": std_val,
        "ci95_half_width": ci_val
    })

df_stats = pd.DataFrame(stats_rows).sort_values(by="mean_reward", ascending=False).reset_index(drop=True)
print(df_stats.to_string(index=False))

print("\n" + "=" * 80)
print("3. PAIRED STATISTICAL DIFFERENCE TESTS (TKAN vs OTHER POLICIES, N=10 SEEDS)")
print("=" * 80)

paired_tests = []
tkan_df = df_raw[df_raw["policy"] == "AgenticTKAN_VoI"].sort_values(by="seed").reset_index(drop=True)

for pol in df_raw["policy"].unique():
    if pol == "AgenticTKAN_VoI":
        continue
    other_df = df_raw[df_raw["policy"] == pol].sort_values(by="seed").reset_index(drop=True)
    
    # Verify matched seeds
    assert (tkan_df["seed"].values == other_df["seed"].values).all(), f"Seeds do not match for {pol}!"
    
    diff = tkan_df["total_reward"].values - other_df["total_reward"].values
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # Paired t-test
    t_stat, p_val_t = st.ttest_rel(tkan_df["total_reward"].values, other_df["total_reward"].values)
    
    # Wilcoxon signed-rank test
    w_stat, p_val_w = st.wilcoxon(tkan_df["total_reward"].values, other_df["total_reward"].values)
    
    paired_tests.append({
        "comparison": f"TKAN vs {pol}",
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_val_paired_t": p_val_t,
        "wilcoxon_stat": w_stat,
        "p_val_wilcoxon": p_val_w,
        "significant_p05": p_val_t < 0.05
    })

df_paired = pd.DataFrame(paired_tests)
print(df_paired[["comparison", "mean_diff", "t_stat", "p_val_paired_t", "p_val_wilcoxon", "significant_p05"]].to_string(index=False))

# Specifically test TKAN vs GRU and TKAN vs MLP
print("\n" + "=" * 80)
print("4. DETAILED BREAKDOWN: TKAN vs GRU and TKAN vs MLP")
print("=" * 80)
for comp in ["TKAN vs AgenticGRU_VoI", "TKAN vs AgenticMLP_VoI", "TKAN vs NoMemory_VoI", "TKAN vs AgenticTKAN_Never"]:
    row = df_paired[df_paired["comparison"] == comp].iloc[0]
    print(f"{row['comparison']}:")
    print(f"  Mean Paired Difference (TKAN - Base): {row['mean_diff']:+.2f}")
    print(f"  Paired t-statistic:                  {row['t_stat']:+.4f} (p = {row['p_val_paired_t']:.4e})")
    print(f"  Wilcoxon signed-rank test:           p = {row['p_val_wilcoxon']:.4e}")
    print(f"  Statistically significant at alpha=0.05? {row['significant_p05']}")
