# Author: Harsh
# Title: Noise-Model Misspecification in Neural Regression: Bias, Variance, and Adaptive Loss Selection
# Main Experiment Script: Benchmarking baselines, Bias-Variance decomposition, and Adaptive RCOS.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ground_truth import f_star, generate_dataset
from models import (
    RegressionMLP, train_mlp,
    mse_loss, mae_loss, huber_loss,
    student_t_nll, centered_gamma_nll
)
from adaptive_selector import ResidualCalibratedSelector
from metrics import bias_variance_decomposition, compute_ise

os.makedirs("figures", exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("NOISE-MODEL MISSPECIFICATION IN NEURAL REGRESSION")
print("Central Research Question: Can an adaptive method identify noise misspecification")
print("and adapt its objective without sacrificing Gaussian efficiency?")
print("=" * 80)

# Setup evaluation grid
n_test = 500
x_test = np.linspace(0.0, 1.0, n_test)[:, None].astype(np.float32)
y_test_clean = f_star(x_test).astype(np.float32)
x_test_t = torch.as_tensor(x_test, dtype=torch.float32)

sigma_noise = 0.30
M_trials = 25 # Number of independent Monte Carlo trials for Bias-Variance decomposition
sample_sizes = [150, 400] # Controlled dataset sizes
primary_N = 250

regimes = {
    "Gaussian": {
        "noise_type": "gaussian",
        "kwargs": {},
        "matched_name": "Gaussian MSE",
        "matched_loss": mse_loss
    },
    "Centered Gamma (k=3)": {
        "noise_type": "gamma",
        "kwargs": {"k": 3.0},
        "matched_name": "Centered Gamma NLL",
        "matched_loss": lambda p, t: centered_gamma_nll(p, t, k=3.0, theta=sigma_noise / np.sqrt(3.0))
    },
    "Student-t (nu=3)": {
        "noise_type": "student_t",
        "kwargs": {"nu": 3.0},
        "matched_name": "Student-t NLL",
        "matched_loss": lambda p, t: student_t_nll(p, t, nu=3.0, scale=sigma_noise)
    }
}

method_names = ["MSE (Assumed Normal)", "MAE (Assumed Laplace)", "Huber Loss", "Matched Likelihood", "Adaptive (RCOS)"]

# =========================================================================
# EXPERIMENT 1: Full Bias-Variance Decomposition across Noise Regimes
# =========================================================================
print(f"\n[EXPERIMENT 1] Running {M_trials} Monte Carlo trials at N = {primary_N} samples...")

bv_results = []
adaptive_selections = {regime_name: [] for regime_name in regimes}
sample_preds_for_plot = {regime_name: {} for regime_name in regimes}

for regime_name, r_config in regimes.items():
    print(f"\n>>> Evaluating Regime: {regime_name} <<<")
    preds_by_method = {m: np.zeros((M_trials, n_test)) for m in method_names}
    
    for trial in range(M_trials):
        seed = 1000 + trial * 7
        X_tr, y_tr, _ = generate_dataset(
            primary_N, r_config["noise_type"], sigma=sigma_noise, random_state=seed, **r_config["kwargs"]
        )
        
        # 1. MSE
        m_mse = RegressionMLP()
        m_mse = train_mlp(m_mse, X_tr, y_tr, mse_loss, epochs=250)
        with torch.no_grad():
            preds_by_method["MSE (Assumed Normal)"][trial] = m_mse(x_test_t).numpy().flatten()
            
        # 2. MAE
        m_mae = RegressionMLP()
        m_mae = train_mlp(m_mae, X_tr, y_tr, mae_loss, epochs=250)
        with torch.no_grad():
            preds_by_method["MAE (Assumed Laplace)"][trial] = m_mae(x_test_t).numpy().flatten()
            
        # 3. Huber
        m_hub = RegressionMLP()
        m_hub = train_mlp(m_hub, X_tr, y_tr, lambda p, t: huber_loss(p, t, delta=1.345 * sigma_noise), epochs=250)
        with torch.no_grad():
            preds_by_method["Huber Loss"][trial] = m_hub(x_test_t).numpy().flatten()
            
        # 4. Matched Likelihood
        m_mat = RegressionMLP()
        m_mat = train_mlp(m_mat, X_tr, y_tr, r_config["matched_loss"], epochs=250)
        with torch.no_grad():
            preds_by_method["Matched Likelihood"][trial] = m_mat(x_test_t).numpy().flatten()
            
        # 5. Adaptive Method (RCOS)
        rcos = ResidualCalibratedSelector(cal_ratio=0.25, pilot_epochs=150, final_epochs=250)
        m_adap, chosen_family, diag, _ = rcos.fit_and_select(X_tr, y_tr)
        with torch.no_grad():
            preds_by_method["Adaptive (RCOS)"][trial] = m_adap(x_test_t).numpy().flatten()
        adaptive_selections[regime_name].append(chosen_family)
        
        # Save trial 0 predictions for visualization
        if trial == 0:
            for m in method_names:
                sample_preds_for_plot[regime_name][m] = preds_by_method[m][0].copy()
                
    # Compute Bias-Variance Decomposition for this regime
    matched_risk = None
    regime_rows = []
    for m in method_names:
        decomp = bias_variance_decomposition(preds_by_method[m], y_test_clean)
        if m == "Matched Likelihood":
            matched_risk = decomp["total_risk"]
        regime_rows.append({
            "Regime": regime_name,
            "Method": m,
            "Bias^2": decomp["bias_squared"],
            "Variance": decomp["variance"],
            "Total Risk (ISE)": decomp["total_risk"],
        })
        
    for r in regime_rows:
        r["Risk Ratio vs Matched"] = r["Total Risk (ISE)"] / matched_risk
        bv_results.append(r)

df_bv = pd.DataFrame(bv_results)

print("\n" + "=" * 90)
print("BIAS-VARIANCE DECOMPOSITION & RISK RATIO TABLE (N = 250, M = 25 trials):")
print("=" * 90)
fmt_cols = ["Regime", "Method", "Bias^2", "Variance", "Total Risk (ISE)", "Risk Ratio vs Matched"]
print(df_bv[fmt_cols].round(4).to_string(index=False))

print("\n" + "=" * 90)
print("ADAPTIVE SELECTOR (RCOS) DIAGNOSTIC SELECTION ACCURACY:")
print("=" * 90)
for reg, sels in adaptive_selections.items():
    unique, counts = np.unique(sels, return_counts=True)
    distr_str = ", ".join([f"{u}: {c}/{len(sels)} ({c/len(sels)*100:.1f}%)" for u, c in zip(unique, counts)])
    print(f"True Noise: {reg:<22} -> Selections: {distr_str}")

# =========================================================================
# EXPERIMENT 2: Sample Size Scaling (N in [100, 250, 600])
# =========================================================================
print("\n[EXPERIMENT 2] Evaluating Sample Size Scaling under Student-t (nu=3) and Gaussian...")
n_sweep = [100, 250, 600]
scaling_results = []

for n_val in n_sweep:
    for reg_target in ["Gaussian", "Student-t (nu=3)"]:
        r_cfg = regimes[reg_target]
        for tr in range(12):
            X_tr, y_tr, _ = generate_dataset(
                n_val, r_cfg["noise_type"], sigma=sigma_noise, random_state=500 + tr * 13, **r_cfg["kwargs"]
            )
            # MSE
            m_mse = train_mlp(RegressionMLP(), X_tr, y_tr, mse_loss, epochs=250)
            # Matched
            m_mat = train_mlp(RegressionMLP(), X_tr, y_tr, r_cfg["matched_loss"], epochs=250)
            # Adaptive
            rcos = ResidualCalibratedSelector(cal_ratio=0.25, pilot_epochs=120, final_epochs=250)
            m_ad, _, _, _ = rcos.fit_and_select(X_tr, y_tr)
            
            with torch.no_grad():
                ise_mse = compute_ise(m_mse(x_test_t).numpy().flatten(), y_test_clean)
                ise_mat = compute_ise(m_mat(x_test_t).numpy().flatten(), y_test_clean)
                ise_ad  = compute_ise(m_ad(x_test_t).numpy().flatten(), y_test_clean)
                
            scaling_results.append({
                "N": n_val, "Regime": reg_target, "Trial": tr,
                "MSE": ise_mse, "Matched": ise_mat, "Adaptive": ise_ad
            })

df_scale = pd.DataFrame(scaling_results)
scale_summary = df_scale.groupby(["Regime", "N"])[["MSE", "Matched", "Adaptive"]].mean().reset_index()
print("\n" + "=" * 90)
print("SAMPLE SIZE SCALING SUMMARY (AVERAGE ISE):")
print("=" * 90)
print(scale_summary.round(4).to_string(index=False))

# =========================================================================
# VISUALIZATIONS
# =========================================================================

# Plot 1: Qualitative Function Reconstruction Fits
plt.figure(figsize=(14, 4.5))
for i, reg in enumerate(regimes.keys()):
    plt.subplot(1, 3, i + 1)
    plt.plot(x_test, y_test_clean, "k-", linewidth=2.2, label=r"True $f^*(x)$")
    plt.plot(x_test, sample_preds_for_plot[reg]["MSE (Assumed Normal)"], "r--", label="MSE (Assumed Normal)")
    plt.plot(x_test, sample_preds_for_plot[reg]["Matched Likelihood"], "g-", label="Matched Likelihood")
    plt.plot(x_test, sample_preds_for_plot[reg]["Adaptive (RCOS)"], "b-.", label="Adaptive (RCOS)")
    plt.title(f"Regime: {reg}", fontsize=11, fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(alpha=0.3)
    if i == 0:
        plt.legend(fontsize=8, loc="lower center")

plt.tight_layout()
plt.savefig("figures/simNN_function_reconstruction.png", dpi=200)
print("\nSaved figures/simNN_function_reconstruction.png")

# Plot 2: Bias-Variance Decomposition Breakdown Bar Chart
plt.figure(figsize=(12, 5))
methods_plot = ["MSE (Assumed Normal)", "MAE (Assumed Laplace)", "Huber Loss", "Matched Likelihood", "Adaptive (RCOS)"]
x_indices = np.arange(len(methods_plot))
width = 0.25

for i, reg in enumerate(regimes.keys()):
    sub = df_bv[df_bv["Regime"] == reg].set_index("Method").reindex(methods_plot)
    b_sq = sub["Bias^2"].values
    var = sub["Variance"].values
    
    offset = (i - 1) * width
    plt.bar(x_indices + offset, b_sq, width=width, color=f"C{i}", hatch="//", alpha=0.9, label=f"{reg} (Bias²)" if i==0 else None)
    plt.bar(x_indices + offset, var, width=width, bottom=b_sq, color=f"C{i}", alpha=0.5, label=f"{reg} (Total Risk)")

plt.xticks(x_indices, ["MSE", "MAE", "Huber", "Matched", "Adaptive (RCOS)"], fontsize=11, fontweight="bold")
plt.ylabel("Total Risk (Integrated Squared Error)", fontsize=11)
plt.title("Bias-Variance Decomposition Across Noise Regimes", fontsize=13, fontweight="bold")
plt.legend(loc="upper right")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("figures/simNN_bias_variance_bars.png", dpi=200)
print("Saved figures/simNN_bias_variance_bars.png")

# Plot 3: Risk Ratio Bar Chart (The Decisive Hypothesis Test)
plt.figure(figsize=(9, 4.5))
pivot_rr = df_bv.pivot(index="Method", columns="Regime", values="Risk Ratio vs Matched").reindex(methods_plot)
ax = pivot_rr.plot(kind="bar", figsize=(10, 4.8), rot=15, colormap="viridis")
plt.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Matched Baseline (Ratio = 1.0)")
plt.ylabel("Risk Ratio vs Matched Objective", fontsize=11)
plt.title("Decisive Hypothesis: Does Adaptive RCOS Match the Optimal Loss Everywhere?", fontsize=12, fontweight="bold")
plt.legend(fontsize=9)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("figures/simNN_risk_ratios.png", dpi=200)
print("Saved figures/simNN_risk_ratios.png")

# Plot 4: Sample Size Scaling
plt.figure(figsize=(9, 4.5))
for reg in ["Gaussian", "Student-t (nu=3)"]:
    sub = scale_summary[scale_summary["Regime"] == reg]
    ls = "-" if reg == "Gaussian" else "--"
    plt.plot(sub["N"], sub["MSE"], f"ro{ls}", label=f"MSE ({reg})")
    plt.plot(sub["N"], sub["Matched"], f"go{ls}", label=f"Matched ({reg})")
    plt.plot(sub["N"], sub["Adaptive"], f"bo{ls}", label=f"Adaptive ({reg})")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Training Sample Size N (log scale)", fontsize=11)
plt.ylabel("Integrated Squared Error (log scale)", fontsize=11)
plt.title("Sample Efficiency Scaling: MSE vs Matched vs Adaptive", fontsize=12, fontweight="bold")
plt.legend(loc="upper right", ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simNN_sample_scaling.png", dpi=200)
print("Saved figures/simNN_sample_scaling.png")

print("\nAll experiments and visualizations completed successfully!")

