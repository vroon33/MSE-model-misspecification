# Author: Harsh
# Simulation NN 1: Deep Regression under Noise Model Misspecification
# Investigating MSE, MAE, Huber, Cauchy, Barron Adaptive Loss, and EML-FNN
# on synthetic nonparametric regression: Y = g*(X) + w
#
# Ground truth: g*(x) = sin(2*pi*x_1) + 2*x_2^2 + cos(pi*x_3)
# We test across 4 noise distributions:
#   1. Gaussian N(0, 1)       - MSE is the true MLE
#   2. Student-t (nu=3)       - Heavy-tailed, infinite 4th moment
#   3. Contaminated Gaussian  - 95% N(0, 1) + 5% N(0, 25) (Huber outlier model)
#   4. Shifted Lognormal      - Highly skewed, mean=0, median=-0.649
#
# Metrics computed on unseen clean test set (evaluating recovery of g*(x)):
#   - Root Mean Square Error vs g*(X): sqrt(E[(f_hat(X) - g*(X))^2])
#   - Mean Absolute Error vs g*(X): E[|f_hat(X) - g*(X)|]
#   - Barron's learned shape alpha (does it discover the true noise tail?)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure losses.py is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from losses import MSELoss, MAELoss, HuberLoss, CauchyLoss, BarronAdaptiveLoss, EMLLoss

os.makedirs("figures", exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# ----------------- Target Function & Data Generator -----------------
def ground_truth_g(X):
    # X is [N, 3] in [0, 1]^3
    return np.sin(2 * np.pi * X[:, 0]) + 2.0 * (X[:, 1]**2) + np.cos(np.pi * X[:, 2])

def sample_noise(noise_type, size):
    if noise_type == "Gaussian":
        return np.random.normal(0, 1.0, size)
    elif noise_type == "Student-t (nu=3)":
        return np.random.standard_t(3.0, size)
    elif noise_type == "Contaminated Gaussian":
        mask = (np.random.random(size) < 0.05)
        return np.where(mask, np.random.normal(0, 5.0, size), np.random.normal(0, 1.0, size))
    elif noise_type == "Shifted Lognormal":
        return np.exp(np.random.normal(0, 1.0, size)) - np.exp(0.5)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

# ----------------- Neural Network Architecture -----------------
class RegressionMLP(nn.Module):
    def __init__(self, in_features=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ----------------- Training Pipeline -----------------
def train_model(loss_name, X_train, y_train, epochs=300, lr=0.003):
    model = RegressionMLP(in_features=3, hidden_dim=64)
    barron_loss_obj = None
    
    if loss_name == "MSE":
        criterion = MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif loss_name == "MAE":
        criterion = MAELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif loss_name == "Huber":
        criterion = HuberLoss(delta=1.345)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif loss_name == "Cauchy":
        criterion = CauchyLoss(c=1.0)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif loss_name == "Barron Adaptive":
        criterion = BarronAdaptiveLoss(init_alpha=2.0, c=1.0)
        barron_loss_obj = criterion
        optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr)
    elif loss_name == "EML-FNN":
        criterion = EMLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_t)
        loss = criterion(y_pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    learned_alpha = barron_loss_obj.get_alpha().item() if barron_loss_obj is not None else np.nan
    return model, learned_alpha

# ----------------- Experiment Execution -----------------
n_train = 600
n_test = 2000
num_replications = 10 # Repeat to measure stability

X_test = np.random.uniform(0, 1, size=(n_test, 3))
g_test = ground_truth_g(X_test)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

noise_distributions = ["Gaussian", "Student-t (nu=3)", "Contaminated Gaussian", "Shifted Lognormal"]
loss_methods = ["MSE", "MAE", "Huber", "Cauchy", "Barron Adaptive", "EML-FNN"]

results_list = []
barron_alphas = {dist: [] for dist in noise_distributions}

print("Running Deep Regression Simulations across Noise Distributions...")
for dist in noise_distributions:
    print(f"\n--- Testing Noise Distribution: {dist} ---")
    for rep in range(num_replications):
        # Generate training data
        X_train = np.random.uniform(0, 1, size=(n_train, 3))
        g_train = ground_truth_g(X_train)
        w_train = sample_noise(dist, size=n_train)
        y_train = g_train + w_train
        
        for loss_m in loss_methods:
            model, alpha_val = train_model(loss_m, X_train, y_train, epochs=250, lr=0.003)
            if loss_m == "Barron Adaptive":
                barron_alphas[dist].append(alpha_val)
                
            model.eval()
            with torch.no_grad():
                pred_test = model(X_test_t).numpy()
                
            # Error vs ground-truth g*(x)
            rmse = np.sqrt(np.mean((pred_test - g_test)**2))
            mae = np.mean(np.abs(pred_test - g_test))
            
            results_list.append({
                "Noise": dist,
                "Method": loss_m,
                "Replication": rep,
                "RMSE_vs_Truth": rmse,
                "MAE_vs_Truth": mae
            })
            
            # Save checkpoint on final replication
            if rep == num_replications - 1:
                clean_dist = dist.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
                clean_method = loss_m.lower().replace(" ", "_").replace("-", "_")
                ckpt_p = f"saved_models/synth3d_{clean_dist}_{clean_method}.pt"
                torch.save({"model_state_dict": model.state_dict(), "noise": dist, "method": loss_m, "learned_alpha": alpha_val}, ckpt_p)

df_all = pd.DataFrame(results_list)

# Aggregate results table: Mean and Std of RMSE vs Truth
summary = df_all.groupby(["Noise", "Method"])["RMSE_vs_Truth"].agg(["mean", "std"]).reset_index()
summary.rename(columns={"mean": "RMSE (mean)", "std": "RMSE (std)"}, inplace=True)
pivot_rmse = summary.pivot(index="Method", columns="Noise", values="RMSE (mean)")

print("\n" + "=" * 80)
print("AVERAGE RMSE ON GROUND TRUTH g*(X) OVER 10 RUNS (LOWER IS BETTER):")
print("=" * 80)
print(pivot_rmse.round(3).to_string())
print("\nDetailed Summary with Variance (Stability):")
print(summary.round(3).to_string(index=False))

# Barron's learned alpha values across noise distributions
print("\n" + "=" * 80)
print("BARRON'S ADAPTIVE LOSS: LEARNED SHAPE PARAMETER alpha:")
print("=" * 80)
for dist, alphas in barron_alphas.items():
    print(f"Noise: {dist:<25} Mean alpha = {np.mean(alphas):.3f} (std = {np.std(alphas):.3f})")

# ----------------- Visualizations -----------------
# Plot 1: Comparison Bar Chart of RMSE under each noise
plt.figure(figsize=(12, 6))
noise_names = noise_distributions
x_idx = np.arange(len(loss_methods))
width = 0.2

for i, dist in enumerate(noise_names):
    sub = summary[summary["Noise"] == dist].set_index("Method").reindex(loss_methods)
    plt.bar(x_idx + (i - 1.5) * width, sub["RMSE (mean)"], width=width, yerr=sub["RMSE (std)"],
            capsize=3, label=dist)

plt.xticks(x_idx, loss_methods, fontsize=11, fontweight="bold")
plt.ylabel("RMSE vs True Function $g^*(X)$", fontsize=12)
plt.title("Deep Regression Accuracy: MSE vs Robust vs Adaptive vs EML-FNN", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("figures/simNN_loss_comparison.png", dpi=200)
print("\nSaved figures/simNN_loss_comparison.png")

# Plot 2: Barron's Learned alpha distribution
plt.figure(figsize=(8, 4.5))
dist_labels = [d.replace(" (nu=3)", "") for d in noise_distributions]
alpha_data = [barron_alphas[d] for d in noise_distributions]
plt.boxplot(alpha_data, labels=dist_labels, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2))
plt.axhline(2.0, color="gray", linestyle="--", label=r"Gaussian Target ($\alpha=2$)")
plt.axhline(1.0, color="orange", linestyle="--", label=r"Laplace Target ($\alpha=1$)")
plt.axhline(0.0, color="purple", linestyle="--", label=r"Cauchy Target ($\alpha=0$)")
plt.ylabel(r"Learned Shape Parameter $\alpha$", fontsize=12)
plt.title(r"Barron Adaptive Loss Automatically Recovers Noise Geometry", fontsize=13, fontweight="bold")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simNN_barron_alphas.png", dpi=200)
print("Saved figures/simNN_barron_alphas.png")

print("\nNeural Network synthetic simulations completed successfully!")
