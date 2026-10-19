# Author: Anshitha Penubolu
# Simulation L: gamma noise below k = 2, where the CRLB stops working
# Model is DC under noise w[n] = Y - k*theta, Y ~ Gamma(shape k, scale theta), as in simulation D
# The data have a left edge at c = A - k*theta. Near the edge the gamma density behaves like y^(k-1), so
#.....k > 2  : regular, CRLB = (k-2) theta^2 / N, best rate 1/N
#.....k <= 2 : Fisher information is infinite, the CRLB gives only 0 (Section 7 of the report)
# The edge itself is informative. The edge estimator  A_edge = min(x) + k*theta  uses only the smallest sample.
# Since P(Y < y) ~ (y/theta)^k / Gamma(k+1) for small y, the minimum of N samples is about
#.....min ~ theta * (Gamma(k+1)/N)^(1/k) * E^(1/k),  E ~ Exp(1)
# hence the prediction is
#.....MSE(edge) ~ theta^2 * Gamma(1 + 2/k) * Gamma(k+1)^(2/k) * N^(-2/k)     (exact for k = 1)
#.....MSE(mean) = k theta^2 / N                                               (exact, every k)
# so for k < 2 the edge estimator wins for large N (rate N^(-2/k) beats 1/N), and for k > 2 the mean wins

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gamma as G
np.random.seed(0)

A = 1.0 # true DC
theta = 1.0 # gamma scale
k_values = [0.5, 1.0, 1.5, 3.0]
N_values = [25, 50, 100, 200, 400, 800, 1600]
trials = 4000 # num-montecarlo trials

def gamma_mle(x, k):
    # same as simulation D, only defined for k > 1
    target = len(x) / (theta * (k - 1))
    g = lambda c: np.sum(1 / (x - c)) - target
    x_min = np.min(x)
    return brentq(g, x_min - 100 * k * theta, x_min - 1e-12) + k * theta

def edge_mse_theory(k, N):
    return theta**2 * G(1 + 2 / k) * G(k + 1)**(2 / k) * N**(-2 / k)

rows = []
for k in k_values:
    for N in N_values:
        x = A + np.random.gamma(k, theta, size=(trials, N)) - k * theta
        est = {
            "mean": np.mean(x, axis=1),
            "median": np.median(x, axis=1),
            "edge": np.min(x, axis=1) + k * theta,
        }
        if k > 1:
            est["MLE"] = np.array([gamma_mle(x[t], k) for t in range(trials)])
        row = {"k": k, "N": N,
               "mean MSE (pred)": k * theta**2 / N,
               "edge MSE (pred)": edge_mse_theory(k, N)}
        for name, e in est.items():
            row[name + " MSE"] = np.mean((e - A)**2) # MSE about the true A, so bias counts
        row["CRLB"] = (k - 2) * theta**2 / N if k > 2 else np.nan
        rows.append(row)

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Gamma noise, theta =", theta, "true A =", A, "with", trials, "trials")
print(table.to_string(index=False, float_format=lambda v: "%.3g" % v))

# measured log-log slope of MSE against N, using N >= 200
print("\nlog-log slope of MSE vs N (N >= 200). Predicted: mean -1, edge -2/k")
big = table[table["N"] >= 200]
for k in k_values:
    t = big[big["k"] == k]
    out = []
    for name in ["mean", "edge", "MLE"]:
        col = name + " MSE"
        if col in t and t[col].notna().all():
            out.append("%s %.2f" % (name, np.polyfit(np.log(t["N"]), np.log(t[col]), 1)[0]))
    print("k = %.1f: predicted edge %.2f | measured: %s" % (k, -2 / k, ", ".join(out)))

# N at which the edge estimator starts to beat the mean (from the predicted formulas, k < 2)
print("\nN beyond which the edge beats the mean (theory):")
for k in [kk for kk in k_values if kk < 2]:
    c = G(1 + 2 / k) * G(k + 1)**(2 / k)
    print("k = %.1f: N* = %.1f" % (k, (c / k)**(1 / (2 / k - 1))))

# k = 3: how fast does the gamma MLE approach the CRLB? (simulation D finds 1.22 at N = 100)
# close to the non-regular boundary k = 2 the MLE converges slowly, so we follow Var/CRB against N
print("\nk = 3, gamma MLE: Var/CRB against N")
np.random.seed(1)
for N, T in [(100, 10000), (300, 8000), (1000, 6000), (3000, 3000)]:
    x = A + np.random.gamma(3.0, theta, size=(T, N)) - 3.0 * theta
    mles = np.array([gamma_mle(x[t], 3.0) for t in range(T)])
    ratio = np.var(mles) / (theta**2 / N)
    print("N = %4d, %5d trials: Var/CRB = %.3f  (Monte Carlo error about %.3f)" % (N, T, ratio, ratio * np.sqrt(2 / T)))

# plot: MSE vs N for each k, theory lines and simulated markers
fig, axes = plt.subplots(1, len(k_values), figsize=(15, 4), sharey=False)
N_fine = np.logspace(np.log10(N_values[0]), np.log10(N_values[-1]), 100)
for ax, k in zip(axes, k_values):
    t = table[table["k"] == k]
    ax.loglog(N_fine, k * theta**2 / N_fine, color="tab:blue", label="mean (MSE), theory")
    ax.loglog(t["N"], t["mean MSE"], "o", color="tab:blue", label="mean, simulated")
    ax.loglog(N_fine, edge_mse_theory(k, N_fine), color="tab:red", label="edge, theory")
    ax.loglog(t["N"], t["edge MSE"], "s", color="tab:red", label="edge, simulated")
    if k > 1:
        ax.loglog(t["N"], t["MLE MSE"], "^", color="tab:green", label="gamma MLE, simulated")
    if k > 2:
        ax.loglog(N_fine, (k - 2) * theta**2 / N_fine, "k--", label="CRLB")
    ax.set_title("k = %g" % k)
    ax.set_xlabel("N")
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("MSE of the estimate of A")
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
fig.suptitle("Gamma noise below and above k = 2: the edge against the mean")
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig("figures/simL_gamma_edge.png", dpi=200)
print("Saved figures/simL_gamma_edge.png")
