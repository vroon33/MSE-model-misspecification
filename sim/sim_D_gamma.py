# Author: Varun Shastry
# Simulation D: mean, median and gamma-MLE under gamma noise
# Model is DC under noise w[n] = Y - k*theta, Y ~ Gamma(shape k, scale theta), so w has zero mean but is skewed
# k = 1 is the exponential noise of simulation B, large k is close to gaussian
# hence the prediction is
#.....mean -> A, with Var/CRB = k/(k-2) for k > 2 (CRB = theta^2 (k-2) / N)
#.....median -> A + median(w), median(w) = (gamma median) - k*theta, Var ~ 1/(4 N f(m)^2)
#.....gamma-MLE -> A, with Var/CRB -> 1
# for k <= 2 the CRB does not exist (the density has a hard edge at its left end)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq
np.random.seed(0)

A = 1.0 # true DC
theta = 1.0 # gamma scale
N = 100 # samples per trial
trials = 5000 # num-montecarlo trials

def gamma_mle(x, k):
    # MLE of A when the noise is Gamma(k, theta) - k*theta, for one trial x
    # the left edge of the data is c = A - k*theta, and the score equation is
    #     sum 1/(x - c) = N / (theta (k-1))
    # the left side increases with c, so there is exactly one root below min(x)
    target = len(x) / (theta * (k - 1))
    g = lambda c: np.sum(1 / (x - c)) - target
    x_min = np.min(x)
    c_hat = brentq(g, x_min - 100 * k * theta, x_min - 1e-12)
    return c_hat + k * theta

k_values = [1, 3, 5, 10, 30]
rows = []
for k in k_values:
    w = np.random.gamma(k, theta, size=(trials, N)) - k * theta
    x = A + w
    means = np.mean(x, axis=1)
    medians = np.median(x, axis=1)

    median_w = stats.gamma.ppf(0.5, k, scale=theta) - k * theta # median of the noise
    f_m = stats.gamma.pdf(median_w + k * theta, k, scale=theta) # density height at the median
    row = {
        "k": k,
        "skewness": 2 / np.sqrt(k),
        "median limit (pred)": A + median_w,
        "median avg (meas)": np.mean(medians),
        "median std (pred)": np.sqrt(1 / (4 * N * f_m**2)),
        "median std (meas)": np.std(medians),
    }
    if k > 2:
        crb = theta**2 * (k - 2) / N
        mles = np.array([gamma_mle(x[t], k) for t in range(trials)])
        row["mean/CRB (pred)"] = k / (k - 2)
        row["mean/CRB (meas)"] = np.var(means) / crb
        row["MLE/CRB (meas)"] = np.var(mles) / crb
    else:
        row["mean/CRB (pred)"] = np.nan # CRB does not exist
        row["mean/CRB (meas)"] = np.nan
        row["MLE/CRB (meas)"] = np.nan
    rows.append(row)

table = pd.DataFrame(rows)
print("Gamma noise, N =", N, "with", trials, "trials, true A =", A)
print(table[["k", "skewness", "mean/CRB (pred)", "mean/CRB (meas)", "MLE/CRB (meas)"]].round(3).to_string(index=False))
print()
print(table[["k", "median limit (pred)", "median avg (meas)", "median std (pred)", "median std (meas)"]].round(3).to_string(index=False))

# plot 1: the shape of the gamma noise, scaled to zero mean and unit variance so the shapes can be compared
# z = w / sqrt(k theta^2) has density sqrt(k) theta * gamma_pdf(sqrt(k) theta z + k theta)
z = np.linspace(-3, 5, 2000)
plt.figure(figsize=(7, 4))
for k in k_values:
    pdf = np.sqrt(k) * theta * stats.gamma.pdf(np.sqrt(k) * theta * z + k * theta, k, scale=theta)
    plt.plot(z, pdf, label="k = %d" % k)
plt.plot(z, stats.norm.pdf(z), color="gray", linestyle="--", label="Gaussian N(0, 1)")
plt.ylim(0, 1.05)
plt.xlabel("w / std(w)")
plt.ylabel("density")
plt.title("Gamma noise scaled to zero mean and unit variance")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simD_gamma_density.png", dpi=200)
print("Saved figures/simD_gamma_density.png")

# plot 2: mean ratio k/(k-2) against k, theory curve and simulated points
k_fine = np.linspace(2.2, 40, 500)
sim_k = table[table["k"] > 2]
plt.figure(figsize=(7, 4.5))
plt.plot(k_fine, k_fine / (k_fine - 2), color="tab:blue", label="Mean (MSE), theory k/(k-2)")
plt.plot(sim_k["k"], sim_k["mean/CRB (meas)"], "o", color="tab:blue", label="Mean, simulated")
plt.axhline(1, color="tab:green", label="gamma-MLE, theory = CRB")
plt.plot(sim_k["k"], sim_k["MLE/CRB (meas)"], "^", color="tab:green", label="gamma-MLE, simulated")
plt.xscale("log")
plt.ylim(0.8, 4)
plt.xlabel("Gamma shape k (log scale)")
plt.ylabel("Var/CRB")
plt.title("Gamma noise: efficiency of the mean and the gamma-MLE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simD_gamma_ratio_vs_k.png", dpi=200)
print("Saved figures/simD_gamma_ratio_vs_k.png")
