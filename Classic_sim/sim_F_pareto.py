# Author: Varun Shastry
# Simulation F: Pareto noise, where the CRB breaks down
# Model is DC under noise w[n] = P - alpha/(alpha-1), P ~ Pareto(alpha, 1), so w has zero mean (alpha > 1)
# the data x = A + w start at A - 1/(alpha-1), an edge that moves with A, and the density jumps to alpha there
# so the CRB regularity conditions fail. We compare
#.....mean:      Var = alpha / ((alpha-1)^2 (alpha-2) N), for alpha > 2
#.....median:    goes to A + 2^(1/alpha) - alpha/(alpha-1), Var ~ 2^(2/alpha) / (N alpha^2)
#.....min-based: A_hat = min(x) + alpha/(alpha-1) - N alpha/(N alpha - 1), Var = N alpha/((N alpha-1)^2 (N alpha-2)) ~ 1/(N alpha)^2
#.....naive CRB (invalid here): (alpha+2) / (N alpha (alpha+1)^2)
#.....Chapman-Robbins bound (valid, no regularity needed), computed numerically

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
np.random.seed(0)

A = 1.0 # true DC
trials = 5000 # num-montecarlo trials

def pareto_noise(alpha, size):
    # inverse CDF method: P = U^(-1/alpha) is Pareto(alpha, 1)
    P = (1 - np.random.random(size)) ** (-1 / alpha)
    return P - alpha / (alpha - 1)

def min_based(x, alpha):
    N = x.shape[1]
    return np.min(x, axis=1) + alpha / (alpha - 1) - N * alpha / (N * alpha - 1)

def naive_crb(alpha, N):
    # 1 / (N E[score^2]) with score (alpha+1)/P, as if the CRB applied
    return (alpha + 2) / (N * alpha * (alpha + 1)**2)

def chapman_robbins(alpha, N):
    # Var >= sup over h > 0 of h^2 / (E1(h)^N - 1),  E1(h) = integral of p(x; A+h)^2 / p(x; A)
    # for Pareto, with u = distance from the edge: E1(h) = int_{1+h}^inf alpha u^(alpha+1) / (u-h)^(2 alpha+2) du
    # E1 grows with h, so once E1^N is huge the bound only gets smaller and we can stop
    best = 0
    for h in np.logspace(-7, 0, 300):
        E1 = integrate.quad(lambda u: alpha * u**(alpha + 1) / (u - h)**(2 * alpha + 2), 1 + h, np.inf, limit=200)[0]
        exponent = N * np.log(E1)
        if exponent > 700:
            break
        bound = h**2 / np.expm1(exponent)
        best = max(best, bound)
    return best

# ---------------- Part 1: table at N = 100 ----------------
N = 100
rows = []
for alpha in [2.5, 3.0, 5.0]:
    x = A + pareto_noise(alpha, (trials, N))
    means = np.mean(x, axis=1)
    medians = np.median(x, axis=1)
    mins = min_based(x, alpha)
    rows.append({
        "alpha": alpha,
        "Var mean (pred)": alpha / ((alpha - 1)**2 * (alpha - 2) * N),
        "Var mean (meas)": np.var(means),
        "Var median (pred)": 2**(2 / alpha) / (N * alpha**2),
        "Var median (meas)": np.var(medians),
        "Var min-based (pred)": N * alpha / ((N * alpha - 1)**2 * (N * alpha - 2)),
        "Var min-based (meas)": np.var(mins),
        "naive CRB": naive_crb(alpha, N),
        "Chapman-Robbins": chapman_robbins(alpha, N),
    })
table = pd.DataFrame(rows)
pd.set_option("display.float_format", "{:.2e}".format)
print("Pareto noise, N =", N, "with", trials, "trials")
print(table.to_string(index=False))
print()
print("median limit (pred) at alpha = 3: %.3f" % (A + 2**(1 / 3) - 1.5))

# ---------------- Part 2: variance against N, alpha = 3 ----------------
alpha = 3.0
N_values = [10, 20, 50, 100, 200, 500, 1000]
var_mean, var_median, var_min, cr, ncrb = [], [], [], [], []
for N in N_values:
    x = A + pareto_noise(alpha, (trials, N))
    var_mean.append(np.var(np.mean(x, axis=1)))
    var_median.append(np.var(np.median(x, axis=1)))
    var_min.append(np.var(min_based(x, alpha)))
    cr.append(chapman_robbins(alpha, N))
    ncrb.append(naive_crb(alpha, N))

sweep = pd.DataFrame({"N": N_values, "Var mean": var_mean, "Var median": var_median,
                      "Var min-based": var_min, "naive CRB": ncrb, "Chapman-Robbins": cr})
print("alpha = 3, variance against N")
print(sweep.to_string(index=False))

# slopes on the log-log plot
slope_mean = np.polyfit(np.log(N_values), np.log(var_mean), 1)[0]
slope_min = np.polyfit(np.log(N_values), np.log(var_min), 1)[0]
slope_cr = np.polyfit(np.log(N_values), np.log(cr), 1)[0]
print("log-log slopes: mean %.2f, min-based %.2f, Chapman-Robbins %.2f" % (slope_mean, slope_min, slope_cr))

plt.figure(figsize=(7, 4.5))
plt.plot(N_values, var_mean, "o-", label="Mean (MSE), measured")
plt.plot(N_values, var_median, "s-", label="Median (MAE), measured")
plt.plot(N_values, var_min, "^-", label="Min-based, measured")
plt.plot(N_values, ncrb, "k--", label="Naive CRB (not valid here)")
plt.plot(N_values, cr, "r:", linewidth=2, label="Chapman-Robbins bound (valid)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of samples N")
plt.ylabel("Variance of the estimate")
plt.title("Pareto noise, alpha = 3")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simF_pareto_var_vs_N.png", dpi=200)
print("Saved figures/simF_pareto_var_vs_N.png")

# plot the noise density
p = np.linspace(1, 6, 1000)
plt.figure(figsize=(7, 4))
for a in [2.5, 3.0, 5.0]:
    edge = 1 - a / (a - 1) # left edge of the noise, the density jumps from 0 to alpha here
    w = np.concatenate(([edge - 0.3, edge], p - a / (a - 1)))
    pdf = np.concatenate(([0, 0], a / p**(a + 1)))
    plt.plot(w, pdf, label="alpha = %.1f" % a)
plt.xlabel("w")
plt.ylabel("p(w)")
plt.title("Pareto noise shifted to zero mean")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simF_pareto_density.png", dpi=200)
print("Saved figures/simF_pareto_density.png")
