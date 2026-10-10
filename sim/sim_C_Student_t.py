# Simulation C: mean vs median under Student-t noise (symmetric, heavy tails)
# Model is DC under noise: x[n] = A + w[n], w[n] ~ Student-t with nu degrees of freedom (scale 1)
# The noise is symmetric about zero, so mean and median are both centred on A and only differ in variance.
# So we use the same efficiency ratio Var/CRLB as in Simulation A.
#
# Predictions (all per sample, scale 1):
#.....Fisher information of one sample: i(nu) = (nu + 1) / (nu + 3), so CRLB = 1 / (N i)
#.....noise variance = nu / (nu - 2), needs nu > 2, so Var(mean) = nu / ((nu - 2) N)
#.....density at the centre f0 = Gamma((nu+1)/2) / (sqrt(nu pi) Gamma(nu/2)), so Var(median) ~ 1 / (4 N f0^2)
#.....hence
#.........mean ratio   = nu / (nu - 2) * i(nu)
#.........median ratio = i(nu) / (4 f0^2)
#.....nu = 1 (Cauchy) and nu = 2 have infinite variance, so the mean is not 1/sqrt(N) accurate any more,
#.....there we compare the spread (interquartile range) of the estimates instead of Var/CRLB.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import brentq
from scipy.stats import norm

os.makedirs("figures", exist_ok=True)
np.random.seed(0)

A = 1.0
trials = 5000
nus = [3, 5, 10]
N_values = [10, 20, 50, 100, 200, 500, 1000]

def fisher_info(nu):
    return (nu + 1) / (nu + 3)

def noise_var(nu):
    return nu / (nu - 2)

def f_centre(nu):
    return np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(nu * np.pi)

def pred_mean(nu):
    return noise_var(nu) * fisher_info(nu)

def pred_median(nu):
    return fisher_info(nu) / (4 * f_centre(nu) ** 2)

def efficiency_ratios(nu, N):
    x = A + np.random.standard_t(nu, size=(trials, N))
    crlb = 1 / (N * fisher_info(nu))
    return np.var(np.mean(x, axis=1)) / crlb, np.var(np.median(x, axis=1)) / crlb

# print table @ N=100
N = 100
rows = []
for nu in nus:
    r_mean, r_median = efficiency_ratios(nu, N)
    rows.append(["Student-t, nu = %g" % nu, "Mean (MSE)", pred_mean(nu), r_mean])
    rows.append(["Student-t, nu = %g" % nu, "Median (MAE)", pred_median(nu), r_median])
table = pd.DataFrame(rows, columns=["Noise", "Estimator", "Predicted", "Measured"])
print("Efficiency ratio Var/CRLB at N =", N, "with", trials, "trials")
print(table.round(3).to_string(index=False))
print()

# nu at which mean and median are equally good (asymptotically)
nu_star = brentq(lambda v: pred_mean(v) - pred_median(v), 3, 5)
print("Predicted ratios are equal at nu = %.3f: below it the median is better, above it the mean is better" % nu_star)
print()

# how ratios change with N
sweep = {}
for nu in nus:
    mean_list = []
    median_list = []
    for N in N_values:
        r_mean, r_median = efficiency_ratios(nu, N)
        mean_list.append(r_mean)
        median_list.append(r_median)
    sweep[nu] = (mean_list, median_list)
    df = pd.DataFrame({"N": N_values, "Mean": mean_list, "Median": median_list})
    print("Efficiency ratio Var/CRLB as N grows, nu = %g (predicted: mean %.3f, median %.3f)"
          % (nu, pred_mean(nu), pred_median(nu)))
    print(df.round(3).to_string(index=False))
    print()

plt.figure(figsize=(12, 4))
for j, nu in enumerate(nus):
    plt.subplot(1, 3, j + 1)
    mean_list, median_list = sweep[nu]
    plt.plot(N_values, mean_list, "o-", color="tab:blue", label="Mean (MSE)")
    plt.plot(N_values, median_list, "s-", color="tab:orange", label="Median (MAE)")
    plt.axhline(pred_mean(nu), color="tab:blue", linewidth=0.9, linestyle=":", label="Predicted (mean)")
    plt.axhline(pred_median(nu), color="tab:orange", linewidth=0.9, linestyle=":", label="Predicted (median)")
    # predicted values written at the right end of each line
    plt.text(1300, pred_mean(nu), "%.2f" % pred_mean(nu), va="top", color="tab:blue")
    plt.text(1300, pred_median(nu), "%.2f" % pred_median(nu), va="bottom", color="tab:orange")
    plt.xscale("log")
    plt.xlim(8, 2000)
    plt.ylim(0.8, 3.2)
    plt.xlabel("Number of samples N")
    if j == 0:
        plt.ylabel("Var/CRLB")
    plt.title(r"Student-t noise, $\nu$ = %g" % nu)
    plt.legend(loc="upper center", ncol=2, fontsize=7.5)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_ratio_vs_N.png", dpi=200)
print("Saved figures/simC_ratio_vs_N.png")

# predicted ratios against nu, with measured values at N = 100
nu_grid = np.logspace(np.log10(2.05), np.log10(50), 400)
nu_meas = [3, 4, 5, 7, 10, 20, 50]
meas_mean = []
meas_median = []
for nu in nu_meas:
    r_mean, r_median = efficiency_ratios(nu, 100)
    meas_mean.append(r_mean)
    meas_median.append(r_median)

plt.figure(figsize=(7, 5))
plt.plot(nu_grid, pred_mean(nu_grid), color="tab:blue", label="Mean, predicted")
plt.plot(nu_grid, pred_median(nu_grid), color="tab:orange", label="Median, predicted")
plt.plot(nu_meas, meas_mean, "o", color="tab:blue", label="Mean, measured (N = 100)")
plt.plot(nu_meas, meas_median, "s", color="tab:orange", label="Median, measured (N = 100)")
plt.axhline(1, color="gray", linewidth=0.8)
plt.axhline(np.pi / 2, color="gray", linewidth=0.8)
plt.text(12, 1, "1 (Gaussian mean)", va="bottom", color="gray")
plt.text(12, np.pi / 2, r"$\pi/2$ (Gaussian median)", va="bottom", color="gray")
plt.axvline(nu_star, color="black", linestyle="--", linewidth=0.8)
plt.text(nu_star * 1.04, 3.6, r"$\nu^*$ = %.2f" % nu_star)
plt.xscale("log")
plt.ylim(0.8, 4)
plt.xlabel(r"Degrees of freedom $\nu$")
plt.ylabel("Var/CRLB")
plt.title("Efficiency ratio against the tail heaviness")
plt.legend(loc="upper right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_ratio_vs_nu.png", dpi=200)
print("Saved figures/simC_ratio_vs_nu.png")
print()

# infinite variance: nu = 1 (Cauchy) and nu = 2, compare interquartile range (IQR) of the estimates
# median: asymptotically normal, so IQR ~ 2 * z_0.75 * 1 / (2 f0 sqrt(N)).
# mean for nu = 1: the mean of N Cauchy samples is again standard Cauchy, so its IQR stays at 2 for every N.
nus_inf = [1, 2]
iqr = {}
for nu in nus_inf:
    iqr_mean = []
    iqr_median = []
    for N in N_values:
        x = A + np.random.standard_t(nu, size=(trials, N))
        m = np.mean(x, axis=1)
        md = np.median(x, axis=1)
        iqr_mean.append(np.subtract(*np.percentile(m, [75, 25])))
        iqr_median.append(np.subtract(*np.percentile(md, [75, 25])))
    iqr[nu] = (iqr_mean, iqr_median)
    pred_iqr_median = norm.ppf(0.75) / (f_centre(nu) * np.sqrt(np.array(N_values)))
    df = pd.DataFrame({"N": N_values, "Mean IQR": iqr_mean, "Median IQR": iqr_median,
                       "Median IQR (pred)": pred_iqr_median})
    print("Interquartile range of the estimates, nu = %g (mean predicted: %s)"
          % (nu, "2 for every N" if nu == 1 else "no closed form"))
    print(df.round(3).to_string(index=False))
    print()

plt.figure(figsize=(10, 4))
Nf = np.array(N_values)
for j, nu in enumerate(nus_inf):
    plt.subplot(1, 2, j + 1)
    iqr_mean, iqr_median = iqr[nu]
    plt.plot(N_values, iqr_mean, "o-", color="tab:blue", label="Mean (MSE)")
    plt.plot(N_values, iqr_median, "s-", color="tab:orange", label="Median (MAE)")
    plt.plot(Nf, norm.ppf(0.75) / (f_centre(nu) * np.sqrt(Nf)), color="tab:orange", linestyle=":",
             label="Median, predicted")
    if nu == 1:
        plt.axhline(2, color="tab:blue", linestyle=":", label="Mean, predicted (2)")
    plt.plot(Nf, iqr_mean[0] * np.sqrt(N_values[0] / Nf), color="gray", linestyle="--",
             linewidth=0.8, label=r"$\propto N^{-1/2}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of samples N")
    if j == 0:
        plt.ylabel("Interquartile range of the estimate")
    plt.title(r"Student-t noise, $\nu$ = %g (infinite variance)" % nu)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_infinite_variance.png", dpi=200)
print("Saved figures/simC_infinite_variance.png")