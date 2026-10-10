# Authors: Varun Shastry and HARSH070105
# Simulation C: mean, median and t-MLE under Student-t noise
# Model is DC under noise w[n] ~ t(nu) with scale 1, which is symmetric with heavier tails than gaussian
# nu large is gaussian, nu = 1 is Cauchy
# hence the prediction is (each divided by the CRB, scale 1)
#.....Fisher information of one sample i = (nu+1)/(nu+3), so CRB = (nu+3)/((nu+1) N)
#.....mean/CRB   = nu/(nu-2) * i, for nu > 2
#.....median/CRB = i / (4 f0^2), f0 = gamma((nu+1)/2) / (sqrt(nu pi) gamma(nu/2)) is the density at the centre
#.....t-MLE/CRB  = 1, the t-MLE is a weighted mean with weights (nu+1)/(nu + (x-A)^2), Lange et al. 1989 eq. (3)
# for nu <= 2 the noise variance is infinite, so there we compare the interquartile range (IQR) of the estimates

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import brentq
from scipy.stats import norm
np.random.seed(0)

A = 1.0 # true DC
trials = 5000 # num-montecarlo trials

def fisher_info(nu):
    return (nu + 1) / (nu + 3)

def noise_var(nu):
    return nu / (nu - 2)

def f_centre(nu):
    return np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(nu * np.pi)

def pred_mean(nu):
    return noise_var(nu) * fisher_info(nu)

def pred_median(nu):
    return fisher_info(nu) / (4 * f_centre(nu)**2)

def t_mle(x, nu):
    # x has one row per trial, start from the median and repeat the weighted mean
    A_hat = np.median(x, axis=1)
    for it in range(50):
        r = x - A_hat[:, None]
        weights = (nu + 1) / (nu + r**2)
        A_hat = np.sum(weights * x, axis=1) / np.sum(weights, axis=1)
    return A_hat

def efficiency_ratios(nu, N, with_mle=False):
    x = A + np.random.standard_t(nu, size=(trials, N))
    crb = 1 / (N * fisher_info(nu))
    r_mean = np.var(np.mean(x, axis=1)) / crb
    r_median = np.var(np.median(x, axis=1)) / crb
    r_mle = np.var(t_mle(x, nu)) / crb if with_mle else np.nan
    return r_mean, r_median, r_mle

# ---------------- Part 1: the densities ----------------
w = np.linspace(-6, 6, 1000)
plt.figure(figsize=(7, 4))
for nu in [3, 5, 30]:
    pdf = f_centre(nu) * (1 + w**2 / nu) ** (-(nu + 1) / 2)
    plt.plot(w, pdf, label="t, nu = %d" % nu)
plt.plot(w, norm.pdf(w), color="gray", linestyle="--", label="Gaussian N(0, 1)")
plt.yscale("log")
plt.ylim(1e-5, 1)
plt.xlabel("w")
plt.ylabel("p(w), log scale")
plt.title("Student-t densities (scale 1)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_t_density.png", dpi=200)
print("Saved figures/simC_t_density.png")

# ---------------- Part 2: efficiency against nu, at N = 100 ----------------
N = 100
nu_meas = [3, 4, 5, 7, 10, 20, 50]
rows = []
for nu in nu_meas:
    r_mean, r_median, r_mle = efficiency_ratios(nu, N, with_mle=True)
    rows.append({"nu": nu,
                 "mean (pred)": pred_mean(nu), "mean (meas)": r_mean,
                 "median (pred)": pred_median(nu), "median (meas)": r_median,
                 "t-MLE (pred)": 1.0, "t-MLE (meas)": r_mle})
table = pd.DataFrame(rows)
print("Efficiency ratio Var/CRB under Student-t noise, N =", N, "with", trials, "trials")
print(table.round(3).to_string(index=False))

nu_star = brentq(lambda v: pred_mean(v) - pred_median(v), 3, 5)
print("mean and median equally efficient at nu = %.3f (below it the median wins, above it the mean wins)" % nu_star)
print()

nu_grid = np.logspace(np.log10(2.05), np.log10(60), 400)
plt.figure(figsize=(7, 5))
plt.plot(nu_grid, pred_mean(nu_grid), color="tab:blue", label="Mean (MSE), predicted")
plt.plot(nu_grid, pred_median(nu_grid), color="tab:orange", label="Median (MAE), predicted")
plt.plot(table["nu"], table["mean (meas)"], "o", color="tab:blue", label="Mean, measured")
plt.plot(table["nu"], table["median (meas)"], "s", color="tab:orange", label="Median, measured")
plt.plot(table["nu"], table["t-MLE (meas)"], "^", color="tab:green", label="t-MLE, measured (predicted 1)")
plt.axhline(1, color="gray", linewidth=0.8)
plt.axhline(np.pi / 2, color="gray", linewidth=0.8)
plt.text(2.15, 0.9, "1 = CRB (and the Gaussian mean)", color="gray")
plt.text(2.15, np.pi / 2 + 0.03, "pi/2 (Gaussian median)", color="gray")
plt.axvline(nu_star, color="black", linestyle="--", linewidth=0.8)
plt.text(nu_star * 1.04, 3.6, "nu* = %.2f" % nu_star)
plt.xscale("log")
plt.ylim(0.8, 4)
plt.xlabel("Degrees of freedom nu (log scale)")
plt.ylabel("Var/CRB")
plt.title("Student-t noise: efficiency against the tail heaviness, N = 100")
plt.legend(loc="upper right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_ratio_vs_nu.png", dpi=200)
print("Saved figures/simC_ratio_vs_nu.png")

# ---------------- Part 3: how the ratios change with N ----------------
N_values = [10, 20, 50, 100, 200, 500, 1000]
nus = [3, 5, 10]
plt.figure(figsize=(12, 4))
for j, nu in enumerate(nus):
    mean_list = []
    median_list = []
    for N in N_values:
        r_mean, r_median, _ = efficiency_ratios(nu, N)
        mean_list.append(r_mean)
        median_list.append(r_median)
    df = pd.DataFrame({"N": N_values, "Mean": mean_list, "Median": median_list})
    print("Var/CRB as N grows, nu = %g (predicted: mean %.3f, median %.3f)" % (nu, pred_mean(nu), pred_median(nu)))
    print(df.round(3).to_string(index=False))
    print()

    plt.subplot(1, 3, j + 1)
    plt.plot(N_values, mean_list, "o-", color="tab:blue", label="Mean (MSE)")
    plt.plot(N_values, median_list, "s-", color="tab:orange", label="Median (MAE)")
    plt.axhline(pred_mean(nu), color="tab:blue", linewidth=0.9, linestyle=":", label="Predicted (mean)")
    plt.axhline(pred_median(nu), color="tab:orange", linewidth=0.9, linestyle=":", label="Predicted (median)")
    plt.xscale("log")
    plt.ylim(0.8, 3.2)
    plt.xlabel("Number of samples N")
    if j == 0:
        plt.ylabel("Var/CRB")
    plt.title("Student-t noise, nu = %g" % nu)
    plt.legend(loc="upper center", ncol=2, fontsize=7.5)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_ratio_vs_N.png", dpi=200)
print("Saved figures/simC_ratio_vs_N.png")

# ---------------- Part 4: infinite variance, nu = 1 (Cauchy) and nu = 2 ----------------
# Var/CRB is meaningless here, so we use the interquartile range (IQR) of the estimates
# median: approximately gaussian with std 1/(2 f0 sqrt(N)), so IQR = 2 * 0.6745 * std = 0.6745 / (f0 sqrt(N))
# mean for nu = 1: the mean of N Cauchy samples is again standard Cauchy, whose IQR is 2, for every N
plt.figure(figsize=(10, 4))
for j, nu in enumerate([1, 2]):
    iqr_mean = []
    iqr_median = []
    for N in N_values:
        x = A + np.random.standard_t(nu, size=(trials, N))
        q75, q25 = np.percentile(np.mean(x, axis=1), [75, 25])
        iqr_mean.append(q75 - q25)
        q75, q25 = np.percentile(np.median(x, axis=1), [75, 25])
        iqr_median.append(q75 - q25)
    pred_iqr_median = norm.ppf(0.75) / (f_centre(nu) * np.sqrt(np.array(N_values)))
    df = pd.DataFrame({"N": N_values, "Mean IQR": iqr_mean, "Median IQR": iqr_median,
                       "Median IQR (pred)": pred_iqr_median})
    print("Interquartile range of the estimates, nu = %g (mean predicted: %s)"
          % (nu, "2 for every N" if nu == 1 else "shrinks, slower than 1/sqrt(N)"))
    print(df.round(3).to_string(index=False))
    print()

    plt.subplot(1, 2, j + 1)
    plt.plot(N_values, iqr_mean, "o-", color="tab:blue", label="Mean (MSE)")
    plt.plot(N_values, iqr_median, "s-", color="tab:orange", label="Median (MAE)")
    plt.plot(N_values, pred_iqr_median, color="tab:orange", linestyle=":", label="Median, predicted")
    if nu == 1:
        plt.axhline(2, color="tab:blue", linestyle=":", label="Mean, predicted (2)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of samples N")
    if j == 0:
        plt.ylabel("Interquartile range of the estimate")
    plt.title("Student-t noise, nu = %g (infinite variance)" % nu)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_infinite_variance.png", dpi=200)
print("Saved figures/simC_infinite_variance.png")

# ---------------- Part 5: one long run of Cauchy samples (nu = 1) ----------------
# mean and median of the first n samples, for n = 1 to 10000
np.random.seed(0) # fixed so the same long run is shown every time
n_max = 10000
x = A + np.random.standard_cauchy(n_max)
n_values = np.arange(1, n_max + 1)
running_mean = np.cumsum(x) / n_values
running_median = np.zeros(n_max)
for n in range(1, n_max + 1):
    running_median[n - 1] = np.median(x[:n])
print("largest sample in the run: %.0f at n = %d" % (x[np.argmax(np.abs(x))], np.argmax(np.abs(x)) + 1))
print("mean after %d samples: %.2f, median: %.3f" % (n_max, running_mean[-1], running_median[-1]))

plt.figure(figsize=(7, 4.5))
plt.plot(n_values, running_mean, label="Mean of first n samples (MSE)")
plt.plot(n_values, running_median, label="Median of first n samples (MAE)")
plt.axhline(A, color="black", linestyle="--", label="True A")
plt.xscale("log")
plt.ylim(-18, 7)
plt.xlabel("Number of samples n")
plt.ylabel("Estimate of A")
plt.title("Cauchy noise (nu = 1): one long run of samples")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simC_cauchy_running.png", dpi=200)
print("Saved figures/simC_cauchy_running.png")
