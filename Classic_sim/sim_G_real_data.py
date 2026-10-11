# Author: Varun Shastry
# Simulation G: noise matching on real data, French motor insurance claim amounts (freMTPL2sev, OpenML 41215)
# Steps
#.....fit candidate distributions by maximum likelihood: gaussian, exponential, gamma, lognormal, weibull
#.....compare them with AIC and the KS statistic, and look at histograms and QQ plots
#.....read off the gamma shape k, which says how skewed the data is (k = 1 is exponential, large k is gaussian-like)
#.....estimate the typical claim with the mean and the median, and compare three standard errors for the mean:
#.........the one implied by the fitted gamma model, the sandwich (sample MCRB, Fortunati et al. Sec 4.1), and the bootstrap

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import fetch_openml
warnings.filterwarnings("ignore") # openml prints version warnings
np.random.seed(0)

# load the data (downloads once, then cached by sklearn)
data = fetch_openml(data_id=41215, as_frame=True).frame
x = data["ClaimAmount"].to_numpy()
N = len(x)
print("Number of claims:", N)
print("mean %.1f   median %.1f   std %.1f   skewness %.1f" % (np.mean(x), np.median(x), np.std(x), stats.skew(x)))

# many claims sit on a few fixed amounts
values, counts = np.unique(x, return_counts=True)
top = np.argsort(counts)[::-1][:5]
print("most common amounts:", [(values[i], counts[i]) for i in top])
print("fraction of claims on these 5 amounts: %.2f" % (np.sum(counts[top]) / N))
print()

# ---------------- Part 1: fit candidate distributions ----------------
# for the positive families we fix the location at 0 (floc=0), so they have 1 or 2 free parameters
families = {
    "Gaussian":    (stats.norm, None),
    "Exponential": (stats.expon, 0),
    "Gamma":       (stats.gamma, 0),
    "Lognormal":   (stats.lognorm, 0),
    "Weibull":     (stats.weibull_min, 0),
}
rows = []
fits = {}
for name, (dist, floc) in families.items():
    if floc is None:
        params = dist.fit(x)
        n_free = len(params)
    else:
        params = dist.fit(x, floc=floc)
        n_free = len(params) - 1 # the location is fixed, not fitted
    fits[name] = (dist, params)
    log_lik = np.sum(dist.logpdf(x, *params))
    rows.append({
        "family": name,
        "free params": n_free,
        "log-likelihood": log_lik,
        "AIC": 2 * n_free - 2 * log_lik,
        "KS statistic": stats.kstest(x, dist.cdf, args=params).statistic,
    })
table = pd.DataFrame(rows).sort_values("AIC")
print("Distribution fits (lower AIC is better, smaller KS is better)")
print(table.round(3).to_string(index=False))
print()

# the gamma shape
k, _, theta = fits["Gamma"][1]
print("Gamma fit: shape k = %.3f, scale theta = %.1f, skewness 2/sqrt(k) = %.2f" % (k, theta, 2 / np.sqrt(k)))
print("Gamma-implied std = sqrt(k)*theta = %.1f   vs   sample std = %.1f" % (np.sqrt(k) * theta, np.std(x)))
print()

# ---------------- Part 2: mean vs median, and their standard errors ----------------
B = 2000 # bootstrap resamples
boot_means = np.zeros(B)
boot_medians = np.zeros(B)
for b in range(B):
    sample = np.random.choice(x, size=N, replace=True)
    boot_means[b] = np.mean(sample)
    boot_medians[b] = np.median(sample)

se_mean_gamma = np.sqrt(k) * theta / np.sqrt(N) # if the gamma model were true
se_mean_sandwich = np.std(x) / np.sqrt(N) # sample MCRB of the mean = sample variance / N
se_mean_boot = np.std(boot_means)
se_median_boot = np.std(boot_medians)

se_table = pd.DataFrame({
    "estimate": ["mean (MSE)", "median (MAE)"],
    "value": [np.mean(x), np.median(x)],
    "SE from gamma model": [se_mean_gamma, np.nan],
    "SE sandwich (sample MCRB)": [se_mean_sandwich, np.nan],
    "SE bootstrap": [se_mean_boot, se_median_boot],
})
print("Typical claim: mean vs median, with standard errors")
print(se_table.round(2).to_string(index=False))
print("the gamma model underestimates the SE of the mean by a factor %.1f" % (se_mean_sandwich / se_mean_gamma))
print()

# ---------------- plots ----------------
# plot 1: histogram of log10(claim) with the fitted densities
# if X has density f, then L = log10(X) has density f(10^L) * 10^L * ln(10)
L = np.log10(x)
grid = np.linspace(L.min(), L.max(), 600)
plt.figure(figsize=(8, 4.5))
plt.hist(L, bins=150, density=True, color="lightgray", label="Claims")
for name in ["Exponential", "Gamma", "Lognormal", "Weibull"]:
    dist, params = fits[name]
    pdf_log = dist.pdf(10**grid, *params) * 10**grid * np.log(10)
    plt.plot(grid, pdf_log, label=name)
plt.ylim(0, 1.5) # the spike of fixed amounts goes far above this, we cut it to see the fits
plt.annotate("spike of fixed amounts\n(1128 to 1204, cut off)", xy=(np.log10(1172), 1.45), xytext=(4.3, 1.2),
             arrowprops=dict(arrowstyle="->"))
plt.xlabel("log10(claim amount)")
plt.ylabel("density")
plt.title("Claim amounts and fitted distributions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simG_claims_hist.png", dpi=200)
print("Saved figures/simG_claims_hist.png")

# plot 2: QQ plots on log scales (empirical quantiles against fitted quantiles)
probs = (np.arange(1, N + 1) - 0.5) / N
x_sorted = np.sort(x)
plt.figure(figsize=(12, 4))
for i, name in enumerate(["Gamma", "Lognormal", "Weibull"]):
    dist, params = fits[name]
    q_model = dist.ppf(probs, *params)
    plt.subplot(1, 3, i + 1)
    plt.loglog(q_model, x_sorted, ".", markersize=2)
    lo, hi = x_sorted.min(), x_sorted.max()
    plt.loglog([lo, hi], [lo, hi], "k--", linewidth=0.8)
    plt.xlabel("%s quantiles" % name)
    plt.ylabel("claim quantiles")
    plt.title("QQ plot, %s" % name)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simG_claims_qq.png", dpi=200)
print("Saved figures/simG_claims_qq.png")
