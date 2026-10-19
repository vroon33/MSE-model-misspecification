# Author: Anshitha Penubolu
# Simulation M: how fragile are the mean and its standard error on the claims data (freMTPL2sev)?
# Simulation I fits a generalized Pareto tail above the 95th percentile and finds tail index alpha = 1/xi ~ 1.09.
# Here we check
#.....1. that alpha < 2 (infinite variance) does not depend on the choice of threshold
#.....2. what the largest claims do to the mean, the median and the standard error of the mean
#.....3. how the running mean and median behave as claims come in, as for Cauchy noise in simulation C
# Standard error of the GPD shape from its MLE (valid for xi > -1/2): SE(xi) ~ (1 + xi) / sqrt(n_exceed)

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import fetch_openml
warnings.filterwarnings("ignore")
np.random.seed(0)

data = fetch_openml(data_id=41215, as_frame=True).frame
claims = data["ClaimAmount"].to_numpy()
N = len(claims)
print("Number of claims:", N)

# ---------- 1. tail index against the threshold ----------
quantiles = [0.85, 0.875, 0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99]
rows = []
for q in quantiles:
    u = np.quantile(claims, q)
    exc = claims[claims > u] - u
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    se = (1 + xi) / np.sqrt(len(exc))
    rows.append({"quantile": q, "threshold u": u, "exceedances": len(exc), "xi": xi,
                 "xi low": xi - 1.96 * se, "xi high": xi + 1.96 * se,
                 "alpha": 1 / xi, "alpha low": 1 / (xi + 1.96 * se), "alpha high": 1 / (xi - 1.96 * se)})
tail = pd.DataFrame(rows)
print("\nGPD tail index against the threshold (95% intervals)")
print(tail.round(3).to_string(index=False))
print("largest upper end of alpha over all thresholds: %.2f (infinite variance needs alpha < 2)" % tail["alpha high"].max())

# ---------- 2. removing the largest claims ----------
s = np.sort(claims)[::-1] # largest first
rows = []
for m in [0, 1, 2, 5, 10, 20, 50]:
    rest = s[m:]
    rows.append({"removed": m, "mean": rest.mean(), "1% trimmed mean": stats.trim_mean(rest, 0.01),
                 "median": np.median(rest), "SE of mean": rest.std(ddof=1) / np.sqrt(len(rest))})
drop = pd.DataFrame(rows)
print("\nRemoving the m largest claims out of", N)
print(drop.round(1).to_string(index=False))
print("largest claims:", np.round(s[:5]))

# ---------- 3. running mean and median in a random order ----------
order = np.random.permutation(N)
x = claims[order]
n = np.arange(1, N + 1)
run_mean = np.cumsum(x) / n
checkpoints = np.unique(np.logspace(1, np.log10(N), 300).astype(int))
run_median = np.array([np.median(x[:c]) for c in checkpoints])
print("\nRunning estimates, random order: n, mean, median")
for c in [100, 1000, 5000, 10000, 20000, N]:
    print("%6d  %8.1f  %8.1f" % (c, run_mean[c - 1], np.median(x[:c])))

# ---------- plots ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.errorbar(tail["quantile"] * 100, tail["alpha"],
            yerr=[tail["alpha"] - tail["alpha low"], tail["alpha high"] - tail["alpha"]],
            fmt="o", color="tab:red", capsize=3, label="tail index alpha = 1/xi, 95% interval")
ax.axhline(2, color="k", linestyle="--", label="alpha = 2: variance infinite below")
ax.axhline(1, color="gray", linestyle=":", label="alpha = 1: mean infinite below")
ax.set_xlabel("threshold, percentile of the claims")
ax.set_ylabel("tail index alpha")
ax.set_ylim(0, 2.5)
ax.set_title("Tail index against the threshold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(drop["removed"], drop["mean"], "o-", label="mean (MSE)")
ax.plot(drop["removed"], drop["1% trimmed mean"], "s-", label="1% trimmed mean")
ax.plot(drop["removed"], drop["median"], "^-", label="median (MAE)")
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("number of largest claims removed")
ax.set_ylabel("estimate of the typical claim")
ax.set_title("Effect of the largest claims")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(drop["removed"], drop["SE of mean"], "k--", alpha=0.6)
ax2.set_ylabel("SE of the mean (dashed)")

ax = axes[2]
ax.semilogx(n, run_mean, color="tab:blue", label="running mean (MSE)")
ax.semilogx(checkpoints, run_median, color="tab:orange", label="running median (MAE)")
ax.set_ylim(0, 3500)
ax.set_xlabel("number of claims n (random order)")
ax.set_ylabel("estimate after n claims")
ax.set_title("Running estimates")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("figures/simM_claims_tail.png", dpi=200)
print("Saved figures/simM_claims_tail.png")
