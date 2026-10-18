# Author: Bhuvi
# Simulation K: estimating the MCRB from data (the sandwich standard error) and checking its coverage
# Part 1: heteroscedastic regression, y = 1 + 2x + sd(x) z with sd(x) = 0.5 + |x|, fitted by OLS (Gaussian
#         homoscedastic model). MCRB = (X^T X)^-1 X^T Omega X (X^T X)^-1 (White 1980). Compare the classical
#         model-based standard error, HC0 and HC3 (MacKinnon and White 1985) by the coverage of 95% intervals.
# Part 2: the variance-estimation example of report Section 8.4, theta_hat = mean(x^2) under a Gaussian
#         model: model-based SE sqrt(2/N) theta_hat against the sample-MCRB SE sqrt((m4 - theta_hat^2)/N).
# Part 3: Huber regression under heavy-tailed noise, sample sandwich A^-1 B A^-1 from the residuals.
# Run from the repository root: python Classic_sim/sim_K_sandwich_coverage.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import mcrb as m

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(15)
theta = np.array([1.0, 2.0])
R = 5000
pd.set_option("display.width", 200)


def covered(est, se, true, q):
    return np.mean(np.abs(est - true) <= q * se)


# ---------------- Part 1: heteroscedastic OLS ----------------
sd = lambda x: 0.5 + np.abs(x)
unit_noises = {"Gaussian z": lambda size: rng.normal(0, 1, size),
               "Laplace z": lambda size: rng.laplace(0, 1 / np.sqrt(2), size)} # unit variance
rows = []
for zname, zs in unit_noises.items():
    for N in [20, 50, 200, 1000]:
        X = m.design(N, rng)
        s = sd(X[:, 1])
        XtXi = np.linalg.inv(X.T @ X)
        mcrb = XtXi @ (X.T * s**2) @ X @ XtXi             # sandwich with the true Omega
        classical = np.mean(s**2) * XtXi                   # CRB of the assumed model at its pseudo-true variance
        Y = theta @ X.T + s * zs((R, N))
        est = m.ols(X, Y)
        r = Y - est @ X.T
        h = np.einsum("ni,ij,nj->n", X, XtXi, X)           # leverages
        s2 = np.sum(r**2, axis=1) / (N - 2)
        se_cl = np.sqrt(s2 * XtXi[1, 1])
        hc0 = np.einsum("ij,ni,rn,nk,kl->rjl", XtXi, X, r**2, X, XtXi)[:, 1, 1]
        hc3 = np.einsum("ij,ni,rn,nk,kl->rjl", XtXi, X, r**2 / (1 - h)**2, X, XtXi)[:, 1, 1]
        q = stats.t.ppf(0.975, N - 2)
        rows.append({"z": zname, "N": N,
                     "N*Var slope (meas)": N * np.var(est[:, 1]), "N*MCRB": N * mcrb[1, 1],
                     "N*classical": N * classical[1, 1],
                     "cover classical": covered(est[:, 1], se_cl, theta[1], q),
                     "cover HC0": covered(est[:, 1], np.sqrt(hc0), theta[1], q),
                     "cover HC3": covered(est[:, 1], np.sqrt(hc3), theta[1], q)})
het = pd.DataFrame(rows)
print("Part 1: heteroscedastic regression, OLS slope, %d trials, nominal coverage 0.95" % R)
print(het.round(3).to_string(index=False))
print()

# ---------------- Part 2: variance estimation under a Gaussian model ----------------
true_noise = {"Gaussian": (lambda size: rng.normal(0, 1, size), 1.0),
              "Laplace b=1": (lambda size: rng.laplace(0, 1, size), 2.0),
              "Student-t nu=10": (lambda size: rng.standard_t(10, size), 10 / 8),
              "Student-t nu=5": (lambda size: rng.standard_t(5, size), 5 / 3)}
rows = []
for name, (sampler, var) in true_noise.items():
    for N in [50, 200, 1000]:
        x = sampler((R, N))
        th = np.mean(x**2, axis=1)
        se_model = np.sqrt(2 / N) * th
        se_sand = np.sqrt(np.maximum(np.mean(x**4, axis=1) - th**2, 0) / N)
        rows.append({"noise": name, "N": N, "cover Gaussian-model SE": covered(th, se_model, var, 1.96),
                     "cover sample-MCRB SE": covered(th, se_sand, var, 1.96),
                     "mean SE model / true sd": np.mean(se_model) / np.std(th),
                     "mean SE sandwich / true sd": np.mean(se_sand) / np.std(th)})
varest = pd.DataFrame(rows)
print("Part 2: 95%% intervals for the noise variance, theta_hat = mean(x^2), %d trials" % R)
print(varest.round(3).to_string(index=False))
print()

# ---------------- Part 3: Huber regression, sample sandwich ----------------
loss = m.Huber()
rows = []
for nz in [m.Gaussian(), m.Laplace(), m.StudentT(3), m.Contaminated()]:
    for N in [50, 200, 1000]:
        X = m.design(N, rng)
        Y = theta @ X.T + nz.sample(rng, (R, N))
        est, failed, c = m.m_estimate(X, Y, loss, return_info=True)
        r = Y - est @ X.T
        a = loss.dpsi(r, c[:, None])
        b = loss.psi(r, c[:, None])**2
        Ainv = np.linalg.inv(np.einsum("ni,rn,nj->rij", X, a, X))
        Bm = np.einsum("ni,rn,nj->rij", X, b, X)
        S = Ainv @ Bm @ Ainv
        se = np.sqrt(S[:, 1, 1])
        M = m.mcrb_matrix(X, loss, nz)
        rows.append({"noise": nz.name, "N": N, "N*Var slope (meas)": N * np.var(est[:, 1]),
                     "N*MCRB": N * M[1, 1], "N*mean sandwich": N * np.mean(S[:, 1, 1]),
                     "cover sandwich": covered(est[:, 1], se, theta[1], stats.t.ppf(0.975, N - 2)),
                     "failed": int(failed.sum())})
hub = pd.DataFrame(rows)
print("Part 3: Huber regression slope, sample sandwich standard error, %d trials" % R)
print(hub.round(3).to_string(index=False))

# figure: coverage against N for Part 1 and Part 2
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for zname, g in het.groupby("z", sort=False):
    for col, mk in [("cover classical", "o-"), ("cover HC0", "s--"), ("cover HC3", "^:")]:
        axes[0].plot(g["N"], g[col], mk, label="%s, %s" % (col.replace("cover ", ""), zname))
axes[0].set_title("Heteroscedastic regression, OLS slope")
for name, g in varest.groupby("noise", sort=False):
    l, = axes[1].plot(g["N"], g["cover Gaussian-model SE"], "o-", label="Gaussian-model SE, " + name)
    axes[1].plot(g["N"], g["cover sample-MCRB SE"], "s--", color=l.get_color(), label="sample MCRB, " + name)
axes[1].set_title(r"Noise variance, $\hat\theta = \overline{x^2}$")
for ax in axes:
    ax.axhline(0.95, color="gray", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("coverage of nominal 95% interval")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simK_sandwich_coverage.png", dpi=200)
plt.close()
print("Saved figures/simK_sandwich_coverage.png")
