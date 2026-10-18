# Author: Bhuvi
# Simulation J: the MCRB (sandwich) for robust losses in linear regression, the Regime A mismatch matrix
# Model (plan Section 4.1): y_i = theta_0 + theta_1 x_i + w_i, x_i ~ U(-2, 2) drawn once and fixed, theta = [1, 2]
# Losses (rows):  MSE, MAE, Huber (k = 1.345), t3-loss, Cauchy-loss, scales set by the MAD rule in mcrb.py
# Noises (cols):  Gaussian, Laplace, Student-t3, Cauchy, contaminated Gaussian (eps = 0.1, sigma1 = 5)
# Predicted: MCRB = (kappa/lam^2) (X^T X)^-1 and CRB = (X^T X)^-1 / I_loc (sign convention in mcrb.py)
# Part 1: predicted efficiency ratio tr(MCRB)/tr(CRB) for every cell
# Part 2: Monte Carlo at N = 100, R = 5000: bias, measured ratio, full-matrix check, the three gates
# Part 3: how tight is the MCRB at finite N, N * Var / (N * MCRB) for N = 10 ... 2000
# Run from the repository root: python Classic_sim/sim_J_mcrb_regression.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mcrb as m

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(2023112015)

theta = np.array([1.0, 2.0])
losses = m.default_losses()
noises = m.default_noises()
B = 20 # batches, for the Monte Carlo standard error of each measured ratio


def run_cell(X, loss, noise, R):
    # R trials of one loss-noise pair with the design X held fixed
    Y = theta @ X.T + noise.sample(rng, (R, len(X)))
    est, failed, _ = m.m_estimate(X, Y, loss, return_info=True)
    return est, failed


def trace_ratio(est, target_trace):
    # measured tr(Cov) / target, and its batch-means standard error
    full = np.trace(np.cov(est.T)) / target_trace
    batches = [np.trace(np.cov(b.T)) / target_trace for b in np.array_split(est, B)]
    return full, np.std(batches, ddof=1) / np.sqrt(B)


# ---------------- Part 1: predicted matrix ----------------
pred = pd.DataFrame({nz.name: [m.efficiency_ratio(ls, nz) for ls in losses] for nz in noises},
                    index=[ls.name for ls in losses])
print("Part 1: predicted efficiency ratio tr(MCRB)/tr(CRB) (independent of X and N)")
print("I_loc per noise:", {nz.name: round(nz.fisher(), 4) for nz in noises})
print(pred.round(3).to_string())
print()

# ---------------- Part 2: Monte Carlo at N = 100 ----------------
N, R = 100, 5000
X = m.design(N, rng)
rows = []
for ls in losses:
    for nz in noises:
        crb = m.crb_matrix(X, nz)
        mc = m.mcrb_matrix(X, ls, nz)
        row = {"loss": ls.name, "noise": nz.name, "predicted": pred.loc[ls.name, nz.name]}
        if not np.isfinite(row["predicted"]):
            # MSE under Cauchy noise: the variance does not exist, so no ratio can be measured
            est, failed = run_cell(X, ls, nz, R)
            row.update({"measured": np.nan, "mc_se": np.nan, "bias0": np.nan, "bias1": np.nan,
                        "frob_err": np.nan, "failed": int(failed.sum()),
                        "IQR slope": np.subtract(*np.percentile(est[:, 1], [75, 25]))})
            rows.append(row)
            continue
        est, failed = run_cell(X, ls, nz, R)
        ratio, se = trace_ratio(est, np.trace(crb))
        cov = np.cov(est.T)
        bias = est.mean(axis=0) - theta
        row.update({"measured": ratio, "mc_se": se, "bias0": bias[0], "bias1": bias[1],
                    "frob_err": np.linalg.norm(cov - mc) / np.linalg.norm(mc), "failed": int(failed.sum())})
        rows.append(row)
res = pd.DataFrame(rows)
res["z"] = (res["measured"] - res["predicted"]) / res["mc_se"]
pd.set_option("display.width", 200)
print("Part 2: N = %d, R = %d trials, fixed design" % (N, R))
print(res.drop(columns=["IQR slope"]).round(3).to_string(index=False))
print("MSE under Cauchy: IQR of the slope estimates = %.3f (no variance exists)"
      % res.loc[res["IQR slope"].notna(), "IQR slope"].iloc[0])
print()

meas = res.pivot(index="loss", columns="noise", values="measured").loc[pred.index, pred.columns]
print("Measured efficiency ratio tr(Cov)/tr(CRB):")
print(meas.round(3).to_string())
print()

gates = [("MSE", "Gaussian", 1.0), ("MAE", "Laplace", 1.0), ("t3-loss", "Student-t3", 1.0),
         ("Cauchy-loss", "Cauchy", 1.0), ("MSE", "Laplace", 2.0), ("MSE", "Student-t3", 2.0)]
print("Correctness gates (predicted value is exact; finite-N gap shown with its MC standard error):")
for ls, nz, target in gates:
    r = res[(res["loss"] == ls) & (res["noise"] == nz)].iloc[0]
    print("  %-12s under %-12s target %.2f  measured %.3f +- %.3f" % (ls, nz, target, r["measured"], r["mc_se"]))
print()

# heatmap of predicted and measured ratios
fig, ax = plt.subplots(figsize=(9, 4.8))
shown = np.log(np.clip(np.where(np.isfinite(pred.values), pred.values, 4.0), 1, 4))
ax.imshow(shown, cmap="Blues", vmin=0, vmax=np.log(4))
for i in range(len(losses)):
    for j in range(len(noises)):
        p, q = pred.values[i, j], meas.values[i, j]
        txt = "pred $\\infty$\nundefined" if not np.isfinite(p) else "pred %.3f\nmeas %.3f" % (p, q)
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5)
ax.set_xticks(range(len(noises)), pred.columns)
ax.set_yticks(range(len(losses)), pred.index)
ax.set_xlabel("true noise")
ax.set_ylabel("training loss (assumed noise)")
ax.set_title("tr(MCRB)/tr(CRB), predicted and measured, N = %d, %d trials" % (N, R))
plt.tight_layout()
plt.savefig("figures/simJ_mismatch_matrix.png", dpi=200)
plt.close()

# ---------------- Part 3: tightness against N ----------------
Ns = [10, 20, 50, 100, 200, 500, 1000, 2000]
R3 = 4000
cells = [("MSE", "Laplace"), ("MAE", "Gaussian"), ("MAE", "Laplace"), ("Huber", "Contaminated"),
         ("t3-loss", "Gaussian"), ("Cauchy-loss", "Cauchy")]
by_name_l = {ls.name: ls for ls in losses}
by_name_n = {nz.name: nz for nz in noises}
rows = []
for n in Ns:
    Xn = m.design(n, rng)
    for ls, nz in cells:
        est, failed = run_cell(Xn, by_name_l[ls], by_name_n[nz], R3)
        target = np.trace(m.mcrb_matrix(Xn, by_name_l[ls], by_name_n[nz]))
        ratio, se = trace_ratio(est, target)
        rows.append({"N": n, "cell": "%s / %s" % (ls, nz), "Var/MCRB": ratio, "mc_se": se,
                     "failed": int(failed.sum())})
tight = pd.DataFrame(rows)
print("Part 3: measured tr(Cov) / tr(MCRB) against N, %d trials each" % R3)
print(tight.pivot(index="N", columns="cell", values="Var/MCRB").round(3).to_string())
print("max MC standard error in this table: %.3f" % tight["mc_se"].max())
print("non-converged fits:", int(tight["failed"].sum()))

plt.figure(figsize=(8, 4.8))
for cell, g in tight.groupby("cell", sort=False):
    plt.errorbar(g["N"], g["Var/MCRB"], yerr=2 * g["mc_se"], marker="o", capsize=2, label=cell)
plt.axhline(1, color="gray", ls="--")
plt.xscale("log")
plt.xlabel("number of samples $N$")
plt.ylabel(r"measured tr(Cov) / tr(MCRB)")
plt.title("How tight is the MCRB at finite N (error bars: 2 MC standard errors)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simJ_tightness_vs_N.png", dpi=200)
plt.close()
print("Saved figures/simJ_mismatch_matrix.png and figures/simJ_tightness_vs_N.png")
