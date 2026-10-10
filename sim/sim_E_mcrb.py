# Author: Varun Shastry
# Simulation E: the misspecified CRB (MCRB), Fortunati et al. 2017
# When the assumed model is wrong, the estimator goes to the pseudo-true value theta0 and
#.....MCRB = (1/N) B / A^2,  A = E[d^2/dtheta^2 ln f(x|theta0)],  B = E[(d/dtheta ln f(x|theta0))^2]
# (expectations under the TRUE distribution, f is the ASSUMED model)
# Part 1: reproduce the variance example of Fortunati et al. 2017, Section 4.2 (wrong mean)
# Part 2: prof's case, estimate the noise variance assuming gaussian when the noise is laplace or student-t
# Part 3: every location case in the report, CRB vs MCRB vs measured

import numpy as np
import pandas as pd
from scipy import stats
np.random.seed(0)

N = 100 # samples per trial
trials = 20000 # num-montecarlo trials (more than before, variance estimates need it)

# ---------------- Part 1: Fortunati Section 4.2 ----------------
# true data N(mu, sigma^2), assumed model N(0, theta), theta estimated by mean(x^2)
# pseudo-true theta0 = sigma^2 + mu^2,  MCRB = (2 sigma^4 + 4 mu^2 sigma^2)/N,  CRB = 2 sigma^4/N
sigma = 1.0
rows = []
for mu in [0.0, 0.5, 1.0]:
    x = np.random.normal(mu, sigma, size=(trials, N))
    theta_hat = np.mean(x**2, axis=1)
    theta0 = sigma**2 + mu**2
    rows.append({
        "mu": mu,
        "theta0 (pred)": theta0,
        "avg estimate": np.mean(theta_hat),
        "N*CRB": 2 * sigma**4,
        "N*MCRB": 2 * sigma**4 + 4 * mu**2 * sigma**2,
        "N*Var (meas)": N * np.var(theta_hat),
        "N*MSE vs sigma^2 (meas)": N * np.mean((theta_hat - sigma**2)**2),
        "N*(MCRB + bias^2)": 2 * sigma**4 + 4 * mu**2 * sigma**2 + N * mu**4,
    })
print("Part 1: Fortunati et al. 2017, Section 4.2 (assumed mean 0, true mean mu), N =", N)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print()

# ---------------- Part 2: wrong noise shape, variance estimation ----------------
# true noise has zero mean, assumed model N(0, theta), estimate theta by mean(x^2)
# pseudo-true theta0 = E[x^2] = true variance, MCRB = (mu4 - sigma^4)/N
# wrong-model CRB = 2 sigma^4/N (what you get if you trust the gaussian model)
# true CRB for the variance, using the true noise model:
#.....laplace(b):  sigma^2 = 2 b^2, I(b) = 1/b^2      -> CRB = (4b)^2 b^2 / N = 16 b^4 / N
#.....t(nu, s):    sigma^2 = s^2 nu/(nu-2), I(s) = 2 nu/((nu+3) s^2) -> CRB = 2 nu (nu+3) s^4 / ((nu-2)^2 N)
rows = []
b = 1.0
x = np.random.laplace(0, b, size=(trials, N))
theta_hat = np.mean(x**2, axis=1)
rows.append({"true noise": "Laplace b=1", "true variance": 2 * b**2, "avg estimate": np.mean(theta_hat),
             "N*wrong-model CRB": 2 * (2 * b**2)**2, "N*true CRB": 16 * b**4,
             "N*MCRB": 24 * b**4 - (2 * b**2)**2, "N*Var (meas)": N * np.var(theta_hat)})
s = 1.0
for nu in [10, 20]:
    x = s * np.random.standard_t(nu, size=(trials, N))
    theta_hat = np.mean(x**2, axis=1)
    var = s**2 * nu / (nu - 2)
    mu4 = 3 * nu**2 * s**4 / ((nu - 2) * (nu - 4))
    rows.append({"true noise": "Student-t nu=%d" % nu, "true variance": var, "avg estimate": np.mean(theta_hat),
                 "N*wrong-model CRB": 2 * var**2, "N*true CRB": 2 * nu * (nu + 3) * s**4 / (nu - 2)**2,
                 "N*MCRB": mu4 - var**2, "N*Var (meas)": N * np.var(theta_hat)})
x = np.random.normal(0, 1, size=(trials, N))
theta_hat = np.mean(x**2, axis=1)
rows.append({"true noise": "Gaussian (no mismatch)", "true variance": 1.0, "avg estimate": np.mean(theta_hat),
             "N*wrong-model CRB": 2.0, "N*true CRB": 2.0, "N*MCRB": 2.0, "N*Var (meas)": N * np.var(theta_hat)})
print("Part 2: estimate the noise variance assuming gaussian noise, N =", N)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print()

# ---------------- Part 3: every location case, CRB vs MCRB vs measured ----------------
# mean = MLE of the gaussian model:  pseudo-true A + E[w] = A,           MCRB = Var(w)/N
# median = MLE of the laplace model: pseudo-true A + median(w),          MCRB = 1/(4 N f(m)^2)
A = 1.0
trials = 5000
noises = {
    # name: (sampler, Var(w), median(w), density at the median, CRB*N or nan)
    "Gaussian":        (lambda size: np.random.normal(0, 1, size), 1.0, 0.0, stats.norm.pdf(0), 1.0),
    "Laplace b=1":     (lambda size: np.random.laplace(0, 1, size), 2.0, 0.0, 0.5, 1.0),
    "Student-t nu=5":  (lambda size: np.random.standard_t(5, size), 5 / 3, 0.0, stats.t.pdf(0, 5), 8 / 6),
    "Gamma k=5":       (lambda size: np.random.gamma(5, 1, size) - 5, 5.0,
                        stats.gamma.ppf(0.5, 5) - 5, stats.gamma.pdf(stats.gamma.ppf(0.5, 5), 5), 3.0),
    "Exponential":     (lambda size: np.random.exponential(1, size) - 1, 1.0,
                        np.log(2) - 1, 0.5, np.nan),
}
rows = []
for name, (sampler, var_w, med_w, f_m, crb_n) in noises.items():
    x = A + sampler((trials, N))
    means = np.mean(x, axis=1)
    medians = np.median(x, axis=1)
    rows.append({"noise": name, "estimator": "mean (MSE)", "pseudo-true": A, "avg estimate": np.mean(means),
                 "N*CRB": crb_n, "N*MCRB": var_w, "N*Var (meas)": N * np.var(means)})
    rows.append({"noise": name, "estimator": "median (MAE)", "pseudo-true": A + med_w, "avg estimate": np.mean(medians),
                 "N*CRB": crb_n, "N*MCRB": 1 / (4 * f_m**2), "N*Var (meas)": N * np.var(medians)})
print("Part 3: location estimation, CRB vs MCRB vs measured, N =", N, "with", trials, "trials")
print(pd.DataFrame(rows).round(3).to_string(index=False))
