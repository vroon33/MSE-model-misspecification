# Author: Varun Shastry
# Simulation B: mean vs median under asymmetric (skewed) noise
# Model is DC under noise w[n], which has zero mean but is NOT symmetric.
# hence the prediction is
#.....sample mean -> A + mean(w) = A
#.....sample median -> A + median(w) != A

# Two skewed noises, shifted so that their mean is zero:
#.....Exponential: w = E - 1, E ~ Exp(1); median(w) = ln(2) - 1
#.....Lognormal: w = exp(Z) - e^0.5, Z ~ N(0, 1); median(w) = 1 - e^0.5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)

A = 1.0
trials = 5000
median_exp = np.log(2) - 1 # shifted exponential noise median
median_logn = 1 - np.exp(0.5) # shifted lognormal noise median

def exponential_noise(size):
    return np.random.exponential(1.0, size) - 1.0

def lognormal_noise(size):
    return np.exp(np.random.normal(0.0, 1.0, size)) - np.exp(0.5)

# density of the two shifted noises, with their mean (0) and median marked
# shaded area = half of the probability, to the left of the median
w = np.linspace(-3, 5, 2000)
gauss_pdf = np.exp(-w**2 / 2) / np.sqrt(2 * np.pi)
exp_pdf = np.where(w >= -1, np.exp(-(w + 1)), 0) # E = w + 1 >= 0
y = np.maximum(w + np.exp(0.5), 1e-12) # Y = w + e^0.5 > 0
logn_pdf = np.where(w + np.exp(0.5) > 0, np.exp(-np.log(y)**2 / 2) / (y * np.sqrt(2 * np.pi)), 0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(w, exp_pdf, color="black", label="Shifted exponential")
plt.plot(w, gauss_pdf, color="gray", linestyle="--", label="Gaussian N(0, 1)")
plt.fill_between(w, exp_pdf, where=(w <= median_exp), alpha=0.25, color="tab:orange")
plt.axvline(0, color="tab:blue", label="mean = 0")
plt.axvline(median_exp, color="tab:orange", linestyle="--", label="median = %.3f" % median_exp)
plt.xlabel("w")
plt.ylabel("p(w)")
plt.title("Exponential noise, w = E - 1")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(w, logn_pdf, color="black", label="Shifted lognormal")
plt.plot(w, gauss_pdf, color="gray", linestyle="--", label="Gaussian N(0, 1)")
plt.fill_between(w, logn_pdf, where=(w <= median_logn), alpha=0.25, color="tab:orange")
plt.axvline(0, color="tab:blue", label="mean = 0")
plt.axvline(median_logn, color="tab:orange", linestyle="--", label="median = %.3f" % median_logn)
plt.xlabel("w")
plt.ylabel("p(w)")
plt.title("Lognormal noise, w = exp(Z) - e^0.5")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simB_noise_density.png", dpi=200)
print("Saved figures/simB_noise_density.png")

# estimates at N = 100
N = 100
x_exp = A + exponential_noise((trials, N))
means_exp = np.mean(x_exp, axis=1)
medians_exp = np.median(x_exp, axis=1)

x_logn = A + lognormal_noise((trials, N))
means_logn = np.mean(x_logn, axis=1)
medians_logn = np.median(x_logn, axis=1)

# predicted std of each estimate
#.....mean: sqrt(Var(w)/N), Var(w) = 1 (exponential), (e - 1) e (lognormal)
#.....median: sqrt(1 / (4Nf^2)), f = density of w at its median = 1/2 (exponential), 1/sqrt(2 pi) (lognormal)
table = pd.DataFrame({
    "Noise": ["Exponential", "Exponential", "Lognormal", "Lognormal"],
    "Estimator": ["Mean (MSE)", "Median (MAE)", "Mean (MSE)", "Median (MAE)"],
    "Predicted limit": [A, A + median_exp, A, A + median_logn],
    "Average": [np.mean(means_exp), np.mean(medians_exp), np.mean(means_logn), np.mean(medians_logn)],
    "Predicted std": [np.sqrt(1 / N), np.sqrt(1 / N),
                      np.sqrt((np.e - 1) * np.e / N), np.sqrt(np.pi / (2 * N))],
    "Measured std": [np.std(means_exp), np.std(medians_exp), np.std(means_logn), np.std(medians_logn)],
})

print("Estimates at N =", N, "with", trials, "trials, true A =", A)
print(table.round(3).to_string(index=False))
print()

# histograms of estimates
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(means_exp, bins=50, alpha=0.6, label="Mean (MSE)")
plt.hist(medians_exp, bins=50, alpha=0.6, label="Median (MAE)")
plt.axvline(A, color="black", linestyle="--", label="A")
plt.axvline(A + median_exp, color="red", linestyle="--", label="A + median(w)")
plt.xlabel("Estimate of A")
plt.ylabel("Number of trials")
plt.title("Exponential noise, N = 100")
plt.ylim(0, 480)
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(means_logn, bins=50, alpha=0.6, label="Mean (MSE)")
plt.hist(medians_logn, bins=50, alpha=0.6, label="Median (MAE)")
plt.axvline(A, color="black", linestyle="--", label="A")
plt.axvline(A + median_logn, color="red", linestyle="--", label="A + median(w)")
plt.xlabel("Estimate of A")
plt.ylabel("Number of trials")
plt.title("Lognormal noise, N = 100")
plt.ylim(0, 480)
plt.legend()
plt.tight_layout()
plt.savefig("figures/simB_histograms.png", dpi=200)
print("Saved figures/simB_histograms.png")

# avg estimate as N grows
N_values = [10, 20, 50, 100, 200, 500, 1000]
avg_mean_exp = []
avg_median_exp = []
avg_mean_logn = []
avg_median_logn = []

for N in N_values:
    x = A + exponential_noise((trials, N))
    avg_mean_exp.append(np.mean(np.mean(x, axis=1)))
    avg_median_exp.append(np.mean(np.median(x, axis=1)))

    x = A + lognormal_noise((trials, N))
    avg_mean_logn.append(np.mean(np.mean(x, axis=1)))
    avg_median_logn.append(np.mean(np.median(x, axis=1)))

sweep = pd.DataFrame({
    "N": N_values,
    "Mean, Exp": avg_mean_exp,
    "Median, Exp": avg_median_exp,
    "Mean, Logn": avg_mean_logn,
    "Median, Logn": avg_median_logn,
})
print("Average estimate of A as N grows (predicted: mean %.3f, median exp %.3f, median logn %.3f)"
      % (A, A + median_exp, A + median_logn))
print(sweep.round(3).to_string(index=False))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(N_values, avg_mean_exp, "o-", label="Mean (MSE)")
plt.plot(N_values, avg_median_exp, "s-", label="Median (MAE)")
plt.axhline(A, color="black", linestyle="--", label="A")
plt.axhline(A + median_exp, color="red", linestyle="--", label="A + median(w)")
plt.xscale("log")
plt.xlabel("Number of samples N")
plt.ylabel("Average estimate of A")
plt.title("Exponential noise")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(N_values, avg_mean_logn, "o-", label="Mean (MSE)")
plt.plot(N_values, avg_median_logn, "s-", label="Median (MAE)")
plt.axhline(A, color="black", linestyle="--", label="A")
plt.axhline(A + median_logn, color="red", linestyle="--", label="A + median(w)")
plt.xscale("log")
plt.xlabel("Number of samples N")
plt.ylabel("Average estimate of A")
plt.title("Lognormal noise")
plt.legend()
plt.tight_layout()
plt.savefig("figures/simB_vs_N.png")
print("Saved figures/simB_vs_N.png")
