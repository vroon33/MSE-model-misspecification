# Author: Varun Shastry
# Simulation A: mean vs median under Gaussian and Laplace noise
# Model is DC under WGN
# We measure the variance of each estimate over many trials, and divide it by CRLB of the true noise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
np.random.seed(0)

A=1.0 # true DC
sigma=1.0 # gauss noise std dev
b=1.0 # laplacian noise scale
trials = 5000 # num-montecarlo trials

def efficiency_ratios(N):
    # gaussian noise: each row is a trial of N samples
    x = A + np.random.normal(0, sigma, size=(trials, N))
    means = np.mean(x, axis=1) # per trial
    medians = np.median(x, axis=1)
    crlb = sigma**2/N
    mean_gauss = np.var(means) / crlb
    median_gauss = np.var(medians) / crlb
    # laplace noise
    x = A + np.random.laplace(0, b, size=(trials, N))
    means = np.mean(x, axis=1)
    medians = np.median(x, axis=1)
    crlb = b**2/N
    mean_laplace = np.var(means) / crlb
    median_laplace = np.var(medians) / crlb

    return mean_gauss, median_gauss, mean_laplace, median_laplace

# print table @ N=100
N = 100
mean_gauss, median_gauss, mean_laplace, median_laplace = efficiency_ratios(N)

table = pd.DataFrame({
    "Gaussian (predicted)": [1.0, np.pi/2],
    "Gaussian (measured)": [mean_gauss, median_gauss],
    "Laplace (predicted)": [2.0, 1.0],
    "Laplace (measured)": [mean_laplace, median_laplace],
}, index=["Mean (MSE)", "Median (MAE)"])

print("Efficiency ratio Var/CRLB at N =", N, "with", trials, "trials")
print(table.round(3).to_string())
print()

# how ratios change with N
N_values = [10, 20, 50, 100, 200, 500, 1000]
mean_gauss_list = []
median_gauss_list = []
mean_laplace_list = []
median_laplace_list = []

for N in N_values:
    r1, r2, r3, r4 = efficiency_ratios(N)
    mean_gauss_list.append(r1)
    median_gauss_list.append(r2)
    mean_laplace_list.append(r3)
    median_laplace_list.append(r4)

sweep = pd.DataFrame({
    "N": N_values,
    "Mean, Gauss": mean_gauss_list,
    "Median, Gauss": median_gauss_list,
    "Mean, Laplace": mean_laplace_list,
    "Median, Laplace": median_laplace_list,
})
print("Efficiency ratio Var/CRLB as N grows")
print(sweep.round(3).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.plot(N_values, mean_gauss_list, "o-", color="tab:blue", label="Mean, Gaussian noise")
plt.plot(N_values, median_gauss_list, "s-", color="tab:orange", label="Median, Gaussian noise")
plt.plot(N_values, mean_laplace_list, "o--", color="tab:blue", label="Mean, Laplace noise")
plt.plot(N_values, median_laplace_list, "s--", color="tab:orange", label="Median, Laplace noise")

# predicted values, with their value written at the right end of each line
plt.axhline(1, color="gray", linewidth=0.8, label="Predicted value")
plt.axhline(2, color="gray", linewidth=0.8)
plt.axhline(np.pi/2, color="gray", linewidth=0.8)
plt.text(1300, 1, "1", va="center")
plt.text(1300, 2, "2", va="center")
plt.text(1300, np.pi/2, r"$\pi/2$", va="center")
plt.xscale("log")
plt.xlim(8, 2000)
plt.xlabel("Number of samples N")
plt.ylabel("Var/CRLB")
plt.title("Efficiency ratio of mean and median")
plt.ylim(0.8, 2.5)
plt.legend(loc="upper center", ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/simA_ratio_vs_N.png")
print("Saved figures/simA_ratio_vs_N.png")