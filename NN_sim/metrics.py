# Author: Harsh
# Module: metrics.py
# Evaluation metrics: ISE, Bias-Variance decomposition, Risk Ratio, NLL, CRPS, and Coverage.

import numpy as np
from scipy import stats

def compute_ise(y_pred, y_true_clean):
    """
    Computes Integrated Squared Error against the clean ground truth f*(x).
    ISE = (1 / N) * sum((y_pred - f*(x))^2)
    """
    return float(np.mean((y_pred - y_true_clean) ** 2))

def bias_variance_decomposition(predictions_matrix, y_true_clean):
    """
    Decomposes total integrated risk into squared bias and variance across M independent trials.
    
    predictions_matrix: shape [M, N_test]
      M = number of independent trials
      N_test = number of test points
    y_true_clean: shape [N_test] or [N_test, 1]
    
    Returns:
      bias_squared: integrated squared bias across the domain
      variance: integrated variance across the domain
      total_risk: bias_squared + variance (equals average ISE)
    """
    y_true_clean = y_true_clean.flatten()
    M, N_test = predictions_matrix.shape
    
    # Mean prediction across M trials for each test point x_j
    mean_prediction = np.mean(predictions_matrix, axis=0) # [N_test]
    
    # Pointwise bias and variance
    pointwise_bias_sq = (mean_prediction - y_true_clean) ** 2
    pointwise_variance = np.mean((predictions_matrix - mean_prediction[None, :]) ** 2, axis=0)
    
    # Integrated over domain
    integrated_bias_sq = float(np.mean(pointwise_bias_sq))
    integrated_variance = float(np.mean(pointwise_variance))
    total_risk = integrated_bias_sq + integrated_variance
    
    return {
        "bias_squared": integrated_bias_sq,
        "variance": integrated_variance,
        "total_risk": total_risk,
        "mean_ise": float(np.mean(np.mean((predictions_matrix - y_true_clean[None, :]) ** 2, axis=1)))
    }

def compute_gaussian_crps(y, mu, sigma):
    """
    Computes Continuous Ranked Probability Score (CRPS) for Gaussian predictive distribution.
    Closed-form expression:
    CRPS(y; mu, sigma) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.
    """
    z = (y - mu) / sigma
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    crps = sigma * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))

def compute_probabilistic_metrics(y_test, mu_pred, sigma_pred, confidence=0.90):
    """
    Evaluates probabilistic regression metrics:
      - NLL (Negative Log-Likelihood)
      - CRPS
      - Empirical coverage rate of (1 - alpha) prediction interval
      - Average interval width
    """
    y_test = y_test.flatten()
    mu_pred = mu_pred.flatten()
    sigma_pred = np.maximum(sigma_pred.flatten(), 1e-6)
    
    # NLL
    nll = 0.5 * np.log(2.0 * np.pi * sigma_pred**2) + ((y_test - mu_pred)**2) / (2.0 * sigma_pred**2)
    mean_nll = float(np.mean(nll))
    
    # CRPS
    mean_crps = compute_gaussian_crps(y_test, mu_pred, sigma_pred)
    
    # Coverage & Width
    z_alpha = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    lower = mu_pred - z_alpha * sigma_pred
    upper = mu_pred + z_alpha * sigma_pred
    
    covered = (y_test >= lower) & (y_test <= upper)
    coverage_rate = float(np.mean(covered))
    avg_width = float(np.mean(upper - lower))
    
    return {
        "nll": mean_nll,
        "crps": mean_crps,
        "coverage": coverage_rate,
        "interval_width": avg_width
    }

