# Author: Harsh
# Module: ground_truth.py
# Controlled regression ground truth and synthetic noise generators.

import numpy as np

def f_star(x):
    """
    Ground truth regression function: f*(x) = sin(2*pi*x) + 0.5*x
    Domain: x in [0, 1]
    """
    return np.sin(2.0 * np.pi * x) + 0.5 * x

def sample_noise(noise_type, size, sigma=0.3, **kwargs):
    """
    Generates synthetic noise with zero mean and controlled variance/shape.
    
    noise_type:
      - 'gaussian': N(0, sigma^2)
      - 'gamma': Centered Gamma with shape k, scale theta = sigma / sqrt(k)
                 w = Y - k*theta, so E[w] = 0, Var(w) = sigma^2
      - 'student_t': Student-t with nu degrees of freedom, scaled by s = sigma * sqrt((nu-2)/nu)
                     so E[w] = 0, Var(w) = sigma^2 (for nu > 2)
      - 'laplace': Laplace(0, b) with b = sigma / sqrt(2), so Var(w) = sigma^2
    """
    if noise_type == "gaussian":
        return np.random.normal(0.0, sigma, size=size)
        
    elif noise_type == "gamma":
        k = kwargs.get("k", 3.0)
        theta = sigma / np.sqrt(k)
        # Y ~ Gamma(shape=k, scale=theta), mean=k*theta, var=k*theta^2 = sigma^2
        y = np.random.gamma(shape=k, scale=theta, size=size)
        return y - k * theta
        
    elif noise_type == "student_t":
        nu = kwargs.get("nu", 3.0)
        if nu > 2:
            scale = sigma * np.sqrt((nu - 2.0) / nu)
        else:
            scale = sigma
        t_samples = np.random.standard_t(df=nu, size=size)
        return t_samples * scale
        
    elif noise_type == "laplace":
        b = sigma / np.sqrt(2.0)
        return np.random.laplace(0.0, b, size=size)
        
    elif noise_type == "contaminated":
        eps = kwargs.get("eps", 0.05)
        k_outlier = kwargs.get("k_outlier", 5.0)
        # Var(w) = (1-eps)*sigma0^2 + eps*sigma1^2 = sigma^2
        sigma0 = sigma / np.sqrt(1.0 - eps + eps * k_outlier**2)
        sigma1 = k_outlier * sigma0
        mask = (np.random.random(size=size) < eps)
        return np.where(mask, np.random.normal(0, sigma1, size), np.random.normal(0, sigma0, size))
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def generate_dataset(n_samples, noise_type, sigma=0.3, random_state=None, **kwargs):
    """
    Generates (X, y) regression dataset where X ~ Uniform(0, 1) and y = f*(X) + w.
    """
    if random_state is not None:
        np.random.seed(random_state)
    x = np.random.uniform(0.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    y_clean = f_star(x)
    w = sample_noise(noise_type, size=(n_samples, 1), sigma=sigma, **kwargs).astype(np.float32)
    y = y_clean + w
    return x, y, y_clean

