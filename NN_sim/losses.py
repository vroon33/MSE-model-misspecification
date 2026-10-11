# basic files for some common functions and loss functions for neural network regression
# by Harsh

import torch
import torch.nn as nn
import numpy as np

class MSELoss(nn.Module):
    """Gaussian Maximum Likelihood Loss (Standard L2)"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)


class MAELoss(nn.Module):
    """Laplace Maximum Likelihood Loss (Standard L1)"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)


class HuberLoss(nn.Module):
    """Huber M-Estimator Loss (Gaussian Core + Linear Tails)"""
    def __init__(self, delta=1.345):
        super().__init__()
        self.criterion = nn.HuberLoss(delta=delta)
        
    def forward(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)


class CauchyLoss(nn.Module):
    """Cauchy Maximum Likelihood Loss: rho(r) = log(1 + (r/c)^2)"""
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
        
    def forward(self, y_pred, y_true):
        diff = (y_pred - y_true) / self.c
        return torch.mean(torch.log1p(diff**2))


class QuantileLoss(nn.Module):
    """Pinball / Asymmetric Laplace Loss for Quantile Regression"""
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
        
    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        loss = torch.max((self.tau - 1.0) * diff, self.tau * diff)
        return torch.mean(loss)


class BarronAdaptiveLoss(nn.Module):
    """
    General and Adaptive Robust Loss (Barron, CVPR 2019).
    rho(r, alpha, c) = (|alpha - 2| / alpha) * ( (( (r/c)^2 / |alpha - 2| ) + 1)^(alpha/2) - 1 )
    where alpha is a learnable parameter constrained via sigmoid or softplus.
    alpha = 2  -> L2 (Gaussian)
    alpha = 1  -> Charbonnier / Smooth L1 (Laplace)
    alpha = 0  -> Cauchy
    alpha = -2 -> Geman-McClure
    alpha -> -inf -> Welsch
    """
    def __init__(self, init_alpha=2.0, c=1.0):
        super().__init__()
        self.c = c
        # Map raw parameter to (-2, 2.5)
        # alpha = 4.5 * sigmoid(param) - 2.0
        # If init_alpha = 2.0, param = logit((2.0 + 2.0)/4.5) = logit(4.0/4.5)
        p = np.clip((init_alpha + 2.0) / 4.5, 0.01, 0.99)
        self.raw_alpha = nn.Parameter(torch.tensor(float(np.log(p / (1.0 - p)))))

    def get_alpha(self):
        return 4.5 * torch.sigmoid(self.raw_alpha) - 2.0

    def forward(self, y_pred, y_true):
        alpha = self.get_alpha()
        r = (y_pred - y_true) / self.c
        sq = r**2
        
        # Numerical stability handling for alpha near 0 and alpha near 2
        eps = 1e-4
        if torch.abs(alpha) < eps:
            # Cauchy limit: 0.5 * log(1 + 0.5 * (r/c)^2)
            loss = torch.log1p(0.5 * sq)
        elif torch.abs(alpha - 2.0) < eps:
            # L2 limit: 0.5 * (r/c)^2
            loss = 0.5 * sq
        else:
            abs_diff = torch.abs(alpha - 2.0)
            inner = (sq / abs_diff) + 1.0
            loss = (abs_diff / alpha) * (torch.pow(inner, alpha / 2.0) - 1.0)
            
        return torch.mean(loss)


class EMLLoss(nn.Module):
    """
    Estimated Maximum Likelihood Loss (Wang, Zhou, Lin, JASA 2024).
    Nonparametrically estimates the residual density using Gaussian Kernel:
      f_hat(r_i) = (1 / (n * h)) * sum_j K((r_i - r_j) / h)
    Loss = - (1 / n) * sum_i log(f_hat(r_i) + eps)
    Asymptotically achieves the oracle CRLB for arbitrary unknown error distributions!
    """
    def __init__(self, bandwidth=None, eps=1e-5):
        super().__init__()
        self.bandwidth = bandwidth
        self.eps = eps

    def forward(self, y_pred, y_true):
        # residuals r_i = y_true - y_pred
        r = (y_true - y_pred).view(-1, 1) # [n, 1]
        n = r.shape[0]
        
        # Adaptive bandwidth if not fixed: Silverman's rule of thumb
        if self.bandwidth is None:
            r_det = r.detach()
            std_r = torch.std(r_det) + 1e-6
            iqr_r = (torch.quantile(r_det, 0.75) - torch.quantile(r_det, 0.25)) / 1.349 + 1e-6
            h = 0.9 * torch.min(std_r, iqr_r) * (n ** (-0.2))
            h = torch.clamp(h, min=0.05, max=5.0)
        else:
            h = torch.tensor(self.bandwidth, device=r.device)
            
        # Pairwise differences [n, n]
        diff = (r - r.t()) / h
        # Gaussian kernel: (1 / sqrt(2*pi)) * exp(-diff^2 / 2)
        inv_sqrt_2pi = 0.3989422804014327
        K = inv_sqrt_2pi * torch.exp(-0.5 * (diff**2))
        
        # f_hat at each r_i: mean across column j
        f_hat = torch.mean(K, dim=1) / h
        
        # Negative log-likelihood
        loss = -torch.mean(torch.log(torch.clamp(f_hat, min=self.eps)))
        return loss
