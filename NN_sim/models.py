# Author: Harsh
# Module: models.py
# Neural network architecture, loss objectives, and training pipelines.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RegressionMLP(nn.Module):
    """
    Standard Feedforward Neural Network (FNN / MLP).
    Architecture: 1 -> 64 -> 64 -> 64 -> 1 with GELU activations.
    """
    def __init__(self, in_dim=1, hidden_dim=64, num_hidden=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Consistent Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ProbabilisticMLP(nn.Module):
    """
    Probabilistic MLP outputting both location mu(x) and log_scale s(x).
    """
    def __init__(self, in_dim=1, hidden_dim=64, num_hidden=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        self.backbone = nn.Sequential(*layers)
        self.head_mu = nn.Linear(hidden_dim, 1)
        self.head_log_scale = nn.Linear(hidden_dim, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.head_mu(h)
        log_scale = self.head_log_scale(h)
        return mu, log_scale


# ---------------- Loss Objectives ----------------

def mse_loss(y_pred, y_true):
    """Gaussian Maximum Likelihood Loss"""
    return torch.mean((y_pred - y_true)**2)

def mae_loss(y_pred, y_true):
    """Laplace Maximum Likelihood Loss"""
    return torch.mean(torch.abs(y_pred - y_true))

def huber_loss(y_pred, y_true, delta=0.40):
    """Huber Robust M-Estimator Loss"""
    diff = torch.abs(y_pred - y_true)
    quadratic = torch.clamp(diff, max=delta)
    linear = diff - quadratic
    return torch.mean(0.5 * (quadratic ** 2) + delta * linear)

def student_t_nll(y_pred, y_true, nu=3.0, scale=0.3):
    """
    Matched Student-t Negative Log-Likelihood Loss.
    Scale s = scale * sqrt((nu-2)/nu)
    """
    s = scale * np.sqrt((nu - 2.0) / nu) if nu > 2 else scale
    r = (y_true - y_pred) / s
    # - ln p_t(r) = 0.5*(nu+1) * ln(1 + r^2/nu) + const
    return torch.mean(0.5 * (nu + 1.0) * torch.log1p((r ** 2) / nu))

def centered_gamma_nll(y_pred, y_true, k=3.0, theta=0.1732, eps=1e-5):
    """
    Matched Centered Gamma Negative Log-Likelihood Loss.
    w = Y - k*theta, so Y = (y_true - y_pred) + k*theta > 0
    - ln p_Y(Y) = -(k-1)*ln(Y) + Y/theta + const
    Smooth barrier (softplus) used outside positive support to maintain stable gradient.
    """
    c = k * theta
    shift = y_true - y_pred + c
    safe_shift = torch.nn.functional.softplus(shift, beta=10.0) + eps
    nll = -(k - 1.0) * torch.log(safe_shift) + (safe_shift / theta)
    # Penalty for violating support boundary
    violation = torch.clamp(-shift + eps, min=0.0)
    penalty = 100.0 * torch.mean(violation ** 2)
    return torch.mean(nll) + penalty


# ---------------- Training Routine ----------------

def train_mlp(model, X_train, y_train, loss_fn, epochs=300, lr=0.003, weight_decay=1e-4):
    """
    Trains a RegressionMLP on (X_train, y_train) using loss_fn.
    """
    device = torch.device("cpu")
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    X_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_t)
        loss = loss_fn(y_pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
    model.eval()
    return model

