# Author: Harsh
# Module: utils.py
# Checkpointing utilities, deterministic seeding, and dataset loading pipeline.

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

from ground_truth import f_star, sample_noise

def set_seed(seed=42):
    """
    Ensures exact reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model_checkpoint(model, filepath, optimizer=None, config=None, metrics=None):
    """
    Saves PyTorch model weights alongside architectural metadata and metrics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "metrics": metrics or {},
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "torch_version": torch.__version__
    }
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint Saved] -> {filepath}")

def load_model_checkpoint(filepath, model=None, device="cpu"):
    """
    Loads checkpoint from disk. If model is provided, restores weights.
    Returns (model, config, metrics).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    metrics = checkpoint.get("metrics", {})
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
    print(f"[Checkpoint Loaded] <- {filepath}")
    return model, config, metrics

def load_dataset(dataset_name="synthetic", n_samples=300, noise_type="gaussian",
                 sigma=0.3, test_ratio=0.2, random_state=42, **kwargs):
    """
    Universal dataset loader supporting:
      1. 'synthetic': 1D ground truth f*(x) = sin(2*pi*x) + 0.5*x
      2. 'synthetic_2d': 2D ground truth f*(x1, x2) = sin(pi*x1)*cos(pi*x2) + 0.5*x1^2
      3. 'claims': French Motor Liability Claims (freMTPL2sev + freMTPL2freq)
      4. 'boston': Boston Housing Prices
    
    Returns:
      (X_train, y_train, X_test, y_test, metadata)
    """
    set_seed(random_state)
    metadata = {"dataset": dataset_name, "noise_type": noise_type}

    if dataset_name == "synthetic":
        X = np.random.uniform(0.0, 1.0, size=(n_samples, 1)).astype(np.float32)
        y_clean = f_star(X).astype(np.float32)
        w = sample_noise(noise_type, size=(n_samples, 1), sigma=sigma, **kwargs).astype(np.float32)
        y = y_clean + w
        
        # Fixed evaluation test grid
        n_eval = 500
        X_test = np.linspace(0.0, 1.0, n_eval)[:, None].astype(np.float32)
        y_test = f_star(X_test).astype(np.float32) # Clean ground truth for ISE evaluation
        metadata["y_test_is_clean"] = True
        return X, y, X_test, y_test, metadata

    elif dataset_name == "synthetic_2d":
        X = np.random.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
        y_clean = (np.sin(np.pi * X[:, 0:1]) * np.cos(np.pi * X[:, 1:2]) + 0.5 * (X[:, 0:1]**2)).astype(np.float32)
        w = sample_noise(noise_type, size=(n_samples, 1), sigma=sigma, **kwargs).astype(np.float32)
        y = y_clean + w
        
        # Fixed 2D evaluation grid
        grid_1d = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
        gx, gy = np.meshgrid(grid_1d, grid_1d)
        X_test = np.column_stack([gx.ravel(), gy.ravel()])
        y_test = (np.sin(np.pi * X_test[:, 0:1]) * np.cos(np.pi * X_test[:, 1:2]) + 0.5 * (X_test[:, 0:1]**2)).astype(np.float32)
        metadata["y_test_is_clean"] = True
        return X, y, X_test, y_test, metadata

    elif dataset_name == "claims":
        print("Fetching and preprocessing freMTPL2 datasets from OpenML...")
        sev = fetch_openml(data_id=41215, as_frame=True).frame
        freq = fetch_openml(data_id=41214, as_frame=True).frame
        df = pd.merge(sev, freq, on="IDpol", how="inner")
        
        feature_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
        X_df = df[feature_cols].copy()
        X_df["Density"] = np.log1p(X_df["Density"])
        
        y = df["ClaimAmount"].to_numpy().astype(np.float32)[:, None]
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df).astype(np.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state
        )
        metadata["y_test_is_clean"] = False
        metadata["scaler"] = scaler
        return X_train, y_train, X_test, y_test, metadata

    elif dataset_name == "boston":
        print("Fetching Boston Housing dataset from OpenML...")
        boston = fetch_openml(name="boston", version=1, as_frame=True).frame
        X_df = boston.drop(columns=["MEDV"]).select_dtypes(include=[np.number])
        y = boston["MEDV"].to_numpy().astype(np.float32)[:, None]
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state
        )
        metadata["y_test_is_clean"] = False
        return X_train, y_train, X_test, y_test, metadata

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Options: 'synthetic', 'synthetic_2d', 'claims', 'boston'.")

