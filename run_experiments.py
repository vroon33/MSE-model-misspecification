#!/usr/bin/env python
# Author: Harsh
# Unified CLI Runner: Train and evaluate diverse neural models, loss functions, and datasets.

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Add NN_sim to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_sim"))
from utils import set_seed, save_model_checkpoint, load_model_checkpoint, load_dataset
from models import (
    RegressionMLP, ProbabilisticMLP,
    mse_loss, mae_loss, huber_loss,
    student_t_nll, centered_gamma_nll
)
from losses import CauchyLoss, BarronAdaptiveLoss, EMLLoss, QuantileLoss
from adaptive_selector import ResidualCalibratedSelector
from metrics import compute_ise

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Regression Testbed: Train and evaluate neural networks under various loss functions and datasets."
    )
    # Dataset & Noise
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "synthetic_2d", "claims", "boston"],
                        help="Dataset name to train and evaluate on.")
    parser.add_argument("--noise", type=str, default="gaussian",
                        choices=["gaussian", "gamma", "student_t", "laplace", "contaminated"],
                        help="Observation noise type for synthetic datasets.")
    parser.add_argument("--sigma", type=float, default=0.3, help="Noise standard deviation.")
    parser.add_argument("--nu", type=float, default=3.0, help="Degrees of freedom for Student-t noise.")
    parser.add_argument("--k", type=float, default=3.0, help="Shape parameter for Gamma noise.")
    parser.add_argument("--n_samples", type=int, default=300, help="Number of training samples (synthetic).")
    
    # Model Architecture
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "deep_mlp", "probabilistic_mlp"],
                        help="Neural network architecture.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers.")
    
    # Loss Function
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "mae", "huber", "cauchy", "barron", "eml", "quantile", "rcos"],
                        help="Training loss objective.")
    parser.add_argument("--delta", type=float, default=0.4, help="Huber delta threshold.")
    parser.add_argument("--tau", type=float, default=0.5, help="Quantile level for quantile loss.")
    
    # Optimization Hyperparameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 for full-batch gradient descent).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for exact reproducibility.")
    
    # Model Checkpointing & Execution
    parser.add_argument("--save_path", type=str, default=None, help="Filepath to save trained model checkpoint (.pt).")
    parser.add_argument("--load_path", type=str, default=None, help="Filepath to load pre-trained checkpoint (.pt).")
    parser.add_argument("--eval_only", action="store_true", help="Skip training; load checkpoint and evaluate on test set.")
    parser.add_argument("--plot", action="store_true", help="Save prediction visualization to figures/.")
    parser.add_argument("--plot_name", type=str, default=None, help="Custom filename for the output plot.")
    
    return parser.parse_args()


def build_model(model_name, in_features, hidden_dim, num_layers):
    if model_name == "mlp":
        return RegressionMLP(in_dim=in_features, hidden_dim=hidden_dim, num_hidden=num_layers)
    elif model_name == "deep_mlp":
        return RegressionMLP(in_dim=in_features, hidden_dim=hidden_dim, num_hidden=max(4, num_layers))
    elif model_name == "probabilistic_mlp":
        return ProbabilisticMLP(in_dim=in_features, hidden_dim=hidden_dim, num_hidden=num_layers)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    print("\n" + "=" * 75)
    print("NOISE-MODEL MISSPECIFICATION & ADAPTIVE REGRESSION RUNNER")
    print(f"Dataset: {args.dataset.upper()} | Model: {args.model.upper()} | Loss: {args.loss.upper()}")
    if args.dataset in ["synthetic", "synthetic_2d"]:
        print(f"Noise Regime: {args.noise.upper()} (sigma={args.sigma}, nu={args.nu}, k={args.k})")
    print("=" * 75)

    # 1. Load Data
    X_tr, y_tr, X_te, y_te, meta = load_dataset(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        noise_type=args.noise,
        sigma=args.sigma,
        nu=args.nu,
        k=args.k,
        random_state=args.seed
    )
    in_dim = X_tr.shape[1]
    print(f"Loaded {len(X_tr)} training samples and {len(X_te)} testing samples. Feature dimension: {in_dim}")

    # 2. Build or Load Model
    model = build_model(args.model, in_dim, args.hidden_dim, args.num_layers)
    
    if args.load_path and os.path.exists(args.load_path):
        model, cfg, _ = load_model_checkpoint(args.load_path, model=model)
        if args.eval_only:
            print("[Mode: Evaluation Only]")
    elif args.eval_only:
        raise FileNotFoundError(f"Requested --eval_only but checkpoint {args.load_path} was not found.")

    # 3. Training Loop (if not eval_only)
    if not args.eval_only:
        if args.loss == "rcos":
            print("\nExecuting Residual-Calibrated Objective Selection (RCOS)...")
            rcos = ResidualCalibratedSelector(pilot_epochs=max(100, args.epochs // 2), final_epochs=args.epochs)
            model, chosen_family, diag, params = rcos.fit_and_select(X_tr, y_tr)
            print(f"RCOS Diagnostics -> Skewness: {diag['skewness']:.3f}, Kurtosis: {diag['excess_kurtosis']:.3f}")
            print(f"RCOS Decision -> Selected Family: {chosen_family} with parameters: {params}")
        else:
            # Instantiate loss function
            extra_params = []
            if args.loss == "mse":
                criterion = mse_loss
            elif args.loss == "mae":
                criterion = mae_loss
            elif args.loss == "huber":
                criterion = lambda p, t: huber_loss(p, t, delta=args.delta)
            elif args.loss == "cauchy":
                loss_module = CauchyLoss(c=args.sigma)
                criterion = loss_module
            elif args.loss == "barron":
                loss_module = BarronAdaptiveLoss(init_alpha=2.0, c=args.sigma)
                extra_params = list(loss_module.parameters())
                criterion = loss_module
            elif args.loss == "eml":
                criterion = EMLLoss()
            elif args.loss == "quantile":
                criterion = QuantileLoss(tau=args.tau)
            else:
                raise ValueError(f"Unknown loss: {args.loss}")

            # Optimizer & Training
            optimizer = optim.AdamW(list(model.parameters()) + extra_params, lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
            
            X_t = torch.as_tensor(X_tr, dtype=torch.float32)
            y_t = torch.as_tensor(y_tr, dtype=torch.float32)
            
            print(f"\nTraining {args.model.upper()} with {args.loss.upper()} for {args.epochs} epochs...")
            model.train()
            
            if args.batch_size > 0 and args.batch_size < len(X_tr):
                dataset = torch.utils.data.TensorDataset(X_t, y_t)
                loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                for ep in range(args.epochs):
                    for bx, by in loader:
                        optimizer.zero_grad()
                        pred = model(bx)
                        loss = criterion(pred, by)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                    scheduler.step()
            else:
                for ep in range(args.epochs):
                    optimizer.zero_grad()
                    pred = model(X_t)
                    loss = criterion(pred, y_t)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    scheduler.step()

    # 4. Out-of-Sample Evaluation
    model.eval()
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_te_t).numpy()

    metrics = {}
    if meta.get("y_test_is_clean", False):
        ise = compute_ise(y_pred, y_te)
        mae = float(np.mean(np.abs(y_pred - y_te)))
        max_err = float(np.max(np.abs(y_pred - y_te)))
        metrics = {"Test_ISE_vs_Truth": ise, "Test_MAE_vs_Truth": mae, "Max_Error": max_err}
        print(f"\n[Test Metrics vs True f*(x)] -> ISE: {ise:.5f} | MAE: {mae:.5f} | Max Error: {max_err:.5f}")
    else:
        test_mae = float(np.mean(np.abs(y_pred - y_te)))
        test_median_ae = float(np.median(np.abs(y_pred - y_te)))
        test_rmse = float(np.sqrt(np.mean((y_pred - y_te)**2)))
        metrics = {"Test_MAE": test_mae, "Test_Median_AE": test_median_ae, "Test_RMSE": test_rmse}
        print(f"\n[Test Metrics] -> MAE: {test_mae:.3f} | Median AE: {test_median_ae:.3f} | RMSE: {test_rmse:.3f}")

    # 5. Checkpointing
    if args.save_path:
        save_model_checkpoint(
            model=model,
            filepath=args.save_path,
            config=vars(args),
            metrics=metrics
        )

    # 6. Optional Visualization
    if args.plot and in_dim == 1:
        plt.figure(figsize=(8, 4.5))
        plt.plot(X_te, y_te, "k-", linewidth=2.0, label="True f*(x)" if meta.get("y_test_is_clean") else "Test Targets")
        plt.plot(X_te, y_pred, "r--", linewidth=1.8, label=f"Prediction ({args.loss.upper()})")
        plt.scatter(X_tr[:100], y_tr[:100], color="gray", alpha=0.4, s=15, label="Train samples (subset)")
        plt.title(f"Model: {args.model.upper()} | Loss: {args.loss.upper()} | Dataset: {args.dataset}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plot_file = args.plot_name or f"figures/eval_{args.dataset}_{args.loss}.png"
        plt.savefig(plot_file, dpi=200)
        print(f"[Plot Saved] -> {plot_file}")

    print("Experiment completed successfully!\n")


if __name__ == "__main__":
    main()

