# Author: Harsh
# Module: adaptive_selector.py
# Residual-Calibrated Objective Selection (RCOS).
# Diagnoses calibration residuals to select/adapt the regression loss function.

import numpy as np
import torch
from scipy import stats
from models import (
    RegressionMLP, train_mlp,
    mse_loss, mae_loss, huber_loss,
    student_t_nll, centered_gamma_nll
)

class ResidualCalibratedSelector:
    """
    RCOS: Residual-Calibrated Objective Selector
    
    1. Splits training data into fit (75%) and calibration (25%) sets.
    2. Trains a pilot model on fit set using Huber loss.
    3. Evaluates residuals on the held-out calibration set.
    4. Computes shape diagnostics (skewness, kurtosis, MAD, tail ratio).
    5. Performs hypothesis testing for Gaussianity vs heavy-tail vs skewness.
    6. Selects and calibrates the optimal loss function.
    7. Retrains final model on full training data with selected loss.
    """
    def __init__(self, cal_ratio=0.25, pilot_epochs=200, final_epochs=300):
        self.cal_ratio = cal_ratio
        self.pilot_epochs = pilot_epochs
        self.final_epochs = final_epochs
        self.diagnostics = {}
        self.selected_family = None
        self.selected_loss_fn = None
        self.calibrated_params = {}

    def diagnose_residuals(self, r):
        """
        Computes formal statistical diagnostics on calibration residuals r.
        """
        r = r.flatten()
        n = len(r)
        
        mean_r = np.mean(r)
        std_r = np.std(r, ddof=1) + 1e-8
        med_r = np.median(r)
        
        # Median Absolute Deviation (MAD)
        mad = stats.median_abs_deviation(r, scale="normal") + 1e-8
        tail_ratio = std_r / mad  # ~1.0 for Gaussian, >1.25 for heavy tails
        
        # Sample skewness and excess kurtosis
        skew = stats.skew(r)
        kurt = stats.kurtosis(r) # excess kurtosis (0 for normal)
        
        # Statistical hypothesis tests
        # D'Agostino skewness test
        try:
            _, p_skew = stats.skewtest(r)
        except Exception:
            p_skew = 1.0 if abs(skew) < 0.5 else 0.01
            
        # Jarque-Bera test for composite normality
        try:
            _, p_jb = stats.jarque_bera(r)
        except Exception:
            p_jb = 1.0 if (abs(skew) < 0.5 and kurt < 1.0) else 0.01

        return {
            "n": n,
            "mean": mean_r,
            "std": std_r,
            "median": med_r,
            "mad": mad,
            "tail_ratio": tail_ratio,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "p_skew": p_skew,
            "p_jb": p_jb
        }

    def select_objective(self, diag):
        """
        Principled decision tree based on statistical hypothesis tests.
        """
        skew = diag["skewness"]
        kurt = diag["excess_kurtosis"]
        tail_ratio = diag["tail_ratio"]
        p_jb = diag["p_jb"]
        p_skew = diag["p_skew"]
        std_r = diag["std"]
        mad = diag["mad"]
        
        # 1. Test for Normality (Gaussian Baseline)
        # If residuals do not reject normality, use MSE!
        if (p_jb > 0.05 or (abs(skew) < 0.40 and kurt < 0.8 and tail_ratio < 1.22)):
            self.selected_family = "Gaussian (MSE)"
            self.calibrated_params = {"sigma": std_r}
            return mse_loss

        # 2. Test for Significant Asymmetry / Skewness
        # If skewness is significant, MSE is inefficient and MAE is biased
        if (abs(skew) >= 0.45 and p_skew < 0.05):
            self.selected_family = "Centered Gamma NLL"
            # Calibrate Gamma shape k from skewness: skew = 2 / sqrt(k) => k = 4 / skew^2
            k_cal = float(np.clip(4.0 / (skew**2), 1.2, 20.0))
            theta_cal = float(std_r / np.sqrt(k_cal))
            self.calibrated_params = {"k": k_cal, "theta": theta_cal}
            
            def adaptive_gamma_loss(y_pred, y_true):
                return centered_gamma_nll(y_pred, y_true, k=k_cal, theta=theta_cal)
            return adaptive_gamma_loss

        # 3. Test for Heavy Tails (Symmetric with Kurtosis > 0.8 or Tail Ratio > 1.22)
        if (kurt >= 0.8 or tail_ratio >= 1.22):
            self.selected_family = "Student-t NLL"
            # Calibrate nu degrees of freedom via kurtosis or MLE
            # kurt = 6 / (nu - 4) => nu = 4 + 6 / kurt
            if kurt > 0.5:
                nu_cal = float(np.clip(4.0 + 6.0 / kurt, 2.2, 15.0))
            else:
                nu_cal = 4.0
            scale_cal = float(mad)
            self.calibrated_params = {"nu": nu_cal, "scale": scale_cal}
            
            def adaptive_t_loss(y_pred, y_true):
                return student_t_nll(y_pred, y_true, nu=nu_cal, scale=scale_cal)
            return adaptive_t_loss

        # Default fallback: Huber loss calibrated to MAD
        self.selected_family = "Calibrated Huber"
        delta_cal = float(1.345 * mad)
        self.calibrated_params = {"delta": delta_cal}
        def adaptive_huber_loss(y_pred, y_true):
            return huber_loss(y_pred, y_true, delta=delta_cal)
        return adaptive_huber_loss

    def fit_and_select(self, X_train, y_train):
        """
        Executes the full adaptation pipeline:
        Pilot -> Residual Diagnosis -> Loss Selection -> Full Retraining.
        """
        n_total = len(X_train)
        n_cal = max(15, int(n_total * self.cal_ratio))
        n_fit = n_total - n_cal
        
        # Split into fit and calibration subsets
        perm = np.random.permutation(n_total)
        fit_idx = perm[:n_fit]
        cal_idx = perm[n_fit:]
        
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]
        
        # Stage 1: Pilot model using neutral Huber loss
        pilot_model = RegressionMLP()
        pilot_model = train_mlp(
            pilot_model, X_fit, y_fit,
            loss_fn=lambda p, t: huber_loss(p, t, delta=0.40),
            epochs=self.pilot_epochs
        )
        
        # Stage 2: Residuals on held-out calibration set
        with torch.no_grad():
            y_cal_pred = pilot_model(torch.as_tensor(X_cal, dtype=torch.float32)).numpy()
        residuals_cal = y_cal - y_cal_pred
        
        # Stage 3: Diagnosis & Objective Selection
        self.diagnostics = self.diagnose_residuals(residuals_cal)
        self.selected_loss_fn = self.select_objective(self.diagnostics)
        
        # Stage 4: Retrain final model on the FULL training set
        final_model = RegressionMLP()
        final_model = train_mlp(
            final_model, X_train, y_train,
            loss_fn=self.selected_loss_fn,
            epochs=self.final_epochs
        )
        
        return final_model, self.selected_family, self.diagnostics, self.calibrated_params

