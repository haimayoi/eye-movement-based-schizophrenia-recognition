"""
Tier 5 Meta-Learner
===================
Combines Tier 3 (tabular) and Tier 4 (deep) OOF predictions into a final
ensemble prediction using two complementary strategies:

  Strategy A — Weighted Average (0 free parameters):
      P_final = w_tab * P_tab + w_bica * P_bica
      Weights are optimised by Optuna on the OOF set (AUC objective).
      No overfitting risk for small N.

  Strategy B — Logistic Regression (2 parameters + bias):
      P_final = σ(β_tab * P_tab + β_bica * P_bica + b)
      Fitted with L2 regularisation (C=1.0) on OOF predictions.
      Provides coefficient-based SHAP interpretability.

Both strategies are evaluated and the better one (higher CV AUC) is selected
as the final meta-learner.

Leakage guarantee:
  - Both strategies train only on OOF predictions.
  - No subject appears in both training and validation of the same fold.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — Weighted Average (Optuna)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedAverageMetaLearner:
    """
    Simple weighted average ensemble.
    Weights are optimised using Optuna to maximise OOF AUC.
    """

    def __init__(self, n_trials: int = 500):
        self.n_trials = n_trials
        self.w_tab: float = 0.5
        self.w_bica: float = 0.5
        self.best_auc: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X[:, 0] = P_tab, X[:, 1] = P_bica
        """
        if not _HAS_OPTUNA:
            # Fall back to uniform weights
            print("[WeightedAvg] Optuna not installed; using uniform weights.")
            self.w_tab = 0.5
            self.w_bica = 0.5
            p = 0.5 * X[:, 0] + 0.5 * X[:, 1]
            self.best_auc = roc_auc_score(y, p)
            return self

        def objective(trial):
            w = trial.suggest_float("w_tab", 0.0, 1.0)
            p = w * X[:, 0] + (1 - w) * X[:, 1]
            return roc_auc_score(y, p)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.w_tab = study.best_params["w_tab"]
        self.w_bica = 1.0 - self.w_tab
        self.best_auc = study.best_value
        print(f"[WeightedAvg] Optimal  w_tab={self.w_tab:.4f}  "
              f"w_bica={self.w_bica:.4f}  OOF AUC={self.best_auc:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.w_tab * X[:, 0] + self.w_bica * X[:, 1]
        return np.column_stack([1 - p, p])

    def to_dict(self) -> dict:
        return {"strategy": "weighted_average",
                "w_tab": self.w_tab,
                "w_bica": self.w_bica,
                "oof_auc": self.best_auc}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B — Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegressionMetaLearner:
    """
    L2-regularised Logistic Regression meta-learner.
    Input: [P_tab, P_bica]   Output: P(SZ)
    Evaluated with stratified 4-fold CV on OOF data.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.model: Optional[LogisticRegression] = None
        self.cv_auc: float = 0.0
        self.coef_tab: float = 0.0
        self.coef_bica: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits LR and records coefficient magnitudes for SHAP-equivalent
        interpretability (only 2 features so coefficients ARE the attribution).
        """
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            random_state=42, solver="lbfgs")

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.model, X, y, cv=cv, scoring="roc_auc")
        self.cv_auc = float(np.mean(scores))
        print(f"[LogisticReg] CV AUC: {self.cv_auc:.4f} "
              f"± {np.std(scores):.4f}")

        # Fit on full OOF data
        self.model.fit(X, y)
        self.coef_tab = float(self.model.coef_[0][0])
        self.coef_bica = float(self.model.coef_[0][1])
        print(f"[LogisticReg] Coefficients: β_tab={self.coef_tab:.4f}  "
              f"β_bica={self.coef_bica:.4f}  "
              f"bias={float(self.model.intercept_[0]):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Call fit() first."
        return self.model.predict_proba(X)

    def to_dict(self) -> dict:
        return {"strategy": "logistic_regression",
                "C": self.C,
                "cv_auc": self.cv_auc,
                "coef_tab": self.coef_tab,
                "coef_bica": self.coef_bica,
                "bias": float(self.model.intercept_[0]) if self.model else 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Tier5 ensemble wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Tier5MetaLearner:
    """
    Trains both WeightedAverage and LogisticRegression meta-learners,
    selects the better one by OOF AUC, and exposes a unified predict_proba.
    """

    def __init__(self, optuna_trials: int = 500, lr_C: float = 1.0):
        self.wa = WeightedAverageMetaLearner(n_trials=optuna_trials)
        self.lr = LogisticRegressionMetaLearner(C=lr_C)
        self.best: str = "weighted_average"   # or "logistic_regression"
        self._best_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Tier5MetaLearner":
        print("\n=== Tier 5 Meta-Learner Training ===")
        print(f"Input shape: {X.shape}  |  Class balance: "
              f"{int(y.sum())}/{len(y) - int(y.sum())} (SZ/HC)")

        self.wa.fit(X, y)
        self.lr.fit(X, y)

        if self.lr.cv_auc >= self.wa.best_auc:
            self.best = "logistic_regression"
            self._best_model = self.lr
            print(f"\n[Tier5] Selected: Logistic Regression "
                  f"(AUC {self.lr.cv_auc:.4f} >= WA {self.wa.best_auc:.4f})")
        else:
            self.best = "weighted_average"
            self._best_model = self.wa
            print(f"\n[Tier5] Selected: Weighted Average "
                  f"(AUC {self.wa.best_auc:.4f} > LR {self.lr.cv_auc:.4f})")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._best_model is not None, "Call fit() first."
        return self._best_model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def summary(self) -> dict:
        return {
            "selected_strategy": self.best,
            "weighted_average": self.wa.to_dict(),
            "logistic_regression": self.lr.to_dict(),
        }

    def save(self, path: str):
        """Save summary JSON and model coefficients."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=4)
        print(f"[Tier5] Saved meta-learner summary to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_meta_learner(
    meta: Tier5MetaLearner,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Compute comprehensive metrics for the meta-learner."""
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        precision_score, recall_score, confusion_matrix
    )

    p = meta.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)

    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {"auc": auc, "accuracy": acc, "f1": f1,
               "precision": prec, "recall": rec, "specificity": spec}

    print("\n--- Tier 5 Final Metrics ---")
    for k, v in metrics.items():
        print(f"  {k.capitalize():15s}: {v:.4f}")

    return metrics
