"""
SHAP Analysis for Tier 3 and Tier 4 Handcrafted/Biomarker Streams
==================================================================
Provides TreeExplainer for XGBoost/LightGBM/CatBoost models (Tier 3)
and DeepExplainer / KernelExplainer for the HandcraftedStream MLP (Tier 4).

Usage example:
    from src.explainability.shap_analysis import (
        explain_xgboost, explain_handcrafted_stream, plot_shap_summary)

    shap_values, X_shap = explain_xgboost(model, X_val, feature_names)
    plot_shap_summary(shap_values, X_shap, feature_names, "results/shap_summary.png")
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("[SHAP] shap not installed. Install with: pip install shap")

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — TreeExplainer (XGBoost / LightGBM / CatBoost)
# ─────────────────────────────────────────────────────────────────────────────

def explain_xgboost(
    model,                          # trained XGB/LGB/CatBoost model
    X: np.ndarray,
    feature_names: list[str],
    n_background: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes SHAP values for a tree model using TreeExplainer.

    Returns:
        shap_values : [N, n_features]  (for positive class)
        X_explain   : [N, n_features]  (the input data)
    """
    assert _HAS_SHAP, "Install shap: pip install shap"

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classifiers, shap_values may be list of [N, F] per class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # positive class

    print(f"[SHAP TreeExplainer] Computed SHAP values: shape {shap_values.shape}")
    return shap_values, X


def explain_handcrafted_stream(
    handcrafted_stream: nn.Module,
    classification_head: nn.Module,
    X_background: np.ndarray,       # background samples for KernelExplainer
    X_explain: np.ndarray,          # samples to explain
    feature_names: list[str],
    n_background: int = 50,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes SHAP values for the HandcraftedStream MLP using KernelExplainer.
    We pass the handcrafted_stream embedding output through the classification_head
    to correctly yield 2-class probabilities.

    Returns:
        shap_values : [N, n_features]
        X_explain   : input data (passed through)
    """
    assert _HAS_SHAP, "Install shap: pip install shap"

    handcrafted_stream = handcrafted_stream.eval().to(device)
    classification_head = classification_head.eval().to(device)

    def model_fn(x: np.ndarray) -> np.ndarray:
        """Wrapper: numpy → torch → numpy (positive class proba)."""
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32, device=device)
            embedding = handcrafted_stream(t)
            logits = classification_head(embedding)
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return probs.cpu().numpy()

    # Use a small background set
    bg = X_background[:n_background]

    print(f"[SHAP KernelExplainer] Running on {len(X_explain)} samples "
          f"(background={len(bg)})... (this may take a few minutes)")

    explainer = shap.KernelExplainer(model_fn, bg)
    shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)

    print(f"[SHAP KernelExplainer] Done. SHAP shape: {np.array(shap_values).shape}")
    return np.array(shap_values), X_explain


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str,
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
):
    """
    Saves a SHAP beeswarm + bar plot to output_path.
    """
    assert _HAS_SHAP, "Install shap: pip install shap"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Beeswarm
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=max_display, show=False)
    axes[0].set_title("SHAP Beeswarm", fontsize=11)

    # Bar (mean |SHAP|)
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type="bar", max_display=max_display, show=False)
    axes[1].set_title("Mean |SHAP| (Feature Importance)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] Saved summary plot to {output_path}")


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str,
    sample_idx: int = 0,
    title: str = "SHAP Waterfall — Single Subject",
):
    """Waterfall plot for a single sample (case study)."""
    assert _HAS_SHAP, "Install shap: pip install shap"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=shap_values.mean(),
        data=X[sample_idx],
        feature_names=feature_names,
    )

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(explanation, max_display=20, show=False)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] Saved waterfall plot to {output_path}")


def compute_mean_shap_by_group(
    shap_values: np.ndarray,
    feature_names: list[str],
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Aggregates mean |SHAP| by feature group (e.g. 'fixation', 'saccade', 'delta').

    Args:
        groups : {'group_name': [feature_col_1, ...]}

    Returns:
        DataFrame with columns ['group', 'mean_abs_shap']
    """
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    rows = []
    for group_name, feat_list in groups.items():
        idxs = [feat_to_idx[f] for f in feat_list if f in feat_to_idx]
        if not idxs:
            continue
        group_shap = np.abs(shap_values[:, idxs]).mean()
        rows.append({"group": group_name, "mean_abs_shap": group_shap})

    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
