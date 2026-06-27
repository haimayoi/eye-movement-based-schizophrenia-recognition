"""
Calibration Plots — Reliability Diagram, ECE, Brier Score, Decision Curve Analysis
====================================================================================
Required for IEEE JBHI / clinical AI papers (2020+).

Decision Curve Analysis (DCA) is mandatory for any paper claiming clinical utility.
It answers: "At what clinical decision threshold is this model useful, relative to
treating all patients or treating none?"

Reference:
    Vickers AJ, Elkin EB. (2006). Decision curve analysis: a novel method for
    evaluating prediction models. Medical Decision Making, 26(6), 565-574.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error (ECE)
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes ECE: weighted average of |accuracy − confidence| over bins.
    Lower is better. Perfect calibration = 0.0.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Brier Score
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier Score = mean squared error of probability predictions.
    Range [0, 1]. Lower is better. Baseline (no-skill): 0.25 for balanced binary.

    Recommended for IEEE JBHI: reports calibration quality independently of
    threshold choice and accounts for both discrimination and calibration.
    """
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier Skill Score (BSS): improvement over climatological baseline.
    BSS = 1 − BS / BS_ref, where BS_ref = prevalence * (1 − prevalence).
    BSS = 0: no skill. BSS = 1: perfect. BSS < 0: worse than baseline.
    """
    bs = brier_score(y_true, y_prob)
    prevalence = y_true.mean()
    bs_ref = prevalence * (1 - prevalence)
    if bs_ref == 0:
        return 0.0
    return float(1.0 - bs / bs_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler:
    """
    Post-hoc temperature scaling.
    Scales logits by 1/T before sigmoid: T>1 = softer, T<1 = harder.
    Used AFTER classification to calibrate P_final, not as a classifier itself.
    """

    def __init__(self):
        self.T: float = 1.0

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray,
            n_grid: int = 200) -> "TemperatureScaler":
        from scipy.special import logit, expit

        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logits = logit(y_prob_clipped)

        best_T, best_nll = 1.0, float('inf')
        for T in np.linspace(0.1, 5.0, n_grid):
            scaled = expit(logits / T)
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            nll = -np.mean(
                y_true * np.log(scaled) + (1 - y_true) * np.log(1 - scaled))
            if nll < best_nll:
                best_nll = nll
                best_T = T

        self.T = best_T
        print(f"[TemperatureScaler] Optimal T={self.T:.4f}  NLL={best_nll:.4f}")
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        from scipy.special import logit, expit
        return expit(logit(np.clip(y_prob, 1e-7, 1 - 1e-7)) / self.T)


# ─────────────────────────────────────────────────────────────────────────────
# Decision Curve Analysis (DCA) — MANDATORY for IEEE JBHI
# ─────────────────────────────────────────────────────────────────────────────

def net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """
    Net Benefit at a single decision threshold pt:
        NB(pt) = (TP/N) − (FP/N) × (pt / (1 − pt))

    Interpretation: at threshold pt (treating if P(SZ) ≥ pt),
    NB > 0 means the model provides clinical value.
    NB compared to "treat all" and "treat none" baselines.

    Args:
        y_true     : binary labels [N]
        y_prob     : predicted probabilities [N]
        threshold  : decision threshold (0, 1)

    Returns:
        net benefit at this threshold
    """
    n = len(y_true)
    if threshold <= 0 or threshold >= 1:
        return 0.0
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


def decision_curve_analysis(
    y_true: np.ndarray,
    pred_dict: dict,           # {label: y_prob_array}
    thresholds: np.ndarray = None,
    output_path: str = "results/dca.png",
    title: str = "Decision Curve Analysis",
) -> dict:
    """
    Computes and plots Decision Curve Analysis for multiple models.

    Plots:
      - Net benefit curve for each model vs threshold
      - "Treat all" baseline: NB(pt) = prevalence − (1−prevalence) × pt/(1−pt)
      - "Treat none" baseline: NB = 0 for all pt

    Clinical interpretation:
      A model curve ABOVE "treat all" and "treat none" indicates net clinical
      benefit at that threshold. The range of thresholds where a model dominates
      both baselines defines its clinically useful range.

    Args:
        pred_dict : {model_name: predicted_probability_array}

    Returns:
        nb_results : {model_name: array of net benefits at each threshold}
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    n = len(y_true)
    prevalence = float(y_true.mean())

    # "Treat all" NB = TP_rate − FP_rate × pt/(1−pt)
    # When treating all: TP_rate = prevalence, FP_rate = 1 − prevalence
    treat_all_nb = np.array([
        prevalence - (1 - prevalence) * (t / (1 - t))
        for t in thresholds
    ])
    treat_none_nb = np.zeros_like(thresholds)

    nb_results = {}
    for label, y_prob in pred_dict.items():
        nb_results[label] = np.array([
            net_benefit(y_true, y_prob, t) for t in thresholds
        ])

    # ── Plot ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baselines
    ax.plot(thresholds, treat_none_nb, 'k-', linewidth=1.5,
            label="Treat None (NB = 0)", alpha=0.7)
    ax.plot(thresholds, np.clip(treat_all_nb, -0.1, None), 'k--',
            linewidth=1.5, label=f"Treat All (prevalence={prevalence:.2f})",
            alpha=0.7)

    # Model curves
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(pred_dict)))
    for (label, nb_vals), color in zip(nb_results.items(), colors):
        ax.plot(thresholds, np.clip(nb_vals, -0.1, None),
                linewidth=2, color=color, label=label)

    ax.set_xlabel("Decision Threshold (pt)", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, prevalence + 0.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

    # Annotation
    ax.text(0.98, 0.98,
            f"Prevalence (SZ rate) = {prevalence:.2f}\n"
            f"N = {n} subjects",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[DCA] Saved Decision Curve Analysis to {output_path}")

    return nb_results


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram (calibration curve)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    y_true: np.ndarray,
    pred_dict: dict,       # {label: y_prob_array}
    output_path: str,
    n_bins: int = 10,
    title: str = "Calibration (Reliability Diagram)",
) -> dict:
    """
    Plots reliability diagram + prediction distribution for multiple models.
    Also computes ECE and Brier Score for each model.

    Returns:
        {model_label: {"ece": float, "brier": float, "bss": float}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax_cal, ax_hist = axes

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(pred_dict)))
    score_results = {}

    for (label, y_prob), color in zip(pred_dict.items(), colors):
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform')
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        bs = brier_score(y_true, y_prob)
        bss = brier_skill_score(y_true, y_prob)

        score_results[label] = {"ece": ece, "brier": bs, "bss": bss}

        ax_cal.plot(mean_pred, frac_pos, 'o-', color=color,
                    label=f"{label}\nECE={ece:.3f}  BS={bs:.3f}  BSS={bss:.3f}",
                    linewidth=2, markersize=6)
        ax_hist.hist(y_prob, bins=20, alpha=0.5, color=color,
                     label=label, density=True)

    ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Perfect calibration")
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax_cal.set_ylabel("Fraction of Positives (SZ)", fontsize=12)
    ax_cal.set_title("Reliability Diagram", fontsize=12)
    ax_cal.legend(fontsize=8)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)

    ax_hist.set_xlabel("Predicted Probability", fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.set_title("Prediction Distribution", fontsize=12)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Calibration] Saved reliability diagram to {output_path}")

    return score_results
