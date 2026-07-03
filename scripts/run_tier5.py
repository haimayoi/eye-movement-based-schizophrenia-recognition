"""
Tier 5 — Meta-Learner Training & Evaluation Script
====================================================
Combines OOF predictions from Tier 3 (tabular) and Tier 4 (deep)
to train a final ensemble meta-learner.

Usage:
    python scripts/run_tier5.py \\
        --tier3-oof results/baselines/xgboost_oof_subject_preds.csv \\
        --tier4-oof results/cefam/cefam_oof_subject_preds.csv \\
        --output-dir results/tier5/

Optional:
    --tier3-test results/baselines/xgboost_test_predictions.csv \\
    --tier4-test results/cefam/cefam_test_predictions.csv \\
    --calibrate         (apply temperature scaling)
    --plot              (generate all visualisations)
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tier5_fusion.oof_collector import (
    load_oof_predictions,
    load_test_predictions,
    assert_no_test_train_overlap,
)
from src.tier5_fusion.meta_learner import Tier5MetaLearner, evaluate_meta_learner
from src.tier5_fusion.calibration import (
    TemperatureScaler,
    plot_reliability_diagram,
    expected_calibration_error,
    brier_score,
    brier_skill_score,
    decision_curve_analysis,
)
from src.explainability.embedding_viz import plot_tier5_decision_space


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tier 5: Meta-Learner over Tier3 + Tier4 OOF predictions")
    parser.add_argument(
        "--tier3-oof",
        default="results/baselines/xgboost_oof_subject_preds.csv",
        help="Path to Tier 3 OOF subject predictions CSV")
    parser.add_argument(
        "--tier4-oof",
        default="results/cefam/cefam_oof_subject_preds.csv",
        help="Path to Tier 4 OOF subject predictions CSV")
    parser.add_argument(
        "--tier3-test",
        default="results/baselines/xgboost_test_predictions.csv",
        help="Path to Tier 3 test predictions CSV (optional)")
    parser.add_argument(
        "--tier4-test",
        default="results/cefam/cefam_test_predictions.csv",
        help="Path to Tier 4 test predictions CSV (optional)")
    parser.add_argument(
        "--output-dir",
        default="results/tier5/",
        help="Directory for outputs")
    parser.add_argument(
        "--optuna-trials", type=int, default=500,
        help="Optuna trials for weighted average optimisation")
    parser.add_argument(
        "--lr-C", type=float, default=1.0,
        help="L2 regularisation strength for Logistic Regression (C=1/λ)")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Apply temperature scaling calibration")
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate all visualisation plots")
    return parser.parse_args()


def evaluate_probabilities(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        precision_score, recall_score, confusion_matrix
    )
    pred = (p >= threshold).astype(int)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, zero_division=0)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"auc": auc, "accuracy": acc, "f1": f1,
            "precision": prec, "recall": rec, "specificity": spec}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 65)
    print(" TIER 5 META-LEARNER")
    print("=" * 65)

    # ── Load OOF predictions ───────────────────────────────────────────────
    df_meta, X_oof, y_oof = load_oof_predictions(
        tier3_oof_path=args.tier3_oof,
        tier4_oof_path=args.tier4_oof,
    )

    # ── Leakage verification ───────────────────────────────────────────────
    # Load test predictions to check for overlap
    df_test, X_test = load_test_predictions(
        tier3_test_path=args.tier3_test,
        tier4_test_path=args.tier4_test,
    )

    train_subjects = set(df_meta['Subject_ID'].astype(int).tolist())
    if df_test is not None:
        # EMS test set has Label=-1 (no ground truth released) — skip leakage
        # check and test evaluation; ID collision with training subjects is
        # expected since train/test subject numbering is independent.
        if 'Label' in df_test.columns and df_test['Label'].eq(-1).all():
            print("[Test] Test labels are unknown (Label=-1) — "
                  "skipping test evaluation.")
            df_test = None
            X_test = None
        else:
            test_subjects = set(df_test['Subject_ID'].astype(int).tolist())
            assert_no_test_train_overlap(train_subjects, test_subjects)

    # ── Print baseline metrics (Tier 3 and Tier 4 independently) ──────────
    print("\n--- Baseline: Tier 3 alone (XGBoost/LGB/CatBoost) ---")
    auc_t3 = roc_auc_score(y_oof, X_oof[:, 0])
    acc_t3 = ((X_oof[:, 0] >= 0.5).astype(int) == y_oof).mean()
    bs_t3 = brier_score(y_oof, X_oof[:, 0])
    bss_t3 = brier_skill_score(y_oof, X_oof[:, 0])
    ece_t3 = expected_calibration_error(y_oof, X_oof[:, 0])
    print(f"  AUC: {auc_t3:.4f}  ACC: {acc_t3:.4f}  Brier: {bs_t3:.4f}  BSS: {bss_t3:.4f}  ECE: {ece_t3:.4f}")

    print("\n--- Baseline: Tier 4 alone (BiCA-HS / CEFAM) ---")
    auc_t4 = roc_auc_score(y_oof, X_oof[:, 1])
    acc_t4 = ((X_oof[:, 1] >= 0.5).astype(int) == y_oof).mean()
    bs_t4 = brier_score(y_oof, X_oof[:, 1])
    bss_t4 = brier_skill_score(y_oof, X_oof[:, 1])
    ece_t4 = expected_calibration_error(y_oof, X_oof[:, 1])
    print(f"  AUC: {auc_t4:.4f}  ACC: {acc_t4:.4f}  Brier: {bs_t4:.4f}  BSS: {bss_t4:.4f}  ECE: {ece_t4:.4f}")

    # ── Train Tier 5 ───────────────────────────────────────────────────────
    meta = Tier5MetaLearner(
        optuna_trials=args.optuna_trials,
        lr_C=args.lr_C
    )
    meta.fit(X_oof, y_oof)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n--- Tier 5 Final Evaluation ---")
    p_oof = meta.predict_proba(X_oof)[:, 1]
    metrics = evaluate_probabilities(y_oof, p_oof)
    metrics['brier_score'] = brier_score(y_oof, p_oof)
    metrics['brier_skill_score'] = brier_skill_score(y_oof, p_oof)
    metrics['ece'] = expected_calibration_error(y_oof, p_oof)

    print("\n--- Uncalibrated Tier 5 Metrics ---")
    for k, v in metrics.items():
        print(f"  {k.capitalize():15s}: {v:.4f}")

    # Calibration
    scaler = None
    if args.calibrate:
        print("\n--- Calibration (Temperature Scaling) ---")
        scaler = TemperatureScaler()
        scaler.fit(y_oof, p_oof)
        p_oof_cal = scaler.transform(p_oof)
        
        metrics_cal = evaluate_probabilities(y_oof, p_oof_cal)
        metrics_cal['brier_score'] = brier_score(y_oof, p_oof_cal)
        metrics_cal['brier_skill_score'] = brier_skill_score(y_oof, p_oof_cal)
        metrics_cal['ece'] = expected_calibration_error(y_oof, p_oof_cal)

        print("\n--- Calibrated Tier 5 Metrics ---")
        for k, v in metrics_cal.items():
            print(f"  {k.capitalize():15s}: {v:.4f}")

        # Update metrics and p_oof with calibrated values
        metrics = metrics_cal
        p_oof = p_oof_cal

    # ── ROC comparison ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for p, lbl, color in [
        (X_oof[:, 0], f"Tier 3 Tabular (AUC={auc_t3:.3f})", "#3498db"),
        (X_oof[:, 1], f"Tier 4 BiCA-HS (AUC={auc_t4:.3f})", "#e74c3c"),
        (p_oof,        f"Tier 5 Meta    (AUC={metrics['auc']:.3f})", "#2ecc71"),
    ]:
        fpr, tpr, _ = roc_curve(y_oof, p)
        ax.plot(fpr, tpr, linewidth=2, color=color, label=lbl)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves: Tier 3 vs Tier 4 vs Tier 5", fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, "tier5_roc_comparison.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Tier5] Saved ROC comparison to {roc_path}")

    # ── Additional visualisations ──────────────────────────────────────────
    if args.plot:
        # Reliability diagram
        plot_reliability_diagram(
            y_true=y_oof,
            pred_dict={
                "Tier3 (XGBoost)": X_oof[:, 0],
                "Tier4 (BiCA-HS)": X_oof[:, 1],
                "Tier5 (Meta)":    p_oof,
            },
            output_path=os.path.join(fig_dir, "tier5_reliability_diagram.png"),
            title="Calibration Curves"
        )

        # Decision Curve Analysis
        decision_curve_analysis(
            y_true=y_oof,
            pred_dict={
                "Tier3 (XGBoost)": X_oof[:, 0],
                "Tier4 (BiCA-HS)": X_oof[:, 1],
                "Tier5 (Meta)":    p_oof,
            },
            output_path=os.path.join(fig_dir, "tier5_dca.png"),
            title="Decision Curve Analysis (Net Benefit)"
        )

        # Tier5 decision space
        fold_col = df_meta.get('Fold_tab') if 'Fold_tab' in df_meta.columns else None
        plot_tier5_decision_space(
            p_tab=X_oof[:, 0],
            p_bica=X_oof[:, 1],
            labels=y_oof,
            p_final=p_oof,
            output_path=os.path.join(fig_dir, "tier5_decision_space.png"),
        )

        # Meta-learner coefficient bar plot (interpretability)
        if meta.best == "logistic_regression":
            coefs = {
                "β_tab (XGBoost)": meta.lr.coef_tab,
                "β_bica (BiCA-HS)": meta.lr.coef_bica,
            }
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#3498db' if v >= 0 else '#e74c3c'
                      for v in coefs.values()]
            ax.barh(list(coefs.keys()), list(coefs.values()),
                    color=colors, edgecolor='black', height=0.5)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel("Logistic Regression Coefficient", fontsize=11)
            ax.set_title("Tier 5 Meta-Learner: Feature Attribution", fontsize=12,
                         fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            coef_path = os.path.join(fig_dir, "tier5_meta_coefficients.png")
            plt.savefig(coef_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[Tier5] Saved coefficient plot to {coef_path}")

    # ── Test inference ─────────────────────────────────────────────────────
    if X_test is not None:
        p_test = meta.predict_proba(X_test)[:, 1]
        if args.calibrate and scaler is not None:
            p_test = scaler.transform(p_test)
        df_test['Pred_Proba_Tier5'] = p_test
        test_path = os.path.join(args.output_dir, "tier5_test_predictions.csv")
        df_test.to_csv(test_path, index=False)
        print(f"\n[Tier5] Saved test predictions to {test_path}")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "baselines": {
            "tier3_oof_auc": float(auc_t3),
            "tier3_oof_acc": float(acc_t3),
            "tier3_oof_brier": float(bs_t3),
            "tier3_oof_bss": float(bss_t3),
            "tier3_oof_ece": float(ece_t3),
            "tier4_oof_auc": float(auc_t4),
            "tier4_oof_acc": float(acc_t4),
            "tier4_oof_brier": float(bs_t4),
            "tier4_oof_bss": float(bss_t4),
            "tier4_oof_ece": float(ece_t4),
        },
        "tier5_metrics": metrics,
        "meta_learner": meta.summary(),
        "n_subjects": int(len(df_meta)),
    }
    summary_path = os.path.join(args.output_dir, "tier5_results_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\n[Tier5] Saved results summary to {summary_path}")

    # Save OOF subject predictions for bootstrap significance testing
    df_meta['Pred_Proba_Subject'] = p_oof
    oof_path = os.path.join(args.output_dir, "tier5_oof_subject_preds.csv")
    df_meta.to_csv(oof_path, index=False)
    print(f"[Tier5] Saved OOF subject predictions to {oof_path}")

    print("\n" + "=" * 65)
    print(f" Tier 3  AUC: {auc_t3:.4f}")
    print(f" Tier 4  AUC: {auc_t4:.4f}")
    print(f" Tier 5  AUC: {metrics['auc']:.4f}  "
          f"({'↑' if metrics['auc'] > max(auc_t3, auc_t4) else '↓'} vs best single)")
    print(f" Strategy: {meta.best}")
    print("=" * 65)

    meta.save(os.path.join(args.output_dir, "tier5_meta_learner.json"))


if __name__ == "__main__":
    main()
