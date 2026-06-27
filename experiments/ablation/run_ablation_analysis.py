# -*- coding: utf-8 -*-
"""
Ablation Study Analysis for Eye Movement-Based Schizophrenia Recognition
=========================================================================

Analyzes existing model results to evaluate the contribution of each
architectural component to overall classification performance.

Ablation Studies:
  F1: Full GNN+CEFAM Hybrid          — Baseline (full model)
  F2: GNN Stream Only (ST-GNN)       — Remove handcrafted stream + CEFAM
  F3: Handcrafted Only (XGBoost)     — Remove GNN stream entirely
  F4: BiCA-HS (Transformer+CrossAttn)— Alternative fusion architecture
  F5: Category-level analysis        — Which stimulus categories contribute most
  F6: Threshold sensitivity          — How does threshold affect each model
  F7: Per-fold stability             — Variance analysis across folds

Usage:
    python experiments/ablation/run_ablation_analysis.py
    python experiments/ablation/run_ablation_analysis.py --output-dir results/ablation
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)

# ============================================================
# Utility Functions
# ============================================================

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive classification metrics at a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = rec  # sensitivity = recall

    # Youden's J statistic
    j_stat = sens + spec - 1.0

    return {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "sensitivity": sens,
        "specificity": spec,
        "youden_j": j_stat,
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def find_optimal_threshold(y_true, y_pred_proba, metric="accuracy"):
    """Find the optimal threshold using grid search."""
    best_th = 0.5
    best_score = 0.0

    for th in np.linspace(0.01, 0.99, 500):
        y_pred = (y_pred_proba >= th).astype(int)
        if metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = sens + spec - 1.0
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_th = th

    return best_th


def _resolve(results_dir, subdir, filename):
    """Try subdir first, then subdir_s42 (seed-based layout), then subdir_42."""
    for candidate in [subdir, f"{subdir}_s42", f"{subdir}_42"]:
        p = os.path.join(results_dir, candidate, filename)
        if os.path.exists(p):
            return p
    return None


def load_model_predictions(results_dir):
    """Load all model prediction files from the results directory."""
    models = {}

    # CEFAM (GNN + CEFAM Hybrid)
    cefam_path = _resolve(results_dir, "cefam", "cefam_subject_val_predictions.csv")
    if cefam_path:
        models["GNN+CEFAM (Full Hybrid)"] = pd.read_csv(cefam_path)

    # STGNN (GNN Stream Only)
    stgnn_path = _resolve(results_dir, "stgnn", "stgnn_subject_val_predictions.csv")
    if stgnn_path:
        models["ST-GNN (GNN Only)"] = pd.read_csv(stgnn_path)

    # BiCA-HS (Transformer + Cross-Attention)
    bica_path = _resolve(results_dir, "bica", "bica_subject_val_predictions.csv")
    if bica_path:
        models["BiCA-HS (Transformer)"] = pd.read_csv(bica_path)

    # XGBoost (Tabular Baseline)
    xgb_path = _resolve(results_dir, "baselines", "xgboost_subject_val_preds.csv")
    if xgb_path:
        models["XGBoost (Tabular Only)"] = pd.read_csv(xgb_path)

    # Load JSON summaries for per-fold data
    summaries = {}
    for name, prefix, subdir in [
        ("GNN+CEFAM (Full Hybrid)", "cefam", "cefam"),
        ("ST-GNN (GNN Only)", "stgnn", "stgnn"),
        ("BiCA-HS (Transformer)", "bica", "bica"),
    ]:
        json_path = _resolve(results_dir, subdir, f"{prefix}_results_summary.json")
        if json_path:
            with open(json_path, 'r') as f:
                summaries[name] = json.load(f)

    return models, summaries


# ============================================================
# Ablation Experiments
# ============================================================

def ablation_f1_model_comparison(models, output_dir):
    """
    F1: Full Model Comparison — Main ablation table.
    Compare all models at default threshold (0.5) and optimal threshold.
    """
    print("\n" + "=" * 70)
    print("F1: FULL MODEL COMPARISON (Main Ablation Table)")
    print("=" * 70)

    results = []

    for model_name, df in models.items():
        y_true = df['Label'].values
        y_prob = df['Pred_Proba_Subject'].values

        # Default threshold (0.5)
        metrics_default = calculate_metrics(y_true, y_prob, threshold=0.5)
        metrics_default['model'] = model_name
        metrics_default['threshold_type'] = 'default (0.5)'

        # Optimal threshold (accuracy)
        opt_th = find_optimal_threshold(y_true, y_prob, metric="accuracy")
        metrics_optimal = calculate_metrics(y_true, y_prob, threshold=opt_th)
        metrics_optimal['model'] = model_name
        metrics_optimal['threshold_type'] = f'optimal ({opt_th:.4f})'

        results.append(metrics_default)
        results.append(metrics_optimal)

        print(f"\n--- {model_name} ---")
        print(f"  AUC-ROC:  {metrics_default['auc']:.4f}")
        print(f"  ACC @0.5: {metrics_default['accuracy']:.4f}")
        print(f"  F1  @0.5: {metrics_default['f1']:.4f}")
        print(f"  ACC @opt: {metrics_optimal['accuracy']:.4f} (th={opt_th:.4f})")
        print(f"  F1  @opt: {metrics_optimal['f1']:.4f}")
        print(f"  Sens@opt: {metrics_optimal['sensitivity']:.4f}")
        print(f"  Spec@opt: {metrics_optimal['specificity']:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "F1_model_comparison.csv"), index=False)
    return df_results


def ablation_f2_component_contribution(models, output_dir):
    """
    F2: Component Contribution Analysis.
    Quantify what each architectural component adds by comparing models.

    Key comparisons:
    - CEFAM Fusion value: Full Hybrid vs GNN Only
    - Handcrafted Stream value: Full Hybrid vs GNN Only
    - GNN Stream value: Full Hybrid vs Tabular Only
    - Bidirectional Cross-Attention: BiCA-HS vs others
    """
    print("\n" + "=" * 70)
    print("F2: COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)

    # Get AUC for each model
    aucs = {}
    accs = {}
    for name, df in models.items():
        y_true = df['Label'].values
        y_prob = df['Pred_Proba_Subject'].values
        aucs[name] = roc_auc_score(y_true, y_prob)
        opt_th = find_optimal_threshold(y_true, y_prob)
        accs[name] = accuracy_score(y_true, (y_prob >= opt_th).astype(int))

    contributions = []

    # CEFAM Fusion contribution
    if "GNN+CEFAM (Full Hybrid)" in aucs and "ST-GNN (GNN Only)" in aucs:
        delta_auc = aucs["GNN+CEFAM (Full Hybrid)"] - aucs["ST-GNN (GNN Only)"]
        delta_acc = accs["GNN+CEFAM (Full Hybrid)"] - accs["ST-GNN (GNN Only)"]
        contributions.append({
            "component": "CEFAM Fusion + Handcrafted Stream",
            "comparison": "Full Hybrid vs GNN Only",
            "delta_auc": delta_auc,
            "delta_acc": delta_acc,
            "interpretation": "Value of adding handcrafted features + bidirectional cross-attention"
        })
        print(f"\n  CEFAM + Handcrafted Stream:")
        print(f"    dAUC = {delta_auc:+.4f}  |  dACC = {delta_acc:+.4f}")

    # GNN contribution over tabular
    if "GNN+CEFAM (Full Hybrid)" in aucs and "XGBoost (Tabular Only)" in aucs:
        delta_auc = aucs["GNN+CEFAM (Full Hybrid)"] - aucs["XGBoost (Tabular Only)"]
        delta_acc = accs["GNN+CEFAM (Full Hybrid)"] - accs["XGBoost (Tabular Only)"]
        contributions.append({
            "component": "GNN Stream (Graph Topology)",
            "comparison": "Full Hybrid vs Tabular Only",
            "delta_auc": delta_auc,
            "delta_acc": delta_acc,
            "interpretation": "Value of graph-based representation over flat features"
        })
        print(f"\n  GNN Stream (Graph Topology):")
        print(f"    dAUC = {delta_auc:+.4f}  |  dACC = {delta_acc:+.4f}")

    # Transformer vs GNN comparison
    if "BiCA-HS (Transformer)" in aucs and "GNN+CEFAM (Full Hybrid)" in aucs:
        delta_auc = aucs["BiCA-HS (Transformer)"] - aucs["GNN+CEFAM (Full Hybrid)"]
        delta_acc = accs["BiCA-HS (Transformer)"] - accs["GNN+CEFAM (Full Hybrid)"]
        contributions.append({
            "component": "Transformer Encoder (vs GAT)",
            "comparison": "BiCA-HS vs GNN+CEFAM",
            "delta_auc": delta_auc,
            "delta_acc": delta_acc,
            "interpretation": "Transformer sequence modeling vs graph topology modeling"
        })
        print(f"\n  Transformer vs GAT:")
        print(f"    dAUC = {delta_auc:+.4f}  |  dACC = {delta_acc:+.4f}")

    # BiCA-HS vs GNN-only
    if "BiCA-HS (Transformer)" in aucs and "ST-GNN (GNN Only)" in aucs:
        delta_auc = aucs["BiCA-HS (Transformer)"] - aucs["ST-GNN (GNN Only)"]
        delta_acc = accs["BiCA-HS (Transformer)"] - accs["ST-GNN (GNN Only)"]
        contributions.append({
            "component": "Full BiCA-HS vs GNN Standalone",
            "comparison": "BiCA-HS vs ST-GNN",
            "delta_auc": delta_auc,
            "delta_acc": delta_acc,
            "interpretation": "Total gain from using Transformer+CrossAttn over GNN only"
        })
        print(f"\n  BiCA-HS vs GNN Standalone:")
        print(f"    dAUC = {delta_auc:+.4f}  |  dACC = {delta_acc:+.4f}")

    df_contrib = pd.DataFrame(contributions)
    df_contrib.to_csv(os.path.join(output_dir, "F2_component_contribution.csv"), index=False)
    return df_contrib


def ablation_f3_category_analysis(models, output_dir):
    """
    F3: Category-Level Analysis.
    Analyze which stimulus categories (Social, Natural, Synthetic, Manipulated)
    are most discriminative for each model.
    """
    print("\n" + "=" * 70)
    print("F3: STIMULUS CATEGORY DISCRIMINATIVE ANALYSIS")
    print("=" * 70)

    categories = ['Social', 'Manipulated', 'Natural', 'Synthetic']
    results = []

    for model_name, df in models.items():
        y_true = df['Label'].values

        # Check if category columns exist
        available_cats = [c for c in categories if c in df.columns]
        if not available_cats:
            continue

        print(f"\n--- {model_name} ---")

        for cat in available_cats:
            if cat not in df.columns:
                continue

            cat_proba = df[cat].values

            # Handle NaN values
            valid_mask = ~np.isnan(cat_proba)
            if valid_mask.sum() < len(y_true):
                cat_proba = np.where(np.isnan(cat_proba), 0.5, cat_proba)

            cat_auc = roc_auc_score(y_true, cat_proba) if len(np.unique(y_true)) > 1 else 0.5

            results.append({
                "model": model_name,
                "category": cat,
                "auc": cat_auc,
            })
            print(f"  {cat:15s}: AUC = {cat_auc:.4f}")

        # Overall for reference
        overall_auc = roc_auc_score(y_true, df['Pred_Proba_Subject'].values)
        results.append({
            "model": model_name,
            "category": "Overall (aggregated)",
            "auc": overall_auc,
        })
        print(f"  {'Overall':15s}: AUC = {overall_auc:.4f}")

    df_cats = pd.DataFrame(results)
    df_cats.to_csv(os.path.join(output_dir, "F3_category_analysis.csv"), index=False)
    return df_cats


def ablation_f4_threshold_sensitivity(models, output_dir):
    """
    F4: Threshold Sensitivity Analysis.
    Evaluate how each model performs across different decision thresholds.
    """
    print("\n" + "=" * 70)
    print("F4: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)

    thresholds = np.linspace(0.05, 0.95, 19)
    results = []

    for model_name, df in models.items():
        y_true = df['Label'].values
        y_prob = df['Pred_Proba_Subject'].values

        print(f"\n--- {model_name} ---")
        print(f"  {'Threshold':>10s} | {'ACC':>6s} | {'F1':>6s} | {'Sens':>6s} | {'Spec':>6s} | {'Youden':>6s}")
        print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

        for th in thresholds:
            metrics = calculate_metrics(y_true, y_prob, threshold=th)
            results.append({
                "model": model_name,
                "threshold": th,
                **metrics
            })
            print(f"  {th:10.2f} | {metrics['accuracy']:6.4f} | {metrics['f1']:6.4f} | "
                  f"{metrics['sensitivity']:6.4f} | {metrics['specificity']:6.4f} | {metrics['youden_j']:6.4f}")

    df_thresh = pd.DataFrame(results)
    df_thresh.to_csv(os.path.join(output_dir, "F4_threshold_sensitivity.csv"), index=False)
    return df_thresh


def ablation_f5_fold_stability(summaries, output_dir):
    """
    F5: Per-Fold Stability Analysis.
    Analyze variance and stability of each model across cross-validation folds.
    """
    print("\n" + "=" * 70)
    print("F5: PER-FOLD STABILITY ANALYSIS")
    print("=" * 70)

    results = []

    for model_name, summary in summaries.items():
        folds = summary.get('folds', [])
        if not folds:
            continue

        aucs = [f['auc'] for f in folds]
        accs = [f['accuracy'] for f in folds]
        f1s = [f['f1'] for f in folds]

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        cv_auc = std_auc / mean_auc if mean_auc > 0 else 0.0  # Coefficient of Variation

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)

        results.append({
            "model": model_name,
            "n_folds": len(folds),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "cv_auc": cv_auc,
            "min_auc": min(aucs),
            "max_auc": max(aucs),
            "range_auc": max(aucs) - min(aucs),
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        })

        print(f"\n--- {model_name} ---")
        print(f"  AUC: {mean_auc:.4f} +/- {std_auc:.4f}  (CV={cv_auc:.4f})")
        print(f"  ACC: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  F1:  {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"  AUC Range: [{min(aucs):.4f}, {max(aucs):.4f}]  (d={max(aucs) - min(aucs):.4f})")
        print(f"  Per-fold AUCs: {[f'{a:.4f}' for a in aucs]}")

    df_stability = pd.DataFrame(results)
    df_stability.to_csv(os.path.join(output_dir, "F5_fold_stability.csv"), index=False)
    return df_stability


def ablation_f6_aggregation_weights(models, output_dir):
    """
    F6: Aggregation Weight Analysis.
    Test different category weighting strategies for probability aggregation.
    """
    print("\n" + "=" * 70)
    print("F6: AGGREGATION WEIGHT ANALYSIS")
    print("=" * 70)

    categories = ['Social', 'Manipulated', 'Natural', 'Synthetic']

    # Define weight strategies
    weight_strategies = {
        "Uniform": {"Social": 0.25, "Manipulated": 0.25, "Natural": 0.25, "Synthetic": 0.25},
        "Social-Focused": {"Social": 0.40, "Manipulated": 0.30, "Natural": 0.15, "Synthetic": 0.15},
        "Social+Manip": {"Social": 0.35, "Manipulated": 0.35, "Natural": 0.15, "Synthetic": 0.15},
        "Drop-Natural": {"Social": 0.33, "Manipulated": 0.34, "Natural": 0.0, "Synthetic": 0.33},
        "Drop-Synthetic": {"Social": 0.34, "Manipulated": 0.33, "Natural": 0.33, "Synthetic": 0.0},
        "Social-Only": {"Social": 1.0, "Manipulated": 0.0, "Natural": 0.0, "Synthetic": 0.0},
        "Manip-Only": {"Social": 0.0, "Manipulated": 1.0, "Natural": 0.0, "Synthetic": 0.0},
    }

    results = []

    for model_name, df in models.items():
        y_true = df['Label'].values
        available_cats = [c for c in categories if c in df.columns]
        if len(available_cats) < 4:
            continue

        print(f"\n--- {model_name} ---")
        print(f"  {'Strategy':25s} | {'AUC':>6s} | {'ACC@opt':>7s} | {'F1@opt':>7s}")
        print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

        for strategy_name, weights in weight_strategies.items():
            # Compute weighted probability
            p_subject = np.zeros(len(df))
            total_w = sum(weights.values())
            if total_w == 0:
                continue

            for cat, w in weights.items():
                if cat in df.columns:
                    p_subject += (w / total_w) * df[cat].fillna(0.5).values

            auc = roc_auc_score(y_true, p_subject)
            opt_th = find_optimal_threshold(y_true, p_subject)
            metrics = calculate_metrics(y_true, p_subject, threshold=opt_th)

            results.append({
                "model": model_name,
                "strategy": strategy_name,
                "auc": auc,
                "accuracy_optimal": metrics['accuracy'],
                "f1_optimal": metrics['f1'],
                "optimal_threshold": opt_th,
                **weights
            })

            print(f"  {strategy_name:25s} | {auc:6.4f} | {metrics['accuracy']:7.4f} | {metrics['f1']:7.4f}")

    df_weights = pd.DataFrame(results)
    df_weights.to_csv(os.path.join(output_dir, "F6_aggregation_weights.csv"), index=False)
    return df_weights


def ablation_f7_roc_curves(models, output_dir):
    """
    F7: ROC Curve Data.
    Generate ROC curve data for all models for plotting.
    """
    print("\n" + "=" * 70)
    print("F7: ROC CURVE DATA (saved for plotting)")
    print("=" * 70)

    all_roc_data = []

    for model_name, df in models.items():
        y_true = df['Label'].values
        y_prob = df['Pred_Proba_Subject'].values

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        for i in range(len(fpr)):
            all_roc_data.append({
                "model": model_name,
                "fpr": fpr[i],
                "tpr": tpr[i],
                "threshold": thresholds[i] if i < len(thresholds) else np.nan,
                "auc": auc
            })

        print(f"  {model_name}: AUC = {auc:.4f}  ({len(fpr)} points)")

    df_roc = pd.DataFrame(all_roc_data)
    df_roc.to_csv(os.path.join(output_dir, "F7_roc_curve_data.csv"), index=False)
    return df_roc


def generate_summary_report(all_results, output_dir):
    """Generate a comprehensive JSON summary report of all ablation studies."""
    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE ABLATION SUMMARY")
    print("=" * 70)

    report = {
        "experiment": "Ablation Study - Eye Movement-Based Schizophrenia Recognition",
        "models_evaluated": list(all_results.get("model_comparison", pd.DataFrame()).get("model", pd.Series()).unique()) if "model_comparison" in all_results else [],
        "sota_reference": {
            "model": "MSNet (Song et al., IEEE TNNLS 2025)",
            "accuracy": 0.8125,
            "auc": 0.8854
        },
        "key_findings": [],
        "files_generated": []
    }

    # Key findings
    if "model_comparison" in all_results:
        df = all_results["model_comparison"]
        df_opt = df[df['threshold_type'].str.startswith('optimal')]
        if not df_opt.empty:
            best_row = df_opt.loc[df_opt['auc'].idxmax()]
            report["key_findings"].append(
                f"Best model: {best_row['model']} with AUC={best_row['auc']:.4f}, "
                f"ACC={best_row['accuracy']:.4f}"
            )

    if "component_contribution" in all_results:
        df = all_results["component_contribution"]
        for _, row in df.iterrows():
            report["key_findings"].append(
                f"{row['component']}: dAUC={row['delta_auc']:+.4f}, dACC={row['delta_acc']:+.4f}"
            )

    if "fold_stability" in all_results:
        df = all_results["fold_stability"]
        most_stable = df.loc[df['cv_auc'].idxmin()]
        report["key_findings"].append(
            f"Most stable model: {most_stable['model']} (CV_AUC={most_stable['cv_auc']:.4f})"
        )

    # Files
    for fname in os.listdir(output_dir):
        if fname.endswith('.csv') or fname.endswith('.json'):
            report["files_generated"].append(fname)

    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)

    print(f"\n  Summary saved to {summary_path}")
    return report


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run ablation study analysis")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Path to the results directory")
    parser.add_argument("--output-dir", type=str, default="experiments/ablation/results",
                        help="Path to save ablation results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY - Eye Movement-Based Schizophrenia Recognition")
    print("=" * 70)

    # Load all model predictions
    print("\nLoading model predictions...")
    models, summaries = load_model_predictions(args.results_dir)

    if not models:
        print("Error: No model predictions found in results directory!")
        print(f"Searched in: {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    # Run all ablation experiments
    all_results = {}

    # F1: Full model comparison
    all_results["model_comparison"] = ablation_f1_model_comparison(models, args.output_dir)

    # F2: Component contribution
    all_results["component_contribution"] = ablation_f2_component_contribution(models, args.output_dir)

    # F3: Category analysis
    all_results["category_analysis"] = ablation_f3_category_analysis(models, args.output_dir)

    # F4: Threshold sensitivity
    all_results["threshold_sensitivity"] = ablation_f4_threshold_sensitivity(models, args.output_dir)

    # F5: Fold stability
    all_results["fold_stability"] = ablation_f5_fold_stability(summaries, args.output_dir)

    # F6: Aggregation weight analysis
    all_results["aggregation_weights"] = ablation_f6_aggregation_weights(models, args.output_dir)

    # F7: ROC curve data
    all_results["roc_curves"] = ablation_f7_roc_curves(models, args.output_dir)

    # Generate summary
    report = generate_summary_report(all_results, args.output_dir)

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {args.output_dir}/")
    print("\nKey Findings:")
    for finding in report.get("key_findings", []):
        print(f"  * {finding}")


if __name__ == "__main__":
    main()
