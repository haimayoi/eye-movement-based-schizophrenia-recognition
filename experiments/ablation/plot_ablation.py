"""
Ablation Study Visualization
=============================

Generates publication-quality figures from ablation study results.

Usage:
    python experiments/ablation/plot_ablation.py
    python experiments/ablation/plot_ablation.py --input-dir experiments/ablation/results
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'BiCA-HS (Transformer)': '#2196F3',
    'GNN+CEFAM (Full Hybrid)': '#4CAF50',
    'ST-GNN (GNN Only)': '#FF9800',
    'XGBoost (Tabular Only)': '#9C27B0',
    'CatBoost (Tabular Only)': '#795548',
    'MSNet (SOTA)': '#F44336',
}


def plot_model_comparison_bar(input_dir, output_dir):
    """Plot F1: Model comparison bar chart."""
    csv_path = os.path.join(input_dir, "F1_model_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df_opt = df[df['threshold_type'].str.startswith('optimal')]

    if df_opt.empty:
        return

    models = df_opt['model'].values
    aucs = df_opt['auc'].values
    accs = df_opt['accuracy'].values
    f1s = df_opt['f1'].values

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, aucs, width, label='AUC-ROC',
                   color=[COLORS.get(m, '#666') for m in models], alpha=0.9)
    bars2 = ax.bar(x, accs, width, label='Accuracy',
                   color=[COLORS.get(m, '#666') for m in models], alpha=0.6)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score',
                   color=[COLORS.get(m, '#666') for m in models], alpha=0.4)

    # SOTA reference line
    ax.axhline(y=0.8854, color='#F44336', linestyle='--', linewidth=1.5,
               label='MSNet SOTA AUC (88.54%)')
    ax.axhline(y=0.8125, color='#F44336', linestyle=':', linewidth=1.5,
               label='MSNet SOTA ACC (81.25%)')

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Ablation Study (Optimal Threshold)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in models], fontsize=9)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "F1_model_comparison.png"))
    plt.close()
    print(f"  [OK] F1_model_comparison.png")


def plot_component_contribution(input_dir, output_dir):
    """Plot F2: Component contribution waterfall/delta chart."""
    csv_path = os.path.join(input_dir, "F2_component_contribution.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    components = df['component'].values
    delta_auc = df['delta_auc'].values
    delta_acc = df['delta_acc'].values

    colors_auc = ['#4CAF50' if d >= 0 else '#F44336' for d in delta_auc]
    colors_acc = ['#2196F3' if d >= 0 else '#F44336' for d in delta_acc]

    y_pos = np.arange(len(components))

    ax1.barh(y_pos, delta_auc * 100, color=colors_auc, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([c[:35] for c in components], fontsize=9)
    ax1.set_xlabel('dAUC (%)')
    ax1.set_title('Component Contribution - AUC')
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(delta_auc):
        ax1.text(v * 100 + 0.5 * np.sign(v), i, f'{v * 100:+.1f}%', va='center', fontsize=9)

    ax2.barh(y_pos, delta_acc * 100, color=colors_acc, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([c[:35] for c in components], fontsize=9)
    ax2.set_xlabel('dACC (%)')
    ax2.set_title('Component Contribution - Accuracy')
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(delta_acc):
        ax2.text(v * 100 + 0.5 * np.sign(v), i, f'{v * 100:+.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "F2_component_contribution.png"))
    plt.close()
    print(f"  [OK] F2_component_contribution.png")


def plot_category_analysis(input_dir, output_dir):
    """Plot F3: Category-level AUC comparison."""
    csv_path = os.path.join(input_dir, "F3_category_analysis.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df_cats = df[df['category'] != 'Overall (aggregated)']

    if df_cats.empty:
        return

    models = df_cats['model'].unique()
    categories = df_cats['category'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = df_cats[df_cats['model'] == model]
        aucs = [model_data[model_data['category'] == cat]['auc'].values[0]
                if cat in model_data['category'].values else 0.5
                for cat in categories]
        color = COLORS.get(model, f'C{i}')
        bars = ax.bar(x + i * width - (len(models) - 1) * width / 2, aucs,
                      width, label=model, color=color, alpha=0.8)

    ax.set_ylabel('AUC-ROC')
    ax.set_title('Category-Level Discriminative Power')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Chance level')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "F3_category_analysis.png"))
    plt.close()
    print(f"  [OK] F3_category_analysis.png")


def plot_roc_curves(input_dir, output_dir):
    """Plot F7: ROC curves for all models."""
    csv_path = os.path.join(input_dir, "F7_roc_curve_data.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('fpr')
        auc_val = model_data['auc'].iloc[0]
        color = COLORS.get(model, None)
        ax.plot(model_data['fpr'], model_data['tpr'],
                label=f'{model} (AUC={auc_val:.4f})',
                color=color, linewidth=2)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Chance')

    # SOTA reference point
    ax.scatter([1 - 0.85], [0.8125], color='#F44336', s=100, marker='*',
               zorder=5, label='MSNet SOTA')

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves - All Models')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "F7_roc_curves.png"))
    plt.close()
    print(f"  [OK] F7_roc_curves.png")


def plot_fold_stability(input_dir, output_dir):
    """Plot F5: Fold stability comparison."""
    csv_path = os.path.join(input_dir, "F5_fold_stability.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    models = df['model'].values
    x = np.arange(len(models))

    colors = [COLORS.get(m, '#666') for m in models]

    ax.bar(x, df['mean_auc'], yerr=df['std_auc'],
           color=colors, alpha=0.8, capsize=8, edgecolor='black', linewidth=0.5)

    # Error range indicators
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([i, i], [row['min_auc'], row['max_auc']], 'k-', linewidth=2, alpha=0.5)

    ax.set_ylabel('AUC-ROC')
    ax.set_title('Cross-Validation Fold Stability (Mean +/- Std)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in models], fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add CV values
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row['mean_auc'] + row['std_auc'] + 0.02,
                f'CV={row["cv_auc"]:.3f}', ha='center', fontsize=9, color='gray')

    ax.axhline(y=0.8854, color='#F44336', linestyle='--', linewidth=1.5,
               label='MSNet SOTA (88.54%)')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "F5_fold_stability.png"))
    plt.close()
    print(f"  [OK] F5_fold_stability.png")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--input-dir", type=str, default="experiments/ablation/results",
                        help="Path to ablation results CSVs")
    parser.add_argument("--output-dir", type=str, default="experiments/ablation/figures",
                        help="Path to save figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("GENERATING ABLATION FIGURES")
    print("=" * 50)

    plot_model_comparison_bar(args.input_dir, args.output_dir)
    plot_component_contribution(args.input_dir, args.output_dir)
    plot_category_analysis(args.input_dir, args.output_dir)
    plot_roc_curves(args.input_dir, args.output_dir)
    plot_fold_stability(args.input_dir, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
