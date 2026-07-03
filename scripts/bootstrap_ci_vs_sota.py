"""
Bootstrap 95% CI for AUC — statistical comparison vs SOTA (MSNet, AUC=0.8854).

Approach:
  For each model and each seed, resample 160 subjects 10,000 times with
  replacement and compute AUC each time → empirical 95% CI.
  If the 2.5th percentile > SOTA AUC → our model is significantly better
  at alpha=0.05 (one-sided: p < 0.025).

Usage:
    python scripts/bootstrap_ci_vs_sota.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SOTA_AUC = 0.8854
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(0)

SEEDS = [42, 123, 456]

MODELS = {
    "XGBoost": {
        42:  "results/baselines_s42/xgboost_oof_subject_preds.csv",
        123: "results/baselines_s123/xgboost_oof_subject_preds.csv",
        456: "results/baselines_s456/xgboost_oof_subject_preds.csv",
    },
    "CatBoost": {
        42:  "results/baselines_s42/catboost_oof_subject_preds.csv",
        123: "results/baselines_s123/catboost_oof_subject_preds.csv",
        456: "results/baselines_s456/catboost_oof_subject_preds.csv",
    },
    "ST-GNN": {
        42:  "results/stgnn_s42/stgnn_oof_subject_preds.csv",
        123: "results/stgnn_s123/stgnn_oof_subject_preds.csv",
        456: "results/stgnn_s456/stgnn_oof_subject_preds.csv",
    },
    "GNN-CEFAM": {
        42:  "results/cefam_s42/cefam_oof_subject_preds.csv",
        123: "results/cefam_s123/cefam_oof_subject_preds.csv",
        456: "results/cefam_s456/cefam_oof_subject_preds.csv",
    },
    "BiCA-HS": {
        42:  "results/bica_s42/bica_subject_val_predictions.csv",
        123: "results/bica_s123/bica_subject_val_predictions.csv",
        456: "results/bica_s456/bica_subject_val_predictions.csv",
    },
    "Tier5 (Ensemble+CEFAM)": {
        42:  "results/tier5_cefam_s42/tier5_oof_subject_preds.csv",
        123: "results/tier5_cefam_s123/tier5_oof_subject_preds.csv",
        456: "results/tier5_cefam_s456/tier5_oof_subject_preds.csv",
    },
    "Tier5 (Ensemble+BiCA-HS)": {
        42:  "results/tier5_bica_s42/tier5_oof_subject_preds.csv",
        123: "results/tier5_bica_s123/tier5_oof_subject_preds.csv",
        456: "results/tier5_bica_s456/tier5_oof_subject_preds.csv",
    },
}


def bootstrap_auc(y, p, n=N_BOOTSTRAP):
    """Return (mean, ci_low, ci_high, p_value_vs_sota)."""
    n_subj = len(y)
    boot_aucs = []
    for _ in range(n):
        idx = RNG.integers(0, n_subj, size=n_subj)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, pb))
    boot_aucs = np.array(boot_aucs)
    ci_low  = np.percentile(boot_aucs, 2.5)
    ci_high = np.percentile(boot_aucs, 97.5)
    # One-sided p-value: fraction of bootstrap samples <= SOTA_AUC
    p_val = (boot_aucs <= SOTA_AUC).mean()
    return boot_aucs.mean(), ci_low, ci_high, p_val, boot_aucs


print("=" * 72)
print(f"Bootstrap 95% CI (n={N_BOOTSTRAP:,}) — vs SOTA AUC = {SOTA_AUC}")
print("=" * 72)

for model_name, paths in MODELS.items():
    print(f"\n{model_name}")
    print(f"  {'Seed':>6} | {'AUC':>7} | {'95% CI':>18} | {'p-value':>9} | Sig?")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*18}-+-{'-'*9}-+------")

    all_boot = []
    for seed in SEEDS:
        path = paths.get(seed)
        if not path:
            continue
        import os
        if not os.path.exists(path):
            print(f"  {seed:>6} | MISSING")
            continue
        df = pd.read_csv(path)
        y = df["Label"].values.astype(int)
        p = df["Pred_Proba_Subject"].values
        observed_auc = roc_auc_score(y, p)

        mean_b, lo, hi, pval, boot = bootstrap_auc(y, p)
        all_boot.append(boot)
        sig = "YES" if lo > SOTA_AUC else "NO"
        print(f"  {seed:>6} | {observed_auc:.4f} | [{lo:.4f}, {hi:.4f}] | {pval:>9.4f} | {sig}")

    # Pooled across seeds
    if len(all_boot) > 1:
        pooled = np.concatenate(all_boot)
        pool_lo  = np.percentile(pooled, 2.5)
        pool_hi  = np.percentile(pooled, 97.5)
        pool_p   = (pooled <= SOTA_AUC).mean()
        pool_sig = "YES" if pool_lo > SOTA_AUC else "NO"
        print(f"  {'POOL':>6} |        | [{pool_lo:.4f}, {pool_hi:.4f}] | {pool_p:>9.4f} | {pool_sig}")

print()
print("=" * 72)
print("Interpretation:")
print(f"  YES = 95% CI lower bound > SOTA ({SOTA_AUC}) => p < 0.025 (significant)")
print(f"  NO  = CI overlaps SOTA => not significant at alpha=0.05")
print("=" * 72)
