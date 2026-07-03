"""
Multi-seed result aggregation.
Reads OOF predictions from all seed runs and reports mean ± std.

Usage:
    python scripts/aggregate_multiseed.py
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

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
        42:  "results/tier5_cefam_s42/tier5_results_summary.json",
        123: "results/tier5_cefam_s123/tier5_results_summary.json",
        456: "results/tier5_cefam_s456/tier5_results_summary.json",
    },
    "Tier5 (Ensemble+BiCA-HS)": {
        42:  "results/tier5_bica_s42/tier5_results_summary.json",
        123: "results/tier5_bica_s123/tier5_results_summary.json",
        456: "results/tier5_bica_s456/tier5_results_summary.json",
    },
}


def compute_metrics_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Label" not in df.columns or "Pred_Proba_Subject" not in df.columns:
        return None
    y = df["Label"].values
    p = df["Pred_Proba_Subject"].values
    if len(np.unique(y)) < 2:
        return None
    pred = (p >= 0.5).astype(int)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc":  roc_auc_score(y, p),
        "acc":  accuracy_score(y, pred),
        "f1":   f1_score(y, pred),
        "sens": recall_score(y, pred),
        "spec": spec,
    }


def compute_metrics_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    m = d.get("tier5_metrics") or d.get("calibrated") or d.get("best_model") or d.get("overall")
    if m is None:
        return None
    return {
        "auc":  m.get("auc", float("nan")),
        "acc":  m.get("accuracy", m.get("acc", float("nan"))),
        "f1":   m.get("f1", float("nan")),
        "sens": m.get("recall", m.get("sensitivity", float("nan"))),
        "spec": m.get("specificity", m.get("spec", float("nan"))),
    }


def compute_metrics(path):
    if path.endswith(".json"):
        return compute_metrics_json(path)
    return compute_metrics_csv(path)


print("=" * 98)
print(f"{'Model':<26} {'AUC':>16} {'ACC':>16} {'F1':>16} {'Sens':>16} {'Spec':>16}")
print("=" * 98)

all_results = {}
for model_name, paths in MODELS.items():
    seed_metrics = []
    missing = []
    for seed in SEEDS:
        m = compute_metrics(paths[seed])
        if m is None:
            missing.append(seed)
        else:
            seed_metrics.append(m)

    if not seed_metrics:
        print(f"{model_name:<26}  (no results found)")
        continue

    all_results[model_name] = seed_metrics

    def fmt(key):
        vals = [m[key] for m in seed_metrics]
        mean = np.mean(vals)
        std  = np.std(vals)
        n    = len(vals)
        tag  = f"*" if missing else ""
        return f"{mean:.4f}±{std:.4f}{tag}"

    print(f"{model_name:<26} {fmt('auc'):>16} {fmt('acc'):>16} "
          f"{fmt('f1'):>16} {fmt('sens'):>16} {fmt('spec'):>16}")
    if missing:
        print(f"  ⚠ Missing seeds: {missing} — partial result")

print("=" * 98)
print(f"{'SOTA (MSNet 2025)':<26} {'0.8854':>16} {'0.8125':>16}")
print("=" * 98)

# Per-seed breakdown
print("\n--- Per-seed AUC ---")
for model_name, seed_metrics in all_results.items():
    available = [s for s in SEEDS if s not in
                 [s2 for s2, m in zip(SEEDS, [compute_metrics(MODELS[model_name][s]) for s in SEEDS]) if m is None]]
    aucs = [f"s{s}={m['auc']:.4f}" for s, m in zip(available, seed_metrics)]
    print(f"  {model_name:<26}: {', '.join(aucs)}")
