"""
Tier 3 Tabular Models — XGBoost / LightGBM / CatBoost
=======================================================
Changes vs. original:
  1. Delta features (features_subject_level.csv) are now loaded and merged
     into the training features so that the subject-level contextual deltas
     (Δ_Social-Natural, etc.) actually contribute to classification.
  2. OOF (out-of-fold) subject-level predictions are saved to disk so that
     Tier 5 can train a meta-learner without any data leakage.
"""

import os
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.tier3_tabular.group_kfold import get_subject_folds


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(model_name: str, seed: int = 42):
    """Returns the specified tabular model with default/standard parameters."""
    if model_name == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is not installed. Please install it to use this baseline.")
        return xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    elif model_name == "lightgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed. Please install it to use this baseline.")
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1
        )
    elif model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. Please install it to use this baseline.")
        return CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=5,
            random_seed=seed,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred_proba):
    """Computes classification performance metrics."""
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc, "auc": auc, "f1": f1,
        "precision": prec, "recall": rec, "specificity": spec
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction aggregation (stimulus → subject)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_predictions(df_preds: pd.DataFrame,
                          weights: dict = None) -> pd.DataFrame:
    """
    Aggregates stimulus-level probabilities to subject-level.
    weights: {Category: weight} summing to 1.0. Defaults to uniform.
    """
    if weights is None:
        weights = {"Social": 0.25, "Manipulated": 0.25,
                   "Natural": 0.25, "Synthetic": 0.25}

    grouped = (df_preds
               .groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba']
               .mean()
               .reset_index())
    pivoted = grouped.pivot(
        index=['Subject_ID', 'Label'], columns='Category',
        values='Pred_Proba'
    ).reset_index()

    p_sz = np.zeros(len(pivoted))
    for cat, w in weights.items():
        if cat in pivoted.columns:
            p_sz += w * pivoted[cat].fillna(0.5).values

    pivoted['Pred_Proba_Subject'] = p_sz
    return pivoted


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_features_with_delta(
    features_stimulus_path: str,
    features_subject_path: str,
    categories_path: str,
) -> tuple:
    """
    Loads stimulus-level features and merges subject-level delta features.

    Delta features encode cross-category differential responses
    (e.g. Δ_Social-Natural_fix_count) and are now actually used for training.

    Returns:
        df_merged  : DataFrame with merged stimulus + delta features
        feature_cols: list of feature column names (excl. meta columns)
    """
    df_stim = pd.read_csv(features_stimulus_path)

    meta_cols = ['Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test', 'Category', 'Fold']

    if os.path.exists(features_subject_path):
        df_subj = pd.read_csv(features_subject_path)
        # Delta feature columns
        delta_cols = [c for c in df_subj.columns
                      if c.startswith('delta_') or c.startswith('Social_')
                      or c.startswith('Natural_') or c.startswith('Synthetic_')
                      or c.startswith('Manipulated_')]
        # Keep only delta / category-mean columns (not Label/Is_Test duplicates)
        delta_cols = [c for c in delta_cols if c not in meta_cols]

        if delta_cols:
            df_delta = df_subj[['Subject_ID'] + delta_cols].drop_duplicates('Subject_ID')
            df_merged = df_stim.merge(df_delta, on='Subject_ID', how='left')
            df_merged[delta_cols] = df_merged[delta_cols].fillna(0.0)
            print(f"  [Delta] Merged {len(delta_cols)} delta/category-mean features "
                  f"from {features_subject_path}")
        else:
            df_merged = df_stim
            delta_cols = []
            print(f"  [Delta] No delta columns found in {features_subject_path}")
    else:
        df_merged = df_stim
        delta_cols = []
        print(f"  [Delta] Subject-level file not found: {features_subject_path}. "
              "Continuing without delta features.")

    # Add category mapping
    if os.path.exists(categories_path):
        df_cat = pd.read_csv(categories_path)
        category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))
        df_merged['Category'] = df_merged['Stimulus_ID'].map(category_map)
    else:
        df_merged['Category'] = 'Unknown'

    # Feature columns = everything except meta
    all_meta = {'Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test', 'Category', 'Fold'}
    feature_cols = [c for c in df_merged.columns if c not in all_meta]

    return df_merged, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tier 3 Tabular Baseline: Train and evaluate models")
    parser.add_argument("--config", type=str,
                        default="configs/cefam_config.yaml")
    parser.add_argument("--data", type=str, default=None,
                        help="Override path to features_stimulus_level.csv")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config, usually 42)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    features_stimulus_path = (args.data
                               or data_config.get('features_stimulus_path',
                                                   'data/processed/features_stimulus_level.csv'))
    features_subject_path = data_config.get('features_subject_path',
                                             'data/processed/features_subject_level.csv')
    categories_path = data_config.get('categories_path',
                                       'data/metadata/stimulus_categories.csv')
    cv_path = os.path.join(data_config.get('raw_dir', 'EMS'), "Train_Valid.xlsx")

    # ── Load features (stimulus + delta) ──────────────────────────────────
    print(f"Loading stimulus-level features from {features_stimulus_path}...")
    df_stim, feature_cols = load_features_with_delta(
        features_stimulus_path, features_subject_path, categories_path)

    # ── Subject fold mapping ───────────────────────────────────────────────
    print(f"Loading CV splits from {cv_path}...")
    subject_to_fold = get_subject_folds(cv_path)

    df_train_valid = (df_stim[(df_stim['Is_Test'] == 0)
                               & (df_stim['Subject_ID'].isin(subject_to_fold))]
                      .copy())
    df_train_valid['Fold'] = df_train_valid['Subject_ID'].map(subject_to_fold)

    df_test = df_stim[df_stim['Is_Test'] == 1].copy()

    # Recompute feature_cols after adding Fold (exclude it)
    all_meta = {'Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test', 'Category', 'Fold'}
    feature_cols = [c for c in df_train_valid.columns if c not in all_meta]

    print(f"Number of features (stimulus + delta): {len(feature_cols)}")
    print(f"Training on {df_train_valid['Subject_ID'].nunique()} subjects "
          f"({len(df_train_valid)} trials)...")

    # ── Cross-validation ───────────────────────────────────────────────────
    df_train_valid['Pred_Proba'] = 0.0
    if not df_test.empty:
        df_test['Pred_Proba'] = 0.0

    for fold in range(args.cv_folds):
        print(f"\n--- Fold {fold} ---")

        train_idx = df_train_valid[df_train_valid['Fold'] != fold].index
        val_idx = df_train_valid[df_train_valid['Fold'] == fold].index

        X_train = df_train_valid.loc[train_idx, feature_cols]
        y_train = df_train_valid.loc[train_idx, 'Label']
        X_val = df_train_valid.loc[val_idx, feature_cols]
        y_val = df_train_valid.loc[val_idx, 'Label']

        _seed = args.seed if args.seed is not None else config.get('seed', 42)
        model = get_model(args.model, seed=_seed)
        model.fit(X_train, y_train)

        val_preds = model.predict_proba(X_val)[:, 1]
        df_train_valid.loc[val_idx, 'Pred_Proba'] = val_preds

        if not df_test.empty:
            test_preds = model.predict_proba(df_test[feature_cols])[:, 1]
            df_test['Pred_Proba'] += test_preds / args.cv_folds

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\n--- Evaluation on Stimulus Level ---")
    stim_metrics = calculate_metrics(df_train_valid['Label'],
                                     df_train_valid['Pred_Proba'])
    for k, v in stim_metrics.items():
        print(f"  Stimulus-Level {k.capitalize()}: {v:.4f}")

    print("\n--- Evaluation on Subject Level (Uniform Weights) ---")
    df_subject_preds = aggregate_predictions(df_train_valid)
    sub_metrics = calculate_metrics(df_subject_preds['Label'],
                                    df_subject_preds['Pred_Proba_Subject'])
    for k, v in sub_metrics.items():
        print(f"  Subject-Level {k.capitalize()}: {v:.4f}")

    # ── Save predictions (OOF + test) ──────────────────────────────────────
    _seed_used = args.seed if args.seed is not None else config.get('seed', 42)
    _default_seed = config.get('seed', 42)
    _out_dir = (f"results/baselines_s{_seed_used}"
                if _seed_used != _default_seed else "results/baselines")
    os.makedirs(_out_dir, exist_ok=True)

    df_train_valid.to_csv(
        f"{_out_dir}/{args.model}_stimulus_val_preds.csv", index=False)
    df_subject_preds.to_csv(
        f"{_out_dir}/{args.model}_subject_val_preds.csv", index=False)

    # OOF subject predictions also saved for Tier 5 meta-learner
    oof_path = f"{_out_dir}/{args.model}_oof_subject_preds.csv"
    df_subject_preds.to_csv(oof_path, index=False)
    print(f"\n[Tier5 OOF] Saved OOF subject predictions to {oof_path}")

    import json
    summary = {
        "stimulus_level": stim_metrics,
        "subject_level": sub_metrics,
        "features_count": len(feature_cols),
    }
    summary_path = f"{_out_dir}/{args.model}_results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved results summary to {summary_path}")

    if not df_test.empty:
        df_test_subject = aggregate_predictions(df_test)
        df_test_subject.to_csv(
            f"{_out_dir}/{args.model}_test_predictions.csv", index=False)
        print(f"Saved test predictions to "
              f"{_out_dir}/{args.model}_test_predictions.csv")


if __name__ == "__main__":
    main()
