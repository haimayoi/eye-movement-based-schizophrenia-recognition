import os
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.tier3_tabular.group_kfold import get_subject_folds

def get_model(model_name: str, seed: int = 42):
    """
    Returns the specified tabular model with default/standard parameters.
    """
    if model_name == "xgboost":
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
        return CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=5,
            random_seed=seed,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def calculate_metrics(y_true, y_pred_proba):
    """
    Computes classification performance metrics.
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": spec
    }

def aggregate_predictions(df_preds: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Aggregates stimulus-level probabilities to subject-level.
    weights: dict of {Category: weight} summing to 1.0. Defaults to uniform.
    """
    if weights is None:
        weights = {"Social": 0.25, "Manipulated": 0.25, "Natural": 0.25, "Synthetic": 0.25}
        
    # Group by Subject_ID and Category, compute mean prediction probability
    grouped = df_preds.groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba'].mean().reset_index()
    
    # Pivot to wide format: Category as columns
    pivoted = grouped.pivot(index=['Subject_ID', 'Label'], columns='Category', values='Pred_Proba').reset_index()
    
    # Compute weighted sum
    p_sz = np.zeros(len(pivoted))
    for cat, w in weights.items():
        if cat in pivoted.columns:
            p_sz += w * pivoted[cat].fillna(0.5).values
            
    pivoted['Pred_Proba_Subject'] = p_sz
    return pivoted

def main():
    parser = argparse.ArgumentParser(description="Tier 3 Tabular Baseline: Train and evaluate models")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default=None, help="Override path to features_stimulus_level.csv")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm", "catboost"], help="Tabular model choice")
    parser.add_argument("--cv-folds", type=int, default=4, help="Number of cross-validation folds")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    features_path = args.data or data_config.get('features_stimulus_path', 'data/processed/features_stimulus_level.csv')
    categories_path = data_config.get('categories_path', 'data/metadata/stimulus_categories.csv')
    cv_path = os.path.join(data_config.get('raw_dir', 'EMS'), "Train_Valid.xlsx")
    
    # Load data
    print(f"Loading stimulus-level features from {features_path}...")
    df_stim = pd.read_csv(features_path)
    
    # Load categories mapping
    df_cat = pd.read_csv(categories_path)
    category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))
    df_stim['Category'] = df_stim['Stimulus_ID'].map(category_map)
    
    # Map subject folds
    print(f"Loading CV splits from {cv_path}...")
    subject_to_fold = get_subject_folds(cv_path)
    
    # Split into Train_Valid and Test set
    df_train_valid = df_stim[(df_stim['Is_Test'] == 0) & (df_stim['Subject_ID'].isin(subject_to_fold.keys()))].copy()
    df_train_valid['Fold'] = df_train_valid['Subject_ID'].map(subject_to_fold)
    
    df_test = df_stim[df_stim['Is_Test'] == 1].copy()
    
    meta_cols = ['Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test', 'Category', 'Fold']
    feature_cols = [col for col in df_train_valid.columns if col not in meta_cols]
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Training on {len(df_train_valid['Subject_ID'].unique())} subjects ({len(df_train_valid)} trials)...")
    
    # Cross-validation prediction storage
    df_train_valid['Pred_Proba'] = 0.0
    
    # For test predictions (average over 4 folds)
    if not df_test.empty:
        df_test['Pred_Proba'] = 0.0
        
    # Perform 4-Fold CV
    for fold in range(args.cv_folds):
        print(f"\n--- Fold {fold} ---")
        
        train_idx = df_train_valid[df_train_valid['Fold'] != fold].index
        val_idx = df_train_valid[df_train_valid['Fold'] == fold].index
        
        X_train = df_train_valid.loc[train_idx, feature_cols]
        y_train = df_train_valid.loc[train_idx, 'Label']
        
        X_val = df_train_valid.loc[val_idx, feature_cols]
        y_val = df_train_valid.loc[val_idx, 'Label']
        
        model = get_model(args.model, seed=config.get('seed', 42))
        model.fit(X_train, y_train)
        
        # Predict on validation set
        val_preds = model.predict_proba(X_val)[:, 1]
        df_train_valid.loc[val_idx, 'Pred_Proba'] = val_preds
        
        # Predict on test set (average over folds)
        if not df_test.empty:
            test_preds = model.predict_proba(df_test[feature_cols])[:, 1]
            df_test['Pred_Proba'] += test_preds / args.cv_folds
            
    # Evaluation
    print("\n--- Evaluation on Stimulus Level ---")
    stim_metrics = calculate_metrics(df_train_valid['Label'], df_train_valid['Pred_Proba'])
    for k, v in stim_metrics.items():
        print(f"Stimulus-Level {k.capitalize()}: {v:.4f}")
        
    print("\n--- Evaluation on Subject Level (Uniform Weights) ---")
    df_subject_preds = aggregate_predictions(df_train_valid)
    sub_metrics = calculate_metrics(df_subject_preds['Label'], df_subject_preds['Pred_Proba_Subject'])
    for k, v in sub_metrics.items():
        print(f"Subject-Level {k.capitalize()}: {v:.4f}")
        
    # Save validation predictions for Optuna tuning
    os.makedirs("results/baselines", exist_ok=True)
    df_train_valid.to_csv(f"results/baselines/{args.model}_stimulus_val_preds.csv", index=False)
    df_subject_preds.to_csv(f"results/baselines/{args.model}_subject_val_preds.csv", index=False)
    
    if not df_test.empty:
        df_test_subject = aggregate_predictions(df_test)
        df_test_subject.to_csv(f"results/baselines/{args.model}_test_predictions.csv", index=False)
        print(f"\nSaved test predictions to results/baselines/{args.model}_test_predictions.csv")

if __name__ == "__main__":
    main()
