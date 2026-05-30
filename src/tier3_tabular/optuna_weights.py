import os
import argparse
import yaml
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred_proba):
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

def main():
    parser = argparse.ArgumentParser(description="Tier 3 Tabular Baseline: Optimize category weights using Optuna")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml", help="Path to config file")
    parser.add_argument("--preds", type=str, default="results/baselines/xgboost_stimulus_val_preds.csv", help="Path to validation predictions CSV")
    parser.add_argument("--n-trials", type=int, default=500, help="Number of Optuna trials")
    args = parser.parse_args()
    
    if not os.path.exists(args.preds):
        print(f"Error: Predictions file not found at {args.preds}. Run tabular_models.py first.")
        return
        
    print(f"Loading validation predictions from {args.preds}...")
    df_preds = pd.read_csv(args.preds)
    
    # Group by Subject_ID, Category, Label, and compute average prediction
    grouped = df_preds.groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba'].mean().reset_index()
    
    # Pivot to wide format
    pivoted = grouped.pivot(index=['Subject_ID', 'Label'], columns='Category', values='Pred_Proba').reset_index()
    
    categories = ['Social', 'Manipulated', 'Natural', 'Synthetic']
    for cat in categories:
        if cat not in pivoted.columns:
            print(f"Warning: Category {cat} is missing. Filling with 0.5.")
            pivoted[cat] = 0.5
            
    # Labels and features
    y_true = pivoted['Label'].values
    
    # Define objective function for Optuna
    def objective(trial):
        # Suggest raw weights
        w_social = trial.suggest_float('w_Social', 0.0, 1.0)
        w_manip = trial.suggest_float('w_Manipulated', 0.0, 1.0)
        w_nat = trial.suggest_float('w_Natural', 0.0, 1.0)
        w_synth = trial.suggest_float('w_Synthetic', 0.0, 1.0)
        
        # Normalize weights to sum to 1.0
        total = w_social + w_manip + w_nat + w_synth
        if total == 0:
            return 0.0
            
        alpha = w_social / total
        beta = w_manip / total
        gamma = w_nat / total
        delta = w_synth / total
        
        # Weighted probability aggregation
        p_subject = (
            alpha * pivoted['Social'].values +
            beta * pivoted['Manipulated'].values +
            gamma * pivoted['Natural'].values +
            delta * pivoted['Synthetic'].values
        )
        
        # We optimize for AUC-ROC
        return roc_auc_score(y_true, p_subject)
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    print(f"Running {args.n_trials} Optuna trials to optimize weights...")
    study.optimize(objective, n_trials=args.n_trials)
    
    best_params = study.best_params
    best_auc = study.best_value
    
    # Compute normalized best weights
    total_w = sum(best_params.values())
    norm_weights = {k.split('_')[-1]: v / total_w for k, v in best_params.items()}
    
    print("\n--- Optimization Completed ---")
    print(f"Best Subject-Level AUC: {best_auc:.4f}")
    print("Optimized Weights:")
    for k, v in norm_weights.items():
        print(f"  {k}: {v:.4f}")
        
    # Calculate all metrics using optimized weights
    alpha = norm_weights['Social']
    beta = norm_weights['Manipulated']
    gamma = norm_weights['Natural']
    delta = norm_weights['Synthetic']
    
    p_subject_opt = (
        alpha * pivoted['Social'].values +
        beta * pivoted['Manipulated'].values +
        gamma * pivoted['Natural'].values +
        delta * pivoted['Synthetic'].values
    )
    
    opt_metrics = calculate_metrics(y_true, p_subject_opt)
    print("\nSubject-Level Metrics with Optimized Weights:")
    for k, v in opt_metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
        
    # Save optimized prediction results
    pivoted['Pred_Proba_Subject_Optimized'] = p_subject_opt
    output_path = args.preds.replace("_stimulus_val_preds.csv", "_subject_val_opt_preds.csv")
    pivoted.to_csv(output_path, index=False)
    print(f"\nSaved optimized subject predictions to {output_path}")

if __name__ == "__main__":
    main()
