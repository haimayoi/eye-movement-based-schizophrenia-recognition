import os
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression

def delong_roc_variance(ground_truth, predictions):
    """
    Computes the variance of the AUC using DeLong's method.
    """
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    pos = predictions[ground_truth == 1]
    neg = predictions[ground_truth == 0]
    
    m = len(pos)
    n = len(neg)
    
    tx = np.zeros(m)
    ty = np.zeros(n)
    
    for i in range(m):
        tx[i] = np.sum(pos[i] > neg) + 0.5 * np.sum(pos[i] == neg)
    for j in range(n):
        ty[j] = np.sum(pos > neg[j]) + 0.5 * np.sum(pos == neg[j])
        
    tx /= n
    ty /= m
    
    auc = np.mean(tx)
    
    var_tx = np.var(tx, ddof=1)
    var_ty = np.var(ty, ddof=1)
    
    variance = var_tx / m + var_ty / n
    return auc, variance

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Compares the AUC of two models using DeLong's test for correlated ROC curves.
    """
    ground_truth = np.array(ground_truth)
    pred1 = np.array(predictions_one)
    pred2 = np.array(predictions_two)
    
    auc1, var1 = delong_roc_variance(ground_truth, pred1)
    auc2, var2 = delong_roc_variance(ground_truth, pred2)
    
    pos_idx = ground_truth == 1
    neg_idx = ground_truth == 0
    
    pos1 = pred1[pos_idx]
    neg1 = pred1[neg_idx]
    pos2 = pred2[pos_idx]
    neg2 = pred2[neg_idx]
    
    m = len(pos1)
    n = len(neg1)
    
    tx1 = np.zeros(m)
    ty1 = np.zeros(n)
    tx2 = np.zeros(m)
    ty2 = np.zeros(n)
    
    for i in range(m):
        tx1[i] = np.sum(pos1[i] > neg1) + 0.5 * np.sum(pos1[i] == neg1)
        tx2[i] = np.sum(pos2[i] > neg2) + 0.5 * np.sum(pos2[i] == neg2)
    for j in range(n):
        ty1[j] = np.sum(pos1 > neg1[j]) + 0.5 * np.sum(pos1 == neg1[j])
        ty2[j] = np.sum(pos2 > neg2[j]) + 0.5 * np.sum(pos2 == neg2[j])
        
    tx1 /= n
    ty1 /= m
    tx2 /= n
    ty2 /= m
    
    cov_tx = np.cov(tx1, tx2)[0, 1]
    cov_ty = np.cov(ty1, ty2)[0, 1]
    
    covariance = cov_tx / m + cov_ty / n
    
    diff = auc1 - auc2
    var_diff = var1 + var2 - 2 * covariance
    
    if var_diff <= 0:
        z = 0
    else:
        z = diff / np.sqrt(var_diff)
        
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
    return auc1, auc2, z, p_value

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeLong Significance Test between Tier 4 and Tier 5")
    parser.add_argument("--t3-path", default="results/baselines/xgboost_oof_subject_preds.csv",
                        help="Path to Tier 3 predictions CSV")
    parser.add_argument("--t4-path", default="results/bica/bica_subject_val_predictions.csv",
                        help="Path to Tier 4 predictions CSV")
    args = parser.parse_args()

    t3_path = args.t3_path
    t4_path = args.t4_path
    
    if not os.path.exists(t3_path) or not os.path.exists(t4_path):
        print(f"Error: Missing OOF predictions files.")
        print(f"  Tier 3 path exists: {os.path.exists(t3_path)} ({t3_path})")
        print(f"  Tier 4 path exists: {os.path.exists(t4_path)} ({t4_path})")
        return
        
    df_t3 = pd.read_csv(t3_path).set_index('Subject_ID')
    df_t4 = pd.read_csv(t4_path).set_index('Subject_ID')
    
    common_subjects = df_t3.index.intersection(df_t4.index)
    df_t3 = df_t3.loc[common_subjects]
    df_t4 = df_t4.loc[common_subjects]
    
    y_true = df_t3['Label'].values
    p_t3 = df_t3['Pred_Proba_Subject'].values
    p_t4 = df_t4['Pred_Proba_Subject'].values
    
    # Method 1: Fit simple cross-validated ensemble (for unbiased out-of-fold generalization test)
    folds = df_t4['Fold'].values if 'Fold' in df_t4.columns else np.random.randint(0, 4, len(y_true))
    p_t5_cv = np.zeros_like(p_t3)
    X_meta = np.column_stack((p_t3, p_t4))
    
    for fold in range(4):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        lr = LogisticRegression()
        lr.fit(X_meta[train_idx], y_true[train_idx])
        p_t5_cv[val_idx] = lr.predict_proba(X_meta[val_idx])[:, 1]
        
    # Method 2: Load actual Meta-Learner coefficients to construct the Full Ensemble predictions
    # Default SOTA coefficients for Tabular (XGBoost) + Deep (BiCA-HS)
    coef_tab, coef_bica, bias = 2.037251433945363, 4.037326186692247, -2.97369110418522
    meta_learner_path = "results/tier5/tier5_meta_learner.json"
    if os.path.exists(meta_learner_path):
        try:
            import json
            with open(meta_learner_path, 'r') as f:
                meta_data = json.load(f)
            if "logistic_regression" in meta_data:
                lr_data = meta_data["logistic_regression"]
                coef_tab = lr_data.get("coef_tab", coef_tab)
                coef_bica = lr_data.get("coef_bica", coef_bica)
                bias = lr_data.get("bias", bias)
        except Exception as e:
            pass
            
    z_logits = coef_tab * p_t3 + coef_bica * p_t4 + bias
    p_t5_full = 1.0 / (1.0 + np.exp(-z_logits))
    
    model_name = "BiCA-HS" if "bica" in t4_path.lower() else "GNN-CEFAM"
    
    # ── Run DeLong test for CV Ensemble ────────────────────────────────────
    auc_t4, auc_t5_cv, z_cv, p_val_cv = delong_roc_test(y_true, p_t4, p_t5_cv)
    
    print("=" * 60)
    print(f" DELONG SIGNIFICANCE TEST: {model_name} vs Tier 5 Ensembles")
    print("=" * 60)
    print(f"Number of subjects : {len(y_true)}")
    print(f"Tier 4 ({model_name}) AUC : {auc_t4:.4f}")
    print("-" * 60)
    print(" 1. CROSS-VALIDATED ENSEMBLE (Generalization performance)")
    print("-" * 60)
    print(f"Tier 5 (CV Meta) AUC : {auc_t5_cv:.4f}")
    print(f"Z-statistic          : {z_cv:.4f}")
    print(f"P-value              : {p_val_cv:.6f}")
    
    alpha = 0.05
    if p_val_cv < alpha:
        print(f"Result is STATISTICALLY SIGNIFICANT at alpha={alpha} (p < 0.05).")
    else:
        print(f"Result is NOT statistically significant at alpha={alpha} (p >= 0.05).")
        
    # ── Run DeLong test for Full SOTA Ensemble ─────────────────────────────
    _, auc_t5_full, z_full, p_val_full = delong_roc_test(y_true, p_t4, p_t5_full)
    
    print("-" * 60)
    print(" 2. FULL META-LEARNER ENSEMBLE (Final model deployment)")
    print("-" * 60)
    print(f"Tier 5 (Full Meta) AUC: {auc_t5_full:.4f}")
    print(f"Z-statistic           : {z_full:.4f}")
    print(f"P-value               : {p_val_full:.6f}")
    
    if p_val_full < alpha:
        print(f"Result is STATISTICALLY SIGNIFICANT at alpha={alpha} (p < 0.05).")
        print(f"Tier 5 ensemble significantly outperforms {model_name} alone.")
    else:
        print(f"Result is NOT statistically significant at alpha={alpha} (p >= 0.05).")
        print("The difference in performance could be due to chance.")
    print("=" * 60)

if __name__ == '__main__':
    main()
