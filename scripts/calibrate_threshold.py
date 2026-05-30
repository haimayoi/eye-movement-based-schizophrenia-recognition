import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score

def calibrate():
    parser = argparse.ArgumentParser(description="Calibrate decision threshold for predictions")
    parser.add_argument("--preds", type=str, default="results/cefam_subject_val_predictions.csv", help="Path to predictions CSV")
    args = parser.parse_args()
    
    if not os.path.exists(args.preds):
        print(f"Error: Predictions file not found at {args.preds}")
        return
        
    print(f"Loading predictions from {args.preds}...")
    df = pd.read_csv(args.preds)
    y_true = df['Label'].values
    y_prob = df['Pred_Proba_Subject'].values
    
    print(f"Overall Subject-Level AUC: {roc_auc_score(y_true, y_prob):.4f}")
    
    # Grid search for the best threshold based on Accuracy
    best_th = 0.5
    best_acc = 0.0
    
    for th in np.linspace(0.01, 0.99, 500):
        y_pred = (y_prob >= th).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    # Compute metrics at the best threshold
    y_pred = (y_prob >= best_th).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)
    
    print("\n--- Metrics at Default Threshold (0.5000) ---")
    y_pred_def = (y_prob >= 0.5).astype(int)
    print(f"  Accuracy: {accuracy_score(y_true, y_pred_def):.4f}")
    print(f"  F1-Score: {f1_score(y_true, y_pred_def):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred_def):.4f}")
    print(f"  Recall: {recall_score(y_true, y_pred_def):.4f}")
    
    print(f"\n--- Optimized Metrics at Calibrated Threshold ({best_th:.4f}) ---")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall (Sensitivity): {rec:.4f}")
    print(f"  Specificity: {spec:.4f}")

if __name__ == "__main__":
    calibrate()

