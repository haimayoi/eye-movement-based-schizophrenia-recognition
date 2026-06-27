import os
import pandas as pd
import numpy as np
import xgboost as xgb
from src.tier3_tabular.tabular_models import load_features_with_delta
from src.tier3_tabular.group_kfold import get_subject_folds

def main():
    features_stimulus_path = 'data/processed/features_stimulus_level.csv'
    features_subject_path = 'data/processed/features_subject_level.csv'
    categories_path = 'data/metadata/stimulus_categories.csv'
    cv_path = 'EMS/Train_Valid.xlsx'
    
    if not os.path.exists(features_stimulus_path):
        print("Error: Features path not found.")
        return
        
    # Load features (135 features total)
    df_stim, feature_cols = load_features_with_delta(
        features_stimulus_path, features_subject_path, categories_path)
        
    subject_to_fold = get_subject_folds(cv_path)
    df_train_valid = df_stim[(df_stim['Is_Test'] == 0) & (df_stim['Subject_ID'].isin(subject_to_fold))].copy()
    df_train_valid['Fold'] = df_train_valid['Subject_ID'].map(subject_to_fold)
    
    # Exclude meta
    all_meta = {'Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test', 'Category', 'Fold'}
    feature_cols = [c for c in df_train_valid.columns if c not in all_meta]
    
    X = df_train_valid[feature_cols]
    y = df_train_valid['Label']
    
    # Train XGBoost on 4 folds and collect feature importances
    importances = np.zeros(len(feature_cols))
    
    for fold in range(4):
        train_idx = df_train_valid[df_train_valid['Fold'] != fold].index
        val_idx = df_train_valid[df_train_valid['Fold'] == fold].index
        
        X_train = df_train_valid.loc[train_idx, feature_cols]
        y_train = df_train_valid.loc[train_idx, 'Label']
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        importances += model.feature_importances_ / 4.0
        
    # Sort importances
    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    print("Top 20 Features by XGBoost Importance:")
    for idx, row in df_imp.head(20).iterrows():
        print(f"{idx+1:02d}. {row['Feature']}: {row['Importance']:.6f}")

if __name__ == '__main__':
    main()
