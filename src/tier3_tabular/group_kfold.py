import os
import pandas as pd

def get_subject_folds(excel_path: str = "EMS/Train_Valid.xlsx") -> dict:
    """
    Reads the cross-validation splits defined in EMS/Train_Valid.xlsx
    and returns a mapping of {Subject_ID: fold_index (0, 1, 2, 3)}.
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"CV split file not found at {excel_path}. Please check path.")
        
    df = pd.read_excel(excel_path)
    
    subject_to_fold = {}
    # Iterate through columns: Set_0, Set_1, Set_2, Set_3
    for col in df.columns:
        if col.startswith("Set_"):
            try:
                fold_idx = int(col.split("_")[-1])
            except ValueError:
                continue
                
            # Each cell in the column is a Subject_ID
            subjects = df[col].dropna().astype(int).tolist()
            for sub_id in subjects:
                subject_to_fold[sub_id] = fold_idx
                
    return subject_to_fold

def split_by_subject_folds(df: pd.DataFrame, subject_to_fold: dict):
    """
    Splits the dataframe into train/val indices for each of the 4 folds.
    Only includes subjects that are present in the subject_to_fold mapping.
    """
    # Filter out subjects that are not in the mapping (e.g. test subjects)
    df_cv = df[df['Subject_ID'].isin(subject_to_fold.keys())].copy()
    df_cv['Fold'] = df_cv['Subject_ID'].map(subject_to_fold)
    
    splits = []
    for fold in range(4):
        train_idx = df_cv[df_cv['Fold'] != fold].index.values
        val_idx = df_cv[df_cv['Fold'] == fold].index.values
        splits.append((train_idx, val_idx))
        
    return splits, df_cv
