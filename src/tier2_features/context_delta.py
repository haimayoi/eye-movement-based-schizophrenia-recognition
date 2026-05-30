import pandas as pd
import numpy as np

DELTA_PAIRS = [
    ('Social', 'Natural'),
    ('Manipulated', 'Natural'),
    ('Social', 'Synthetic'),
    ('Manipulated', 'Synthetic'),
]

def compute_category_means(df_stim: pd.DataFrame, category_map: dict, feature_cols: list) -> pd.DataFrame:
    """
    Computes the mean feature values per subject, per category.
    category_map: dict of {stimulus_id: category_name}
    """
    df = df_stim.copy()
    df['Category'] = df['Stimulus_ID'].map(category_map)
    
    # Check for unmapped stimuli
    if df['Category'].isnull().any():
        missing = df[df['Category'].isnull()]['Stimulus_ID'].unique()
        print(f"Warning: The following stimuli were not mapped to any category: {missing}")
        # Default to Natural or drop
        df = df.dropna(subset=['Category'])
        
    # Group by Subject_ID and Category, compute mean
    cat_means = df.groupby(['Subject_ID', 'Category'])[feature_cols].mean().reset_index()
    return cat_means

def compute_delta_features(cat_means_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Computes pairwise delta features between different image categories.
    cat_means_df has columns: Subject_ID, Category, feature_1, feature_2, ...
    """
    subjects = cat_means_df['Subject_ID'].unique()
    
    # Pivot to wide format: Subject_ID as index, Category_feature as columns
    pivoted = cat_means_df.pivot(index='Subject_ID', columns='Category', values=feature_cols)
    # Columns are multi-index (feature, Category)
    
    deltas = {}
    for feat in feature_cols:
        for cat_a, cat_b in DELTA_PAIRS:
            delta_name = f"delta_{cat_a}_{cat_b}_{feat}"
            
            # Extract values safely
            col_a = (feat, cat_a)
            col_b = (feat, cat_b)
            
            if col_a in pivoted.columns and col_b in pivoted.columns:
                val_a = pivoted[col_a]
                val_b = pivoted[col_b]
                deltas[delta_name] = val_a - val_b
            else:
                # If category is missing, set delta to 0
                deltas[delta_name] = pd.Series(0.0, index=pivoted.index)
                
    df_deltas = pd.DataFrame(deltas)
    return df_deltas
