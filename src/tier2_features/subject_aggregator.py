import os
import argparse
import yaml
import pandas as pd
from src.tier2_features.context_delta import compute_category_means, compute_delta_features

def main():
    parser = argparse.ArgumentParser(description="Tier 2 Feature Engineering: Aggregate to subject-level features")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, default=None, help="Override path to stimulus-level features CSV")
    parser.add_argument("--categories", type=str, default=None, help="Override path to stimulus categories CSV")
    parser.add_argument("--output", type=str, default=None, help="Override path to output subject-level features CSV")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    input_path = args.input or data_config.get('features_stimulus_path', 'data/processed/features_stimulus_level.csv')
    categories_path = args.categories or data_config.get('categories_path', 'data/metadata/stimulus_categories.csv')
    output_path = args.output or data_config.get('features_subject_path', 'data/processed/features_subject_level.csv')
    
    print(f"Loading stimulus-level features from {input_path}...")
    df_stim = pd.read_csv(input_path)
    
    print(f"Loading category mapping from {categories_path}...")
    df_cat = pd.read_csv(categories_path)
    category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))
    
    # Identify feature columns
    meta_cols = ['Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test']
    feature_cols = [col for col in df_stim.columns if col not in meta_cols]
    
    # 1. Compute category means
    print("Computing mean feature values per subject, per category...")
    cat_means = compute_category_means(df_stim, category_map, feature_cols)
    
    # 2. Pivot category means to wide format
    # Pivot: index=Subject_ID, columns=Category, values=feature_cols
    # Will result in a MultiIndex columns
    pivoted_means = cat_means.pivot(index='Subject_ID', columns='Category', values=feature_cols)
    
    # Flatten multi-index columns: e.g. ("fix_count", "Social") -> "Social_fix_count"
    flat_cols = []
    for feat, cat in pivoted_means.columns:
        flat_cols.append(f"{cat}_{feat}")
    pivoted_means.columns = flat_cols
    
    # 3. Compute delta features
    print("Computing contextual delta features...")
    df_deltas = compute_delta_features(cat_means, feature_cols)
    
    # 4. Merge all together
    # Both pivoted_means and df_deltas are indexed by Subject_ID
    df_subject_feats = pd.concat([pivoted_means, df_deltas], axis=1)
    
    # Get metadata per subject
    df_meta = df_stim[['Subject_ID', 'Label', 'Is_Test']].drop_duplicates().set_index('Subject_ID')
    
    # Combine metadata and features
    df_final = df_meta.join(df_subject_feats).reset_index()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Aggregated subject-level features for {len(df_final)} subjects. Saved to {output_path}")

if __name__ == "__main__":
    main()
