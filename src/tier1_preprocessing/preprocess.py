import os
import argparse
import yaml
from src.tier1_preprocessing.excel_loader import load_excel_files
from src.tier1_preprocessing.spatial_filter import apply_spatial_filter
from src.tier1_preprocessing.temporal_filter import apply_temporal_filter
from src.tier1_preprocessing.parquet_writer import save_to_parquet

def main():
    parser = argparse.ArgumentParser(description="Tier 1 Preprocessing: Load raw Excel files and filter fixations")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    
    # Paths from config
    raw_dir = data_config.get('raw_dir', 'EMS')
    parquet_path = data_config.get('parquet_path', 'data/processed/clean_fixations.parquet')
    
    # Raw partitions (since EMS dataset is in EMS folder)
    train_valid_dir = os.path.join(raw_dir, "Train_Valid", "Fixations")
    test_dir = os.path.join(raw_dir, "Test", "Fixations")
    
    print("--- Tier 1: Loading raw data ---")
    df_train_valid = load_excel_files(train_valid_dir, is_test=False)
    df_test = load_excel_files(test_dir, is_test=True)
    
    if df_train_valid.empty and df_test.empty:
        print("Error: No data loaded. Check input directories.")
        return
        
    df_all = []
    if not df_train_valid.empty:
        df_all.append(df_train_valid)
    if not df_test.empty:
        df_all.append(df_test)
        
    df = pd_concat_safe(df_all)
    
    print(f"Total raw fixations loaded: {len(df)}")
    
    # Preprocessing filters
    prep_config = data_config.get('preprocessing', {})
    screen_config = data_config.get('screen', {'width': 1024, 'height': 768})
    
    if prep_config.get('spatial_boundary', True):
        df = apply_spatial_filter(df, screen_config['width'], screen_config['height'])
        
    min_dur = prep_config.get('min_fixation_duration_ms', 50)
    df = apply_temporal_filter(df, min_dur)
    
    # Save output
    save_to_parquet(df, parquet_path)
    print("--- Tier 1 Preprocessing Completed Successfully ---")

def pd_concat_safe(dfs):
    import pandas as pd
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    main()
