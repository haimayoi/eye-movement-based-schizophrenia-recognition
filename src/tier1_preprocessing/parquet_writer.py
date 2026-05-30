import os
import pandas as pd

def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the processed DataFrame as a Parquet file using pyarrow.
    """
    if df.empty:
        print("Warning: DataFrame is empty, nothing to save.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Successfully saved {len(df)} fixations to {output_path}")
