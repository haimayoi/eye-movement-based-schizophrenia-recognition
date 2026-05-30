import pandas as pd

def apply_temporal_filter(df: pd.DataFrame, min_duration: int = 50) -> pd.DataFrame:
    """
    Removes fixations with duration less than min_duration ms.
    """
    if df.empty:
        return df
        
    mask = df['FIX_DURATION'] >= min_duration
    
    removed = (~mask).sum()
    pct_removed = (removed / len(df)) * 100
    print(f"Temporal duration filter: removed {removed} out of {len(df)} fixations ({pct_removed:.2f}%)")
    
    return df[mask].reset_index(drop=True)
