import pandas as pd

def apply_spatial_filter(df: pd.DataFrame, screen_w: int = 1024, screen_h: int = 768) -> pd.DataFrame:
    """
    Removes fixations outside screen boundaries:
    0 <= FIX_X < screen_w and 0 <= FIX_Y < screen_h
    """
    if df.empty:
        return df
        
    mask = (
        (df['FIX_X'] >= 0) & (df['FIX_X'] < screen_w) &
        (df['FIX_Y'] >= 0) & (df['FIX_Y'] < screen_h)
    )
    
    removed = (~mask).sum()
    pct_removed = (removed / len(df)) * 100
    print(f"Spatial boundary filter: removed {removed} out of {len(df)} fixations ({pct_removed:.2f}%)")
    
    return df[mask].reset_index(drop=True)
