import os
import glob
import pandas as pd
from tqdm import tqdm

def parse_subject_and_label(file_path: str, is_test: bool = False):
    """
    Parses Subject_ID and Label from the filename/path.
    Train_Valid rules:
    - Files numbered < 200 are Healthy Controls (HC), Label = 0
    - Files numbered >= 200 are Schizophrenics (SZ), Label = 1
    Test rules:
    - Files are shuffled (e.g., Test_001.xlsx), Label = -1 (unknown)
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    if is_test:
        # e.g. "Test_001" -> subject ID can be parsed from number
        try:
            sub_id = int(base_name.split('_')[-1])
        except ValueError:
            sub_id = base_name
        return sub_id, -1
    else:
        # e.g. "003" -> subject ID 3
        try:
            sub_id = int(base_name)
            label = 1 if sub_id >= 200 else 0
        except ValueError:
            sub_id = base_name
            label = -1
        return sub_id, label

def load_excel_files(dir_path: str, is_test: bool = False) -> pd.DataFrame:
    """
    Loads all Excel files in dir_path and aggregates them into a single DataFrame.
    """
    search_path = os.path.join(dir_path, "*.xlsx")
    files = glob.glob(search_path)
    
    dfs = []
    for f in tqdm(files, desc=f"Loading {os.path.basename(dir_path)}"):
        try:
            df = pd.read_excel(f)
            sub_id, label = parse_subject_and_label(f, is_test=is_test)
            
            # Ensure the required columns exist
            required_cols = ['IMAGE', 'FIX_INDEX', 'FIX_DURATION', 'FIX_X', 'FIX_Y', 'FIX_PUPIL']
            for col in required_cols:
                if col not in df.columns:
                    # In case column names are lowercase
                    matched_col = [c for c in df.columns if c.upper() == col]
                    if matched_col:
                        df = df.rename(columns={matched_col[0]: col})
                    else:
                        raise ValueError(f"Missing column: {col} in {f}")
            
            df['Subject_ID'] = sub_id
            df['Label'] = label
            df['Is_Test'] = 1 if is_test else 0
            
            dfs.append(df[required_cols + ['Subject_ID', 'Label', 'Is_Test']])
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)
