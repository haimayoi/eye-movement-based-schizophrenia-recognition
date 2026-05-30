import os
import argparse
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import ConvexHull
from tqdm import tqdm

def compute_spatial_entropy(x_coords, y_coords, screen_w=1024, screen_h=768, grid_cols=10, grid_rows=10):
    """
    Grid the screen into grid_cols x grid_rows and compute entropy of fixation positions.
    """
    if len(x_coords) == 0:
        return 0.0
        
    x_bins = np.linspace(0, screen_w, grid_cols + 1)
    y_bins = np.linspace(0, screen_h, grid_rows + 1)
    
    # Compute 2D histogram
    hist, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
    
    # Flatten and normalize to get probabilities
    probs = hist.flatten() / len(x_coords)
    probs = probs[probs > 0]
    
    if len(probs) <= 1:
        return 0.0
        
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def compute_convex_hull_area(coords):
    """
    Computes the area of the convex hull enclosing the fixation coordinates.
    """
    if len(coords) < 3:
        return 0.0
    try:
        # Check if all points are collinear
        # Adding a tiny amount of noise can prevent degeneracy issues
        hull = ConvexHull(coords)
        return hull.volume # In 2D, volume represents the area
    except Exception:
        return 0.0

def compute_turning_angles(dx, dy):
    """
    Computes the angle changes between consecutive saccades.
    """
    angles = []
    for i in range(len(dx) - 1):
        u = np.array([dx[i], dy[i]])
        v = np.array([dx[i+1], dy[i+1]])
        
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u > 0 and norm_v > 0:
            cos_theta = np.dot(u, v) / (norm_u * norm_v)
            # Clip cos_theta to avoid float precision issues out of [-1, 1]
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            angles.append(angle)
    return angles

def extract_trial_features(df_trial, screen_w=1024, screen_h=768):
    """
    Extracts features for a single subject's viewing of a single stimulus.
    """
    features = {}
    
    # Sort by FIX_INDEX to ensure chronological order
    df_sorted = df_trial.sort_values('FIX_INDEX')
    
    x = df_sorted['FIX_X'].values
    y = df_sorted['FIX_Y'].values
    dur = df_sorted['FIX_DURATION'].values
    pupil = df_sorted['FIX_PUPIL'].values
    
    count = len(df_sorted)
    
    # Fixation statistics
    features['fix_count'] = count
    features['fix_dur_mean'] = np.mean(dur) if count > 0 else 0.0
    features['fix_dur_median'] = np.median(dur) if count > 0 else 0.0
    features['fix_dur_std'] = np.std(dur) if count > 1 else 0.0
    
    # Saccade dynamics
    dx = np.diff(x)
    dy = np.diff(y)
    amplitudes = np.sqrt(dx**2 + dy**2)
    
    # Saccade velocities (amplitude / duration of target fixation)
    # The first fixation dur corresponds to i=0, so the target fixation for first saccade is at i=1
    sacc_durations = dur[1:]
    velocities = []
    for amp, s_dur in zip(amplitudes, sacc_durations):
        if s_dur > 0:
            velocities.append(amp / s_dur)
        else:
            velocities.append(0.0)
            
    turning_angles = compute_turning_angles(dx, dy)
    
    features['sacc_amp_mean'] = np.mean(amplitudes) if len(amplitudes) > 0 else 0.0
    features['sacc_amp_std'] = np.std(amplitudes) if len(amplitudes) > 1 else 0.0
    features['sacc_vel_mean'] = np.mean(velocities) if len(velocities) > 0 else 0.0
    features['sacc_angle_mean'] = np.mean(turning_angles) if len(turning_angles) > 0 else 0.0
    
    # Scanpath geometry
    features['scanpath_length'] = np.sum(amplitudes)
    
    coords = np.column_stack((x, y))
    features['convex_hull_area'] = compute_convex_hull_area(coords)
    
    center_x, center_y = screen_w / 2, screen_h / 2
    distances_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    features['center_bias'] = np.mean(distances_to_center) if count > 0 else 0.0
    
    features['spatial_entropy'] = compute_spatial_entropy(x, y, screen_w, screen_h)
    
    # Pupil dynamics
    features['pupil_mean'] = np.mean(pupil) if count > 0 else 0.0
    features['pupil_std'] = np.std(pupil) if count > 1 else 0.0
    features['pupil_cv'] = (features['pupil_std'] / features['pupil_mean']) if features['pupil_mean'] > 0 else 0.0
    
    return features

def main():
    parser = argparse.ArgumentParser(description="Tier 2 Feature Engineering: Extract stimulus-level features")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, default=None, help="Override path to input Parquet")
    parser.add_argument("--output", type=str, default=None, help="Override path to output CSV")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    input_path = args.input or data_config.get('parquet_path', 'data/processed/clean_fixations.parquet')
    output_path = args.output or data_config.get('features_stimulus_path', 'data/processed/features_stimulus_level.csv')
    
    screen_w = data_config.get('screen', {}).get('width', 1024)
    screen_h = data_config.get('screen', {}).get('height', 768)
    
    print(f"Loading preprocessed fixations from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Group by Subject_ID and IMAGE (which is the stimulus image file name)
    grouped = df.groupby(['Subject_ID', 'IMAGE', 'Label', 'Is_Test'])
    
    records = []
    print("Extracting features per trial...")
    for (sub_id, img_name, label, is_test), df_trial in tqdm(grouped):
        # Extract features
        trial_feats = extract_trial_features(df_trial, screen_w, screen_h)
        
        # Metadata
        trial_feats['Subject_ID'] = sub_id
        trial_feats['Stimulus_ID'] = img_name
        trial_feats['Label'] = label
        trial_feats['Is_Test'] = is_test
        
        records.append(trial_feats)
        
    df_feats = pd.DataFrame(records)
    
    # Reorder columns to have metadata first
    meta_cols = ['Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test']
    feat_cols = [col for col in df_feats.columns if col not in meta_cols]
    df_feats = df_feats[meta_cols + feat_cols]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_feats.to_csv(output_path, index=False)
    print(f"Extracted features for {len(df_feats)} trials. Saved to {output_path}")

if __name__ == "__main__":
    main()
