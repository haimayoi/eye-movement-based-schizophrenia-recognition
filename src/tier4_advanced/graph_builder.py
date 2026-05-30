import os
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
from tqdm import tqdm

def load_rinet_features(rinet_path: str, df_clean: pd.DataFrame = None):
    """
    Loads precomputed 1056-dim RINet features.
    If the file does not exist, generates a dummy dictionary and saves it.
    """
    if os.path.exists(rinet_path):
        print(f"Loading RINet features from {rinet_path}...")
        try:
            return np.load(rinet_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {rinet_path}: {e}. Fallback to dummy features.")
            
    print(f"Warning: RINet features not found at {rinet_path}. Generating dummy features...")
    os.makedirs(os.path.dirname(rinet_path), exist_ok=True)
    
    # We will generate dummy features for each unique (Subject_ID, IMAGE) trial
    dummy_dict = {}
    if df_clean is not None:
        grouped = df_clean.groupby(['Subject_ID', 'IMAGE'])
        for (sub_id, img), df_trial in tqdm(grouped, desc="Generating dummy RINet features"):
            n_fix = len(df_trial)
            # Generate random features of size 1056 for each fixation
            dummy_dict[(sub_id, img)] = np.random.randn(n_fix, 1056).astype(np.float32)
            
    np.save(rinet_path, dummy_dict)
    print(f"Saved dummy RINet features to {rinet_path}")
    return dummy_dict

def build_trial_graph(df_trial, rinet_trial_feats, max_len=24, k_neighbors=3, pupil_mean=0.0, pupil_std=1.0):
    """
    Builds a spatiotemporal PyG Graph from a single trial.
    """
    # Sort chronologically
    df_sorted = df_trial.sort_values('FIX_INDEX').reset_index(drop=True)
    n_fixations = len(df_sorted)
    
    # Coordinates and features
    x = df_sorted['FIX_X'].values
    y = df_sorted['FIX_Y'].values
    dur = df_sorted['FIX_DURATION'].values
    pupil = df_sorted['FIX_PUPIL'].values
    
    # Truncate if necessary
    n_nodes = min(n_fixations, max_len)
    
    x = x[:n_nodes]
    y = y[:n_nodes]
    dur = dur[:n_nodes]
    pupil = pupil[:n_nodes]
    
    # Compute low-level node features: [N, 5]
    # Normalize:
    x_norm = x / 1024.0
    y_norm = y / 768.0
    dur_norm = dur / 1000.0
    pupil_norm = (pupil - pupil_mean) / (pupil_std + 1e-8)
    
    pupil_diff = np.zeros(n_nodes)
    if n_nodes > 1:
        pupil_diff[1:] = np.diff(pupil_norm)
        
    low_level = np.column_stack((x_norm, y_norm, dur_norm, pupil_norm, pupil_diff))
    
    # Get RINet visual features: [N, 1056]
    if rinet_trial_feats is not None:
        # Align length
        rinet_feats = rinet_trial_feats[:n_nodes]
        if len(rinet_feats) < n_nodes:
            pad_r = np.zeros((n_nodes - len(rinet_feats), 1056))
            rinet_feats = np.vstack((rinet_feats, pad_r))
    else:
        rinet_feats = np.zeros((n_nodes, 1056))
        
    # Combine to node features: [n_nodes, 1061]
    # We will split this in the model, or project it.
    node_features_numpy = np.hstack((low_level, rinet_feats)).astype(np.float32)
    
    # Construct Edges
    edges = []
    edge_attrs = []
    
    # 1. Temporal (Sequential) Edges: i -> i+1 (directed)
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        
        # Calculate saccade metrics
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        amp = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        edge_attrs.append([amp, angle])
        
    # 2. Spatial (k-NN) Edges (undirected)
    if k_neighbors > 0 and n_nodes > 1:
        coords = np.column_stack((x, y))
        tree = KDTree(coords)
        for i in range(n_nodes):
            # Query k+1 nearest neighbors (since nearest neighbor is self)
            k_val = min(k_neighbors + 1, n_nodes)
            dists, neighbors = tree.query(coords[i], k=k_val)
            
            # If query returns a 1D array (when k=1)
            if np.isscalar(neighbors):
                neighbors = [neighbors]
                
            for j in neighbors:
                if j == i:
                    continue
                # Add edges both ways (undirected)
                edges.append([i, j])
                edges.append([j, i])
                
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                amp = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                
                edge_attrs.append([amp, angle])
                edge_attrs.append([amp, -angle]) # opposite direction
                
    if len(edges) == 0:
        # Edge case: single node trial
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        
    # Apply Padding to max_len
    # Create mask (True for real, False for padded)
    mask = torch.ones(max_len, dtype=torch.bool)
    mask[n_nodes:] = False
    
    # Pad node features with zeros
    padded_node_features = np.zeros((max_len, 1061), dtype=np.float32)
    padded_node_features[:n_nodes] = node_features_numpy
    x_tensor = torch.tensor(padded_node_features, dtype=torch.float32)
    
    # Metadata
    label = df_sorted['Label'].iloc[0]
    sub_id = df_sorted['Subject_ID'].iloc[0]
    img = df_sorted['IMAGE'].iloc[0]
    is_test = df_sorted['Is_Test'].iloc[0]
    
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mask=mask,
        y=torch.tensor([label], dtype=torch.long),
        subject_id=sub_id,
        stimulus_id=img,
        is_test=is_test
    )
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Tier 4 Advanced: Build graphs from clean fixations")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, default=None, help="Override path to clean_fixations.parquet")
    parser.add_argument("--rinet", type=str, default=None, help="Override path to feature_dict_RINet.npy")
    parser.add_argument("--output", type=str, default=None, help="Override path to output graphs directory")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    input_path = args.input or data_config.get('parquet_path', 'data/processed/clean_fixations.parquet')
    rinet_path = args.rinet or data_config.get('rinet_path', 'data/external/feature_dict_RINet.npy')
    output_dir = args.output or data_config.get('graphs_dir', 'data/processed/graphs/')
    
    # Graph params
    graph_cfg = data_config.get('graph', {})
    padding_cfg = data_config.get('padding', {})
    
    max_len = padding_cfg.get('max_seq_len', 24)
    k_neighbors = graph_cfg.get('k_neighbors', 3)
    
    print(f"Loading preprocessed fixations from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Calculate global pupil stats for normalization
    pupil_mean = df['FIX_PUPIL'].mean()
    pupil_std = df['FIX_PUPIL'].std()
    print(f"Global pupil normalization stats: Mean = {pupil_mean:.2f}, Std = {pupil_std:.2f}")
    
    # Load RINet features
    rinet_dict = load_rinet_features(rinet_path, df)
    
    grouped = df.groupby(['Subject_ID', 'IMAGE'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Building spatiotemporal graphs...")
    graphs = []
    for (sub_id, img), df_trial in tqdm(grouped):
        # Retrieve RINet feature mapping
        rinet_trial_feats = rinet_dict.get((sub_id, img), None)
        
        # Build graph
        graph_data = build_trial_graph(
            df_trial, 
            rinet_trial_feats, 
            max_len=max_len, 
            k_neighbors=k_neighbors,
            pupil_mean=pupil_mean,
            pupil_std=pupil_std
        )
        
        graphs.append(graph_data)
        
    # Save the graphs list
    output_file = os.path.join(output_dir, "graphs.pt")
    torch.save(graphs, output_file)
    print(f"Successfully constructed and saved {len(graphs)} graphs at {output_file}")

if __name__ == "__main__":
    main()
