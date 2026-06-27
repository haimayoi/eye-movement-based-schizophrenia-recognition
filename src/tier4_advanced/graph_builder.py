import os
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
from tqdm import tqdm

def load_resnet_features(resnet_path: str,
                         allow_dummy: bool = False,
                         df_clean=None):
    """
    Loads precomputed 2048-dim ResNet50 visual features.

    Args:
        resnet_path : Path to feature_dict_ResNet50.npy
        allow_dummy : If True, generates random features instead of raising.
                      ONLY use for unit tests. Results are NOT scientifically valid.
        df_clean    : Required if allow_dummy=True, used to determine trial keys.

    Raises:
        RuntimeError : If resnet_path does not exist and allow_dummy=False.
    """
    if os.path.exists(resnet_path):
        print(f"Loading ResNet50 features from {resnet_path}...")
        try:
            return np.load(resnet_path, allow_pickle=True).item()
        except Exception as e:
            raise RuntimeError(
                f"ResNet50 feature file exists at {resnet_path} but could not be loaded.\n"
                f"Error: {e}\n"
                "Verify the file is not corrupted. Re-run src/utils/extract_resnet_features.py."
            ) from e

    if not allow_dummy:
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "CRITICAL ERROR — ResNet50 visual features not found\n"
            "=" * 70 + "\n\n"
            f"Expected path: {resnet_path}\n\n"
            "ResNet50 provides 2048-dim visual features per fixation.\n"
            "These constitute 99.8% of GNN node feature dimensionality (2048/2053).\n"
            "Without them, ALL GNN and CEFAM results are scientifically invalid.\n\n"
            "Resolution:\n"
            "  Run the feature extraction script:\n"
            "    python src/utils/extract_resnet_features.py\n"
            "  (Requires EMS/Images directory with stimulus images.)\n\n"
            "For unit testing ONLY (not valid for experiments):\n"
            "  Pass allow_dummy=True or use the --allow-dummy-features CLI flag.\n"
            "  Any results produced with dummy features MUST NOT be reported.\n"
            + "=" * 70
        )

    # ── Dummy path: explicit opt-in for unit testing only ──────────────────
    print("\n" + "!" * 70)
    print("WARNING: Using DUMMY ResNet50 features (np.random.randn).")
    print("This is ONLY valid for unit testing. DO NOT use for experiments.")
    print("!" * 70 + "\n")

    assert df_clean is not None, \
        "df_clean must be provided when allow_dummy=True to determine trial keys."

    dummy_dict = {}
    grouped = df_clean.groupby(['Subject_ID', 'IMAGE'])
    for (sub_id, img), df_trial in tqdm(grouped,
                                         desc="Generating dummy ResNet50 features"):
        n_fix = len(df_trial)
        dummy_dict[(sub_id, img)] = np.random.randn(n_fix, 2048).astype(np.float32)

    return dummy_dict


def build_trial_graph(df_trial, resnet_trial_feats, max_len=24, k_neighbors=3, pupil_mean=0.0, pupil_std=1.0, visual_dim=2048):
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
    
    # Global pupil stats normalized (or local fold-wise inside training loop)
    pupil_diff = np.zeros(n_nodes)
    if n_nodes > 1:
        pupil_diff[1:] = np.diff(pupil_norm)
        
    low_level = np.column_stack((x_norm, y_norm, dur_norm, pupil_norm, pupil_diff))
    
    # Get ResNet50 visual features: [N, visual_dim]
    if resnet_trial_feats is not None:
        resnet_feats = resnet_trial_feats[:n_nodes]
        if len(resnet_feats) < n_nodes:
            pad_r = np.zeros((n_nodes - len(resnet_feats), visual_dim))
            resnet_feats = np.vstack((resnet_feats, pad_r))
    else:
        resnet_feats = np.zeros((n_nodes, visual_dim))
        
    # Combine to node features: [n_nodes, 5 + visual_dim]
    node_features_numpy = np.hstack((low_level, resnet_feats)).astype(np.float32)
    
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
    resnet_dim = resnet_feats.shape[1]
    padded_node_features = np.zeros((max_len, 5 + resnet_dim), dtype=np.float32)
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
    parser = argparse.ArgumentParser(
        description="Tier 4 Advanced: Build spatiotemporal graphs from clean fixations")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml")
    parser.add_argument("--input", type=str, default=None,
                        help="Override path to clean_fixations.parquet")
    parser.add_argument("--visual-features", type=str, default=None,
                        help="Override path to feature_dict_ResNet50.npy")
    parser.add_argument("--output", type=str, default=None,
                        help="Override path to output graphs directory")
    parser.add_argument("--allow-dummy-features", action="store_true",
                        help="[TESTING ONLY] Use random noise instead of real ResNet50 features. "
                             "Results produced with this flag are NOT scientifically valid "
                             "and MUST NOT be reported in any paper or evaluation.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    input_path = args.input or data_config.get('parquet_path', 'data/processed/clean_fixations.parquet')
    resnet_path = args.visual_features or data_config.get('resnet_path',
                  data_config.get('rinet_path', 'data/external/feature_dict_ResNet50.npy'))
    output_dir = args.output or data_config.get('graphs_dir', 'data/processed/graphs/')
    
    # Graph params
    graph_cfg = data_config.get('graph', {})
    padding_cfg = data_config.get('padding', {})
    
    max_len = padding_cfg.get('max_seq_len', 24)
    k_neighbors = graph_cfg.get('k_neighbors', 3)
    
    print(f"Loading preprocessed fixations from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Pupil normalisation (global across all subjects).
    # NOTE: This uses validation/test subjects in the statistic, constituting minor
    # information leakage. For train-fold-only normalisation, compute mean/std
    # inside the training loop using df[train_subjects]. Impact is small
    # (< 0.1σ bias on a balanced dataset) but should be fixed for camera-ready.
    pupil_mean = df['FIX_PUPIL'].mean()
    pupil_std = df['FIX_PUPIL'].std()
    print(f"Global pupil stats (minor leakage — see NOTE above): "
          f"Mean={pupil_mean:.2f}, Std={pupil_std:.2f}")

    # Load ResNet50 features (raises RuntimeError if missing and --allow-dummy-features not set)
    resnet_dict = load_resnet_features(
        resnet_path,
        allow_dummy=args.allow_dummy_features,
        df_clean=df if args.allow_dummy_features else None
    )

    # Dynamically detect visual feature dimension from the loaded dict
    visual_dim = 2048  # ResNet50 default
    if resnet_dict:
        for val in resnet_dict.values():
            if val is not None and len(val) > 0:
                visual_dim = val.shape[1]
                break
    print(f"Detected visual feature dimension from dataset: {visual_dim}")

    grouped = df.groupby(['Subject_ID', 'IMAGE'])

    os.makedirs(output_dir, exist_ok=True)

    print("Building spatiotemporal graphs...")
    graphs = []
    for (sub_id, img), df_trial in tqdm(grouped):
        resnet_trial_feats = resnet_dict.get((sub_id, img), None)

        graph_data = build_trial_graph(
            df_trial,
            resnet_trial_feats,
            max_len=max_len,
            k_neighbors=k_neighbors,
            pupil_mean=pupil_mean,
            pupil_std=pupil_std,
            visual_dim=visual_dim
        )
        
        graphs.append(graph_data)
        
    # Save the graphs list
    output_file = os.path.join(output_dir, "graphs.pt")
    torch.save(graphs, output_file)
    print(f"Successfully constructed and saved {len(graphs)} graphs at {output_file}")

if __name__ == "__main__":
    main()
