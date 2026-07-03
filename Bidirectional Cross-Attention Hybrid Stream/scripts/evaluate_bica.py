import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier3_tabular.group_kfold import get_subject_folds
from src.models.bica.model import BiCAHSModel
from scripts.train_tier4 import build_flat_features_dict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": spec
    }

def renormalize_bica_trials(trials, global_mean, global_std, fold_mean, fold_std):
    """
    Renormalizes pupil features in BiCA trials fold-wise to prevent global normalisation leakage.
    t['seq'] column 4: pupil size.
    """
    renormalized = []
    for t in trials:
        t_new = t.copy()
        seq = t_new['seq'].copy()
        mask = t_new['mask']
        
        # Backproject globally normalized pupil (column 4) to raw, and normalize with fold stats
        pupil_global = seq[:, 4]
        pupil_fold = (pupil_global * (global_std / (fold_std + 1e-8))) + ((global_mean - fold_mean) / (fold_std + 1e-8))
        
        # Zero out padding elements
        pupil_fold[~mask] = 0.0
        seq[:, 4] = pupil_fold
        t_new['seq'] = seq
        renormalized.append(t_new)
    return renormalized

class BiCADataset(Dataset):
    """
    Sequence dataset for BiCA-HS.
    """
    def __init__(self, df_fixations, flat_features_dict, max_seq_len=200, pupil_mean=0.0, pupil_std=1.0):
        self.max_seq_len = max_seq_len
        
        # Group fixations by Subject_ID and IMAGE (Stimulus_ID)
        self.trials = []
        grouped = df_fixations.groupby(['Subject_ID', 'IMAGE'])
        

        for (sub_id, img), df_trial in grouped:
            key = (int(sub_id), img)
            if key not in flat_features_dict:
                continue
                
            df_sorted = df_trial.sort_values('FIX_INDEX').reset_index(drop=True)
            n_nodes = len(df_sorted)
            
            x = df_sorted['FIX_X'].values / 1024.0
            y = df_sorted['FIX_Y'].values / 768.0
            t = df_sorted['FIX_START'].values / 1000.0 if 'FIX_START' in df_sorted.columns else np.zeros(n_nodes)
            dur = df_sorted['FIX_DURATION'].values / 1000.0
            
            # Reconstruct timestamp if FIX_START is missing
            if 'FIX_START' not in df_sorted.columns and len(dur) > 1:
                t[1:] = np.cumsum(dur[:-1])
                
            pupil = (df_sorted['FIX_PUPIL'].values - pupil_mean) / (pupil_std + 1e-8)
            
            seq_len = min(n_nodes, max_seq_len)
            
            seq_feat = np.column_stack((
                x[:seq_len],
                y[:seq_len],
                t[:seq_len],
                dur[:seq_len],
                pupil[:seq_len]
            )).astype(np.float32)
            
            padded_seq = np.zeros((max_seq_len, 5), dtype=np.float32)
            padded_seq[:seq_len] = seq_feat
            
            mask = np.zeros(max_seq_len, dtype=bool)
            mask[:seq_len] = True
            
            label = int(df_sorted['Label'].iloc[0])
            is_test = int(df_sorted['Is_Test'].iloc[0])
            
            self.trials.append({
                'seq': padded_seq,
                'mask': mask,
                'flat': flat_features_dict[key],
                'label': label,
                'subject_id': int(sub_id),
                'stimulus_id': img,
                'is_test': is_test
            })
            
    def __len__(self):
        return len(self.trials)
        
    def __getitem__(self, idx):
        t = self.trials[idx]
        return (
            torch.tensor(t['seq'], dtype=torch.float32),
            torch.tensor(t['mask'], dtype=torch.bool),
            torch.tensor(t['flat'], dtype=torch.float32),
            torch.tensor(t['label'], dtype=torch.long),
            t['subject_id'],
            t['stimulus_id'],
            t['is_test']
        )

def main():
    parser = argparse.ArgumentParser(description="Evaluate BiCA-HS checkpoints and save predictions")
    parser.add_argument("--config", type=str, default="configs/bica_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config['data']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    parquet_path = data_config.get('parquet_path', 'data/processed/clean_fixations.parquet')
    df_fixations = pd.read_parquet(parquet_path)
    
    features_path = data_config.get('features_stimulus_path', 'data/processed/features_stimulus_level.csv')
    features_subject_path = data_config.get('features_subject_path', 'data/processed/features_subject_level.csv')
    df_stim = pd.read_csv(features_path)
    
    categories_path = data_config.get('categories_path', 'data/metadata/stimulus_categories.csv')
    df_cat = pd.read_csv(categories_path)
    category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))
    
    meta_cols = ['Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test']
    feature_cols = [col for col in df_stim.columns if col not in meta_cols]
    
    flat_features_dict, handcrafted_dim, delta_cols = build_flat_features_dict(
        df_stim, feature_cols, features_subject_path
    )
    print(f"Handcrafted feature dimension: {handcrafted_dim} "
          f"({len(feature_cols)} stimulus + {len(delta_cols)} delta)")
    
    # Calculate global pupil stats for normalization
    global_pupil_mean = df_fixations['FIX_PUPIL'].mean()
    global_pupil_std = df_fixations['FIX_PUPIL'].std()
    
    full_dataset = BiCADataset(
        df_fixations=df_fixations,
        flat_features_dict=flat_features_dict,
        max_seq_len=data_config.get('max_seq_len', 200),
        pupil_mean=global_pupil_mean,
        pupil_std=global_pupil_std
    )
    
    cv_path = os.path.join(data_config.get('raw_dir', 'EMS'), "Train_Valid.xlsx")
    if not os.path.exists(cv_path):
        cv_path = os.path.join(data_config.get('data_dir', 'EMS'), "Train_Valid.xlsx")
    if not os.path.exists(cv_path):
        cv_path = "EMS/Train_Valid.xlsx"
        
    subject_to_fold = get_subject_folds(cv_path)
    train_valid_trials = [t for t in full_dataset.trials if t['is_test'] == 0 and t['subject_id'] in subject_to_fold]
    
    num_split_folds = len(set(subject_to_fold.values()))
    cv_folds = min(config['evaluation'].get('cv_folds', 4), num_split_folds)
    
    class CustomDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            t = self.data[idx]
            return (
                torch.tensor(t['seq'], dtype=torch.float32),
                torch.tensor(t['mask'], dtype=torch.bool),
                torch.tensor(t['flat'], dtype=torch.float32),
                torch.tensor(t['label'], dtype=torch.long),
                t['subject_id'],
                t['stimulus_id'],
                t['is_test']
            )
            
    checkpoint_dir = config['paths'].get('checkpoint_dir', 'Bidirectional Cross-Attention Hybrid Stream/results/checkpoints/')
    
    all_val_preds = []
    cv_results = []
    
    for fold in range(cv_folds):
        print(f"Evaluating Fold {fold}...")
        val_list = [t for t in train_valid_trials if subject_to_fold[t['subject_id']] == fold]
        
        # Compute fold-specific pupil statistics using train subjects of this fold
        train_subjects = [s_id for s_id, f in subject_to_fold.items() if f != fold]
        df_train_pupil = df_fixations[df_fixations['Subject_ID'].isin(train_subjects)]
        fold_pupil_mean = df_train_pupil['FIX_PUPIL'].mean()
        fold_pupil_std = df_train_pupil['FIX_PUPIL'].std()
        
        # Renormalize val list dynamically using fold statistics to match fold checkpoint
        fold_val_list = renormalize_bica_trials(
            val_list, global_pupil_mean, global_pupil_std, fold_pupil_mean, fold_pupil_std)
        
        val_loader = DataLoader(CustomDataset(fold_val_list), batch_size=16, shuffle=False)
        
        d_bio = handcrafted_dim
        model = BiCAHSModel(d_bio=d_bio, config=config).to(device)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"bica_fold_{fold}_best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            continue
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        val_preds = []
        val_targets = []
        val_subj_preds = []
        
        with torch.no_grad():
            for seqs, masks, flats, targets, sub_ids, stim_ids, _ in val_loader:
                seqs = seqs.to(device)
                masks = masks.to(device)
                flats = flats.to(device)
                targets = targets.to(device)
                
                logits, _, _ = model(seqs, masks, flats)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                
                val_preds.extend(probs)
                val_targets.extend(targets.cpu().numpy())
                
                for i in range(seqs.size(0)):
                    val_subj_preds.append({
                        "Subject_ID": int(sub_ids[i]),
                        "Stimulus_ID": stim_ids[i],
                        "Label": int(targets[i].item()),
                        "Pred_Proba": float(probs[i])
                    })
                    
        df_val_subj = pd.DataFrame(val_subj_preds)
        df_val_subj['Category'] = df_val_subj['Stimulus_ID'].map(category_map)
        
        grouped_subj = df_val_subj.groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba'].mean().reset_index()
        pivoted_subj = grouped_subj.pivot(index=['Subject_ID', 'Label'], columns='Category', values='Pred_Proba').reset_index()
        
        for cat in ['Social', 'Manipulated', 'Natural', 'Synthetic']:
            if cat not in pivoted_subj.columns:
                pivoted_subj[cat] = 0.5
                
        pivoted_subj['Pred_Proba_Subject'] = pivoted_subj[['Social', 'Manipulated', 'Natural', 'Synthetic']].mean(axis=1)
        val_auc_subject = roc_auc_score(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
        
        fold_metrics = calculate_metrics(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
        fold_metrics['fold'] = fold
        cv_results.append(fold_metrics)
        
        pivoted_subj['Fold'] = fold
        all_val_preds.append(pivoted_subj)
        print(f"Fold {fold} Subject AUC: {val_auc_subject:.4f}")
        
    if all_val_preds:
        df_all_val_preds = pd.concat(all_val_preds, ignore_index=True)
        results_dir = config['paths'].get('log_dir', 'Bidirectional Cross-Attention Hybrid Stream/results/')
        os.makedirs(results_dir, exist_ok=True)
        val_preds_path = os.path.join(results_dir, "bica_subject_val_predictions.csv")
        df_all_val_preds.to_csv(val_preds_path, index=False)
        print(f"\nSaved overall validation predictions to {val_preds_path}")
        
        overall_metrics = calculate_metrics(df_all_val_preds['Label'].values, df_all_val_preds['Pred_Proba_Subject'].values)
        print("\n--- Overall Subject-Level Validation Metrics ---")
        for k, v in overall_metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")
            
        import json
        summary = {
            "overall": overall_metrics,
            "folds": cv_results,
            "mean_fold_auc": float(np.mean([r['auc'] for r in cv_results])),
            "std_fold_auc": float(np.std([r['auc'] for r in cv_results]))
        }
        summary_path = os.path.join(results_dir, "bica_results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved results summary to {summary_path}")

if __name__ == "__main__":
    main()
