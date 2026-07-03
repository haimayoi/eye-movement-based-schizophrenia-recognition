import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier3_tabular.group_kfold import get_subject_folds
from src.models.bica.model import BiCAHSModel
from src.tier4_advanced.focal_loss import FocalLossWithEntropyReg
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
                continue  # Skip if flat features are not extracted
                
            # Sort fixations chronologically
            df_sorted = df_trial.sort_values('FIX_INDEX').reset_index(drop=True)
            n_nodes = len(df_sorted)
            
            # Extract sequence features:
            # 1. Normalized X [0, 1]
            x = df_sorted['FIX_X'].values / 1024.0
            # 2. Normalized Y [0, 1]
            y = df_sorted['FIX_Y'].values / 768.0
            # 4. Fixation duration in seconds
            dur = df_sorted['FIX_DURATION'].values / 1000.0
            
            # 3. Timestamp in seconds from trial start (reconstructed via cumulative sum of durations)
            t = np.zeros_like(dur)
            if len(dur) > 1:
                t[1:] = np.cumsum(dur[:-1])
                
            # 5. Z-score normalized pupil size
            pupil = (df_sorted['FIX_PUPIL'].values - pupil_mean) / (pupil_std + 1e-8)
            
            seq_len = min(n_nodes, max_seq_len)
            
            seq_feat = np.column_stack((
                x[:seq_len],
                y[:seq_len],
                t[:seq_len],
                dur[:seq_len],
                pupil[:seq_len]
            )).astype(np.float32)
            
            # Pad sequence
            padded_seq = np.zeros((max_seq_len, 5), dtype=np.float32)
            padded_seq[:seq_len] = seq_feat
            
            # Mask: True for real, False for padding
            mask = np.zeros(max_seq_len, dtype=bool)
            mask[:seq_len] = True
            
            # Metadata
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
    parser = argparse.ArgumentParser(description="Tier 4: Train BiCA-HS Transformer Model")
    parser.add_argument("--config", type=str, default="configs/bica_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output results directory")
    parser.add_argument("--overfit-batches", type=int, default=0, help="Sanity check: overfit to N batches")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Override Focal Loss alpha")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Override Focal Loss gamma")
    parser.add_argument("--entropy-lambda", type=float, default=None, help="Override entropy regularization lambda")
    args = parser.parse_args()
    
    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
        
    # Loss config overrides
    if 'loss' not in config['training'] or not isinstance(config['training']['loss'], dict):
        config['training']['loss'] = {}
    if args.focal_alpha is not None:
        config['training']['loss']['focal_alpha'] = args.focal_alpha
    if args.focal_gamma is not None:
        config['training']['loss']['focal_gamma'] = args.focal_gamma
    if args.entropy_lambda is not None:
        config['training']['loss']['entropy_lambda'] = args.entropy_lambda
        
    seed = args.seed or config.get('seed', 42)
    set_seed(seed)

    # Paths — seed-specific subdirectory for non-default seeds
    if args.output_dir:
        results_dir = args.output_dir
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    else:
        _default_seed = config.get('seed', 42)
        _seed_suffix = f'_s{seed}' if seed != _default_seed else ''
        _base_results = 'Bidirectional Cross-Attention Hybrid Stream/results/'
        _base_ckpt = 'Bidirectional Cross-Attention Hybrid Stream/results/checkpoints/'
        results_dir = (config['paths'].get('log_dir', _base_results) if not _seed_suffix
                       else _base_results.rstrip('/') + _seed_suffix + '/')
        checkpoint_dir = (config['paths'].get('checkpoint_dir', _base_ckpt) if not _seed_suffix
                          else _base_ckpt.rstrip('/') + _seed_suffix + '/')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_config = config['data']
    training_config = config['training']
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load datasets
    parquet_path = data_config.get('parquet_path', 'data/processed/clean_fixations.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Clean fixations file not found at {parquet_path}. Run preprocess.py first.")
        
    print(f"Loading preprocessed fixations from {parquet_path}...")
    df_fixations = pd.read_parquet(parquet_path)
    
    features_path = data_config.get('features_stimulus_path', 'data/processed/features_stimulus_level.csv')
    features_subject_path = data_config.get('features_subject_path', 'data/processed/features_subject_level.csv')
    print(f"Loading flat stimulus-level features from {features_path}...")
    df_stim = pd.read_csv(features_path)
    
    # Load categories mapping
    categories_path = data_config.get('categories_path', 'data/metadata/stimulus_categories.csv')
    df_cat = pd.read_csv(categories_path)
    category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))
    
    # Extract features column names and build flat 135 features
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
    
    # Create complete dataset
    max_seq_len = data_config.get('max_seq_len', 200)
    full_dataset = BiCADataset(
        df_fixations=df_fixations,
        flat_features_dict=flat_features_dict,
        max_seq_len=max_seq_len,
        pupil_mean=global_pupil_mean,
        pupil_std=global_pupil_std
    )
    
    # Get subject folds
    cv_path = os.path.join(data_config.get('raw_dir', 'EMS'), "Train_Valid.xlsx")
    if not os.path.exists(cv_path):
        cv_path = os.path.join(data_config.get('data_dir', 'EMS'), "Train_Valid.xlsx")
    if not os.path.exists(cv_path):
        cv_path = "EMS/Train_Valid.xlsx"
        
    subject_to_fold = get_subject_folds(cv_path)
    
    # Separate train/valid and test splits
    train_valid_trials = [t for t in full_dataset.trials if t['is_test'] == 0 and t['subject_id'] in subject_to_fold]
    test_trials = [t for t in full_dataset.trials if t['is_test'] == 1]
    
    print(f"Total train/valid trials: {len(train_valid_trials)}")
    print(f"Total test trials: {len(test_trials)}")
    
    # Cap folds by the actual splits present in the Train_Valid split file (which is 4 sets: Set_0, Set_1, Set_2, Set_3)
    num_split_folds = len(set(subject_to_fold.values()))
    cv_folds = min(config['evaluation'].get('cv_folds', 4), num_split_folds)
    
    best_fold_aucs = []
    cv_results = []
    all_val_preds = []
    
    # Wrap standard PyTorch list datasets
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
            
    for fold in range(cv_folds):
        print(f"\n==================== Training Fold {fold} ====================")
        
        # Split trials based on subject fold assignment
        train_list = [t for t in train_valid_trials if subject_to_fold[t['subject_id']] != fold]
        val_list = [t for t in train_valid_trials if subject_to_fold[t['subject_id']] == fold]
        
        # Compute fold-specific pupil statistics using train subjects of this fold
        train_subjects = [s_id for s_id, f in subject_to_fold.items() if f != fold]
        df_train_pupil = df_fixations[df_fixations['Subject_ID'].isin(train_subjects)]
        fold_pupil_mean = df_train_pupil['FIX_PUPIL'].mean()
        fold_pupil_std = df_train_pupil['FIX_PUPIL'].std()
        print(f"Fold {fold} training pupil stats: Mean={fold_pupil_mean:.4f}, Std={fold_pupil_std:.4f}")
        
        # Dynamically renormalize trials to eliminate pupil normalisation leakage
        train_list = renormalize_bica_trials(
            train_list, global_pupil_mean, global_pupil_std, fold_pupil_mean, fold_pupil_std)
        val_list = renormalize_bica_trials(
            val_list, global_pupil_mean, global_pupil_std, fold_pupil_mean, fold_pupil_std)
        
        print(f"Train trials: {len(train_list)}, Val trials: {len(val_list)}")
        
        batch_size = training_config.get('batch_size', 16)
        train_loader = DataLoader(CustomDataset(train_list), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(val_list), batch_size=batch_size, shuffle=False)
        
        # Model, Loss, Optimizer
        d_bio = handcrafted_dim
        model = BiCAHSModel(d_bio=d_bio, config=config).to(device)
        
        # Use Focal Loss with attention regularization
        loss_cfg = training_config.get('loss', {})
        criterion = FocalLossWithEntropyReg(
            alpha=loss_cfg.get('focal_alpha', 0.5),  # Balanced for diagnostic recall
            gamma=loss_cfg.get('focal_gamma', 2.0),
            entropy_lambda=loss_cfg.get('entropy_lambda', 0.01)
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config.get('lr', 5e-4)),
            weight_decay=float(training_config.get('weight_decay', 1e-4))
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('T_max', 150),
            eta_min=float(training_config.get('eta_min', 1e-6))
        )
        
        best_val_auc = 0.0
        best_metrics = {}
        best_subj_preds = None
        epochs = training_config.get('epochs', 150)
        patience = training_config.get('patience', 15)
        patience_counter = 0
        
        checkpoint_path = os.path.join(checkpoint_dir, f"bica_fold_{fold}_best.pt")
        
        # Checkpoint resumption check (skip training if checkpoint already exists)
        if args.overfit_batches == 0 and os.path.exists(checkpoint_path):
            print(f"Found existing checkpoint at {checkpoint_path}. Resuming and evaluating...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            val_subj_preds = []
            with torch.no_grad():
                for seqs, masks, flats, targets, sub_ids, stim_ids, _ in val_loader:
                    seqs = seqs.to(device)
                    masks = masks.to(device)
                    flats = flats.to(device)
                    targets = targets.to(device)
                    logits, attn_1, attn_2 = model(seqs, masks, flats)
                    loss = criterion(logits, targets, attention_weights=attn_1)
                    val_loss += loss.item() * seqs.size(0)
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(targets.cpu().numpy())
                    for i in range(seqs.size(0)):
                        s_id = int(sub_ids[i])
                        val_subj_preds.append({
                            "Subject_ID": s_id,
                            "Stimulus_ID": stim_ids[i],
                            "Label": int(targets[i].item()),
                            "Pred_Proba": float(probs[i])
                        })
            val_loss = val_loss / len(val_list)
            val_auc_trial = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
            df_val_subj = pd.DataFrame(val_subj_preds)
            df_val_subj['Category'] = df_val_subj['Stimulus_ID'].map(category_map)
            grouped_subj = df_val_subj.groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba'].mean().reset_index()
            pivoted_subj = grouped_subj.pivot(index=['Subject_ID', 'Label'], columns='Category', values='Pred_Proba').reset_index()
            # Ensure all categories exist in validation dataframe
            for cat in ['Social', 'Manipulated', 'Natural', 'Synthetic']:
                if cat not in pivoted_subj.columns:
                    pivoted_subj[cat] = 0.5
            pivoted_subj['Pred_Proba_Subject'] = pivoted_subj[['Social', 'Manipulated', 'Natural', 'Synthetic']].mean(axis=1)
            val_auc_subject = roc_auc_score(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
            print(f"Loaded checkpoint - Val Loss: {val_loss:.4f} | Val Trial AUC: {val_auc_trial:.4f} | Val Subject AUC: {val_auc_subject:.4f}")
            best_val_auc = val_auc_subject
            best_metrics = calculate_metrics(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
            best_metrics['epoch'] = -1
            best_metrics['fold'] = fold
            best_subj_preds = pivoted_subj.copy()
            best_subj_preds['Fold'] = fold
            best_fold_aucs.append(best_val_auc)
            cv_results.append(best_metrics)
            all_val_preds.append(best_subj_preds)
            continue
        
        # Overfit test if requested
        if args.overfit_batches > 0:
            print(f"Sanity Check: Overfitting on {args.overfit_batches} batch(es)...")
            overfit_list = train_list[:batch_size * args.overfit_batches]
            train_loader = DataLoader(CustomDataset(overfit_list), batch_size=batch_size, shuffle=False)
            epochs = 50
            
        for epoch in range(epochs):
            # --- Training Epoch ---
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for seqs, masks, flats, targets, _, _, _ in train_loader:
                seqs = seqs.to(device)
                masks = masks.to(device)
                flats = flats.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                logits, attn_1, attn_2 = model(seqs, masks, flats)
                
                # Apply Focal Loss (regularize on cross-attention weights)
                loss = criterion(logits, targets, attention_weights=attn_1)
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), training_config.get('gradient_clip_val', 1.0))
                optimizer.step()
                
                train_loss += loss.item() * seqs.size(0)
                
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                train_preds.extend(probs)
                train_targets.extend(targets.cpu().numpy())
                
            train_loss = train_loss / len(train_list)
            train_auc = roc_auc_score(train_targets, train_preds) if len(np.unique(train_targets)) > 1 else 0.5
            
            # --- Validation Epoch ---
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            val_subj_preds = []
            
            with torch.no_grad():
                for seqs, masks, flats, targets, sub_ids, stim_ids, _ in val_loader:
                    seqs = seqs.to(device)
                    masks = masks.to(device)
                    flats = flats.to(device)
                    targets = targets.to(device)
                    
                    logits, attn_1, attn_2 = model(seqs, masks, flats)
                    
                    loss = criterion(logits, targets, attention_weights=attn_1)
                    val_loss += loss.item() * seqs.size(0)
                    
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(targets.cpu().numpy())
                    
                    # Record for Subject-Level aggregation
                    for i in range(seqs.size(0)):
                        s_id = int(sub_ids[i])
                        val_subj_preds.append({
                            "Subject_ID": s_id,
                            "Stimulus_ID": stim_ids[i],
                            "Label": int(targets[i].item()),
                            "Pred_Proba": float(probs[i])
                        })
                        
            val_loss = val_loss / len(val_list)
            val_auc_trial = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
            
            # Subject-level aggregation
            df_val_subj = pd.DataFrame(val_subj_preds)
            df_val_subj['Category'] = df_val_subj['Stimulus_ID'].map(category_map)
            
            grouped_subj = df_val_subj.groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba'].mean().reset_index()
            pivoted_subj = grouped_subj.pivot(index=['Subject_ID', 'Label'], columns='Category', values='Pred_Proba').reset_index()
            
            # Ensure all categories exist
            for cat in ['Social', 'Manipulated', 'Natural', 'Synthetic']:
                if cat not in pivoted_subj.columns:
                    pivoted_subj[cat] = 0.5
                    
            pivoted_subj['Pred_Proba_Subject'] = pivoted_subj[['Social', 'Manipulated', 'Natural', 'Synthetic']].mean(axis=1)
            val_auc_subject = roc_auc_score(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
            
            scheduler.step()
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0 or args.overfit_batches > 0:
                print(f"Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val Trial AUC: {val_auc_trial:.4f} | Val Subject AUC: {val_auc_subject:.4f}")
                
            if args.overfit_batches > 0 and train_loss < 0.01:
                print(f"Overfitting successful! Final loss: {train_loss:.4f}")
                break
                
            if val_auc_subject > best_val_auc:
                best_val_auc = val_auc_subject
                torch.save(model.state_dict(), checkpoint_path)
                
                best_metrics = calculate_metrics(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
                best_metrics['epoch'] = epoch + 1
                best_metrics['fold'] = fold
                
                best_subj_preds = pivoted_subj.copy()
                best_subj_preds['Fold'] = fold
                
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (Best Val Subject AUC: {best_val_auc:.4f})")
                break
                
        print(f"Fold {fold} complete. Best Validation Subject AUC: {best_val_auc:.4f}")
        best_fold_aucs.append(best_val_auc)
        cv_results.append(best_metrics)
        if best_subj_preds is not None:
            all_val_preds.append(best_subj_preds)
            
    print(f"\n--- 4-Fold Cross Validation Complete ---")
    print(f"Fold AUCs: {best_fold_aucs}")
    print(f"Mean AUC: {np.mean(best_fold_aucs):.4f} Std: {np.std(best_fold_aucs):.4f}")
    
    # Save overall predictions
    if all_val_preds:
        df_all_val_preds = pd.concat(all_val_preds, ignore_index=True)
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

    # ── Test Set Inference ─────────────────────────────────────────────────────
    if test_trials:
        print(f"\n--- Running Test Set Inference on {len(test_trials)} trials ---")
        test_fold_preds = []

        for fold in range(cv_folds):
            ckpt = os.path.join(checkpoint_dir, f"bica_fold_{fold}_best.pt")
            if not os.path.exists(ckpt):
                print(f"  Fold {fold}: checkpoint not found at {ckpt}, skipping.")
                continue

            d_bio = handcrafted_dim
            m = BiCAHSModel(d_bio=d_bio, config=config).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()

            # Compute fold-specific pupil statistics using train subjects of this fold
            train_subjects = [s_id for s_id, f in subject_to_fold.items() if f != fold]
            df_train_pupil = df_fixations[df_fixations['Subject_ID'].isin(train_subjects)]
            fold_pupil_mean = df_train_pupil['FIX_PUPIL'].mean()
            fold_pupil_std = df_train_pupil['FIX_PUPIL'].std()
            
            # Renormalize test trials for this fold model to eliminate leakage and distribution mismatch
            fold_test_trials = renormalize_bica_trials(
                test_trials, global_pupil_mean, global_pupil_std, fold_pupil_mean, fold_pupil_std)

            test_loader = DataLoader(
                CustomDataset(fold_test_trials),
                batch_size=training_config.get('batch_size', 16),
                shuffle=False)
            fold_subj_preds = []

            with torch.no_grad():
                for seqs, masks, flats, targets, sub_ids, stim_ids, _ in test_loader:
                    seqs = seqs.to(device)
                    masks = masks.to(device)
                    flats = flats.to(device)
                    targets = targets.to(device)

                    logits, _, _ = m(seqs, masks, flats)
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                    for i in range(seqs.size(0)):
                        fold_subj_preds.append({
                            "Subject_ID": int(sub_ids[i]),
                            "Stimulus_ID": stim_ids[i],
                            "Label": int(targets[i].item()),
                            "Pred_Proba": float(probs[i])
                        })

            df_fold = pd.DataFrame(fold_subj_preds)
            df_fold['Category'] = df_fold['Stimulus_ID'].map(category_map)
            grouped = df_fold.groupby(
                ['Subject_ID', 'Category', 'Label'])['Pred_Proba'].mean().reset_index()
            pivoted = grouped.pivot(
                index=['Subject_ID', 'Label'],
                columns='Category', values='Pred_Proba').reset_index()
            for cat in ['Social', 'Manipulated', 'Natural', 'Synthetic']:
                if cat not in pivoted.columns:
                    pivoted[cat] = 0.5
            pivoted['Pred_Proba_Subject'] = pivoted[
                ['Social', 'Manipulated', 'Natural', 'Synthetic']].mean(axis=1)
            pivoted['Fold'] = fold
            test_fold_preds.append(pivoted)
            print(f"  Fold {fold}: {len(pivoted)} test subjects processed.")

        if test_fold_preds:
            df_test_all = pd.concat(test_fold_preds, ignore_index=True)
            df_test_ensemble = (
                df_test_all.groupby('Subject_ID')
                .agg(Label=('Label', 'first'),
                     Pred_Proba_Subject=('Pred_Proba_Subject', 'mean'))
                .reset_index()
            )
            test_path = os.path.join(results_dir, "bica_test_predictions.csv")
            df_test_ensemble.to_csv(test_path, index=False)
            print(f"[Test] Saved test set predictions to {test_path}")

            if len(df_test_ensemble['Label'].unique()) > 1:
                test_metrics = calculate_metrics(
                    df_test_ensemble['Label'].values,
                    df_test_ensemble['Pred_Proba_Subject'].values)
                print("\n--- Test Set Metrics ---")
                for k, v in test_metrics.items():
                    print(f"  {k.capitalize()}: {v:.4f}")
    else:
        print("\n[Test] No test trials found; skipping test inference.")

if __name__ == "__main__":
    main()
