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

from src.tier3_tabular.group_kfold import get_subject_folds
from src.models.bica.model import BiCAHSModel
from src.tier4_advanced.focal_loss import FocalLossWithEntropyReg

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

class BiCADataset(Dataset):
    """
    Sequence dataset for BiCA-HS.
    """
    def __init__(self, df_fixations, df_stim, feature_cols, max_seq_len=200, pupil_mean=0.0, pupil_std=1.0):
        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        
        # Group fixations by Subject_ID and IMAGE (Stimulus_ID)
        self.trials = []
        grouped = df_fixations.groupby(['Subject_ID', 'IMAGE'])
        
        # Create lookup for flat features: (Subject_ID, Stimulus_ID) -> row features
        flat_features_dict = {}
        for _, row in df_stim.iterrows():
            s_id = int(row['Subject_ID'])
            stim_id = row['Stimulus_ID']
            flat_features_dict[(s_id, stim_id)] = row[feature_cols].values.astype(np.float32)
            
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
            # 3. Timestamp in seconds from trial start
            t = df_sorted['FIX_START'].values / 1000.0
            # 4. Fixation duration in seconds
            dur = df_sorted['FIX_DURATION'].values / 1000.0
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
    parser.add_argument("--overfit-batches", type=int, default=0, help="Sanity check: overfit to N batches")
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
        
    seed = args.seed or config.get('seed', 42)
    set_seed(seed)
    
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
    print(f"Loading flat stimulus-level features from {features_path}...")
    df_stim = pd.read_csv(features_path)
    
    # Load categories mapping
    categories_path = data_config.get('categories_path', 'data/metadata/stimulus_categories.csv')
    df_cat = pd.read_csv(categories_path)
    category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))
    
    # Extract features column names
    meta_cols = ['Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test']
    feature_cols = [col for col in df_stim.columns if col not in meta_cols]
    
    # Calculate global pupil stats for normalization
    pupil_mean = df_fixations['FIX_PUPIL'].mean()
    pupil_std = df_fixations['FIX_PUPIL'].std()
    
    # Create complete dataset
    max_seq_len = data_config.get('max_seq_len', 200)
    full_dataset = BiCADataset(
        df_fixations=df_fixations,
        df_stim=df_stim,
        feature_cols=feature_cols,
        max_seq_len=max_seq_len,
        pupil_mean=pupil_mean,
        pupil_std=pupil_std
    )
    
    # Get subject folds
    cv_path = os.path.join(data_config.get('data_dir', 'EMS'), "Train_Valid.xlsx")
    subject_to_fold = get_subject_folds(cv_path)
    
    # Separate train/valid and test splits
    train_valid_trials = [t for t in full_dataset.trials if t['is_test'] == 0 and t['subject_id'] in subject_to_fold]
    test_trials = [t for t in full_dataset.trials if t['is_test'] == 1]
    
    print(f"Total train/valid trials: {len(train_valid_trials)}")
    print(f"Total test trials: {len(test_trials)}")
    
    cv_folds = config['evaluation'].get('cv_folds', 4)
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
        
        print(f"Train trials: {len(train_list)}, Val trials: {len(val_list)}")
        
        batch_size = training_config.get('batch_size', 16)
        train_loader = DataLoader(CustomDataset(train_list), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(val_list), batch_size=batch_size, shuffle=False)
        
        # Model, Loss, Optimizer
        d_bio = len(feature_cols)
        model = BiCAHSModel(d_bio=d_bio, config=config).to(device)
        
        # Use Focal Loss with attention regularization
        criterion = FocalLossWithEntropyReg(
            alpha=0.5,  # Balanced for diagnostic recall
            gamma=2.0,
            entropy_lambda=0.01
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
        
        # Checkpoint directories
        checkpoint_dir = config['paths'].get('checkpoint_dir', 'Bidirectional Cross-Attention Hybrid Stream/results/checkpoints/')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"bica_fold_{fold}_best.pt")
        
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
