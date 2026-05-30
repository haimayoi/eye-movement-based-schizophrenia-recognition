import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from src.tier3_tabular.group_kfold import get_subject_folds
from src.tier4_advanced.hybrid_model import GNNCEFAMHybridModel
from src.tier4_advanced.stgnn_standalone import STGNNStandaloneModel
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

def main():
    parser = argparse.ArgumentParser(description="Tier 4: Train GNN+CEFAM Hybrid Model")
    parser.add_argument("--config", type=str, default="configs/cefam_config.yaml", help="Path to config file")
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
    graphs_file = os.path.join(data_config.get('graphs_dir', 'data/processed/graphs/'), "graphs.pt")
    if not os.path.exists(graphs_file):
        raise FileNotFoundError(f"Graphs file not found at {graphs_file}. Run graph_builder.py first.")
        
    print(f"Loading spatiotemporal graphs from {graphs_file}...")
    all_graphs = torch.load(graphs_file, weights_only=False)
    
    # Load categories mapping
    categories_path = data_config.get('categories_path', 'data/metadata/stimulus_categories.csv')
    df_cat = pd.read_csv(categories_path)
    category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))

    # Check model name
    is_standalone = config['model'].get('name') == "ST-GNN-Standalone"
    prefix = "stgnn" if is_standalone else "cefam"

    # Conditionally load flat features
    flat_features_dict = {}
    handcrafted_dim = 0
    if not is_standalone:
        features_path = data_config.get('features_stimulus_path', 'data/processed/features_stimulus_level.csv')
        print(f"Loading flat stimulus-level features from {features_path}...")
        df_stim = pd.read_csv(features_path)
        
        # Extract features column names
        meta_cols = ['Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test']
        feature_cols = [col for col in df_stim.columns if col not in meta_cols]
        
        # Create lookup for flat features to align with graph loaders
        # Key: (Subject_ID, Stimulus_ID) -> numpy array of features
        for _, row in df_stim.iterrows():
            sub_id = int(row['Subject_ID'])
            stim_id = row['Stimulus_ID']
            flat_features_dict[(sub_id, stim_id)] = row[feature_cols].values.astype(np.float32)
            
        handcrafted_dim = len(feature_cols)
        print(f"Handcrafted features input dimension: {handcrafted_dim}")
    
    # Get subject folds
    cv_path = os.path.join(data_config.get('raw_dir', 'EMS'), "Train_Valid.xlsx")
    subject_to_fold = get_subject_folds(cv_path)
    
    # Separate train/valid and test graphs
    train_valid_graphs = [g for g in all_graphs if g.is_test == 0 and g.subject_id in subject_to_fold]
    test_graphs = [g for g in all_graphs if g.is_test == 1]
    
    print(f"Total train/valid graphs: {len(train_valid_graphs)}")
    print(f"Total test graphs: {len(test_graphs)}")
    
    # 4-Fold CV Training Loop
    cv_folds = config['evaluation'].get('cv_folds', 4)
    best_fold_aucs = []
    cv_results = []
    all_val_preds = []
    
    for fold in range(cv_folds):
        print(f"\n==================== Training Fold {fold} ====================")
        
        # Split graphs based on subject fold assignment
        train_graphs = [g for g in train_valid_graphs if subject_to_fold[g.subject_id] != fold]
        val_graphs = [g for g in train_valid_graphs if subject_to_fold[g.subject_id] == fold]
        
        print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")
        
        # DataLoaders
        batch_size = training_config.get('batch_size', 32)
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        
        # Model, Loss, Optimizer
        if is_standalone:
            model = STGNNStandaloneModel(config=config).to(device)
        else:
            model = GNNCEFAMHybridModel(handcrafted_dim=handcrafted_dim, config=config).to(device)
        
        loss_cfg = training_config.get('loss', {})
        criterion = FocalLossWithEntropyReg(
            alpha=loss_cfg.get('focal_alpha', 0.25),
            gamma=loss_cfg.get('focal_gamma', 2.0),
            entropy_lambda=loss_cfg.get('entropy_lambda', 0.01)
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config.get('lr', 5e-4)),
            weight_decay=float(training_config.get('weight_decay', 1e-4))
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=training_config.get('T_0', 20),
            T_mult=training_config.get('T_mult', 2),
            eta_min=float(training_config.get('eta_min', 1e-6))
        )
        
        best_val_auc = 0.0
        best_metrics = {}
        best_subj_preds = None
        epochs = training_config.get('epochs', 200)
        patience = training_config.get('patience', 20)
        patience_counter = 0
        
        # Checkpoint directories
        checkpoint_dir = config['paths'].get('checkpoint_dir', 'results/checkpoints/')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_fold_{fold}_best.pt")
        
        # Overfit test if requested
        if args.overfit_batches > 0:
            print(f"Sanity Check: Overfitting on {args.overfit_batches} batch(es)...")
            overfit_graphs = train_graphs[:batch_size * args.overfit_batches]
            train_loader = DataLoader(overfit_graphs, batch_size=batch_size, shuffle=False)
            epochs = 50
            
        for epoch in range(epochs):
            # --- Training Epoch ---
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch in train_loader:
                batch = batch.to(device)
                
                if not is_standalone:
                    # Fetch aligned handcrafted features for the batch
                    hc_batch = []
                    for sub_id, stim_id in zip(batch.subject_id, batch.stimulus_id):
                        s_id = int(sub_id.item()) if torch.is_tensor(sub_id) else int(sub_id)
                        hc_batch.append(flat_features_dict[(s_id, stim_id)])
                    hc_tensor = torch.tensor(np.array(hc_batch), dtype=torch.float32).to(device)
                else:
                    hc_tensor = None
                
                optimizer.zero_grad()
                logits, gnn_attn, _, _ = model(batch, hc_tensor)
                
                # Targets
                targets = batch.y
                
                # Calculate loss (regularize on GNN attention weights)
                loss = criterion(logits, targets, attention_weights=gnn_attn)
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), training_config.get('gradient_clip_val', 1.0))
                optimizer.step()
                
                train_loss += loss.item() * batch.num_graphs
                
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                train_preds.extend(probs)
                train_targets.extend(targets.cpu().numpy())
                
            train_loss = train_loss / len(train_graphs)
            train_auc = roc_auc_score(train_targets, train_preds) if len(np.unique(train_targets)) > 1 else 0.5
            
            # --- Validation Epoch ---
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            # Prediction tracking for Subject-Level aggregation
            val_subj_preds = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    
                    if not is_standalone:
                        hc_batch = []
                        for sub_id, stim_id in zip(batch.subject_id, batch.stimulus_id):
                            s_id = int(sub_id.item()) if torch.is_tensor(sub_id) else int(sub_id)
                            hc_batch.append(flat_features_dict[(s_id, stim_id)])
                        hc_tensor = torch.tensor(np.array(hc_batch), dtype=torch.float32).to(device)
                    else:
                        hc_tensor = None
                    
                    logits, gnn_attn, _, _ = model(batch, hc_tensor)
                    targets = batch.y
                    
                    loss = criterion(logits, targets, attention_weights=gnn_attn)
                    val_loss += loss.item() * batch.num_graphs
                    
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(targets.cpu().numpy())
                    
                    # Record for aggregation
                    for i in range(batch.num_graphs):
                        s_id = int(batch.subject_id[i].item()) if torch.is_tensor(batch.subject_id[i]) else int(batch.subject_id[i])
                        val_subj_preds.append({
                            "Subject_ID": s_id,
                            "Stimulus_ID": batch.stimulus_id[i],
                            "Label": int(batch.y[i].item()),
                            "Pred_Proba": float(probs[i])
                        })
                        
            val_loss = val_loss / len(val_graphs)
            val_auc_trial = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
            
            # Subject-level aggregation (Uniform weights)
            df_val_subj = pd.DataFrame(val_subj_preds)
            df_val_subj['Category'] = df_val_subj['Stimulus_ID'].map(category_map)
            
            # Compute category mean probabilities per subject
            grouped_subj = df_val_subj.groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba'].mean().reset_index()
            pivoted_subj = grouped_subj.pivot(index=['Subject_ID', 'Label'], columns='Category', values='Pred_Proba').reset_index()
            
            # Mean aggregation
            pivoted_subj['Pred_Proba_Subject'] = pivoted_subj[['Social', 'Manipulated', 'Natural', 'Synthetic']].mean(axis=1)
            
            val_auc_subject = roc_auc_score(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
            
            scheduler.step()
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0 or args.overfit_batches > 0:
                print(f"Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val Trial AUC: {val_auc_trial:.4f} | Val Subject AUC: {val_auc_subject:.4f}")
                
            # Sanity check training exit
            if args.overfit_batches > 0 and train_loss < 0.01:
                print(f"Overfitting successful! Final loss: {train_loss:.4f}")
                break
                
            # Save best model based on Subject-Level AUC
            if val_auc_subject > best_val_auc:
                best_val_auc = val_auc_subject
                torch.save(model.state_dict(), checkpoint_path)
                
                # Compute and save all other metrics at this best epoch
                best_metrics = calculate_metrics(pivoted_subj['Label'].values, pivoted_subj['Pred_Proba_Subject'].values)
                best_metrics['epoch'] = epoch + 1
                best_metrics['fold'] = fold
                
                # Save the predictions for these validation subjects
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
    
    # Save overall predictions and metrics
    if all_val_preds:
        df_all_val_preds = pd.concat(all_val_preds, ignore_index=True)
        os.makedirs("results", exist_ok=True)
        val_preds_path = f"results/{prefix}_subject_val_predictions.csv"
        df_all_val_preds.to_csv(val_preds_path, index=False)
        print(f"\nSaved overall validation predictions to {val_preds_path}")
        
        # Calculate overall metrics
        overall_metrics = calculate_metrics(df_all_val_preds['Label'].values, df_all_val_preds['Pred_Proba_Subject'].values)
        print("\n--- Overall Subject-Level Validation Metrics ---")
        for k, v in overall_metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")
            
        # Save JSON results summary
        import json
        summary = {
            "overall": overall_metrics,
            "folds": cv_results,
            "mean_fold_auc": float(np.mean([r['auc'] for r in cv_results])),
            "std_fold_auc": float(np.std([r['auc'] for r in cv_results])),
            "mean_fold_accuracy": float(np.mean([r['accuracy'] for r in cv_results])),
            "std_fold_accuracy": float(np.std([r['accuracy'] for r in cv_results]))
        }
        
        summary_path = f"results/{prefix}_results_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved results summary to {summary_path}")

if __name__ == "__main__":
    main()
