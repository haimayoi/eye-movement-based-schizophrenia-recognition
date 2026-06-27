"""
Tier 4 Training — GNN+CEFAM Hybrid or ST-GNN Standalone
=========================================================
Changes vs. original:
  1. Delta features (features_subject_level.csv) are now loaded and
     appended to the stimulus-level flat features, so the HandcraftedStream
     receives [stimulus_features ∥ delta_features] per trial.
  2. OOF (out-of-fold) subject-level predictions are saved to
     results/{prefix}/{prefix}_oof_subject_preds.csv for Tier 5.
  3. train_tier4.py also writes an embedding file (z_final per subject)
     for explainability (t-SNE / UMAP).
  4. CEFAM attention weights (attn_1) are saved per fold for visualization.
"""

import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import sys
# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tier3_tabular.group_kfold import get_subject_folds
from src.tier4_advanced.hybrid_model import GNNCEFAMHybridModel
from src.tier4_advanced.stgnn_standalone import STGNNStandaloneModel
from src.tier4_advanced.focal_loss import FocalLossWithEntropyReg


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"accuracy": acc, "auc": auc, "f1": f1,
            "precision": prec, "recall": rec, "specificity": spec}


# ─────────────────────────────────────────────────────────────────────────────
# Delta-feature loading helper
# ─────────────────────────────────────────────────────────────────────────────

def build_flat_features_dict(
    df_stim: pd.DataFrame,
    feature_cols: list,
    features_subject_path: str,
):
    """
    Returns:
        flat_features_dict : {(Subject_ID, Stimulus_ID): np.ndarray}
        total_dim          : len(feature_cols) + len(delta_cols)
        delta_cols         : list of delta column names (may be empty)
    """
    delta_cols = []
    df_delta_lookup = {}  # {subject_id: np.ndarray of delta features}

    if os.path.exists(features_subject_path):
        df_subj = pd.read_csv(features_subject_path)
        all_meta = {'Subject_ID', 'Label', 'Is_Test', 'Stimulus_ID', 'Category', 'Fold'}
        delta_cols = [c for c in df_subj.columns
                      if c not in all_meta and (
                          c.startswith('delta_')
                          or c.startswith('Social_') or c.startswith('Natural_')
                          or c.startswith('Synthetic_') or c.startswith('Manipulated_'))]

        if delta_cols:
            for _, row in df_subj.drop_duplicates('Subject_ID').iterrows():
                s_id = int(row['Subject_ID'])
                df_delta_lookup[s_id] = row[delta_cols].values.astype(np.float32)
            print(f"  [Delta] Loaded {len(delta_cols)} delta features from "
                  f"{features_subject_path}")
        else:
            print(f"  [Delta] No delta columns in {features_subject_path}")
    else:
        print(f"  [Delta] Subject-level file not found; skipping delta features.")

    n_delta = len(delta_cols)
    total_dim = len(feature_cols) + n_delta

    flat_features_dict = {}
    for _, row in df_stim.iterrows():
        s_id = int(row['Subject_ID'])
        stim_id = row['Stimulus_ID']
        stim_feat = row[feature_cols].values.astype(np.float32)
        if n_delta > 0:
            delta_feat = df_delta_lookup.get(s_id, np.zeros(n_delta, dtype=np.float32))
            combined = np.concatenate([stim_feat, delta_feat])
        else:
            combined = stim_feat
        flat_features_dict[(s_id, stim_id)] = combined

    return flat_features_dict, total_dim, delta_cols


def renormalize_graphs_pupil(graphs, global_mean, global_std, fold_mean, fold_std):
    """
    Renormalizes pupil features fold-wise to prevent global normalisation leakage.
    Column 3: pupil_norm, Column 4: pupil_diff.
    """
    renormalized = []
    for g in graphs:
        g_new = g.clone()
        x = g_new.x.clone()
        mask = g_new.mask
        
        # Extract global pupil norm
        pupil_global = x[:, 3]
        
        # Backproject to raw pupil and normalize with fold stats
        pupil_fold = (pupil_global * (global_std / (fold_std + 1e-8))) + ((global_mean - fold_mean) / (fold_std + 1e-8))
        
        # Zero out padding elements
        pupil_fold[~mask] = 0.0
        x[:, 3] = pupil_fold
        
        # Recompute pupil diff (saccade-to-saccade change)
        pupil_diff = torch.zeros_like(pupil_fold)
        n_nodes = int(mask.sum())
        if n_nodes > 1:
            pupil_diff[1:n_nodes] = torch.diff(pupil_fold[:n_nodes])
        x[:, 4] = pupil_diff
        
        g_new.x = x
        renormalized.append(g_new)
    return renormalized


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tier 4: Train GNN+CEFAM Hybrid Model")
    parser.add_argument("--config", type=str,
                        default="configs/cefam_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit-batches", type=int, default=0,
                        help="Sanity check: overfit to N batches")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load clean fixations parquet to compute global pupil statistics ────
    parquet_path = data_config.get('parquet_path', 'data/processed/clean_fixations.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Clean fixations parquet not found at {parquet_path}.")
    df_clean = pd.read_parquet(parquet_path)
    global_pupil_mean = df_clean['FIX_PUPIL'].mean()
    global_pupil_std = df_clean['FIX_PUPIL'].std()
    print(f"Loaded clean fixations. Global pupil mean: {global_pupil_mean:.4f}, std: {global_pupil_std:.4f}")

    # ── Load graphs ───────────────────────────────────────────────────────
    graphs_file = os.path.join(
        data_config.get('graphs_dir', 'data/processed/graphs/'), "graphs.pt")
    if not os.path.exists(graphs_file):
        raise FileNotFoundError(
            f"Graphs file not found at {graphs_file}. Run graph_builder.py first.")

    print(f"Loading spatiotemporal graphs from {graphs_file}...")
    all_graphs = torch.load(graphs_file, weights_only=False)

    # Category map
    categories_path = data_config.get('categories_path',
                                       'data/metadata/stimulus_categories.csv')
    df_cat = pd.read_csv(categories_path)
    category_map = dict(zip(df_cat['Image_Name'], df_cat['Category']))

    # Model name
    is_standalone = config['model'].get('name') == "ST-GNN-Standalone"
    prefix = "stgnn" if is_standalone else "cefam"

    # ── Load flat features (stimulus + delta) ─────────────────────────────
    flat_features_dict = {}
    handcrafted_dim = 0

    if not is_standalone:
        features_path = data_config.get('features_stimulus_path',
                                         'data/processed/features_stimulus_level.csv')
        features_subject_path = data_config.get('features_subject_path',
                                                 'data/processed/features_subject_level.csv')
        print(f"Loading flat stimulus-level features from {features_path}...")
        df_stim = pd.read_csv(features_path)

        meta_cols_set = {'Subject_ID', 'Stimulus_ID', 'Label', 'Is_Test'}
        feature_cols = [c for c in df_stim.columns if c not in meta_cols_set]

        flat_features_dict, handcrafted_dim, delta_cols = build_flat_features_dict(
            df_stim, feature_cols, features_subject_path)
        print(f"Handcrafted feature dimension: {handcrafted_dim} "
              f"({len(feature_cols)} stimulus + {len(delta_cols)} delta)")

    # ── Subject fold mapping ───────────────────────────────────────────────
    cv_path = os.path.join(data_config.get('raw_dir', 'EMS'), "Train_Valid.xlsx")
    subject_to_fold = get_subject_folds(cv_path)

    train_valid_graphs = [g for g in all_graphs
                          if g.is_test == 0 and g.subject_id in subject_to_fold]
    test_graphs = [g for g in all_graphs if g.is_test == 1]
    print(f"Total train/valid graphs: {len(train_valid_graphs)}")
    print(f"Total test graphs: {len(test_graphs)}")

    # ── Cross-validation ───────────────────────────────────────────────────
    cv_folds = config['evaluation'].get('cv_folds', 4)
    best_fold_aucs = []
    cv_results = []
    all_val_preds = []

    # Paths — seed-specific subdirectory for non-default seeds
    _default_seed = config.get('seed', 42)
    _seed_suffix = f'_s{seed}' if seed != _default_seed else ''
    results_dir = (config['paths'].get('log_dir', f'results/{prefix}/') if not _seed_suffix
                   else f'results/{prefix}{_seed_suffix}/')
    checkpoint_dir = (config['paths'].get('checkpoint_dir', f'results/{prefix}/checkpoints/') if not _seed_suffix
                      else f'results/{prefix}{_seed_suffix}/checkpoints/')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for fold in range(cv_folds):
        print(f"\n==================== Training Fold {fold} ====================")

        train_graphs = [g for g in train_valid_graphs
                        if subject_to_fold[g.subject_id] != fold]
        val_graphs = [g for g in train_valid_graphs
                      if subject_to_fold[g.subject_id] == fold]
        
        # Compute fold-specific pupil statistics using train subjects of this fold
        train_subjects = [s_id for s_id, f in subject_to_fold.items() if f != fold]
        df_train_pupil = df_clean[df_clean['Subject_ID'].isin(train_subjects)]
        fold_pupil_mean = df_train_pupil['FIX_PUPIL'].mean()
        fold_pupil_std = df_train_pupil['FIX_PUPIL'].std()
        print(f"Fold {fold} training pupil stats: Mean={fold_pupil_mean:.4f}, Std={fold_pupil_std:.4f}")
        
        # Dynamically renormalize graphs to eliminate pupil normalisation leakage
        train_graphs = renormalize_graphs_pupil(
            train_graphs, global_pupil_mean, global_pupil_std, fold_pupil_mean, fold_pupil_std)
        val_graphs = renormalize_graphs_pupil(
            val_graphs, global_pupil_mean, global_pupil_std, fold_pupil_mean, fold_pupil_std)

        print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")

        batch_size = training_config.get('batch_size', 32)
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

        # ── Model, Loss, Optimizer ─────────────────────────────────────────
        if is_standalone:
            model = STGNNStandaloneModel(config=config).to(device)
        else:
            model = GNNCEFAMHybridModel(
                handcrafted_dim=handcrafted_dim, config=config).to(device)

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
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{prefix}_fold_{fold}_best.pt")

        # ── Checkpoint resume ─────────────────────────────────────────────
        checkpoint_loaded = False
        if args.overfit_batches == 0 and os.path.exists(checkpoint_path):
            print(f"Found existing checkpoint at {checkpoint_path}. Evaluating...")
            try:
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=device))
                checkpoint_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load checkpoint due to shape mismatch or error: {e}")
                print("Skipping checkpoint resume and training this fold from scratch...")
                # Delete the mismatched checkpoint to prevent issues in subsequent runs
                try:
                    os.remove(checkpoint_path)
                    print(f"Deleted mismatched checkpoint: {checkpoint_path}")
                except Exception as del_err:
                    print(f"Warning: Could not delete checkpoint file: {del_err}")

        if checkpoint_loaded:
            model.eval()
            val_preds, val_targets, val_subj_preds = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    hc_tensor = _get_hc_tensor(
                        batch, flat_features_dict, handcrafted_dim,
                        is_standalone, device)
                    logits, gnn_attn, _, _ = model(batch, hc_tensor)
                    targets = batch.y
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(targets.cpu().numpy())
                    _record_subj_preds(val_subj_preds, batch, probs)

            pivoted_subj = _aggregate_val_subj(val_subj_preds, category_map)
            val_auc_subject = roc_auc_score(
                pivoted_subj['Label'].values,
                pivoted_subj['Pred_Proba_Subject'].values)
            best_val_auc = val_auc_subject
            best_metrics = calculate_metrics(
                pivoted_subj['Label'].values,
                pivoted_subj['Pred_Proba_Subject'].values)
            best_metrics.update({'epoch': -1, 'fold': fold})
            best_subj_preds = pivoted_subj.copy()
            best_subj_preds['Fold'] = fold
            best_fold_aucs.append(best_val_auc)
            cv_results.append(best_metrics)
            all_val_preds.append(best_subj_preds)
            print(f"Loaded checkpoint — Val Subject AUC: {val_auc_subject:.4f}")
            continue

        # ── Overfit sanity check ──────────────────────────────────────────
        if args.overfit_batches > 0:
            print(f"Sanity Check: Overfitting on {args.overfit_batches} batch(es)...")
            overfit_graphs = train_graphs[:batch_size * args.overfit_batches]
            train_loader = DataLoader(overfit_graphs, batch_size=batch_size,
                                      shuffle=False)
            epochs = 50

        # ── Epoch loop ────────────────────────────────────────────────────
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss, train_preds, train_targets = 0.0, [], []

            for batch in train_loader:
                batch = batch.to(device)
                hc_tensor = _get_hc_tensor(
                    batch, flat_features_dict, handcrafted_dim,
                    is_standalone, device)
                optimizer.zero_grad()
                logits, gnn_attn, _, _ = model(batch, hc_tensor)
                targets = batch.y
                loss = criterion(logits, targets, attention_weights=gnn_attn,
                                 node_batch=batch.batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_config.get('gradient_clip_val', 1.0))
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                train_preds.extend(probs)
                train_targets.extend(targets.cpu().numpy())

            train_loss /= len(train_graphs)
            train_auc = (roc_auc_score(train_targets, train_preds)
                         if len(np.unique(train_targets)) > 1 else 0.5)

            # Validation
            model.eval()
            val_loss, val_preds, val_targets, val_subj_preds = 0.0, [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    hc_tensor = _get_hc_tensor(
                        batch, flat_features_dict, handcrafted_dim,
                        is_standalone, device)
                    logits, gnn_attn, _, _ = model(batch, hc_tensor)
                    targets = batch.y
                    loss = criterion(logits, targets, attention_weights=gnn_attn,
                                     node_batch=batch.batch)
                    val_loss += loss.item() * batch.num_graphs
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(targets.cpu().numpy())
                    _record_subj_preds(val_subj_preds, batch, probs)

            val_loss /= len(val_graphs)
            val_auc_trial = (roc_auc_score(val_targets, val_preds)
                             if len(np.unique(val_targets)) > 1 else 0.5)

            pivoted_subj = _aggregate_val_subj(val_subj_preds, category_map)
            val_auc_subject = roc_auc_score(
                pivoted_subj['Label'].values,
                pivoted_subj['Pred_Proba_Subject'].values)

            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0 or args.overfit_batches > 0:
                print(f"Ep {epoch+1:03d}/{epochs} | "
                      f"TrLoss={train_loss:.4f} TrAUC={train_auc:.4f} | "
                      f"VaLoss={val_loss:.4f} VaTrialAUC={val_auc_trial:.4f} "
                      f"VaSubjAUC={val_auc_subject:.4f}")

            if args.overfit_batches > 0 and train_loss < 0.01:
                print(f"Overfitting successful! Loss: {train_loss:.4f}")
                break

            if val_auc_subject > best_val_auc:
                best_val_auc = val_auc_subject
                torch.save(model.state_dict(), checkpoint_path)
                best_metrics = calculate_metrics(
                    pivoted_subj['Label'].values,
                    pivoted_subj['Pred_Proba_Subject'].values)
                best_metrics.update({'epoch': epoch + 1, 'fold': fold})
                best_subj_preds = pivoted_subj.copy()
                best_subj_preds['Fold'] = fold
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(Best Val Subject AUC: {best_val_auc:.4f})")
                break

        print(f"Fold {fold} complete. Best Val Subject AUC: {best_val_auc:.4f}")
        best_fold_aucs.append(best_val_auc)
        cv_results.append(best_metrics)
        if best_subj_preds is not None:
            all_val_preds.append(best_subj_preds)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n--- 4-Fold Cross Validation Complete ---")
    print(f"Fold AUCs: {best_fold_aucs}")
    print(f"Mean AUC: {np.mean(best_fold_aucs):.4f} "
          f"Std: {np.std(best_fold_aucs):.4f}")

    if all_val_preds:
        df_all_val = pd.concat(all_val_preds, ignore_index=True)

        # Standard predictions file
        val_preds_path = os.path.join(
            results_dir, f"{prefix}_subject_val_predictions.csv")
        df_all_val.to_csv(val_preds_path, index=False)
        print(f"Saved overall validation predictions to {val_preds_path}")

        # OOF predictions for Tier 5 meta-learner (same content, explicit name)
        oof_path = os.path.join(results_dir, f"{prefix}_oof_subject_preds.csv")
        df_all_val.to_csv(oof_path, index=False)
        print(f"[Tier5 OOF] Saved OOF subject predictions to {oof_path}")

        overall_metrics = calculate_metrics(
            df_all_val['Label'].values,
            df_all_val['Pred_Proba_Subject'].values)
        print("\n--- Overall Subject-Level Validation Metrics ---")
        for k, v in overall_metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")

        import json
        summary = {
            "overall": overall_metrics,
            "folds": cv_results,
            "mean_fold_auc": float(np.mean([r['auc'] for r in cv_results])),
            "std_fold_auc": float(np.std([r['auc'] for r in cv_results])),
            "mean_fold_accuracy": float(np.mean([r['accuracy'] for r in cv_results])),
            "std_fold_accuracy": float(np.std([r['accuracy'] for r in cv_results])),
            "handcrafted_dim": handcrafted_dim,
        }
        summary_path = os.path.join(results_dir, f"{prefix}_results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved results summary to {summary_path}")

    # ── Test Set Inference ─────────────────────────────────────────────────────
    if test_graphs:
        print(f"\n--- Running Test Set Inference on {len(test_graphs)} graphs ---")
        test_fold_preds = []

        for fold in range(cv_folds):
            ckpt = os.path.join(checkpoint_dir, f"{prefix}_fold_{fold}_best.pt")
            if not os.path.exists(ckpt):
                print(f"  Fold {fold}: checkpoint not found at {ckpt}, skipping.")
                continue
            if is_standalone:
                m = STGNNStandaloneModel(config=config).to(device)
            else:
                m = GNNCEFAMHybridModel(
                    handcrafted_dim=handcrafted_dim, config=config).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()

            test_loader = DataLoader(
                test_graphs, batch_size=batch_size, shuffle=False)
            fold_subj_preds = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    hc_tensor = _get_hc_tensor(
                        batch, flat_features_dict, handcrafted_dim,
                        is_standalone, device)
                    logits, _, _, _ = m(batch, hc_tensor)
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    _record_subj_preds(fold_subj_preds, batch, probs)

            fold_df = _aggregate_val_subj(fold_subj_preds, category_map)
            fold_df['Fold'] = fold
            test_fold_preds.append(fold_df)
            print(f"  Fold {fold}: {len(fold_df)} test subjects processed.")

        if test_fold_preds:
            df_test_all = pd.concat(test_fold_preds, ignore_index=True)
            df_test_ensemble = (
                df_test_all.groupby('Subject_ID')
                .agg(Label=('Label', 'first'),
                     Pred_Proba_Subject=('Pred_Proba_Subject', 'mean'))
                .reset_index()
            )
            test_path = os.path.join(
                results_dir, f"{prefix}_test_predictions.csv")
            df_test_ensemble.to_csv(test_path, index=False)
            print(f"[Test] Saved test set predictions to {test_path}")

            if len(np.unique(df_test_ensemble['Label'].values)) > 1:
                test_metrics = calculate_metrics(
                    df_test_ensemble['Label'].values,
                    df_test_ensemble['Pred_Proba_Subject'].values)
                print("\n--- Test Set Metrics ---")
                for k, v in test_metrics.items():
                    print(f"  {k.capitalize()}: {v:.4f}")
    else:
        print("\n[Test] No test graphs found; skipping test inference.")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────────

def _get_hc_tensor(batch, flat_features_dict, handcrafted_dim,
                   is_standalone, device):
    """Fetch the aligned handcrafted feature tensor for a batch."""
    if is_standalone:
        return None
    hc_batch = []
    for sub_id, stim_id in zip(batch.subject_id, batch.stimulus_id):
        s_id = (int(sub_id.item()) if torch.is_tensor(sub_id)
                else int(sub_id))
        feat = flat_features_dict.get(
            (s_id, stim_id),
            np.zeros(handcrafted_dim, dtype=np.float32))
        hc_batch.append(feat)
    return torch.tensor(np.array(hc_batch), dtype=torch.float32).to(device)


def _record_subj_preds(store: list, batch, probs: np.ndarray):
    """Append per-trial subject predictions to store list."""
    for i in range(batch.num_graphs):
        s_id = (int(batch.subject_id[i].item())
                if torch.is_tensor(batch.subject_id[i])
                else int(batch.subject_id[i]))
        store.append({
            "Subject_ID": s_id,
            "Stimulus_ID": batch.stimulus_id[i],
            "Label": int(batch.y[i].item()),
            "Pred_Proba": float(probs[i])
        })


def _aggregate_val_subj(val_subj_preds: list, category_map: dict) -> pd.DataFrame:
    """Aggregate trial-level predictions to subject-level (uniform mean)."""
    df_val = pd.DataFrame(val_subj_preds)
    df_val['Category'] = df_val['Stimulus_ID'].map(category_map)
    grouped = (df_val.groupby(['Subject_ID', 'Category', 'Label'])['Pred_Proba']
               .mean().reset_index())
    pivoted = grouped.pivot(
        index=['Subject_ID', 'Label'], columns='Category',
        values='Pred_Proba'
    ).reset_index()
    for cat in ['Social', 'Manipulated', 'Natural', 'Synthetic']:
        if cat not in pivoted.columns:
            pivoted[cat] = 0.5
    pivoted['Pred_Proba_Subject'] = (
        pivoted[['Social', 'Manipulated', 'Natural', 'Synthetic']].mean(axis=1))
    return pivoted


if __name__ == "__main__":
    main()
