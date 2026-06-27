"""
Tier 5 OOF Collector
====================
Utilities for loading, aligning, and validating out-of-fold predictions
from Tier 3 (tabular) and Tier 4 (deep) models before training the
Tier 5 meta-learner.

Leakage prevention guarantees enforced here:
  - Every subject in the meta-learner training set must appear exactly once
    in OOF predictions (one prediction per fold).
  - Subject IDs in Tier 3 OOF and Tier 4 OOF must be identical.
  - No subject in the test set may appear in the OOF training set.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_oof_predictions(
    tier3_oof_path: str,
    tier4_oof_path: str,
    tier3_prob_col: str = "Pred_Proba_Subject",
    tier4_prob_col: str = "Pred_Proba_Subject",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and align OOF predictions from Tier 3 and Tier 4.

    Args:
        tier3_oof_path : path to CSV with columns [Subject_ID, Label, Pred_Proba_Subject]
        tier4_oof_path : path to CSV with the same schema
        tier3_prob_col : probability column name in tier3 file
        tier4_prob_col : probability column name in tier4 file

    Returns:
        df_meta   : aligned DataFrame [Subject_ID, Label, P_tab, P_bica, Fold_tab, Fold_bica]
        X         : feature matrix [N, 2]  (P_tab, P_bica)
        y         : label vector   [N]
    """
    if not os.path.exists(tier3_oof_path):
        raise FileNotFoundError(
            f"Tier 3 OOF predictions not found: {tier3_oof_path}\n"
            "Run: python -m src.tier3_tabular.tabular_models --model xgboost")
    if not os.path.exists(tier4_oof_path):
        raise FileNotFoundError(
            f"Tier 4 OOF predictions not found: {tier4_oof_path}\n"
            "Run: python scripts/train_tier4.py")

    df3 = pd.read_csv(tier3_oof_path)
    df4 = pd.read_csv(tier4_oof_path)

    # Normalise column names
    df3 = df3.rename(columns={tier3_prob_col: "P_tab"})
    df4 = df4.rename(columns={tier4_prob_col: "P_bica"})

    # Keep only what we need
    keep3 = ["Subject_ID", "Label", "P_tab"]
    keep4 = ["Subject_ID", "P_bica"]
    if "Fold" in df3.columns:
        keep3.append("Fold")
        df3 = df3.rename(columns={"Fold": "Fold_tab"})
        keep3[-1] = "Fold_tab"
    if "Fold" in df4.columns:
        keep4.append("Fold")
        df4 = df4.rename(columns={"Fold": "Fold_bica"})
        keep4[-1] = "Fold_bica"

    df3 = df3[[c for c in keep3 if c in df3.columns]].drop_duplicates("Subject_ID")
    df4 = df4[[c for c in keep4 if c in df4.columns]].drop_duplicates("Subject_ID")

    df_meta = df3.merge(df4, on="Subject_ID", how="inner")

    # ── Leakage assertions ────────────────────────────────────────────────
    _assert_no_duplicate_subjects(df_meta, "merged OOF")
    _assert_subject_sets_match(df3, df4)

    X = df_meta[["P_tab", "P_bica"]].values.astype(np.float32)
    y = df_meta["Label"].values.astype(int)

    print(f"[Tier5 OOF] Loaded {len(df_meta)} subjects | "
          f"P_tab range [{X[:,0].min():.3f}, {X[:,0].max():.3f}] | "
          f"P_bica range [{X[:,1].min():.3f}, {X[:,1].max():.3f}]")

    return df_meta, X, y


def load_test_predictions(
    tier3_test_path: str,
    tier4_test_path: str,
    tier3_prob_col: str = "Pred_Proba_Subject",
    tier4_prob_col: str = "Pred_Proba_Subject",
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Load test-set predictions from both tiers for inference.

    Returns (df_test_meta, X_test) or (None, None) if test files are missing.
    """
    if not os.path.exists(tier3_test_path) or not os.path.exists(tier4_test_path):
        print("[Tier5] Test prediction files not found; skipping test inference.")
        return None, None

    df3 = pd.read_csv(tier3_test_path).rename(
        columns={tier3_prob_col: "P_tab"})
    df4 = pd.read_csv(tier4_test_path).rename(
        columns={tier4_prob_col: "P_bica"})

    keep3 = ["Subject_ID", "P_tab"]
    if "Label" in df3.columns:
        keep3.insert(1, "Label")
    df3 = df3[keep3].drop_duplicates("Subject_ID")
    df4 = df4[["Subject_ID", "P_bica"]].drop_duplicates("Subject_ID")

    df_test = df3.merge(df4, on="Subject_ID", how="inner")
    X_test = df_test[["P_tab", "P_bica"]].values.astype(np.float32)

    return df_test, X_test


# ─────────────────────────────────────────────────────────────────────────────
# Leakage guards
# ─────────────────────────────────────────────────────────────────────────────

def _assert_no_duplicate_subjects(df: pd.DataFrame, name: str):
    dupes = df[df.duplicated("Subject_ID")]
    if not dupes.empty:
        raise AssertionError(
            f"[Leakage Guard] Duplicate Subject_IDs in {name}:\n"
            f"{dupes['Subject_ID'].tolist()}\n"
            "Each subject must appear exactly once in OOF predictions.")


def _assert_subject_sets_match(df3: pd.DataFrame, df4: pd.DataFrame):
    s3 = set(df3["Subject_ID"].astype(int))
    s4 = set(df4["Subject_ID"].astype(int))
    only_in_3 = s3 - s4
    only_in_4 = s4 - s3
    if only_in_3 or only_in_4:
        raise AssertionError(
            f"[Leakage Guard] Subject set mismatch between Tier3 and Tier4 OOF:\n"
            f"  Only in Tier3: {only_in_3}\n"
            f"  Only in Tier4: {only_in_4}\n"
            "Ensure both models trained with identical GroupKFold splits.")


def assert_no_test_train_overlap(train_subjects: set, test_subjects: set):
    """Call this before training Tier5 to confirm no test leakage."""
    overlap = train_subjects & test_subjects
    if overlap:
        raise AssertionError(
            f"[Leakage Guard] Test subjects appear in training OOF:\n"
            f"  Overlap: {overlap}\n"
            "This indicates test set contamination in fold splitting.")
    print(f"[Leakage Guard] PASSED — no train/test subject overlap "
          f"({len(train_subjects)} train, {len(test_subjects)} test subjects)")
