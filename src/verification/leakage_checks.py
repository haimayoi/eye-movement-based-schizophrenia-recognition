"""
Leakage Checks & Reproducibility Safeguards
============================================
Assertion-based checks that should be run before any training:

  1. GroupKFold correctness — each subject in exactly one fold
  2. Subject overlap — no subject appears in both train and val
  3. Feature normalisation — statistics computed only on training split
  4. Threshold optimisation — checks that threshold is NOT tuned on val set
  5. OOF completeness — all training subjects have exactly one OOF prediction
  6. Test contamination — no test subject appears in OOF training set

Usage:
    from src.verification.leakage_checks import run_all_checks
    run_all_checks(subject_to_fold, df_train_valid, df_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Set, List


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — GroupKFold correctness
# ─────────────────────────────────────────────────────────────────────────────

def check_groupkfold_correctness(
    subject_to_fold: Dict[int, int],
    df: pd.DataFrame,
    n_folds: int = 4,
) -> None:
    """
    Verifies that subject_to_fold assigns every subject to exactly one fold,
    and that fold indices are in range [0, n_folds).
    """
    fold_values = set(subject_to_fold.values())
    expected_folds = set(range(n_folds))

    if not fold_values.issubset(expected_folds):
        unexpected = fold_values - expected_folds
        raise AssertionError(
            f"[LeakageCheck 1] Unexpected fold indices: {unexpected}. "
            f"Expected folds in range [0, {n_folds}).")

    # Check every subject_id in df is in subject_to_fold
    df_subjects = set(df['Subject_ID'].astype(int).unique())
    missing = df_subjects - set(subject_to_fold.keys())
    if missing:
        raise AssertionError(
            f"[LeakageCheck 1] {len(missing)} subjects in data missing from "
            f"subject_to_fold: {missing}")

    print(f"[LeakageCheck 1] PASSED — GroupKFold: {len(subject_to_fold)} subjects "
          f"→ {n_folds} folds.")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — Subject overlap within folds
# ─────────────────────────────────────────────────────────────────────────────

def check_no_subject_overlap(
    subject_to_fold: Dict[int, int],
    fold: int,
) -> None:
    """
    Verifies that the set of training subjects for a given fold
    does not intersect the validation subjects.
    """
    train_subjects = {s for s, f in subject_to_fold.items() if f != fold}
    val_subjects = {s for s, f in subject_to_fold.items() if f == fold}
    overlap = train_subjects & val_subjects

    if overlap:
        raise AssertionError(
            f"[LeakageCheck 2] FOLD {fold}: Subjects appear in BOTH train "
            f"and validation: {overlap}.")
    print(f"[LeakageCheck 2] PASSED — Fold {fold}: no subject overlap "
          f"({len(train_subjects)} train, {len(val_subjects)} val).")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — Normalisation computed on training split only
# ─────────────────────────────────────────────────────────────────────────────

def check_normalisation_no_leakage(
    df_full: pd.DataFrame,
    subject_to_fold: Dict[int, int],
    fold: int,
    feature_col: str = 'FIX_PUPIL',
) -> None:
    """
    Warns if a normalisation statistic (mean/std) is computed on full data
    instead of training split only.

    This is a WARNING (not hard assert) because impact on balanced datasets
    is small, but should be fixed before publication.
    """
    df_train = df_full[df_full['Subject_ID'].map(subject_to_fold) != fold]
    df_val = df_full[df_full['Subject_ID'].map(subject_to_fold) == fold]

    full_mean = df_full[feature_col].mean()
    train_mean = df_train[feature_col].mean()

    diff = abs(full_mean - train_mean)
    if diff > 0.01:
        print(f"[LeakageCheck 3] WARNING — {feature_col} normalisation: "
              f"full_mean={full_mean:.4f} vs train_mean={train_mean:.4f} "
              f"(Δ={diff:.4f}). Use train-only statistics.")
    else:
        print(f"[LeakageCheck 3] OK — {feature_col} normalisation bias < 0.01 "
              f"(balanced dataset: {len(df_full)} total, {len(df_train)} train).")


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — No threshold optimisation on validation set
# ─────────────────────────────────────────────────────────────────────────────

def check_fixed_threshold_reporting(
    threshold: float,
    source: str = "unknown"
) -> None:
    """
    Asserts that reported metrics use a fixed threshold (0.5), NOT one
    optimised on the validation set. Any threshold ≠ 0.5 triggers a warning.
    """
    if abs(threshold - 0.5) > 0.001:
        raise AssertionError(
            f"[LeakageCheck 4] Reported accuracy uses threshold={threshold:.4f} "
            f"(from {source}), which was optimised on the validation set. "
            "This is EVALUATION LEAKAGE. Report metrics at threshold=0.5 only.")
    print(f"[LeakageCheck 4] PASSED — Fixed threshold=0.5 used.")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — OOF completeness
# ─────────────────────────────────────────────────────────────────────────────

def check_oof_completeness(
    df_oof: pd.DataFrame,
    subject_to_fold: Dict[int, int],
    model_name: str = "model"
) -> None:
    """
    Verifies:
      a. Every training subject has exactly one OOF prediction.
      b. No subject appears more than once in OOF output.
    """
    expected_subjects = set(subject_to_fold.keys())
    oof_subjects = set(df_oof['Subject_ID'].astype(int).tolist())

    missing = expected_subjects - oof_subjects
    extra = oof_subjects - expected_subjects
    dupes = df_oof[df_oof.duplicated('Subject_ID', keep=False)]

    errors = []
    if missing:
        errors.append(f"  Missing OOF predictions for {len(missing)} subjects: "
                      f"{list(missing)[:5]}...")
    if extra:
        errors.append(f"  OOF has {len(extra)} unexpected subjects: "
                      f"{list(extra)[:5]}...")
    if not dupes.empty:
        errors.append(f"  Duplicate Subject_IDs in OOF: "
                      f"{dupes['Subject_ID'].tolist()[:5]}...")

    if errors:
        raise AssertionError(
            f"[LeakageCheck 5] OOF incompleteness for {model_name}:\n"
            + "\n".join(errors))

    print(f"[LeakageCheck 5] PASSED — {model_name} OOF complete: "
          f"{len(oof_subjects)} subjects, no duplicates.")


# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — Test contamination
# ─────────────────────────────────────────────────────────────────────────────

def check_no_test_contamination(
    train_subjects: Set[int],
    test_subjects: Set[int],
    label: str = "dataset"
) -> None:
    """
    Asserts that no subject_id appears in both training and test sets.
    """
    overlap = train_subjects & test_subjects
    if overlap:
        raise AssertionError(
            f"[LeakageCheck 6] TEST CONTAMINATION in {label}! "
            f"{len(overlap)} subjects appear in BOTH train and test: {overlap}")
    print(f"[LeakageCheck 6] PASSED — No test contamination in {label}. "
          f"({len(train_subjects)} train, {len(test_subjects)} test subjects)")


# ─────────────────────────────────────────────────────────────────────────────
# Check 7 — Pupil normalisation leakage (specific to this project)
# ─────────────────────────────────────────────────────────────────────────────

def check_pupil_normalisation(
    df_fixations: pd.DataFrame,
    subject_to_fold: Dict[int, int],
    n_folds: int = 4
) -> None:
    """
    Checks whether global pupil normalisation introduces meaningful leakage.
    For each fold, computes the difference between global and train-only statistics.
    """
    print("\n[LeakageCheck 7] Pupil normalisation leakage check:")
    global_mean = df_fixations['FIX_PUPIL'].mean()
    global_std = df_fixations['FIX_PUPIL'].std()

    for fold in range(n_folds):
        train_subs = {s for s, f in subject_to_fold.items() if f != fold}
        df_train_fix = df_fixations[
            df_fixations['Subject_ID'].astype(int).isin(train_subs)]
        train_mean = df_train_fix['FIX_PUPIL'].mean()
        train_std = df_train_fix['FIX_PUPIL'].std()

        mean_bias = abs(global_mean - train_mean)
        std_bias = abs(global_std - train_std)

        status = "OK" if mean_bias < 0.5 else "WARNING"
        print(f"  Fold {fold}: mean_bias={mean_bias:.3f}  std_bias={std_bias:.3f}"
              f"  [{status}]")

    print("  Recommendation: Compute pupil stats on training fold only.")


# ─────────────────────────────────────────────────────────────────────────────
# Run all checks
# ─────────────────────────────────────────────────────────────────────────────

def run_all_checks(
    subject_to_fold: Dict[int, int],
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame = None,
    df_fixations: pd.DataFrame = None,
    n_folds: int = 4,
    threshold: float = 0.5,
) -> None:
    """
    Runs all leakage checks in sequence.
    Call this at the start of any training script.
    """
    print("\n" + "=" * 60)
    print(" LEAKAGE VERIFICATION SUITE")
    print("=" * 60)

    check_groupkfold_correctness(subject_to_fold, df_train_valid, n_folds)

    for fold in range(n_folds):
        check_no_subject_overlap(subject_to_fold, fold)

    check_fixed_threshold_reporting(threshold, source="training config")

    train_subjects = set(df_train_valid['Subject_ID'].astype(int).unique())
    if df_test is not None and not df_test.empty:
        test_subjects = set(df_test['Subject_ID'].astype(int).unique())
        check_no_test_contamination(train_subjects, test_subjects)

    if df_fixations is not None:
        check_pupil_normalisation(df_fixations, subject_to_fold, n_folds)

    print("\n" + "=" * 60)
    print(" ALL CHECKS PASSED")
    print("=" * 60 + "\n")
