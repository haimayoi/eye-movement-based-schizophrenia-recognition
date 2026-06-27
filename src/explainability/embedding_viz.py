"""
Embedding Visualization — t-SNE and UMAP
=========================================
Extracts and visualises learned embeddings from:
  - HandcraftedStream / BiomarkerStream (z_expert) [Tier 4]
  - GNNStream pooled output (z_graph) [Tier 4]
  - CEFAM fused output (z_final) [Tier 4 / Tier 5]
  - Meta-learner input space (P_tab, P_bica) [Tier 5]

Both t-SNE and UMAP are supported.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap as umap_module
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False
    print("[Embedding] umap-learn not installed. Install with: pip install umap-learn")


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Apply t-SNE to embeddings. Returns [N, 2] array."""
    X = StandardScaler().fit_transform(embeddings)
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, n_iter=1000)
    return tsne.fit_transform(X)


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Apply UMAP to embeddings. Returns [N, 2] array."""
    assert _HAS_UMAP, "Install umap-learn: pip install umap-learn"
    X = StandardScaler().fit_transform(embeddings)
    reducer = umap_module.UMAP(n_components=n_components,
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                random_state=random_state)
    return reducer.fit_transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = ["#3498db", "#e74c3c"]   # HC=blue, SZ=red
_CMAP = ListedColormap(_PALETTE)


def _scatter_2d(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    fold: np.ndarray = None,
    title: str = "",
    alpha: float = 0.7,
):
    """Internal: scatter plot coloured by label (HC/SZ) with fold markers."""
    markers = ['o', 's', '^', 'D', 'P']
    class_names = {0: "HC", 1: "SZ"}

    for cls in [0, 1]:
        mask = labels == cls
        if fold is not None:
            for f in np.unique(fold):
                f_mask = mask & (fold == f)
                if not f_mask.any():
                    continue
                ax.scatter(coords[f_mask, 0], coords[f_mask, 1],
                           c=_PALETTE[cls], marker=markers[int(f) % len(markers)],
                           alpha=alpha, s=60, edgecolors='none',
                           label=f"{class_names[cls]} (fold {f})")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=_PALETTE[cls], marker='o', alpha=alpha, s=60,
                       edgecolors='none', label=class_names[cls])

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.legend(fontsize=8, markerscale=1.2)
    ax.grid(True, alpha=0.2)


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    fold: np.ndarray = None,
    output_path: str = "results/embeddings.png",
    title: str = "Learned Embeddings",
    methods: list[str] = None,
):
    """
    Plots both t-SNE and UMAP (if available) side by side.

    Args:
        embeddings : [N, D]  — e.g. z_final from CEFAM
        labels     : [N]     — 0=HC, 1=SZ
        fold       : [N]     — optional fold index for marker shapes
        methods    : list of 'tsne', 'umap' (default: both if available)
    """
    if methods is None:
        methods = ['tsne'] + (['umap'] if _HAS_UMAP else [])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_cols = len(methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        print(f"[Embedding] Computing {method.upper()}... (dim={embeddings.shape[1]})")
        if method == 'tsne':
            coords = compute_tsne(embeddings)
            method_title = f"{title} — t-SNE"
        elif method == 'umap':
            coords = compute_umap(embeddings)
            method_title = f"{title} — UMAP"
        else:
            continue
        _scatter_2d(ax, coords, labels, fold, title=method_title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Embedding] Saved embedding plot to {output_path}")


def plot_tier5_decision_space(
    p_tab: np.ndarray,
    p_bica: np.ndarray,
    labels: np.ndarray,
    p_final: np.ndarray = None,
    output_path: str = "results/tier5_decision_space.png",
):
    """
    2D scatter of (P_tab, P_bica) coloured by true label.
    Optional: overlay decision boundary from Tier5 meta-learner.
    Tier5's 2D input space is directly visualisable — no dim reduction needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    for cls, name, color, marker in [(0, 'HC', '#3498db', 'o'),
                                       (1, 'SZ', '#e74c3c', '^')]:
        mask = labels == cls
        ax.scatter(p_tab[mask], p_bica[mask],
                   c=color, marker=marker, alpha=0.7, s=80,
                   edgecolors='white', linewidths=0.5, label=name)

    # Decision boundary overlay (if Tier5 meta-learner probabilities provided)
    if p_final is not None:
        # Colour points by confidence
        confident_correct = ((p_final >= 0.5) == labels.astype(bool))
        wrong = ~confident_correct
        if wrong.any():
            ax.scatter(p_tab[wrong], p_bica[wrong],
                       facecolors='none', edgecolors='black', s=120,
                       linewidths=1.5, label="Misclassified", zorder=5)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3, linewidth=0.8, label="P_tab = P_bica")

    ax.set_xlabel("P_tab (Tabular / XGBoost probability)", fontsize=12)
    ax.set_ylabel("P_bica (BiCA-HS / Deep model probability)", fontsize=12)
    ax.set_title("Tier 5 Input Space: Tabular vs Deep Model", fontsize=13,
                 fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Embedding] Saved Tier5 decision space to {output_path}")
