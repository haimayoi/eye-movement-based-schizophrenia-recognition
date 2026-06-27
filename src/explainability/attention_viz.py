"""
Attention Visualization
=======================
Visualises:
  1. GNN Global Attention — per-fixation gate weights (saliency over scanpath)
  2. CEFAM Direction-1 Attention — which graph nodes the expert features attend to
     (attn_1 shape [B, 1, N] — now non-degenerate after fix)
  3. Entropy plot — attention entropy over training epochs (sharpening)
  4. Scanpath overlay — plot fixation positions coloured by attention weight
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ─────────────────────────────────────────────────────────────────────────────
# Attention entropy
# ─────────────────────────────────────────────────────────────────────────────

def compute_attention_entropy(
    attn_weights: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Computes Shannon entropy of attention weight distribution.
    Low entropy = sparse (focused), High entropy = diffuse (uniform).
    attn_weights : [N] — should sum to 1.
    """
    a = np.clip(attn_weights, eps, 1.0)
    a = a / a.sum()
    return float(-np.sum(a * np.log(a)))


def plot_entropy_over_epochs(
    entropy_list: list[float],
    output_path: str,
    title: str = "Attention Entropy over Training Epochs"
):
    """
    Plots how attention entropy decreases (sparsity increases) over training.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(entropy_list, linewidth=2, color='#e74c3c')
    ax.fill_between(range(len(entropy_list)), entropy_list, alpha=0.2, color='#e74c3c')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[AttentionViz] Saved entropy plot to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Scanpath attention overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_scanpath_attention(
    x_coords: np.ndarray,           # [N] fixation x coordinates (pixels)
    y_coords: np.ndarray,           # [N] fixation y positions
    gnn_attn: np.ndarray,           # [N] GNN global attention gate weights
    cefam_attn: np.ndarray = None,  # [N] CEFAM direction-1 attention (optional)
    screen_w: int = 1024,
    screen_h: int = 768,
    output_path: str = "results/attention_scanpath.png",
    title: str = "Scanpath Attention Map",
    subject_id: int = None,
    label: int = None,
):
    """
    Plots fixation positions on screen coordinates, coloured by attention weight.
    Lines connect consecutive fixations (scanpath order).
    Circle radius ∝ attention weight.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_cols = 2 if cefam_attn is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    def _draw_one(ax, attn, stream_name):
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        cmap = cm.get_cmap('plasma')
        norm = Normalize(vmin=0, vmax=1)

        # Draw scanpath lines
        for i in range(len(x_coords) - 1):
            ax.plot([x_coords[i], x_coords[i+1]],
                    [y_coords[i], y_coords[i+1]],
                    color='gray', alpha=0.4, linewidth=0.8, zorder=1)

        # Draw fixation circles
        for i, (x, y, a) in enumerate(zip(x_coords, y_coords, attn_norm)):
            color = cmap(norm(a))
            radius = 8 + 24 * float(a)
            circle = plt.Circle((x, screen_h - y),  # flip y for image coords
                                 radius, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            ax.text(x, screen_h - y, str(i+1), ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold', zorder=3)

        ax.set_xlim(0, screen_w)
        ax.set_ylim(0, screen_h)
        ax.set_aspect('equal')
        ax.set_xlabel("X (pixels)", fontsize=11)
        ax.set_ylabel("Y (pixels)", fontsize=11)
        lbl_str = f"  [{'SZ' if label == 1 else 'HC'}]" if label is not None else ""
        sid_str = f"  Subject {subject_id}" if subject_id is not None else ""
        ax.set_title(f"{stream_name} — {title}{sid_str}{lbl_str}", fontsize=11)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Attention weight (normalised)")

    _draw_one(axes[0], gnn_attn, "GNN Gate Attention")
    if cefam_attn is not None:
        _draw_one(axes[1], cefam_attn, "CEFAM Expert→Graph Attention")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[AttentionViz] Saved scanpath attention to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CEFAM attention matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_cefam_attention_heatmap(
    attn_1: np.ndarray,          # [1, N] — expert→graph, from model forward pass
    fixation_labels: list[str],  # label for each fixation node
    output_path: str,
    title: str = "CEFAM Expert→Graph Attention (Direction 1)",
):
    """
    Plots the CEFAM direction-1 attention as a heatmap.
    attn_1 shape after squeeze: [N] — probability over fixation nodes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    attn = attn_1.squeeze()  # [N]
    fig, ax = plt.subplots(figsize=(max(8, len(attn) * 0.5), 3))
    im = ax.imshow(attn[np.newaxis, :], cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=attn.max())
    ax.set_xticks(range(len(fixation_labels)))
    ax.set_xticklabels(fixation_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.3,
                 label="Attention weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[AttentionViz] Saved CEFAM heatmap to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch attention statistics
# ─────────────────────────────────────────────────────────────────────────────

def attention_statistics(attn_weights: np.ndarray) -> dict:
    """
    Returns descriptive statistics of attention weight distribution.
    Useful for monitoring sparsity over training.
    """
    a = np.array(attn_weights).flatten()
    return {
        "entropy": compute_attention_entropy(a),
        "max": float(a.max()),
        "min": float(a.min()),
        "std": float(a.std()),
        "gini": float(1 - (2 * np.sum((np.arange(1, len(a)+1) / len(a))
                                       * np.sort(a)) / (len(a) * a.mean())
                           if a.mean() > 0 else 0)),
        "n_tokens": len(a),
    }
