import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch


class CEFAMFusion(nn.Module):
    """
    Cross-attention Enhanced Fusion Attention Module (CEFAM) — Fixed Version.

    Previous bug: Both z_graph and z_expert were unsqueezed to seq_len=1,
    making softmax trivially = 1.0 with zero learning signal.

    Fix:
        Direction 1 — Expert queries Graph nodes (PRIMARY):
            Q = z_expert  [B, 1, d_model]
            K = h_nodes   [B, N, d_model]  (pre-pooling GNN node embeddings)
            V = h_nodes
            → attn_1 shape [B, 1, N]: "which fixation nodes are relevant to clinical features?"
            → This is interpretable and genuinely non-trivial.

        Direction 2 — Graph queries Expert (SECONDARY):
            Q = z_graph   [B, 1, d_model]  (pooled graph summary)
            K = z_expert  [B, 1, d_model]
            V = z_expert
            → attn_2 shape [B, 1, 1]: still trivial, but this direction is less
              important; the key innovation is Direction 1.

    The fix makes Direction 1 attention produce a non-constant weight vector
    over fixation nodes — directly interpretable as "clinical feature → fixation
    node saliency map."
    """

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 dropout: float = 0.1, node_emb_dim: int = 256,
                 max_nodes: int = 24):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes

        # Project node embeddings (hidden_dim * heads = 256 by default)
        # down to d_model so cross-attention dimensions match.
        self.node_proj = nn.Linear(node_emb_dim, d_model)

        # Direction 1: Expert queries Graph nodes
        # Q = z_expert [B, 1, d_model], K/V = h_nodes [B, N, d_model]
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Direction 2: Graph queries Expert (seq_len=1, kept for symmetry)
        # Q = z_graph [B, 1, d_model], K/V = z_expert [B, 1, d_model]
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Fusion MLP: [B, 2 * d_model] → [B, 2 * d_model]
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_nodes(
        self,
        h_nodes: torch.Tensor,
        node_batch: torch.Tensor,
        batch_size: int
    ):
        """
        Scatter per-graph node embeddings into a padded tensor.

        Args:
            h_nodes    : [total_nodes, node_emb_dim] — raw node embeddings
            node_batch : [total_nodes]               — graph assignment index
            batch_size : B

        Returns:
            h_padded   : [B, max_nodes, d_model]  (projected + padded)
            key_mask   : [B, max_nodes]  bool, True = padding position
        """
        d = self.d_model
        N = self.max_nodes
        device = h_nodes.device

        # Project first (cheaper to project before padding)
        h_proj = self.node_proj(h_nodes)          # [total_nodes, d_model]

        h_padded = torch.zeros(batch_size, N, d, device=device)
        key_mask = torch.ones(batch_size, N, dtype=torch.bool, device=device)  # True = pad

        # Use PyG's unbatch to get per-graph node lists
        graphs = unbatch(h_proj, node_batch)       # list of [n_i, d_model]
        for b, g_nodes in enumerate(graphs):
            n = min(g_nodes.size(0), N)
            h_padded[b, :n] = g_nodes[:n]
            key_mask[b, :n] = False                # False = real token

        return h_padded, key_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_graph: torch.Tensor,
        z_expert: torch.Tensor,
        h_nodes: torch.Tensor = None,
        node_batch: torch.Tensor = None
    ):
        """
        Args:
            z_graph    : Pooled graph embedding  [B, d_model]
            z_expert   : Handcrafted embedding   [B, d_model]
            h_nodes    : Pre-pooling node embs   [total_nodes, node_emb_dim]
                         If None, falls back to the original degenerate seq_len=1
                         mode (backward-compatible for STGNNStandaloneModel).
            node_batch : Batch assignment        [total_nodes]

        Returns:
            z_final  : Fused representation  [B, 2 * d_model]
            attn_1   : Expert→Graph attention [B, 1, N]  (interpretable)
            attn_2   : Graph→Expert attention [B, 1, 1]  (symmetric)
        """
        B = z_graph.size(0)

        z_g = z_graph.unsqueeze(1)     # [B, 1, d_model]
        z_e = z_expert.unsqueeze(1)    # [B, 1, d_model]

        # ---------------------------------------------------------------
        # Direction 1: Expert queries Graph (genuine cross-attention)
        # ---------------------------------------------------------------
        if h_nodes is not None and node_batch is not None:
            h_padded, key_mask = self._pad_nodes(h_nodes, node_batch, B)
            # Q = z_expert [B,1,d], K/V = h_nodes [B,N,d]
            z_fused_1, attn_1 = self.cross_attn_1(
                query=z_e,
                key=h_padded,
                value=h_padded,
                key_padding_mask=key_mask   # ignore padding positions
            )
            # attn_1 shape: [B, 1, N] — per-fixation relevance
        else:
            # Fallback: seq_len=1 (degenerate, backward-compatible)
            z_fused_1, attn_1 = self.cross_attn_1(
                query=z_e,
                key=z_g,
                value=z_g
            )

        # ---------------------------------------------------------------
        # Direction 2: Graph queries Expert (seq_len=1, symmetric)
        # ---------------------------------------------------------------
        z_fused_2, attn_2 = self.cross_attn_2(
            query=z_g,
            key=z_e,
            value=z_e
        )

        # Remove seq_len dimension
        z_fused_1 = z_fused_1.squeeze(1)   # [B, d_model]
        z_fused_2 = z_fused_2.squeeze(1)   # [B, d_model]

        z_concat = torch.cat([z_fused_1, z_fused_2], dim=-1)   # [B, 2*d_model]
        z_final = self.fusion(z_concat)                          # [B, 2*d_model]

        return z_final, attn_1, attn_2
