import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, GlobalAttention


class GNNStream(nn.Module):
    """
    GNN Stream that processes spatiotemporal scanpath graphs.
    Uses 2 layers of GATConv with edge attributes, residual connections,
    and Global Attention Pooling to extract the graph embedding z_graph.

    Returns BOTH:
    - z_graph   : pooled graph embedding [batch_size, out_dim]
    - h_nodes   : pre-pooling node embeddings [total_nodes, hidden_dim * heads]
    - attn_weights : global-attention gate weights [total_nodes]

    h_nodes is exposed so that CEFAMFusion can perform genuine
    cross-attention between graph nodes and expert features, instead of the
    previous degenerate seq_len=1 operation.
    """

    def __init__(self, in_dim=69, edge_dim=2, hidden_dim=64, heads=4,
                 out_dim=128, dropout=0.3):
        super().__init__()

        # --- GAT Layer 1 ---
        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout
        )
        self.bn1 = BatchNorm(hidden_dim * heads)

        # --- GAT Layer 2 ---
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout
        )
        self.bn2 = BatchNorm(hidden_dim * heads)

        # Skip connection from raw input to output of GAT2
        self.residual = nn.Linear(in_dim, hidden_dim * heads)

        # Global Attention Pooling (soft-attention readout)
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim * heads, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pool = GlobalAttention(gate_nn)

        # Final projection to z_graph dimension
        self.proj = nn.Linear(hidden_dim * heads, out_dim)
        self.dropout = nn.Dropout(dropout)

        self._hidden_dim = hidden_dim
        self._heads = heads

    @property
    def node_emb_dim(self) -> int:
        """Dimension of pre-pooling node embeddings (h_nodes)."""
        return self._hidden_dim * self._heads

    def forward(self, x, edge_index, edge_attr, batch, mask=None):
        """
        Args:
            x          : Node features          [total_nodes, in_dim]
            edge_index : Graph edge index        [2, num_edges]
            edge_attr  : Edge features           [num_edges, edge_dim]
            batch      : Batch assignment vector [total_nodes]
            mask       : Node mask (True for real, False for padding) [total_nodes]

        Returns:
            z_graph      : Pooled graph embedding [batch_size, out_dim]
            h_nodes      : Pre-pooling node embs  [total_nodes, hidden_dim * heads]
            attn_weights : Global attention gate  [total_nodes]
        """
        x_in = x

        # --- Layer 1 ---
        h = self.gat1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)

        # --- Layer 2 ---
        h2 = self.gat2(h, edge_index, edge_attr)
        h2 = self.bn2(h2)

        # Residual connection
        h_res = self.residual(x_in)
        h2 = h2 + h_res
        h2 = F.elu(h2)
        h2 = self.dropout(h2)

        # h2 is our pre-pooling node embedding → exposed as h_nodes
        h_nodes = h2  # [total_nodes, hidden_dim * heads]

        # --- Global Attention Pooling ---
        gate = self.pool.gate_nn(h2)                          # [total_nodes, 1]
        if mask is not None:
            # Mask out padding nodes by setting their gate scores to a large negative value
            gate = gate.masked_fill(~mask.unsqueeze(-1), -1e9)
        from torch_geometric.utils import softmax
        attn_weights = softmax(gate, batch)                   # [total_nodes, 1]

        h2_weighted = h2 * attn_weights
        from torch_geometric.nn import global_add_pool
        z_graph_raw = global_add_pool(h2_weighted, batch)     # [batch_size, hidden_dim * heads]

        # Final projection
        z_graph = self.proj(z_graph_raw)                      # [batch_size, out_dim]

        return z_graph, h_nodes, attn_weights.squeeze(-1)
