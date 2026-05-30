import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, GlobalAttention

class GNNStream(nn.Module):
    """
    GNN Stream that processes spatiotemporal scanpath graphs.
    Uses 2 layers of GATConv with edge attributes, residual connections,
    and Global Attention Pooling to extract the graph embedding z_graph.
    """
    def __init__(self, in_dim=69, edge_dim=2, hidden_dim=64, heads=4, out_dim=128, dropout=0.3):
        super().__init__()
        
        # Layer 1
        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout
        )
        self.bn1 = BatchNorm(hidden_dim * heads)
        
        # Layer 2
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout
        )
        self.bn2 = BatchNorm(hidden_dim * heads)
        
        # Skip connection from input to output of GAT2
        self.residual = nn.Linear(in_dim, hidden_dim * heads)
        
        # Global Attention Pooling
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim * heads, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pool = GlobalAttention(gate_nn)
        
        # Final projection to match z_graph dimension (128)
        self.proj = nn.Linear(hidden_dim * heads, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: Node features [num_nodes, in_dim]
        edge_index: Graph edge index [2, num_edges]
        edge_attr: Edge features [num_edges, edge_dim]
        batch: Batch assignment [num_nodes]
        """
        # Save input for skip connection
        x_in = x
        
        # Layer 1
        h = self.gat1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)
        
        # Layer 2
        h2 = self.gat2(h, edge_index, edge_attr)
        h2 = self.bn2(h2)
        
        # Add residual connection
        h_res = self.residual(x_in)
        h2 = h2 + h_res
        
        h2 = F.elu(h2)
        h2 = self.dropout(h2)
        
        # Global Attention Pooling (Readout)
        # Compute attention weights manually for regularization/interpretability
        gate = self.pool.gate_nn(h2) # [num_nodes, 1]
        
        # Apply softmax across nodes for each graph in the batch
        from torch_geometric.utils import softmax
        attn_weights = softmax(gate, batch) # [num_nodes, 1]
        
        # Weighted sum of node embeddings
        h2_weighted = h2 * attn_weights
        
        # Aggregate (sum) over nodes for each graph
        from torch_geometric.nn import global_add_pool
        z_graph_raw = global_add_pool(h2_weighted, batch) # [batch_size, hidden_dim * heads]
        
        # Final projection to [batch_size, out_dim]
        z_graph = self.proj(z_graph_raw)
        
        return z_graph, attn_weights.squeeze(-1)
