import torch
import torch.nn as nn

class CEFAMFusion(nn.Module):
    """
    Cross-attention Enhanced Fusion Attention Module (CEFAM)
    Implements bidirectional cross-attention between GNN embeddings (z_graph) 
    and handcrafted clinical embeddings (z_expert).
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        
        # Direction 1: Expert guides Graph attention
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Direction 2: Graph guides Expert attention
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, z_graph, z_expert):
        """
        z_graph: [batch_size, 128]
        z_expert: [batch_size, 128]
        Returns:
            z_final: [batch_size, 256] fused hybrid representation
            attn_1: cross-attention weights from Direction 1 (Expert guides Graph)
            attn_2: cross-attention weights from Direction 2 (Graph guides Expert)
        """
        # Reshape for PyTorch MultiheadAttention (expects [batch_size, seq_len, embed_dim])
        # Since these are global embeddings, seq_len = 1
        z_g = z_graph.unsqueeze(1) # [batch_size, 1, 128]
        z_e = z_expert.unsqueeze(1) # [batch_size, 1, 128]
        
        # Direction 1: Query=Graph, Key/Value=Expert
        # Answers: "Which graph patterns align with clinical markers?"
        z_fused_1, attn_1 = self.cross_attn_1(
            query=z_g,
            key=z_e,
            value=z_e
        )
        
        # Direction 2: Query=Expert, Key/Value=Graph
        # Answers: "What hidden patterns enhance biomarker meaning?"
        z_fused_2, attn_2 = self.cross_attn_2(
            query=z_e,
            key=z_g,
            value=z_g
        )
        
        # Remove seq_len dimension and concatenate
        z_fused_1 = z_fused_1.squeeze(1) # [batch_size, 128]
        z_fused_2 = z_fused_2.squeeze(1) # [batch_size, 128]
        
        z_concat = torch.cat([z_fused_1, z_fused_2], dim=-1) # [batch_size, 256]
        
        # Project and norm
        z_final = self.fusion(z_concat) # [batch_size, 256]
        
        return z_final, attn_1, attn_2
