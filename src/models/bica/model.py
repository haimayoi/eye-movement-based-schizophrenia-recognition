import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LearnedStream(nn.Module):
    """
    Learned Stream using a Transformer Encoder to process raw fixation sequences.
    """
    def __init__(self, d_input=5, d_model=128, nhead=4, num_layers=4, d_ff=512, dropout=0.1, max_seq_len=200, pos_encoding='sinusoidal', pre_norm=True):
        super().__init__()
        self.input_projection = nn.Linear(d_input, d_model)
        
        if pos_encoding == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        else:
            # Learnable positional embedding
            self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
            self.dropout = nn.Dropout(dropout)
            
        self.pos_encoding_type = pos_encoding
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=pre_norm
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, mask=None):
        """
        src: [batch_size, seq_len, d_input]
        mask: [batch_size, seq_len] (bool, True for valid tokens, False for padding)
        """
        x = self.input_projection(src)  # [batch_size, seq_len, d_model]
        
        if self.pos_encoding_type == 'sinusoidal':
            x = self.pos_encoder(x)
        else:
            x = x + self.pos_encoder[:, :x.size(1)]
            x = self.dropout(x)
            
        # In PyTorch src_key_padding_mask: True means ignore/padding position.
        # Since our mask is True for real and False for padding, we invert it.
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask
            
        out = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)  # [batch_size, seq_len, d_model]
        
        # Mean pooling over non-padded elements
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            out_masked = out * mask_expanded
            z_learned = out_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            z_learned = out.mean(dim=1)
            
        return z_learned

class BiomarkerStream(nn.Module):
    """
    MLP encoder for handcrafted flat features.
    """
    def __init__(self, d_bio, hidden_dims=[256, 128], dropout=[0.2, 0.1], d_out=128):
        super().__init__()
        layers = []
        curr_dim = d_bio
        for h_dim, drop in zip(hidden_dims, dropout):
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, d_out))
        layers.append(nn.BatchNorm1d(d_out))
        layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class BiCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between Transformer stream (z_learned)
    and Biomarker stream (z_expert).
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        # Direction 1: Expert guides Learned Sequence
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        # Direction 2: Learned Sequence guides Expert
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, z_learned, z_expert):
        # Shape: [batch_size, 1, d_model]
        z_l = z_learned.unsqueeze(1)
        z_e = z_expert.unsqueeze(1)
        
        # Dir 1: Q=learned, K=V=expert
        z_fused_1, attn_1 = self.cross_attn_1(query=z_l, key=z_e, value=z_e)
        # Dir 2: Q=expert, K=V=learned
        z_fused_2, attn_2 = self.cross_attn_2(query=z_e, key=z_l, value=z_l)
        
        # Concatenate and project
        z_concat = torch.cat([z_fused_1.squeeze(1), z_fused_2.squeeze(1)], dim=-1)
        z_final = self.fusion(z_concat)
        
        return z_final, attn_1, attn_2

class BiCAHSModel(nn.Module):
    """
    Bidirectional Cross-Attention Hybrid Stream (BiCA-HS) Model.
    """
    def __init__(self, d_bio, config):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        
        # 1. Learned Stream (Transformer Encoder)
        ls_cfg = model_cfg['learned_stream']
        self.learned_stream = LearnedStream(
            d_input=ls_cfg.get('d_input', 5),
            d_model=ls_cfg.get('d_model', 128),
            nhead=ls_cfg.get('nhead', 4),
            num_layers=ls_cfg.get('num_layers', 4),
            d_ff=ls_cfg.get('d_ff', 512),
            dropout=ls_cfg.get('dropout', 0.1),
            max_seq_len=ls_cfg.get('max_seq_len', 200),
            pos_encoding=ls_cfg.get('pos_encoding', 'sinusoidal'),
            pre_norm=ls_cfg.get('pre_norm', True)
        )
        
        # 2. Biomarker Stream (MLP)
        bs_cfg = model_cfg['biomarker_stream']
        self.biomarker_stream = BiomarkerStream(
            d_bio=d_bio,
            hidden_dims=bs_cfg.get('hidden_dims', [256, 128]),
            dropout=bs_cfg.get('dropout', [0.2, 0.1]),
            d_out=ls_cfg.get('d_model', 128)
        )
        
        # 3. Bidirectional Cross-Attention
        ca_cfg = model_cfg['cross_attention']
        self.cross_attention = BiCrossAttention(
            d_model=ca_cfg.get('d_model', 128),
            nhead=ca_cfg.get('nhead', 4),
            dropout=ca_cfg.get('dropout', 0.1)
        )
        
        # 4. Classifier Head
        clf_cfg = model_cfg['classifier']
        self.classifier = nn.Sequential(
            nn.Linear(ca_cfg.get('d_model', 128) * 2, clf_cfg.get('hidden_dims', [256, 128])[0]),
            nn.GELU(),
            nn.Dropout(clf_cfg.get('dropout', [0.3, 0.2])[0]),
            nn.Linear(clf_cfg.get('hidden_dims', [256, 128])[0], clf_cfg.get('hidden_dims', [256, 128])[1]),
            nn.GELU(),
            nn.Dropout(clf_cfg.get('dropout', [0.3, 0.2])[1]),
            nn.Linear(clf_cfg.get('hidden_dims', [256, 128])[1], 2)
        )
        
    def forward(self, sequences, mask, flat_features):
        """
        sequences: [batch_size, seq_len, 5] (normalized sequence features)
        mask: [batch_size, seq_len] (bool mask, True for valid tokens, False for padding)
        flat_features: [batch_size, d_bio] (flat handcrafted features)
        """
        # 1. Learned Stream (Transformer Encoder)
        z_learned = self.learned_stream(sequences, mask)  # [batch_size, d_model]
        
        # 2. Biomarker Stream (MLP)
        z_expert = self.biomarker_stream(flat_features)  # [batch_size, d_model]
        
        # 3. Cross-Attention Fusion
        z_final, attn_1, attn_2 = self.cross_attention(z_learned, z_expert)  # [batch_size, 2 * d_model]
        
        # 4. Classifier Head
        logits = self.classifier(z_final)
        
        return logits, attn_1, attn_2
