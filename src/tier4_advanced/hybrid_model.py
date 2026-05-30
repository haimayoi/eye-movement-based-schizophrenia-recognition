import torch
import torch.nn as nn
from src.tier4_advanced.rinet_projector import RINetProjector
from src.tier4_advanced.gnn_stream import GNNStream
from src.tier4_advanced.handcrafted_stream import HandcraftedStream
from src.tier4_advanced.cefam_fusion import CEFAMFusion

class Classifier(nn.Module):
    """
    Classification head for the hybrid model.
    """
    def __init__(self, input_dim=256, hidden_dims=[128], dropouts=[0.3], num_classes=2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim, drop in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class GNNCEFAMHybridModel(nn.Module):
    """
    Full Hybrid Model combining:
    1. RINet Projector (projects 1056-dim visual features to 64-dim)
    2. GNN Stream (encodes graph structure to z_graph [128-dim])
    3. Handcrafted Stream (encodes flat features to z_expert [128-dim])
    4. CEFAM Fusion (bidirectional cross-attention)
    5. Classifier Head (outputs logits for SZ classification)
    """
    def __init__(self, handcrafted_dim, config):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        
        # 1. RINet Projector
        rinet_cfg = model_cfg['rinet_projector']
        self.rinet_projector = RINetProjector(
            input_dim=rinet_cfg.get('input_dim', 1056),
            hidden_dim=rinet_cfg.get('hidden_dim', 256),
            output_dim=rinet_cfg.get('output_dim', 64),
            dropout=rinet_cfg.get('dropout', 0.3)
        )
        
        # 2. GNN Stream
        gnn_cfg = model_cfg['gnn_stream']
        gat_cfg = gnn_cfg['gat']
        self.gnn_stream = GNNStream(
            in_dim=gnn_cfg.get('node_dim', 69),
            edge_dim=gnn_cfg.get('edge_dim', 2),
            hidden_dim=gat_cfg.get('hidden_dim', 64),
            heads=gat_cfg.get('heads', [4, 4])[0], # use first layer head count for simplicity
            out_dim=gnn_cfg.get('output_dim', 128),
            dropout=gat_cfg.get('dropout', 0.3)
        )
        
        # 3. Handcrafted Stream
        hc_cfg = model_cfg['handcrafted_stream']
        self.handcrafted_stream = HandcraftedStream(
            input_dim=handcrafted_dim,
            hidden_dims=hc_cfg.get('hidden_dims', [256, 128]),
            dropouts=hc_cfg.get('dropout', [0.2, 0.1]),
            output_dim=hc_cfg.get('output_dim', 128)
        )
        
        # 4. CEFAM Fusion
        cefam_cfg = model_cfg['cefam']
        self.cefam_fusion = CEFAMFusion(
            d_model=cefam_cfg.get('d_model', 128),
            nhead=cefam_cfg.get('nhead', 4),
            dropout=cefam_cfg.get('dropout', 0.1)
        )
        
        # 5. Classifier Head
        clf_cfg = model_cfg['classifier']
        self.classifier = Classifier(
            input_dim=clf_cfg.get('input_dim', 256),
            hidden_dims=clf_cfg.get('hidden_dims', [128]),
            dropouts=clf_cfg.get('dropout', [0.3]),
            num_classes=clf_cfg.get('num_classes', 2)
        )
        
    def forward(self, data, handcrafted_features):
        """
        data: PyTorch Geometric Data object containing batch graphs
        handcrafted_features: Tensor of flat features [batch_size, handcrafted_dim]
        """
        # Node features shape in graph construction: [num_nodes, 1061]
        # First 5 are low level: x_norm, y_norm, dur_norm, pupil_norm, pupil_diff
        # Remaining 1056 are raw RINet features
        x_raw = data.x
        low_level = x_raw[:, :5]
        rinet_raw = x_raw[:, 5:]
        
        # Project RINet features: [num_nodes, 64]
        rinet_proj = self.rinet_projector(rinet_raw)
        
        # Concatenate: [num_nodes, 69]
        node_features = torch.cat([low_level, rinet_proj], dim=-1)
        
        # Pass through GNN Stream: z_graph = [batch_size, 128]
        z_graph, gnn_attn_weights = self.gnn_stream(
            x=node_features,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        )
        
        # Pass through Handcrafted Stream: z_expert = [batch_size, 128]
        z_expert = self.handcrafted_stream(handcrafted_features)
        
        # Pass through CEFAM Fusion: z_final = [batch_size, 256]
        z_final, cefam_attn_1, cefam_attn_2 = self.cefam_fusion(z_graph, z_expert)
        
        # Classify
        logits = self.classifier(z_final)
        
        return logits, gnn_attn_weights, cefam_attn_1, cefam_attn_2
