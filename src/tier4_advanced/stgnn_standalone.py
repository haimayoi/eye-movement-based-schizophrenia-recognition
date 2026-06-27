import torch
import torch.nn as nn
from src.tier4_advanced.resnet_projector import ResNetProjector
from src.tier4_advanced.gnn_stream import GNNStream

class Classifier(nn.Module):
    """
    Classification head for the standalone GNN model.
    """
    def __init__(self, input_dim=128, hidden_dims=[256, 128], dropouts=[0.3, 0.2], num_classes=2, activation='relu'):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim, drop in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(curr_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class STGNNStandaloneModel(nn.Module):
    """
    Spatiotemporal GNN Standalone Model (ablation study: GNN stream only).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        
        # 1. ResNet50 Projector (2048-dim ResNet50 features → 64-dim)
        resnet_cfg = model_cfg.get('resnet_projector', model_cfg.get('rinet_projector', {}))
        self.resnet_projector = ResNetProjector(
            input_dim=resnet_cfg.get('input_dim', 2048),
            hidden_dim=resnet_cfg.get('hidden_dim', 256),
            output_dim=resnet_cfg.get('output_dim', 64),
            dropout=resnet_cfg.get('dropout', 0.3)
        )
        
        # 2. GNN Stream
        gnn_cfg = model_cfg['gnn']
        # Node features: 5 low-level + 64 ResNet50-projected = 69
        self.gnn_stream = GNNStream(
            in_dim=gnn_cfg.get('node_dim', 69),
            edge_dim=gnn_cfg.get('edge_dim', 2),
            hidden_dim=gnn_cfg.get('hidden_dim', 64),
            heads=gnn_cfg.get('heads', [4, 4])[0],
            out_dim=gnn_cfg.get('output_dim', 128),
            dropout=gnn_cfg.get('dropout', 0.3)
        )
        
        # 3. Classifier Head (taking 128-dim z_graph to num_classes=2)
        clf_cfg = model_cfg['classifier']
        self.classifier = Classifier(
            input_dim=gnn_cfg.get('output_dim', 128),
            hidden_dims=clf_cfg.get('hidden_dims', [256, 128]),
            dropouts=clf_cfg.get('dropout', [0.3, 0.2]),
            num_classes=clf_cfg.get('num_classes', 2),
            activation=clf_cfg.get('activation', 'relu')
        )
        
    def forward(self, data, handcrafted_features=None):
        """
        data: PyTorch Geometric Data object containing batch graphs
        handcrafted_features: Ignored (kept for compatibility with training loop)
        """
        x_raw = data.x
        low_level = x_raw[:, :5]
        resnet_raw = x_raw[:, 5:]

        # Project ResNet50 features: [num_nodes, 2048] → [num_nodes, 64]
        resnet_proj = self.resnet_projector(resnet_raw)

        # Concatenate: [num_nodes, 69]
        node_features = torch.cat([low_level, resnet_proj], dim=-1)
        
        # GNNStream now returns (z_graph, h_nodes, attn_weights)
        # h_nodes is not used in standalone mode (no CEFAM fusion)
        z_graph, _h_nodes, gnn_attn_weights = self.gnn_stream(
            x=node_features,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            mask=data.mask if hasattr(data, 'mask') else None
        )

        # Classify directly from z_graph
        logits = self.classifier(z_graph)

        # Return None for CEFAM attentions (no fusion in standalone mode)
        return logits, gnn_attn_weights, None, None
