import torch
import torch.nn as nn
from src.tier4_advanced.resnet_projector import ResNetProjector
from src.tier4_advanced.gnn_stream import GNNStream
from src.tier4_advanced.handcrafted_stream import HandcraftedStream
from src.tier4_advanced.cefam_fusion import CEFAMFusion


class Classifier(nn.Module):
    """Classification head for the hybrid model."""

    def __init__(self, input_dim=256, hidden_dims=None, dropouts=None,
                 num_classes=2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]
        if dropouts is None:
            dropouts = [0.3]

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
    Full Hybrid Model:
      1. ResNetProjector  : 2048-dim ResNet50 visual features → 64-dim
      2. GNNStream        : graph topology → z_graph [128-dim]
                            + h_nodes [total_nodes, 256] (pre-pooling node embs)
      3. HandcraftedStream: flat features → z_expert [128-dim]
      4. CEFAMFusion      : bidirectional cross-attention
                            - Direction 1: z_expert queries h_nodes (N fixations)
                              → attn_1 [B, 1, N]: fixation saliency map
                            - Direction 2: z_graph queries z_expert (symmetric)
      5. Classifier Head  : logits [B, 2]
    """

    def __init__(self, handcrafted_dim: int, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config['model']

        # 1. ResNet50 Projector
        resnet_cfg = model_cfg.get('resnet_projector', model_cfg.get('rinet_projector', {}))
        self.resnet_projector = ResNetProjector(
            input_dim=resnet_cfg.get('input_dim', 2048),
            hidden_dim=resnet_cfg.get('hidden_dim', 256),
            output_dim=resnet_cfg.get('output_dim', 64),
            dropout=resnet_cfg.get('dropout', 0.3)
        )

        # 2. GNN Stream
        gnn_cfg = model_cfg['gnn_stream']
        gat_cfg = gnn_cfg['gat']
        hidden_dim = gat_cfg.get('hidden_dim', 64)
        heads = gat_cfg.get('heads', [4, 4])[0]
        node_emb_dim = hidden_dim * heads  # e.g. 64 * 4 = 256

        self.gnn_stream = GNNStream(
            in_dim=gnn_cfg.get('node_dim', 69),
            edge_dim=gnn_cfg.get('edge_dim', 2),
            hidden_dim=hidden_dim,
            heads=heads,
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

        # 4. CEFAM Fusion (fixed: genuine cross-attention on node sequences)
        cefam_cfg = model_cfg['cefam']
        padding_cfg = config.get('data', {}).get('padding', {})
        max_nodes = padding_cfg.get('max_seq_len', 24)

        self.cefam_fusion = CEFAMFusion(
            d_model=cefam_cfg.get('d_model', 128),
            nhead=cefam_cfg.get('nhead', 4),
            dropout=cefam_cfg.get('dropout', 0.1),
            node_emb_dim=node_emb_dim,
            max_nodes=max_nodes
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
        Args:
            data                : PyG batched graph (contains data.x, data.edge_index,
                                  data.edge_attr, data.batch)
            handcrafted_features: [B, handcrafted_dim]  flat expert features

        Returns:
            logits        : [B, 2]
            gnn_attn      : global attention gate weights [total_nodes]
            cefam_attn_1  : Expert→Graph saliency        [B, 1, N]
            cefam_attn_2  : Graph→Expert attention        [B, 1, 1]
        """
        # Split stored node features: [:5] low-level, [5:] raw ResNet50
        x_raw = data.x
        low_level = x_raw[:, :5]
        resnet_raw = x_raw[:, 5:]

        # Project ResNet50: [total_nodes, 2048] → [total_nodes, 64]
        resnet_proj = self.resnet_projector(resnet_raw)

        # Concatenate: [total_nodes, 69]
        node_features = torch.cat([low_level, resnet_proj], dim=-1)

        # GNN Stream: returns z_graph [B,128], h_nodes [total_nodes,256], attn
        z_graph, h_nodes, gnn_attn = self.gnn_stream(
            x=node_features,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            mask=data.mask if hasattr(data, 'mask') else None
        )

        # Handcrafted Stream: [B, handcrafted_dim] → [B, 128]
        z_expert = self.handcrafted_stream(handcrafted_features)

        # CEFAM Fusion (now genuine cross-attention)
        z_final, cefam_attn_1, cefam_attn_2 = self.cefam_fusion(
            z_graph=z_graph,
            z_expert=z_expert,
            h_nodes=h_nodes,
            node_batch=data.batch
        )

        # Classify
        logits = self.classifier(z_final)

        return logits, gnn_attn, cefam_attn_1, cefam_attn_2
