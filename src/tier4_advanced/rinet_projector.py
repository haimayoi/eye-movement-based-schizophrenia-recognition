import torch
import torch.nn as nn

class RINetProjector(nn.Module):
    """
    Projector MLP to compress 1056-dim RINet visual features to 64-dim.
    Reduces parameter space and prevents overfitting.
    """
    def __init__(self, input_dim=1056, hidden_dim=256, output_dim=64, dropout=0.3):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        # x shape: [num_nodes, 1056]
        # BatchNorm1d requires batch dimension > 1. If we have only 1 node, it might fail.
        # So we check if we need to adjust or run with training mode / eval mode safely.
        if x.shape[0] <= 1:
            # Bypass batch norm if only 1 node (or in eval mode, it works fine)
            # To be safe, we temporarily set batchnorm to eval if input is 1-dimensional
            was_training = self.training
            self.eval()
            out = self.projector(x)
            if was_training:
                self.train()
            return out
        return self.projector(x)
