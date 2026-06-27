import torch
import torch.nn as nn


class ResNetProjector(nn.Module):
    """
    Projector MLP to compress 2048-dim ResNet50 visual features to 64-dim.
    Reduces parameter space and prevents overfitting on small datasets.
    """
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=64, dropout=0.3):
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
        # x: [num_nodes, input_dim] (typically 2048 from ResNet50)
        # BatchNorm1d requires batch > 1; fall back to eval mode for single-node edge case.
        if x.shape[0] <= 1:
            was_training = self.training
            self.eval()
            out = self.projector(x)
            if was_training:
                self.train()
            return out
        return self.projector(x)
