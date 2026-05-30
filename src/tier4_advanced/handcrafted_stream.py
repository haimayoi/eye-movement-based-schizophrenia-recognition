import torch
import torch.nn as nn

class HandcraftedStream(nn.Module):
    """
    Encodes Tier 2 flat subject-level / trial-level features
    into a 128-dimensional embedding space (z_expert).
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], dropouts=[0.2, 0.1], output_dim=128):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        
        # Layer 1
        layers.append(nn.Linear(curr_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropouts[0]))
        
        # Layer 2
        layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        layers.append(nn.BatchNorm1d(hidden_dims[1]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropouts[1]))
        
        # Final projection to output_dim (z_expert)
        layers.append(nn.Linear(hidden_dims[1], output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.GELU())
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        x: Handcrafted feature matrix [batch_size, input_dim]
        """
        # If batch size is 1, batchnorm will crash in training mode.
        # Handle batch size = 1 safely.
        if x.shape[0] <= 1:
            was_training = self.training
            self.eval()
            out = self.encoder(x)
            if was_training:
                self.train()
            return out
        return self.encoder(x)
