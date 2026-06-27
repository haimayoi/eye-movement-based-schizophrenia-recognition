import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithEntropyReg(nn.Module):
    """
    Combines Focal Loss (to handle hard examples and class imbalance)
    with Entropy Sparsity Regularization (to encourage sparse, interpretable attention).
    """
    def __init__(self, alpha=0.25, gamma=2.0, entropy_lambda=0.01, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_lambda = entropy_lambda
        self.eps = eps
        
    def forward(self, logits, targets, attention_weights=None, node_batch=None):
        """
        logits: Model predictions before softmax [batch_size, num_classes]
        targets: Ground truth class indices [batch_size]
        attention_weights: Optional attention weights to regularize.
            - [B, 1, seq_len] for BiCA-HS (per-sample distribution over seq tokens)
            - [total_nodes]   for GNN models (flat; requires node_batch)
        node_batch: [total_nodes] graph-assignment vector from PyG batch.
            Required when attention_weights is a flat 1D GNN gate tensor.
        """
        # --- Focal Loss ---
        probs = F.softmax(logits, dim=-1)
        targets_oh = F.one_hot(targets, num_classes=logits.shape[-1]).float()

        pt = (probs * targets_oh).sum(dim=-1)
        alpha_t = self.alpha * targets.float() + (1.0 - self.alpha) * (1.0 - targets.float())
        focal_loss = -alpha_t * ((1.0 - pt) ** self.gamma) * torch.log(pt + self.eps)
        loss = focal_loss.mean()

        # --- Entropy Sparsity Regularization ---
        if attention_weights is not None and self.entropy_lambda > 0:
            if node_batch is not None:
                # GNN case: attention_weights is [total_nodes] flat across the batch.
                # Compute per-graph entropy then average, so the loss scale is
                # independent of batch size and graph size.
                from torch_geometric.utils import unbatch
                graphs = unbatch(attention_weights.unsqueeze(-1), node_batch)
                per_graph_entropy = torch.stack([
                    -(g.squeeze(-1) * torch.log(g.squeeze(-1) + self.eps)).sum()
                    for g in graphs
                ])
                entropy = per_graph_entropy
            else:
                # BiCA-HS case: attention_weights is [B, 1, seq_len] or [B, seq_len].
                # .sum(dim=-1) reduces over the distribution dimension → [B, 1] or [B].
                entropy = -(attention_weights * torch.log(attention_weights + self.eps)).sum(dim=-1)

            loss += self.entropy_lambda * entropy.mean()

        return loss
