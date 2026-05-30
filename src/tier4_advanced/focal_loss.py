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
        
    def forward(self, logits, targets, attention_weights=None):
        """
        logits: Model predictions before softmax [batch_size, num_classes]
        targets: Ground truth class indices [batch_size]
        attention_weights: Optional attention weights tensor to regularize [batch_size, N] or similar
        """
        # --- Focal Loss ---
        probs = F.softmax(logits, dim=-1)
        targets_oh = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        
        # Probability of the true class
        pt = (probs * targets_oh).sum(dim=-1)
        
        # Class balancing factor alpha_t
        # targets is index (e.g. 0 or 1)
        alpha_t = self.alpha * targets.float() + (1.0 - self.alpha) * (1.0 - targets.float())
        
        # Focal Loss computation
        focal_loss = -alpha_t * ((1.0 - pt) ** self.gamma) * torch.log(pt + self.eps)
        loss = focal_loss.mean()
        
        # --- Entropy Sparsity Regularization ---
        if attention_weights is not None and self.entropy_lambda > 0:
            # Normalize attention weights if they don't sum to 1 over the last dimension
            # (e.g. in case of raw weights, though they should be softmaxed already)
            # attention_weights shape: [batch_size, seq_len] or [num_nodes]
            # Calculate entropy: -sum(p * log(p))
            entropy = -(attention_weights * torch.log(attention_weights + self.eps)).sum(dim=-1)
            loss += self.entropy_lambda * entropy.mean()
            
        return loss
