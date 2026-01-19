import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (Batch, Seq_Len, Num_Classes) - raw logits
        targets: (Batch, Seq_Len) - ground truth class indices
        """
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1), reduction='none')
        
        # probabilities for the true class
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class BeTLoss(nn.Module):
    def __init__(self, action_dim, n_clusters, focal_alpha=1.0, focal_gamma=2.0):
        super(BeTLoss, self).__init__()
        self.action_dim = action_dim
        self.n_clusters = n_clusters
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
    def forward(self, pred_logits, pred_offsets, target_labels, true_offsets):
        """
        pred_logits: (B, T, K) - predicted class logits
        pred_offsets: (B, T, K, Action_Dim) - predicted action offsets
        target_labels: (B, T) - ground truth class indices
        true_offsets: (B, T, Action_Dim) - ground truth action offsets
        """
        # Focal Loss
        cls_loss = self.focal_loss_fn(pred_logits, target_labels)

        # Offset Regression Loss (Masked MSE)
        # (B, T) -> (B, T, 1, 1) -> (B, T, 1, Action_Dim)
        target_indices = target_labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.action_dim)

        selected_pred_offsets = torch.gather(pred_offsets, 2, target_indices).squeeze(2)

        offset_loss = F.mse_loss(selected_pred_offsets, true_offsets, reduction='mean')
        total_loss = cls_loss + offset_loss
        return total_loss, cls_loss, offset_loss
    