import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with optional positive and negative prototypes.
    Reference: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, task_ids=None, pos_prototype=None, neg_prototype=None):
        """
        Args:
            features: [B, D]
            labels: [B] (unused in current implementation, kept for compatibility)
            task_ids: [B]
            pos_prototype: [num_tasks, D] or None
            neg_prototype: [num_tasks, D] or None
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize all features
        features = F.normalize(features, dim=1)
        all_features = [features]
        all_task_ids = [task_ids]

        # Add positive prototypes if provided
        if pos_prototype is not None:
            pos_proto_norm = F.normalize(pos_prototype, dim=1)
            all_features.append(pos_proto_norm)
            # assign task ids 0..num_tasks-1 for positive prototypes
            num_tasks = pos_prototype.shape[0]
            pos_task_ids = torch.arange(num_tasks, device=device)
            all_task_ids.append(pos_task_ids)

        # Add negative prototypes if provided
        if neg_prototype is not None:
            neg_proto_norm = F.normalize(neg_prototype, dim=1)
            all_features.append(neg_proto_norm)
            # assign unique task ids that do NOT match any positive task id
            # use negative numbers or a large offset to ensure no overlap
            num_neg = neg_prototype.shape[0]
            # We'll set to -1, -2, ... (or any id not present in other sets)
            neg_task_ids = torch.full((num_neg,), -1, device=device)  # all negative prototypes share same id -1
            # Alternatively, assign unique negative ids to avoid them being positive with each other
            # but sharing -1 means they will be positive with each other (if we don't mask self)
            # Better to assign unique ids that are distinct from all others
            # Use a large offset: e.g., 1000 + i
            neg_task_ids = torch.arange(1000, 1000 + num_neg, device=device)
            all_task_ids.append(neg_task_ids)

        # Concatenate
        all_features = torch.cat(all_features, dim=0)
        all_task_ids = torch.cat(all_task_ids, dim=0)

        # Compute similarity matrix
        sim_matrix = torch.matmul(all_features, all_features.T) / self.temperature
        sim_matrix_exp = torch.exp(sim_matrix)

        # Mask for positive pairs: same task id
        mask = all_task_ids.unsqueeze(1) == all_task_ids.unsqueeze(0)
        mask = mask.float()

        # Remove self-comparison (diagonal)
        mask = mask - torch.eye(mask.shape[0], device=device)

        # Compute log probability
        log_prob = sim_matrix - torch.log(sim_matrix_exp.sum(1, keepdim=True) + 1e-12)

        # Average over positive pairs per sample
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss: negative mean of positive log-probs
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class MultiTaskContrastiveLoss(nn.Module):
    """
    Combine CE loss + Task-aware Contrastive with positive and negative prototypes.
    """
    def __init__(self, temperature=0.07, alpha=0.1):
        super(MultiTaskContrastiveLoss, self).__init__()
        self.supcon = SupConLoss(temperature)
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, logits, labels, features, task_ids, pos_prototype=None, neg_prototype=None):
        """
        Args:
            logits: [B, 2]
            labels: [B]
            features: [B, D]
            task_ids: [B]
            pos_prototype: [num_tasks, D] or None
            neg_prototype: [num_tasks, D] or None
        """
        ce_loss = self.ce(logits, labels)
        con_loss = self.supcon(features, labels, task_ids, pos_prototype, neg_prototype)
        total_loss = ce_loss + self.alpha * con_loss
        return total_loss, ce_loss, con_loss


if __name__ == "__main__":
    B = 8
    D = 384
    num_tasks = 12

    features = torch.randn(B, D)
    labels = torch.randint(0, 2, (B,))
    task_ids = torch.randint(0, num_tasks, (B,))
    pos_proto = torch.randn(num_tasks, D)
    neg_proto = torch.randn(num_tasks, D)
    logits = torch.randn(B, 2)

    loss_fn = MultiTaskContrastiveLoss()
    loss, ce, con = loss_fn(logits, labels, features, task_ids, pos_proto, neg_proto)

    print("Total Loss:", loss.item())
    print("CE Loss:", ce.item())
    print("Contrastive Loss:", con.item())