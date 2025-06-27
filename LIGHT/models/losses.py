import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.temperature = 0.1
    
    def forward(self, A, B):
        """
        Computes the cosine similarity loss.

        Args:
            A (torch.Tensor): First set of embeddings (B, N, C).
            B (torch.Tensor): Second set of embeddings (B, N, C).

        Returns:
            torch.Tensor: Computed loss.
        """
        # Normalize embeddings
        A_norm = F.normalize(A, p=2, dim=-1)
        B_norm = F.normalize(B, p=2, dim=-1)
        emb1 = A_norm
        emb2 = B_norm   
        B, N, C = A_norm.shape
        labels = torch.arange(N, dtype=torch.long).repeat(B).to(A.device) # Shape: (N x B,)
        sim12 = torch.bmm(emb1, emb2.transpose(-1, -2)).reshape(B * N, N) / self.temperature # Shape: (N * B, N)
        sim21 = torch.bmm(emb2, emb1.transpose(-1, -2)).reshape(B * N, N) / self.temperature # Shape: (N * B, N)
        nce_loss = self.ce_loss_fn(sim12, labels)
        nce_loss += self.ce_loss_fn(sim21, labels)
        return nce_loss



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=-1)
        weight_matrix = torch.ones_like(logits).to(logits.device)
        weight_matrix.fill_diagonal_(self.alpha)
        focal_loss = -weight_matrix * (1 - probs) ** self.gamma * target * torch.log(probs + 1e-8)
        focal_loss -= weight_matrix * probs ** self.gamma * (1 - target) * torch.log(1 - probs + 1e-8)
        return focal_loss    
        

        