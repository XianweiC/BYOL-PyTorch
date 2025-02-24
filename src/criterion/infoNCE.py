import torch
import torch.nn as nn
from torch.nn import functional as F


class InfoNCE(nn.Module):
    def __init__(self, device=torch.device('mps')):
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, feat1, feat2, logit_scale=1.):
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        logits_per_image1 = logit_scale * feat1 @ feat2.T
        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        loss = (self.loss_function(logits_per_image1, labels) +
                self.loss_function(logits_per_image2, labels)) / 2
        return loss
