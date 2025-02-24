from torch import nn
from torch.nn import functional as F


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    @staticmethod
    def forward(pred, target):
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        return 2 - 2 * (pred * target).sum(dim=-1)