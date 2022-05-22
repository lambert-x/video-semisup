import torch
import numpy as np
from ..builder import LOSSES
import torch.nn as nn

@LOSSES.register_module()
class CosineSimiLoss(torch.nn.Module):

    def __init__(self, dim=1):
        super(CosineSimiLoss, self).__init__()
        self.criterion = torch.nn.CosineSimilarity(dim=dim)

    def forward(self, v1, v2):
        """Forward head.

        Args:
            positive (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        v1 = nn.functional.normalize(v1, dim=1)
        v2 = nn.functional.normalize(v2, dim=1)
        loss = 1 - self.criterion(v1, v2).mean()
        return loss
