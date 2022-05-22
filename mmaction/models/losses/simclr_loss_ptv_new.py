import torch
import numpy as np
from ..builder import LOSSES
from fvcore.nn.distributed import differentiable_all_gather
import torch.nn.functional as F
import torch.distributed as dist
from .simclr_loss_ptv import SimCLRLoss_PTV

@LOSSES.register_module()
class SimCLRLoss_PTV_New(torch.nn.Module):

    def __init__(self, temperature=0.5, loss_weight=1):
        super(SimCLRLoss_PTV_New, self).__init__()
        self.temperature = temperature
        self.simclr_loss = SimCLRLoss_PTV(temperature=temperature)
        self.loss_weight = loss_weight

    def forward(self, embedding):
        """Forward head.

        Args:
            positive (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x1, x2 = embedding
        loss = self.simclr_loss([x1, x2]) + self.simclr_loss([x2, x1])
        loss = self.loss_weight * loss
        return loss
