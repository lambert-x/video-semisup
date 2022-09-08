import torch
import numpy as np
from ..builder import LOSSES
from fvcore.nn.distributed import differentiable_all_gather
import torch.nn.functional as F
import torch.distributed as dist


@LOSSES.register_module()
class SimCLRLoss_PTV(torch.nn.Module):

    def __init__(self, temperature=0.5):
        super(SimCLRLoss_PTV, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embedding):
        """Forward head.

        Args:
            positive (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x1, x2 = embedding
        x1 = F.normalize(x1, p=2, dim=1)

        x2 = F.normalize(x2, p=2, dim=1)
        x2 = torch.cat(differentiable_all_gather(x2), dim=0)
        prod = torch.einsum("nc,kc->nk", [x1, x2])
        prod = prod.div(self.temperature)
        batch_size = x1.size(0)
        if dist.is_available() and dist.is_initialized():
            device_ind = dist.get_rank()
        else:
            device_ind = 0
        gt = (
            torch.tensor(
                list(range(device_ind * batch_size, (device_ind + 1) * batch_size))
            )
                .long()
                .to(x1.device)
        )
        loss = self.criterion(prod, gt)
        return loss
