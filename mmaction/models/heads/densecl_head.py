import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

from ..builder import HEADS
from ..builder import build_loss


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@HEADS.register_module()
class DenseCLHead(nn.Module):
    '''
    The non-linear neck in DenseCL.
    Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None,
                 loss_contrast=dict(type='SimCLRLoss')
                 ):
        super(DenseCLHead, self).__init__()

        self.with_pool = (num_grid is not None)

        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool3d((1, num_grid, num_grid))

        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, hid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hid_channels, out_channels, kernel_size=1)
        )

        self.avgpool2 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.loss_contrast = build_loss(loss_contrast)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):

        if self.with_pool:
            x = self.pool(x)  # bxcxtxsxs (last layer t=1)
        x = self.mlp(x)  # sxs: bxcxtxsxs
        x = x.view(x.size(0), x.size(1), -1)  # bxcxs^2
        return x
        # avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        # avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd

    def loss(self, embedding):
        return self.loss_contrast(embedding)
