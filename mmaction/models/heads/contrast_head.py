import torch.nn as nn
from mmcv.cnn import normal_init, kaiming_init

from ..builder import HEADS
from .base import BaseHead
from ..builder import build_loss
from mmcv.cnn.bricks import build_norm_layer


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
class ContrastHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 out_channels,
                 in_channels,
                 hidden_channels=None,
                 loss_contrast=dict(type='SimCLRLoss'),
                 spatial_type='avg',
                 norm_cfg=None,
                 fc_layer_num=2,
                 with_final_normalize=False,
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.loss_contrast = build_loss(loss_contrast)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # self.fcs = nn.Sequential(
        #     nn.Linear(self.in_channels, self.in_channels),
        #     nn.BatchNorm1d(self.in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.in_channels, self.out_channels)
        # )
        # self.fc1 = nn.Linear(self.in_channels, self.in_channels)
        if hidden_channels is None:
            self.hidden_channels = in_channels
        else:
            self.hidden_channels = hidden_channels
        if fc_layer_num == 2:
            if norm_cfg is not None:
                norm1_name, self.norm1 = build_norm_layer(norm_cfg, self.hidden_channels)
                self.fcs = nn.Sequential(
                    nn.Linear(self.in_channels, self.hidden_channels, bias=False),
                    self.norm1,
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_channels, self.out_channels, bias=(not with_final_normalize))
                )
            else:
                self.fcs = nn.Sequential(
                    nn.Linear(self.in_channels, self.hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_channels, self.out_channels)
                )
        elif fc_layer_num == 3:
            if norm_cfg is not None:
                norm1_name, self.norm1 = build_norm_layer(norm_cfg, self.hidden_channels)
                norm2_name, self.norm2 = build_norm_layer(norm_cfg, self.hidden_channels)
                self.fcs = nn.Sequential(
                    nn.Linear(self.in_channels, self.hidden_channels, bias=False),
                    self.norm1,
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                    self.norm2,
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_channels, self.out_channels, bias=(not with_final_normalize))
                )
            else:
                self.fcs = nn.Sequential(
                    nn.Linear(self.in_channels, self.hidden_channels, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_channels, self.out_channels)
                )
        else:
            raise NotImplementedError
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(self.in_channels, self.out_channels)
        if with_final_normalize:
            norm_final_cfg = norm_cfg.copy()
            norm_final_cfg['affine'] = False
            norm_final_name, self.norm_final = build_norm_layer(norm_final_cfg, self.out_channels)
            self.fcs.add_module(str(len(self.fcs) + 1), self.norm_final)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # normal_init(self.fc1, std=self.init_std)
        # normal_init(self.fc2, std=self.init_std)
        # normal_init(self.fcs, std=self.init_std)
        _init_weights(self.fcs, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        out = self.fcs(x)
        # [N, in_channels]
        return out

    def loss(self, embedding):
        return self.loss_contrast(embedding)


@HEADS.register_module()
class Contrast2DHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 out_channels,
                 in_channels,
                 hidden_channels=None,
                 loss_contrast=dict(type='SimCLRLoss'),
                 spatial_type='avg',
                 norm_cfg=None,
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.loss_contrast = build_loss(loss_contrast)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # self.fcs = nn.Sequential(
        #     nn.Linear(self.in_channels, self.in_channels),
        #     nn.BatchNorm1d(self.in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.in_channels, self.out_channels)
        # )
        # self.fc1 = nn.Linear(self.in_channels, self.in_channels)
        if hidden_channels is None:
            self.hidden_channels = in_channels
        else:
            self.hidden_channels = hidden_channels
        if norm_cfg is not None:
            norm1_name, self.norm1 = build_norm_layer(norm_cfg, self.hidden_channels)
            self.fcs = nn.Sequential(
                nn.Linear(self.in_channels, self.hidden_channels, bias=False),
                self.norm1,
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_channels, self.out_channels)
            )
        else:
            self.fcs = nn.Sequential(
                nn.Linear(self.in_channels, self.hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_channels, self.out_channels)
            )
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(self.in_channels, self.out_channels)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # normal_init(self.fc1, std=self.init_std)
        # normal_init(self.fc2, std=self.init_std)
        # normal_init(self.fcs, std=self.init_std)
        _init_weights(self.fcs, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        out = self.fcs(x)
        # [N, in_channels]
        return out

    def loss(self, embedding):
        return self.loss_contrast(embedding)
