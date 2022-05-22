from abc import abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

# Custom imports
from .base import BaseRecognizer
from .. import builder

from ..losses.cosine_simi_loss import CosineSimiLoss

class SemiAppTemp_SimCLR_BaseRecognizer(BaseRecognizer):
    """Base class for semi-supervised recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head,
                 cls_head_temp=None, temp_contrast_head=None, temp_backbone='same', temp_sup_head='same',
                 cls_head_app=None, app_contrast_head=None, app_backbone=None, app_sup_head=None,
                 contrast_head_rgb=None, contrast_head_tg=None, contrast_head_shared=None,
                 simsiam_densecl_head=None, contrast_densecl_head=None,
                 app_frame_num=1, train_cfg=None, test_cfg=None):
        super().__init__(backbone, cls_head, train_cfg=train_cfg, test_cfg=test_cfg)

        self.framework = train_cfg.get('framework', 'fixmatch')
        self.warmup_epoch = train_cfg.get('warmup_epoch', 0)
        self.temp_align_indices = train_cfg.get('temp_align_indices', None)
        self.align_loss_func = train_cfg.get('align_loss_func', 'L1')
        self.align_stop_grad = train_cfg.get('align_stop_grad', 'tg')
        self.align_with_correspondence = train_cfg.get('align_with_correspondence', False)

        self.cls_loss_weight = train_cfg.get('cls_loss_weight', dict(labeled=0.5, unlabeled=0.5))
        self.fixmatch_threshold = train_cfg.get('fixmatch_threshold', 0.95)
        self.specific_modality = train_cfg.get('specific_modality', None)
        self.pseudo_label_metric = train_cfg.get('pseudo_label_metric', 'rgb')
        self.crossclip_contrast_loss = train_cfg.get('crossclip_contrast_loss', [])
        self.crossclip_contrast_range = train_cfg.get('crossclip_contrast_range', ['weak', 'strong'])
        self.contrast_warmup_epoch = train_cfg.get('contrast_warmup_epoch', 0)
        self.contrast_loss_weight = train_cfg.get('contrast_loss_weight', 1)
        print('pseudo_label_metric:', self.pseudo_label_metric)

        self.loss_clip_selection = train_cfg.get('loss_clip_selection', None)
        self.densecl_indices = train_cfg.get('densecl_indices', None)
        self.densecl_strategy = train_cfg.get('densecl_strategy', 'simsiam')
        self.loss_lambda = train_cfg.get('loss_lambda', None)

        if temp_backbone == 'same':
            temp_backbone = backbone
        if temp_sup_head == 'same':
            temp_sup_head = cls_head

        self.test_modality = test_cfg.get('test_modality', 'video')
        self.test_return_features = test_cfg.get('return_features', False)

        if self.temp_align_indices is not None:
            align_loss_dict = dict(L1=nn.L1Loss(), L2=nn.MSELoss(), Cosine=CosineSimiLoss(dim=1))
            self.align_criterion = align_loss_dict[self.align_loss_func]
            backbone.out_indices = (0, 1, 2, 3)
            temp_backbone.out_indices = (0, 1, 2, 3)
            # if self.align_loss_func == 'L1':
            #     self.align_criterion = nn.L1Loss()
            # elif self.align_loss_func == 'L2':
            #     self.align_criterion = nn.MSELoss()

        self.backbone = builder.build_backbone(backbone)
        self.temp_backbone = builder.build_backbone(temp_backbone)
        self.temp_sup_head = builder.build_head(temp_sup_head)
        self.temp_backbone.init_weights()
        self.temp_sup_head.init_weights()

        if temp_contrast_head is not None and cls_head_temp is not None:
            self.use_temp_contrast = True

            self.temp_contrast_head = builder.build_head(temp_contrast_head)
            self.cls_head_temp = builder.build_head(cls_head_temp)
            self.temp_contrast_head.init_weights()
            self.cls_head_temp.init_weights()


        else:
            self.use_temp_contrast = False

        if 'rgb' in self.crossclip_contrast_loss:
            self.contrast_head_rgb = builder.build_head(contrast_head_rgb)
            self.contrast_head_rgb.init_weights()

        if 'tg' in self.crossclip_contrast_loss:
            self.contrast_head_tg = builder.build_head(contrast_head_tg)
            self.contrast_head_tg.init_weights()


        if ('crossview' in self.crossclip_contrast_loss) or ('crossview_sameclip' in self.crossclip_contrast_loss):
            self.contrast_head_rgb = builder.build_head(contrast_head_rgb)
            self.contrast_head_rgb.init_weights()

            self.contrast_head_tg = builder.build_head(contrast_head_tg)
            self.contrast_head_tg.init_weights()

        if ('crossview_sharedhead' in self.crossclip_contrast_loss) or ('crossview_sameclip_sharedhead' in self.crossclip_contrast_loss):
            self.contrast_head_shared = builder.build_head(contrast_head_shared)
            self.contrast_head_shared.init_weights()

        if 'crossview_crossclip_densecl' in self.crossclip_contrast_loss or 'sameview_crossclip_densecl' in self.crossclip_contrast_loss:
            self.contrast_densecl_head = builder.build_head(contrast_densecl_head)
            self.contrast_densecl_head.init_weights()

        # Build the appearance branch
        if app_backbone is None:
            self.app_backbone = None
        else:
            self.cls_head_app = builder.build_head(cls_head_app)
            self.app_backbone = builder.build_backbone(app_backbone)
            self.app_contrast_head = builder.build_head(app_contrast_head)
            self.cls_head_app.init_weights()
            self.app_backbone.init_weights()
            self.app_contrast_head.init_weights()

            if app_sup_head is not None:
                self.app_sup_head = builder.build_head(app_sup_head)
                self.app_sup_head.init_weights()

            self.app_frame_num = app_frame_num

        if simsiam_densecl_head is not None:
            self.simsiam_densecl_pred = builder.build_head(simsiam_densecl_head)
            self.simsiam_densecl_pred.init_weights()

        self.init_weights()

    @abstractmethod
    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled,
                      imgs_appearance, imgs_diff_labeled, imgs_diff_weak, imgs_diff_strong):
        """Defines the computation performed at every call when training."""
        pass

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        # print(losses)
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        # print(loss)
        log_vars['loss'] = loss
        if dist.is_available() and dist.is_initialized():
            loss_value = log_vars['num_select'].data.clone()
            dist.all_reduce(loss_value)
            log_vars['num_select'] = loss_value.item()

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                if loss_name == 'num_select':
                    continue

                loss_value = loss_value.data.clone()
                # NOTE: Compute unlabeled data accuracy for selected confident examples only
                if 'acc_weak' in loss_name or 'acc_strong' in loss_name:
                    dist.all_reduce(loss_value)
                    loss_value.div_(log_vars['num_select'] + 1e-6)
                    # TODO: A work-around. Sometimes all 0 accuracy will explode after all_reduce
                    if loss_value > 1. or loss_value < 0.:
                        loss_value = torch.zeros(1).to(loss_value.device)
                else:
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, imgs, label=None, return_loss=True, imgs_weak=None, imgs_strong=None, label_unlabeled=None,
                **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if imgs_weak is None or imgs_strong is None or label_unlabeled is None:
                raise ValueError('Unlabeled data should be available.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
            return self.forward_train(imgs, label, imgs_weak, imgs_strong, label_unlabeled, **kwargs)
        else:
            return self.forward_test(imgs)

    def train_step(self, data_batch_labeled, data_batch_unlabeled, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch_labeled (dict): The output of dataloader (labeled).
            data_batch_unlabeled (dict): The output of dataloader (unlabeled).
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch_labeled['imgs']
        label = data_batch_labeled['label']
        imgs_diff_labeled = data_batch_labeled['imgs_diff']

        imgs_weak = data_batch_unlabeled['imgs_weak']
        imgs_strong = data_batch_unlabeled['imgs_strong']
        imgs_diff_weak = data_batch_unlabeled['imgs_diff_weak']
        imgs_diff_strong = data_batch_unlabeled['imgs_diff_strong']
        if self.app_backbone is None:
            imgs_appearance = None
        else:
            imgs_appearance = torch.cat(
                [data_batch_labeled['imgs_appearance'], data_batch_unlabeled['imgs_appearance']], dim=0)
        label_unlabeled = data_batch_unlabeled['label_unlabeled']

        # losses = self(imgs, label, imgs_weak=imgs_weak, imgs_strong=imgs_strong, label_unlabeled=label_unlabeled, human_mask=human_mask, imagenet_prob=imagenet_prob)
        losses = self(imgs, label, imgs_weak=imgs_weak, imgs_strong=imgs_strong, label_unlabeled=label_unlabeled,
                      imgs_appearance=imgs_appearance, imgs_diff_labeled=imgs_diff_labeled,
                      imgs_diff_weak=imgs_diff_weak, imgs_diff_strong=imgs_diff_strong, **kwargs)
        # print(losses)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch_labeled.values()))))

        return outputs
