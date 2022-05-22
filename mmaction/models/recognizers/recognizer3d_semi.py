import torch
import torch.nn.functional as F
from ..builder import RECOGNIZERS
# Custom imports
from .base_semi import SemiBaseRecognizer
import torch.nn as nn

@RECOGNIZERS.register_module()
class SemiRecognizer3D(SemiBaseRecognizer):
    """Semi-supervised 3D recognizer model framework."""

    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled,
                      human_mask=None, imagenet_prob=None, cur_epoch=None):
        """Defines the computation performed at every call when training."""
        bz_labeled = imgs.shape[0]
        bz_unlabeled = imgs_weak.shape[0]
        imgs_all = torch.cat([imgs, imgs_weak, imgs_strong], dim=0)

        # TODO: If we forward imgs_weak with no_grad, and then jointly forward imgs and imgs_strong,
        # we might be able to save memory for a larger batch size? But not sure if this has
        # negative impact on batch-norm.
        imgs_all = imgs_all.reshape((-1, ) + imgs_all.shape[2:])
        x = self.extract_feat(imgs_all)
        cls_score = self.cls_head(x)

        # NOTE: pre-softmx logit
        cls_score_labeled = cls_score[:bz_labeled, :]
        cls_score_weak = cls_score[bz_labeled:bz_labeled+bz_unlabeled, :]
        cls_score_strong = cls_score[bz_labeled+bz_unlabeled:, :]

        loss = dict()

        ## Labeled data
        gt_labels = labels.squeeze()
        loss_labeled = self.cls_head.loss(cls_score_labeled, gt_labels)
        for k in loss_labeled:
            loss[k+'_labeled'] = loss_labeled[k]

        ## Unlabeled data
        with torch.no_grad():
            cls_prob_weak = F.softmax(cls_score_weak, dim=-1)

        # TODO: Control this threshold with cfg
        thres = self.fixmatch_threshold
        select = (torch.max(cls_prob_weak, 1).values >= thres).nonzero(as_tuple=False).squeeze(1)
        num_select = select.shape[0]
        if num_select > 0 and cur_epoch >= self.warmup_epoch:
            if self.framework == 'fixmatch':
                all_pseudo_labels = cls_score_weak.argmax(1).detach()
                pseudo_labels = torch.index_select(all_pseudo_labels.view(-1, 1), dim=0, index=select).squeeze(1)
            else:    # UDA
                # UDA uses soft (sharpened) pseudo labels
                all_pseudo_labels = F.softmax(cls_score_weak / 0.4, dim=1)
                pseudo_labels = torch.index_select(all_pseudo_labels, dim=0, index=select)
            cls_score_unlabeled = torch.index_select(cls_score_strong, dim=0, index=select)

            # NOTE: When we have ActorCutMix, we should do label smoothing
            # We use the human tube ratio (wrt to the whole clip) to determine the weight

            if self.framework == 'fixmatch':
                loss_unlabeled = self.cls_head.loss(cls_score_unlabeled, pseudo_labels)
            else:    # UDA
                loss_unlabeled = dict()
                loss_unlabeled['loss_cls'] = -torch.mean(torch.sum(F.log_softmax(cls_score_unlabeled, dim=1) * pseudo_labels, dim=1))

            # NOTE: When do loss reduce_mean, we should always divide by the batch size 
            # instead of number of confident samples. So we compensate here.
            if self.framework == 'fixmatch':
                loss['loss_cls_unlabeled'] = loss_unlabeled['loss_cls'] * num_select / bz_unlabeled
            else:
                loss['loss_cls_unlabeled'] = loss_unlabeled['loss_cls']

            ## Get statistics for strong/weak pair for sanity check
            #labels_unlabeled = torch.index_select(labels_unlabeled, dim=0, index=select).squeeze(1)
            #cls_score_weak = torch.index_select(cls_score_weak, dim=0, index=select)
            #cls_score_strong = torch.index_select(cls_score_strong, dim=0, index=select)
            #with torch.no_grad():
            #    loss_weak = self.cls_head.loss(cls_score_weak, labels_unlabeled)
            #    loss_strong = self.cls_head.loss(cls_score_strong, labels_unlabeled)
            #loss['top1_acc_weak'] = loss_weak['top1_acc']
            #loss['top1_acc_strong'] = loss_strong['top1_acc']
            #loss['top5_acc_weak'] = loss_weak['top5_acc']
            #loss['top5_acc_strong'] = loss_strong['top5_acc']

        else:
            loss['loss_cls_unlabeled'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top1_acc_weak'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top1_acc_strong'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top5_acc_weak'] = torch.zeros(1).to(cls_prob_weak.device)
            #loss['top5_acc_strong'] = torch.zeros(1).to(cls_prob_weak.device)

        with torch.no_grad():
            loss['num_select'] = (torch.ones(1)*num_select).to(cls_score_weak.device)


        return loss

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        if self.test_modality == 'diff':
            imgs = imgs[:, :, 1:, ...] - imgs[:, :, :-1, ...]
            if self.temp_align_indices is not None:
                x = self.temp_backbone(imgs)[-1]
            else:
                x = self.temp_backbone(imgs)
            cls_score = self.temp_sup_head(x)
        elif self.test_modality == 'video':
            if self.temp_align_indices is not None:
                x = self.extract_feat(imgs)[-1]
            else:
                x = self.extract_feat(imgs)

            if self.test_return_features:
                x = nn.AdaptiveAvgPool3d((1, 1, 1))(x)
                x = x.view(x.shape[0], -1)
                batch_size = x.shape[0]
                features = x.view(batch_size // num_segs, num_segs, -1)
                features = features.mean(dim=1)
                return features
            cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs):
        raise NotImplementedError

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)