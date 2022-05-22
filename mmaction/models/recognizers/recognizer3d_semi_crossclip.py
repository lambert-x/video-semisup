import torch
import torch.nn.functional as F
from ..builder import RECOGNIZERS
# Custom imports
from .base_semi import SemiBaseRecognizer
from ...utils import GatherLayer


@RECOGNIZERS.register_module()
class Semi_Crossclip_Recognizer3D(SemiBaseRecognizer):
    """Semi-supervised 3D recognizer model framework."""

    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled, cur_epoch=None):

        """Defines the computation performed at every call when training."""
        bz_labeled = imgs.shape[0] * imgs.shape[1]
        bz_unlabeled = imgs_weak.shape[0] * imgs_weak.shape[1]

        imgs = imgs.transpose(0, 1)
        imgs_weak = imgs_weak.transpose(0, 1)
        imgs_strong = imgs_strong.transpose(0, 1)

        imgs_all = torch.cat([imgs, imgs_weak, imgs_strong], dim=0)

        # TODO: If we forward imgs_weak with no_grad, and then jointly forward imgs and imgs_strong,
        # we might be able to save memory for a larger batch size? But not sure if this has
        # negative impact on batch-norm.
        imgs_all = imgs_all.reshape((-1,) + imgs_all.shape[2:])

        labels = labels.transpose(0, 1).reshape((-1, 1))


        vid_feature = self.extract_feat(imgs_all)


        # cls_score = self.cls_head(vid_feature)
        cls_score = dict()
        # cls_score_rgb, cls_score_diff = torch.split(cls_score_concat, bz_labeled+2*bz_unlabeled, dim=0)
        cls_score['rgb'] = self.cls_head(vid_feature)

        batch_total_len = bz_labeled + bz_unlabeled

        # NOTE: pre-softmx logit
        cls_score_labeled = dict()
        cls_score_weak = dict()
        cls_score_strong = dict()
        for view in ['rgb']:
            cls_score_labeled[view] = cls_score[view][:bz_labeled, :]
            cls_score_weak[view] = cls_score[view][bz_labeled:bz_labeled + bz_unlabeled, :]
            cls_score_strong[view] = cls_score[view][bz_labeled + bz_unlabeled:bz_labeled + 2 * bz_unlabeled, :]

        loss = dict()
        #
        # if 'weak' in self.crossclip_contrast_range and 'strong' in self.crossclip_contrast_range:
        #     query_index = torch.cat([torch.arange(bz_labeled // 2), torch.arange(bz_unlabeled // 2) + bz_labeled,
        #                              torch.arange(bz_unlabeled // 2) + bz_labeled + bz_unlabeled])
        #     key_mask = torch.ones(bz_labeled + 2 * bz_unlabeled)
        #     key_mask[query_index] = 0
        #     key_index = torch.arange(bz_labeled + 2 * bz_unlabeled)[key_mask.bool()]
        # elif 'weak' in self.crossclip_contrast_range:
        #     query_index = torch.cat([torch.arange(bz_labeled // 2), torch.arange(bz_unlabeled // 2) + bz_labeled])
        #     key_mask = torch.ones(bz_labeled + bz_unlabeled)
        #     key_mask[query_index] = 0
        #     key_index = torch.arange(bz_labeled + bz_unlabeled)[key_mask.bool()]
        # elif 'strong' in self.crossclip_contrast_range:
        #     query_index = torch.arange(bz_unlabeled // 2) + bz_labeled + bz_unlabeled
        #     key_index = query_index + (bz_unlabeled // 2)
        # else:
        #     pass
        #
        # if 'rgb' in self.crossclip_contrast_loss:
        #     contrast_rgb_feature = self.contrast_head_rgb(vid_feature)
        #     rgb_query = contrast_rgb_feature[query_index]
        #     rgb_key = contrast_rgb_feature[key_index]
        #     rgb_query_pred = self.simsiam_pred_rgb(rgb_query)
        #     rgb_key_pred = self.simsiam_pred_rgb(rgb_key)
        #     embedding_rgb = [rgb_query_pred, rgb_key_pred, rgb_query.detach(), rgb_key.detach()]
        #     loss['loss_rgb_contrast'] = self.contrast_loss_weight * self.simsiam_pred_rgb.loss(embedding_rgb)
        #
        #
        # if cur_epoch < self.contrast_warmup_epoch:
        #     for contrast_loss_type in ['loss_rgb_contrast']:
        #         if contrast_loss_type in loss.keys():
        #             loss[contrast_loss_type] = loss[contrast_loss_type].detach()

        gt_labels = labels.squeeze()

        # inter-frame difference modality (temporal) supervised loss

        # video supervised loss
        # loss_labeled_weight = 1
        loss_labeled_weight = self.loss_labeled_weight
        loss_labeled = {}
        for view in ['rgb']:
            loss_labeled[view] = self.cls_head.loss(cls_score_labeled[view], gt_labels)
            for k in loss_labeled[view]:
                if 'loss' in k:
                    loss[k + f'_labeled_{view}'] = loss_labeled[view][k] * loss_labeled_weight
                else:
                    loss[k + f'_labeled_{view}'] = loss_labeled[view][k]

        ## Unlabeled data
        with torch.no_grad():
            cls_score_weak = cls_score_weak['rgb']
            cls_prob_weak = F.softmax(cls_score_weak, dim=-1)
        # cls_prob_weak = F.softmax(cls_score_weak.detach(), dim=-1)
        # TODO: Control this threshold with cfg
        # if self.framework == 'fixmatch':
        #     thres = 0.95
        # else:    # UDA
        #     thres = 0.8
        thres = self.fixmatch_threshold
        select = (torch.max(cls_prob_weak, 1).values >= thres).nonzero(as_tuple=False).squeeze(1)
        num_select = select.shape[0]

        if num_select > 0 and cur_epoch >= self.warmup_epoch:
            all_pseudo_labels = cls_score_weak.argmax(1).detach()
            pseudo_labels = torch.index_select(all_pseudo_labels.view(-1, 1), dim=0, index=select).squeeze(1)

            cls_score_unlabeled = dict()
            loss_unlabeled = dict()
            loss_unlabeled_weight = self.loss_unlabeled_weight
            # loss_unlabeled_weight = 1
            for view in ['rgb']:
                cls_score_unlabeled[view] = torch.index_select(cls_score_strong[view], dim=0, index=select)
                loss_unlabeled[view] = self.cls_head.loss(cls_score_unlabeled[view], pseudo_labels)
                # NOTE: When do loss reduce_mean, we should always divide by the batch size
                # instead of number of confident samples. So we compensate here.
                loss[f'loss_cls_unlabeled_{view}'] = loss_unlabeled_weight * loss_unlabeled[view][
                    'loss_cls'] * num_select / bz_unlabeled

        else:
            for view in ['rgb']:
                loss[f'loss_cls_unlabeled_{view}'] = torch.zeros(1).to(cls_prob_weak.device)

        with torch.no_grad():
            loss['num_select'] = (torch.ones(1) * num_select).to(cls_prob_weak.device)

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
        elif self.test_modality == 'normalized_tg':
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
            cls_score = self.cls_head(x)
        else:
            raise NotImplementedError
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
