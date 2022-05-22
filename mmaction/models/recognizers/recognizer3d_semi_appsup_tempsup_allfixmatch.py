import torch
import torch.nn.functional as F
from ..builder import RECOGNIZERS
# Custom imports
from .base_semi_apppearance_temporal_allfixmatch import SemiAppTemp_AllFixMatch_BaseRecognizer
from ...utils import GatherLayer


@RECOGNIZERS.register_module()
class Semi_AppSup_TempSup_AllFixMatch_Recognizer3D(SemiAppTemp_AllFixMatch_BaseRecognizer):
    """Semi-supervised 3D recognizer model framework."""

    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled, imgs_appearance,
                      imgs_diff_labeled, imgs_diff_weak, imgs_diff_strong, cur_epoch=None):

        """Defines the computation performed at every call when training."""
        bz_labeled = imgs.shape[0]
        bz_unlabeled = imgs_weak.shape[0]
        imgs_all = torch.cat([imgs, imgs_weak, imgs_strong], dim=0)
        imgs_diff_all = torch.cat([imgs_diff_labeled, imgs_diff_weak, imgs_diff_strong], dim=0)

        # TODO: If we forward imgs_weak with no_grad, and then jointly forward imgs and imgs_strong,
        # we might be able to save memory for a larger batch size? But not sure if this has
        # negative impact on batch-norm.
        imgs_all = imgs_all.reshape((-1, ) + imgs_all.shape[2:])
        imgs_diff_all = imgs_diff_all.reshape((-1,) + imgs_diff_all.shape[2:])

        if self.temp_align_indices is not None:
            vid_features = self.extract_feat(imgs_all)
            vid_feature = vid_features[-1]
            imgs_diff_features = self.temp_backbone(imgs_diff_all)
            imgs_diff_feature = imgs_diff_features[-1]
        else:
            vid_feature = self.extract_feat(imgs_all)
            imgs_diff_feature = self.temp_backbone(imgs_diff_all)

        # cls_score = self.cls_head(vid_feature)
        cls_score = dict()
        # cls_score_rgb, cls_score_diff = torch.split(cls_score_concat, bz_labeled+2*bz_unlabeled, dim=0)
        cls_score['rgb'] = self.cls_head(vid_feature)
        cls_score['diff'] = self.temp_sup_head(imgs_diff_feature)
        clip_len = imgs_all.size(2)
        batch_total_len = bz_labeled + bz_unlabeled


        # NOTE: pre-softmx logit
        cls_score_labeled = dict()
        cls_score_weak = dict()
        cls_score_strong = dict()
        for view in ['rgb', 'diff']:
            cls_score_labeled[view] = cls_score[view][:bz_labeled, :]
            cls_score_weak[view] = cls_score[view][bz_labeled:bz_labeled+bz_unlabeled, :]
            cls_score_strong[view] = cls_score[view][bz_labeled+bz_unlabeled:bz_labeled+2*bz_unlabeled, :]


        loss = dict()

        if self.use_temp_contrast:
            cls_score_temp = self.cls_head_temp(vid_feature)
            vid_temporal = cls_score_temp[:batch_total_len, :]
            imgs_diff_temporal = self.temp_contrast_head(imgs_diff_feature[:batch_total_len, :])
            embedding_temp = torch.cat([vid_temporal, imgs_diff_temporal], dim=0)
            embedding_temp = embedding_temp / (torch.norm(embedding_temp, p=2, dim=1, keepdim=True) + 1e-10)

            embedding_temp = torch.cat(GatherLayer.apply(embedding_temp), dim=0)  # (2N)xd
            loss['loss_temporal_contrast'] = self.cls_head_temp.loss(embedding_temp)

        if self.temp_align_indices is not None:
            for i, layer_idx in enumerate(self.temp_align_indices):
                loss[f'loss_layer{layer_idx}_align'] = self.align_criterion(vid_features[i][:batch_total_len, ...],
                                                                            imgs_diff_features[i][:batch_total_len, ...].detach())

        gt_labels = labels.squeeze()

        # inter-frame difference modality (temporal) supervised loss

        # video supervised loss
        loss_labeled_weight = 0.5
        loss_labeled = {}
        for view in ['rgb', 'diff']:
            loss_labeled[view] = self.cls_head.loss(cls_score_labeled[view], gt_labels)
            for k in loss_labeled[view]:
                if 'loss' in k:
                    loss[k + f'_labeled_{view}'] = loss_labeled[view][k] * loss_labeled_weight
                else:
                    loss[k + f'_labeled_{view}'] = loss_labeled[view][k]


        ## Unlabeled data
        with torch.no_grad():
            if self.pseudo_label_metric == 'rgb':
                cls_score_weak = cls_score_weak['rgb']
            elif self.pseudo_label_metric == 'tg':
                cls_score_weak = cls_score_weak['diff']
            elif self.pseudo_label_metric == 'avg':
                cls_score_weak = (cls_score_weak['rgb'] + cls_score_weak['diff']) / 2
            else:
                raise NotImplementedError
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
            loss_unlabeled_weight = 0.5
            for view in ['rgb', 'diff']:
                cls_score_unlabeled[view] = torch.index_select(cls_score_strong[view], dim=0, index=select)
                loss_unlabeled[view] = self.cls_head.loss(cls_score_unlabeled[view], pseudo_labels)
                # NOTE: When do loss reduce_mean, we should always divide by the batch size
                # instead of number of confident samples. So we compensate here.
                loss[f'loss_cls_unlabeled_{view}'] = loss_unlabeled_weight * loss_unlabeled[view]['loss_cls'] * num_select / bz_unlabeled

        else:
            for view in ['rgb', 'diff']:
                loss[f'loss_cls_unlabeled_{view}'] = torch.zeros(1).to(cls_prob_weak.device)

        with torch.no_grad():
            loss['num_select'] = (torch.ones(1)*num_select).to(cls_prob_weak.device)

        return loss


    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
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