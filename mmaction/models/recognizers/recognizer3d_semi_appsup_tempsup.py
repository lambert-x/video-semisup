import torch
import torch.nn.functional as F
from ..builder import RECOGNIZERS
# Custom imports
from .base_semi_apppearance_temporal import SemiAppTempBaseRecognizer
from ...utils import GatherLayer


@RECOGNIZERS.register_module()
class Semi_AppSup_TempSup_Recognizer3D(SemiAppTempBaseRecognizer):
    """Semi-supervised 3D recognizer model framework."""

    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled, imgs_appearance,
                      imgs_diff_labeled, imgs_diff_unlabeled, cur_epoch=None):
        """Defines the computation performed at every call when training."""
        bz_labeled = imgs.shape[0]
        bz_unlabeled = imgs_weak.shape[0]
        imgs_all = torch.cat([imgs, imgs_weak, imgs_strong], dim=0)
        imgs_diff_all = torch.cat([imgs_diff_labeled, imgs_diff_unlabeled], dim=0)

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

        cls_score = self.cls_head(vid_feature)

        clip_len = imgs_all.size(2)
        batch_total_len = bz_labeled + bz_unlabeled





        if imgs_appearance is not None:
            # contrastive heads
            cls_score_appearence = self.cls_head_app(vid_feature)
            vid_appearance = cls_score_appearence[:bz_labeled + bz_unlabeled, :]
            img_feature = self.app_backbone(imgs_appearance)
            img_cls_score = self.app_sup_head(img_feature)
            img_appearance = self.app_contrast_head(img_feature)
            # print('img_appearance1:', img_appearance.shape)
            if self.app_frame_num > 1:
                img_appearance = img_appearance.view(batch_total_len, self.app_frame_num, -1).mean(dim=1)
            # print('img_appearance2:', img_appearance.shape)
        # NOTE: pre-softmx logit
        cls_score_labeled = cls_score[:bz_labeled, :]
        cls_score_weak = cls_score[bz_labeled:bz_labeled+bz_unlabeled, :]
        cls_score_strong = cls_score[bz_labeled+bz_unlabeled:, :]

        loss = dict()

        if self.use_temp_contrast:
            cls_score_temp = self.cls_head_temp(vid_feature)
            vid_temporal = cls_score_temp[:bz_labeled + bz_unlabeled, :]
            imgs_diff_temporal = self.temp_contrast_head(imgs_diff_feature)

            embedding_temp = torch.cat([vid_temporal, imgs_diff_temporal], dim=0)
            embedding_temp = embedding_temp / (torch.norm(embedding_temp, p=2, dim=1, keepdim=True) + 1e-10)

            embedding_temp = torch.cat(GatherLayer.apply(embedding_temp), dim=0)  # (2N)xd
            loss['loss_temporal_contrast'] = self.cls_head_temp.loss(embedding_temp)

        if self.temp_align_indices is not None:
            for i, layer_idx in enumerate(self.temp_align_indices):
                loss[f'loss_layer{layer_idx}_align'] = self.align_criterion(vid_features[i][:batch_total_len, ...],
                                                                            imgs_diff_features[i].detach())

        if imgs_appearance is not None:
            embedding_app = torch.cat([vid_appearance, img_appearance], dim=0)
            embedding_app = embedding_app / (torch.norm(embedding_app, p=2, dim=1, keepdim=True) + 1e-10)

            embedding_app = torch.cat(GatherLayer.apply(embedding_app), dim=0)  # (2N)xd
            loss['loss_appearance_contrast'] = self.cls_head_app.loss(embedding_app)



        gt_labels = labels.squeeze()
        # img appearance supervised loss
        if imgs_appearance is not None:
            img_cls_score_labeled = img_cls_score[:bz_labeled * self.app_frame_num, :]
            loss_appearance_sup = self.app_sup_head.loss(img_cls_score_labeled, gt_labels.repeat_interleave(self.app_frame_num))
            loss['loss_appearance_sup'] = loss_appearance_sup['loss_cls']

        # inter-frame difference modality (temporal) supervised loss
        imgs_diff_cls_score = self.temp_sup_head(imgs_diff_feature)
        imgs_diff_cls_score_labeled = imgs_diff_cls_score[:bz_labeled, :]
        imgs_diff_cls_score_weak = imgs_diff_cls_score[bz_labeled:bz_labeled+bz_unlabeled, :]
        loss_temporal_sup = self.temp_sup_head.loss(imgs_diff_cls_score_labeled, gt_labels)
        loss['loss_temporal_sup'] = loss_temporal_sup['loss_cls']

        # video supervised loss
        loss_labeled = self.cls_head.loss(cls_score_labeled, gt_labels)
        for k in loss_labeled:
            loss[k+'_labeled'] = loss_labeled[k]

        ## Unlabeled data
        with torch.no_grad():
            if self.pseudo_label_metric == 'rgb':
                cls_prob_weak = F.softmax(cls_score_weak, dim=-1)
            elif self.pseudo_label_metric == 'tg':
                cls_prob_weak = F.softmax(imgs_diff_cls_score_weak, dim=-1)
            elif self.pseudo_label_metric == 'avg':
                cls_prob_weak = F.softmax(0.5 * (cls_score_weak + imgs_diff_cls_score_weak), dim=-1)
            else:
                raise NotImplementedError
        # cls_prob_weak = F.softmax(cls_score_weak.detach(), dim=-1)
        # TODO: Control this threshold with cfg
        if self.framework == 'fixmatch':
            thres = 0.95
        else:    # UDA
            thres = 0.8
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
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        if self.test_modality == 'diff':
            imgs = imgs[:, :, 1:, ...] - imgs[:, :, :-1, ...]
            # print(imgs.shape)
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