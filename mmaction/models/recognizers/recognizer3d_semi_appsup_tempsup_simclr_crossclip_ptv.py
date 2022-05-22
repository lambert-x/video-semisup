import torch
import torch.nn.functional as F
import torch.nn as nn
# Custom imports
from .base_semi_apppearance_temporal_simclr import SemiAppTemp_SimCLR_BaseRecognizer
from ..builder import RECOGNIZERS
from ..losses import CosineSimiLoss

from ...utils import GatherLayer


@RECOGNIZERS.register_module()
class Semi_AppSup_TempSup_SimCLR_Crossclip_PTV_Recognizer3D(SemiAppTemp_SimCLR_BaseRecognizer):
    """Semi-supervised 3D recognizer model framework."""

    def forward_train(self, imgs, labels, imgs_weak, imgs_strong, labels_unlabeled, imgs_appearance,
                      imgs_diff_labeled, imgs_diff_weak, imgs_diff_strong, cur_epoch=None):

        """Defines the computation performed at every call when training."""
        clips_per_video = imgs.shape[1]
        bz_labeled = imgs.shape[0] * imgs.shape[1]
        bz_unlabeled = imgs_weak.shape[0] * imgs_weak.shape[1]

        img_shape_template = imgs.shape[2:]
        imgs = imgs.transpose(0, 1).reshape((-1,) + img_shape_template)
        imgs_weak = imgs_weak.transpose(0, 1).reshape((-1,) + img_shape_template)
        imgs_strong = imgs_strong.transpose(0, 1).reshape((-1,) + img_shape_template)

        imgs_diff_shape_template = imgs_diff_labeled.shape[2:]
        imgs_diff_labeled = imgs_diff_labeled.transpose(0, 1).reshape((-1,) + imgs_diff_shape_template)
        imgs_diff_weak = imgs_diff_weak.transpose(0, 1).reshape((-1,) + imgs_diff_shape_template)
        imgs_diff_strong = imgs_diff_strong.transpose(0, 1).reshape((-1,) + imgs_diff_shape_template)

        imgs_all = torch.cat([imgs, imgs_weak, imgs_strong], dim=0)
        imgs_diff_all = torch.cat([imgs_diff_labeled, imgs_diff_weak, imgs_diff_strong], dim=0)

        # TODO: If we forward imgs_weak with no_grad, and then jointly forward imgs and imgs_strong,
        # we might be able to save memory for a larger batch size? But not sure if this has
        # negative impact on batch-norm.
        # imgs_all = imgs_all.reshape((-1,) + imgs_all.shape[2:])
        # imgs_diff_all = imgs_diff_all.reshape((-1,) + imgs_diff_all.shape[2:])
        labels = labels.transpose(0, 1).reshape((-1, 1))
        # print(labels)
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

        batch_total_len = bz_labeled + bz_unlabeled

        # NOTE: pre-softmx logit
        cls_score_labeled = dict()
        cls_score_weak = dict()
        cls_score_strong = dict()
        for view in ['rgb', 'diff']:
            cls_score_labeled[view] = cls_score[view][:bz_labeled, :]
            if self.loss_clip_selection == 0:
                cls_score_weak[view] = cls_score[view][bz_labeled:(bz_labeled + bz_unlabeled // 2), :]
                cls_score_strong[view] = cls_score[view][
                                         (bz_labeled + bz_unlabeled):(bz_labeled + bz_unlabeled + bz_unlabeled // 2), :]
            else:
                cls_score_weak[view] = cls_score[view][bz_labeled:bz_labeled + bz_unlabeled, :]
                cls_score_strong[view] = cls_score[view][bz_labeled + bz_unlabeled:bz_labeled + 2 * bz_unlabeled, :]

        loss = dict()

        if 'weak' in self.crossclip_contrast_range and 'strong' in self.crossclip_contrast_range:
            query_index = torch.cat([torch.arange(bz_labeled // clips_per_video), torch.arange(bz_unlabeled // clips_per_video) + bz_labeled,
                                     torch.arange(bz_unlabeled // clips_per_video) + bz_labeled + bz_unlabeled])
            key_mask = torch.ones(bz_labeled + 2 * bz_unlabeled)
            key_mask[query_index] = 0
            key_index = torch.arange(bz_labeled + 2 * bz_unlabeled)[key_mask.bool()]
        elif 'weak' in self.crossclip_contrast_range:
            query_index = torch.cat([torch.arange(bz_labeled // clips_per_video), torch.arange(bz_unlabeled // clips_per_video) + bz_labeled])
            key_mask = torch.ones(bz_labeled + bz_unlabeled)
            key_mask[query_index] = 0
            key_index = torch.arange(bz_labeled + bz_unlabeled)[key_mask.bool()]
        elif 'strong' in self.crossclip_contrast_range:
            query_index = torch.arange(bz_unlabeled // clips_per_video) + bz_labeled + bz_unlabeled
            key_index = query_index + (bz_unlabeled // clips_per_video)
        else:
            pass

        # print(query_index, key_index)

        if 'rgb' in self.crossclip_contrast_loss:
            contrast_rgb_feature = self.contrast_head_rgb(vid_feature)
            rgb_query = contrast_rgb_feature[query_index]
            rgb_key = contrast_rgb_feature[key_index]
            loss['loss_rgb_contrast'] = self.contrast_loss_weight * \
                                        self.contrast_head_rgb.loss([rgb_query, rgb_key])

        if 'tg' in self.crossclip_contrast_loss:
            contrast_tg_feature = self.contrast_head_tg(imgs_diff_feature)
            tg_query = contrast_tg_feature[query_index]
            tg_key = contrast_tg_feature[key_index]
            loss['loss_tg_contrast'] = self.contrast_loss_weight * \
                                       self.contrast_head_tg.loss([tg_query, tg_key])

        if 'crossview' in self.crossclip_contrast_loss:
            contrast_rgb_feature = self.contrast_head_rgb(vid_feature)
            rgb_query = contrast_rgb_feature[query_index]
            rgb_key = contrast_rgb_feature[key_index]

            contrast_tg_feature = self.contrast_head_tg(imgs_diff_feature)
            tg_query = contrast_tg_feature[query_index]
            tg_key = contrast_tg_feature[key_index]

            embedding_rgb1_tg2 = [rgb_query, tg_key]

            embedding_rgb2_tg1 = [rgb_key, tg_query]

            loss['loss_crossview_contrast'] = self.contrast_loss_weight * (
                    self.contrast_head_rgb.loss(embedding_rgb1_tg2) +
                    self.contrast_head_tg.loss(embedding_rgb2_tg1)) / 2

        if 'crossview_sharedhead' in self.crossclip_contrast_loss:
            contrast_rgb_feature = self.contrast_head_shared(vid_feature)
            rgb_query = contrast_rgb_feature[query_index]
            rgb_key = contrast_rgb_feature[key_index]

            contrast_tg_feature = self.contrast_head_shared(imgs_diff_feature)
            tg_query = contrast_tg_feature[query_index]
            tg_key = contrast_tg_feature[key_index]

            embedding_rgb1_tg2 = [rgb_query, tg_key]

            embedding_rgb2_tg1 = [rgb_key, tg_query]

            loss['loss_crossview_contrast'] = self.contrast_loss_weight * (
                    self.contrast_head_shared.loss(embedding_rgb1_tg2) +
                    self.contrast_head_shared.loss(embedding_rgb2_tg1)) / 2

        if 'crossview_sameclip' in self.crossclip_contrast_loss:
            if clips_per_video > 1:
                contrast_rgb_feature = self.contrast_head_rgb(vid_feature)
                rgb_query = contrast_rgb_feature[query_index]
                rgb_key = contrast_rgb_feature[key_index]

                contrast_tg_feature = self.contrast_head_tg(imgs_diff_feature)
                tg_query = contrast_tg_feature[query_index]
                tg_key = contrast_tg_feature[key_index]

                embedding_rgb1_tg1 = [rgb_query, tg_query]

                embedding_rgb2_tg2 = [rgb_key, tg_key]

                loss['loss_crossview_contrast'] = self.contrast_loss_weight * (
                        self.contrast_head_rgb.loss(embedding_rgb1_tg1) +
                        self.contrast_head_tg.loss(embedding_rgb2_tg2)) / 2
            else:
                contrast_rgb_feature = self.contrast_head_rgb(vid_feature)
                rgb_query = contrast_rgb_feature[query_index]
                contrast_tg_feature = self.contrast_head_tg(imgs_diff_feature)
                tg_query = contrast_tg_feature[query_index]
                embedding_rgb1_tg1 = [rgb_query, tg_query]
                loss['loss_crossview_contrast'] = self.contrast_loss_weight * \
                                                  self.contrast_head_rgb.loss(embedding_rgb1_tg1)

        if 'crossview_sameclip_sharedhead' in self.crossclip_contrast_loss:
            if clips_per_video > 1:
                contrast_rgb_feature = self.contrast_head_shared(vid_feature)
                rgb_query = contrast_rgb_feature[query_index]
                rgb_key = contrast_rgb_feature[key_index]

                contrast_tg_feature = self.contrast_head_shared(imgs_diff_feature)
                tg_query = contrast_tg_feature[query_index]
                tg_key = contrast_tg_feature[key_index]

                embedding_rgb1_tg1 = [rgb_query, tg_query]

                embedding_rgb2_tg2 = [rgb_key, tg_key]

                loss['loss_crossview_contrast'] = self.contrast_loss_weight * (
                        self.contrast_head_shared.loss(embedding_rgb1_tg1) +
                        self.contrast_head_shared.loss(embedding_rgb2_tg2)) / 2
            else:
                contrast_rgb_feature = self.contrast_head_shared(vid_feature)
                rgb_query = contrast_rgb_feature[query_index]
                contrast_tg_feature = self.contrast_head_shared(imgs_diff_feature)
                tg_query = contrast_tg_feature[query_index]
                embedding_rgb1_tg1 = [rgb_query, tg_query]
                loss['loss_crossview_contrast'] = self.contrast_loss_weight * \
                                                  self.contrast_head_shared.loss(embedding_rgb1_tg1)

        if 'crossview_crossclip_densecl' in self.crossclip_contrast_loss:

            contrast_rgb_feature = self.contrast_densecl_head(vid_feature)

            rgb_query_backbone = vid_feature[query_index]
            rgb_key_backbone = vid_feature[key_index]
            rgb_query = contrast_rgb_feature[query_index]
            rgb_key = contrast_rgb_feature[key_index]

            contrast_tg_feature = self.contrast_densecl_head(imgs_diff_feature)

            tg_query_backbone = imgs_diff_feature[query_index]
            tg_key_backbone = imgs_diff_feature[key_index]
            tg_query = contrast_tg_feature[query_index]
            tg_key = contrast_tg_feature[key_index]

            n = rgb_query.size(0)
            c = vid_feature.size(1)

            loss['loss_crossview_dense_contrast'] = list()
            for rgb_grid, tg_grid, rgb_grid_b, tg_grid_b in \
                    [(rgb_query, tg_key, rgb_query_backbone, tg_key_backbone),
                     (rgb_key, tg_query, rgb_key_backbone, tg_query_backbone)]:
                rgb_grid = rgb_grid.view(n, c, -1)
                rgb_grid = nn.functional.normalize(rgb_grid, dim=1)

                tg_grid = tg_grid.view(n, c, -1)
                tg_grid = nn.functional.normalize(tg_grid, dim=1)

                rgb_grid_b = rgb_grid_b.view(rgb_grid_b.size(0), rgb_grid_b.size(1), -1)
                tg_grid_b = tg_grid_b.view(tg_grid_b.size(0), tg_grid_b.size(1), -1)

                rgb_grid_b = nn.functional.normalize(rgb_grid_b, dim=1)
                tg_grid_b = nn.functional.normalize(tg_grid_b, dim=1)
                # print(rgb_grid_b.shape)
                # print(tg_grid_b.shape)
                backbone_sim_matrix = torch.matmul(rgb_grid_b.permute(0, 2, 1), tg_grid_b)
                densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2
                indexed_tg_grid = torch.gather(tg_grid, 2,
                                               densecl_sim_ind.unsqueeze(1).expand(-1, rgb_grid.size(1), -1))

                rgb_embedding = rgb_grid.view(rgb_grid.size(0), rgb_grid.size(1), -1).permute(0, 2, 1).reshape(-1, c)
                tg_embedding = indexed_tg_grid.view(indexed_tg_grid.size(0), indexed_tg_grid.size(1), -1).permute(0, 2, 1).reshape(-1, c)

                contrastive_embedding = [rgb_embedding, tg_embedding]

                loss['loss_crossview_dense_contrast'].append(self.contrast_densecl_head.loss(contrastive_embedding))

            loss['loss_crossview_dense_contrast'] = torch.mean(torch.stack(loss['loss_crossview_dense_contrast']))


        if 'sameview_crossclip_densecl' in self.crossclip_contrast_loss:

            contrast_rgb_feature = self.contrast_densecl_head(vid_feature)

            rgb_query_backbone = vid_feature[query_index]
            rgb_key_backbone = vid_feature[key_index]
            rgb_query = contrast_rgb_feature[query_index]
            rgb_key = contrast_rgb_feature[key_index]

            contrast_tg_feature = self.contrast_densecl_head(imgs_diff_feature)

            tg_query_backbone = imgs_diff_feature[query_index]
            tg_key_backbone = imgs_diff_feature[key_index]
            tg_query = contrast_tg_feature[query_index]
            tg_key = contrast_tg_feature[key_index]

            n = rgb_query.size(0)
            c = vid_feature.size(1)

            loss['loss_crossview_dense_contrast'] = list()
            for q_grid, k_grid, q_grid_b, k_grid_b in \
                    [(rgb_query, rgb_key, rgb_query_backbone, rgb_key_backbone),
                     (tg_query, tg_key, tg_query_backbone, tg_key_backbone)]:
                q_grid = q_grid.view(n, c, -1)
                q_grid = nn.functional.normalize(q_grid, dim=1)

                k_grid = k_grid.view(n, c, -1)
                k_grid = nn.functional.normalize(k_grid, dim=1)

                q_grid_b = q_grid_b.view(q_grid_b.size(0), q_grid_b.size(1), -1)
                k_grid_b = k_grid_b.view(k_grid_b.size(0), k_grid_b.size(1), -1)

                q_grid_b = nn.functional.normalize(q_grid_b, dim=1)
                k_grid_b = nn.functional.normalize(k_grid_b, dim=1)
                # print(rgb_grid_b.shape)
                # print(tg_grid_b.shape)
                backbone_sim_matrix = torch.matmul(q_grid_b.permute(0, 2, 1), k_grid_b)
                densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2
                indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1))

                q_embedding = q_grid.view(q_grid.size(0), q_grid.size(1), -1).permute(0, 2, 1).reshape(-1, c)
                k_embedding = indexed_k_grid.view(indexed_k_grid.size(0), indexed_k_grid.size(1), -1).permute(0, 2, 1).reshape(-1, c)

                contrastive_embedding = [q_embedding, k_embedding]

                loss['loss_crossview_dense_contrast'].append(self.contrast_densecl_head.loss(contrastive_embedding))

            loss['loss_crossview_dense_contrast'] = torch.mean(torch.stack(loss['loss_crossview_dense_contrast']))


        if self.loss_lambda is not None and 'loss_crossview_dense_contrast' in loss.keys() and 'loss_crossview_contrast' in loss.keys():
            loss['loss_crossview_dense_contrast'] = self.loss_lambda * loss['loss_crossview_dense_contrast']
            loss['loss_crossview_contrast'] = (1 - self.loss_lambda) * loss['loss_crossview_contrast']

        if cur_epoch < self.contrast_warmup_epoch:
            for contrast_loss_type in ['loss_rgb_contrast', 'loss_tg_contrast', 'loss_crossview_contrast',
                                       'loss_crossview_dense_contrast']:
                if contrast_loss_type in loss.keys():
                    loss[contrast_loss_type] = loss[contrast_loss_type].detach()

        # if self.use_temp_contrast:
        #     cls_score_temp = self.cls_head_temp(vid_feature)
        #     vid_temporal = cls_score_temp[:batch_total_len, :]
        #     imgs_diff_temporal = self.temp_contrast_head(imgs_diff_feature[:batch_total_len, :])
        #
        #     vid_temporal_pred = self.simsiam_temp_pred(vid_temporal)
        #     imgs_diff_temporal_pred = self.simsiam_temp_pred(imgs_diff_temporal)
        #     embedding_temp = [vid_temporal_pred, imgs_diff_temporal_pred, vid_temporal.detach(), imgs_diff_temporal.detach()]
        #     loss['loss_temporal_contrast'] = self.simsiam_temp_pred.loss(embedding_temp)

        if self.temp_align_indices is not None:

            for i, layer_idx in enumerate(self.temp_align_indices):
                if self.align_stop_grad == 'tg':
                    #     if self.align_with_correspondence:
                    #         c = vid_features[layer_idx].size(1)
                    #         q_grid = vid_features[layer_idx][:batch_total_len, ...].view(batch_total_len, c, -1)
                    #         k_grid = imgs_diff_features[layer_idx][:batch_total_len, ...].detach().view(batch_total_len, c,
                    #                                                                                     -1)
                    #         if self.align_with_correspondence == 'normalized_cosine':
                    #             q_grid = nn.functional.normalize(q_grid, dim=1)
                    #             k_grid = nn.functional.normalize(k_grid, dim=1)
                    #         backbone_sim_matrix = torch.matmul(q_grid.permute(0, 2, 1), k_grid)
                    #         densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2
                    #         indexed_k_grid = torch.gather(k_grid, 2,
                    #                                       densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1))
                    #         loss[f'loss_layer{layer_idx}_align'] = self.align_criterion(
                    #             q_grid, indexed_k_grid
                    #         )
                    if self.loss_clip_selection == 0:
                        align_selection_index = torch.cat(
                            [torch.arange(bz_labeled // 2), torch.arange(bz_unlabeled // 2) + bz_labeled])
                        loss[f'loss_layer{layer_idx}_align'] = self.align_criterion(
                            vid_features[layer_idx][align_selection_index],
                            imgs_diff_features[layer_idx][align_selection_index].detach())
                    else:
                        loss[f'loss_layer{layer_idx}_align'] = self.align_criterion(
                            vid_features[layer_idx][:batch_total_len, ...],
                            imgs_diff_features[layer_idx][:batch_total_len, ...].detach())
                elif self.align_stop_grad == 'rgb':
                    loss[f'loss_layer{layer_idx}_align'] = self.align_criterion(
                        vid_features[layer_idx][:batch_total_len, ...].detach(),
                        imgs_diff_features[layer_idx][:batch_total_len, ...])
                elif self.align_stop_grad is None:
                    loss[f'loss_layer{layer_idx}_align'] = self.align_criterion(
                        vid_features[layer_idx][:batch_total_len, ...],
                        imgs_diff_features[layer_idx][:batch_total_len, ...])

        if self.densecl_indices is not None:
            if self.densecl_indices == (3,):
                layer_idx = self.densecl_indices[0]

                # b = vid_feature.size(0)
                c = vid_feature.size(1)

                q_grid = vid_feature[:batch_total_len, ...].view(batch_total_len, c, -1)
                q_grid = nn.functional.normalize(q_grid, dim=1)

                k_grid = imgs_diff_feature[:batch_total_len, ...].view(batch_total_len, c, -1)
                k_grid = nn.functional.normalize(k_grid, dim=1)

                q_grid_pred = self.simsiam_densecl_pred(vid_feature)
                k_grid_pred = self.simsiam_densecl_pred(imgs_diff_feature.detach())

                q_grid_pred = q_grid_pred[:batch_total_len, ...].view(batch_total_len, c, -1)
                q_grid_pred = nn.functional.normalize(q_grid_pred, dim=1)

                k_grid_pred = k_grid_pred[:batch_total_len, ...].view(batch_total_len, c, -1)
                k_grid_pred = nn.functional.normalize(k_grid_pred, dim=1)

                # backbone_sim_matrix = torch.matmul(q_grid.permute(0, 2, 1), k_grid)
                #
                # # densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2
                # #
                # # indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1))
                self.cosine_align_criterion = CosineSimiLoss(dim=1)
                if self.densecl_strategy == 'shared_pred_stop_tg':
                    loss[f'loss_layer{layer_idx}_dense'] = self.cosine_align_criterion(q_grid_pred, k_grid_pred)

                elif self.densecl_strategy == 'rgb_pred_stop_tg':
                    loss[f'loss_layer{layer_idx}_dense'] = self.cosine_align_criterion(q_grid_pred, k_grid)

                elif self.densecl_strategy == 'tg_pred_stop_tg':
                    loss[f'loss_layer{layer_idx}_dense'] = self.cosine_align_criterion(q_grid, k_grid_pred)

                elif self.densecl_strategy == 'simsiam':
                    embedding_rgb = [q_grid_pred, k_grid_pred, q_grid.detach(), k_grid.detach()]
                    loss[f'loss_layer{layer_idx}_dense_simsiam'] = self.simsiam_densecl_pred.loss(embedding_rgb)

                with torch.no_grad():
                    loss[f'layer{layer_idx}_align_L2'] = torch.nn.MSELoss()(
                        vid_feature[:batch_total_len, ...],
                        imgs_diff_feature[:batch_total_len, ...]
                    )
            else:
                raise NotImplementedError

        gt_labels = labels.squeeze()

        # if torch.distributed.get_rank() == 0:
        #     print(gt_labels)
        # inter-frame difference modality (temporal) supervised loss

        # video supervised loss
        # loss_labeled_weight = 0.5
        loss_labeled_weight = self.cls_loss_weight['labeled']
        loss_labeled = {}
        # loss_clip1 = {}
        # loss_clip2 = {}
        for view in ['rgb', 'diff']:
            if self.loss_clip_selection is not None:
                if self.loss_clip_selection == 0:
                    loss_labeled[view] = self.cls_head.loss(cls_score_labeled[view][:bz_labeled // 2],
                                                            gt_labels[:bz_labeled // 2])
            else:
                loss_labeled[view] = self.cls_head.loss(cls_score_labeled[view], gt_labels)
            # loss_clip1[view] = self.cls_head.loss(cls_score_labeled[view][bz_labeled//2:bz_labeled],
            #                                         gt_labels[bz_labeled//2:bz_labeled])
            # loss_clip2[view] = self.cls_head.loss(cls_score_labeled[view][:bz_labeled//2],
            #                                         gt_labels[:bz_labeled//2])
            # loss_labeled[view] = {}
            # for k in loss_clip1[view].keys():
            #     loss_labeled[view][k] = (loss_clip1[view][k] + loss_clip2[view][k]) / 2
            #
            #
            for k in loss_labeled[view]:
                if 'loss' in k:
                    loss[k + f'_labeled_{view}'] = loss_labeled[view][k] * loss_labeled_weight
                else:
                    loss[k + f'_labeled_{view}'] = loss_labeled[view][k]

        ## Unlabeled data
        with torch.no_grad():
            if self.pseudo_label_metric == 'rgb':
                cls_prob_weak = cls_score_weak['rgb']
            elif self.pseudo_label_metric == 'tg':
                cls_prob_weak = cls_score_weak['diff']
            elif self.pseudo_label_metric == 'avg':
                cls_prob_weak = (cls_score_weak['rgb'] + cls_score_weak['diff']) / 2
            elif self.pseudo_label_metric == 'individual':
                cls_prob_weak = dict()
                for view in ['rgb', 'diff']:
                    cls_prob_weak[view] = F.softmax(cls_score_weak[view], dim=-1)
            else:
                raise NotImplementedError
            if self.pseudo_label_metric != 'individual':
                cls_prob_weak = F.softmax(cls_prob_weak, dim=-1)
        # if self.loss_clip_selection == 0:
        #     cls_prob_weak = cls_prob_weak[:bz_unlabeled // 2]
        #     for view in ['rgb', 'diff']:
        #         cls_score_strong[view] = cls_score_strong[view][:bz_unlabeled // 2]

        # if torch.distributed.get_rank() == 0:
        #     print(cls_score_strong[view].shape)

        # if self.loss_clip_selection == 0:
        #     bz_unlabeled /= 2
        # cls_prob_weak = F.softmax(cls_score_weak.detach(), dim=-1)
        # TODO: Control this threshold with cfg
        # if self.framework == 'fixmatch':
        #     thres = 0.95
        # else:    # UDA
        #     thres = 0.8

        thres = self.fixmatch_threshold

        if self.pseudo_label_metric != 'individual':

            select = (torch.max(cls_prob_weak, 1).values >= thres).nonzero(as_tuple=False).squeeze(1)
            num_select = select.shape[0]

            if num_select > 0 and cur_epoch >= self.warmup_epoch:
                all_pseudo_labels = cls_prob_weak.argmax(1).detach()
                pseudo_labels = torch.index_select(all_pseudo_labels.view(-1, 1), dim=0, index=select).squeeze(1)

                cls_score_unlabeled = dict()
                loss_unlabeled = dict()
                loss_unlabeled_weight = self.cls_loss_weight['unlabeled']
                # loss_unlabeled_weight = 0.5
                for view in ['rgb', 'diff']:
                    cls_score_unlabeled[view] = torch.index_select(cls_score_strong[view], dim=0, index=select)
                    loss_unlabeled[view] = self.cls_head.loss(cls_score_unlabeled[view], pseudo_labels)
                    # NOTE: When do loss reduce_mean, we should always divide by the batch size
                    # instead of number of confident samples. So we compensate here.
                    loss[f'loss_cls_unlabeled_{view}'] = loss_unlabeled_weight * loss_unlabeled[view][
                        'loss_cls'] * num_select / len(cls_prob_weak)
                    # if torch.distributed.get_rank() == 0:
                    #     print(loss_unlabeled_weight, num_select, len(cls_prob_weak))

            else:
                for view in ['rgb', 'diff']:
                    loss[f'loss_cls_unlabeled_{view}'] = torch.zeros(1).to(cls_prob_weak.device)

        elif self.pseudo_label_metric == 'individual':
            num_select = dict()
            for view in ['rgb', 'diff']:
                select = (torch.max(cls_prob_weak[view], 1).values >= thres).nonzero(as_tuple=False).squeeze(1)
                num_select[view] = select.shape[0]

                if num_select[view] > 0 and cur_epoch >= self.warmup_epoch:
                    all_pseudo_labels = cls_prob_weak[view].argmax(1).detach()
                    pseudo_labels = torch.index_select(all_pseudo_labels.view(-1, 1), dim=0, index=select).squeeze(1)

                    cls_score_unlabeled = dict()
                    loss_unlabeled = dict()
                    loss_unlabeled_weight = self.cls_loss_weight['unlabeled']
                    # loss_unlabeled_weight = 0.5

                    cls_score_unlabeled[view] = torch.index_select(cls_score_strong[view], dim=0, index=select)
                    loss_unlabeled[view] = self.cls_head.loss(cls_score_unlabeled[view], pseudo_labels)
                    # NOTE: When do loss reduce_mean, we should always divide by the batch size
                    # instead of number of confident samples. So we compensate here.
                    loss[f'loss_cls_unlabeled_{view}'] = loss_unlabeled_weight * loss_unlabeled[view][
                        'loss_cls'] * num_select[view] / len(cls_prob_weak[view])
                    # if torch.distributed.get_rank() == 0:
                    #     print(loss_unlabeled_weight, num_select, len(cls_prob_weak))

                else:
                    loss[f'loss_cls_unlabeled_{view}'] = torch.zeros(1).to(cls_prob_weak[view].device)

        with torch.no_grad():
            if self.pseudo_label_metric != 'individual':
                loss['num_select'] = (torch.ones(1) * num_select).to(cls_prob_weak.device)
            elif self.pseudo_label_metric == 'individual':
                num_select_avg = (num_select['rgb'] + num_select['diff']) / 2
                loss['num_select'] = (torch.ones(1) * num_select_avg).to(cls_prob_weak[view].device)
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
            if self.test_return_features:
                x = nn.AdaptiveAvgPool3d((1, 1, 1))(x)
                x = x.view(x.shape[0], -1)
                batch_size = x.shape[0]
                features = x.view(batch_size // num_segs, num_segs, -1)
                features = features.mean(dim=1)
                return features
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
