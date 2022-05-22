import io
import os
import os.path as osp
import shutil
import warnings
import copy
import mmcv
import numpy as np
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..builder import PIPELINES
from .loading import RawFrameDecode, SampleFrames
from .augmentations import Normalize
from collections.abc import Sequence


@PIPELINES.register_module()
class RawFrameDecode_WithDiff(RawFrameDecode):

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        super().__init__(io_backend, decoding_backend, **kwargs)

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        # custom imgs_diff
        imgs_diff = list()
        # last_frame = None
        for frame_idx in results['frame_inds_diff']:
            frame_idx += offset
            filepath = osp.join(directory, filename_tmpl.format(frame_idx))
            img_bytes = self.file_client.get(filepath)
            # Get frame with channel order RGB directly.
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            imgs_diff.append(cur_frame)
            # if last_frame is not None:
            #     imgs_diff.append(cur_frame - last_frame)
            # last_frame = cur_frame

        results['imgs_diff'] = imgs_diff

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results


@PIPELINES.register_module()
class SampleFrames_WithDiff(SampleFrames):

    def __init__(self,
                 clip_len,
                 interval_diff=None,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 ):
        super().__init__(clip_len, frame_interval, num_clips, temporal_jitter, twice_sample,
                         out_of_bound_opt, test_mode, start_index)
        self.interval_diff = interval_diff

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        if self.interval_diff is None:
            frame_inds_diff = np.around(np.linspace(frame_inds[0], frame_inds[-1], self.clip_len + 1))
            results['frame_inds_diff'] = frame_inds_diff.astype(np.int)
        else:
            raise NotImplemented
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class SampleFrames_Custom(SampleFrames):

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 total_frames_offset=0,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 sampling_contrastive=None,
                 test_mode=False,
                 start_index=None,
                 ):
        super().__init__(clip_len, frame_interval, num_clips, temporal_jitter, twice_sample,
                         out_of_bound_opt, test_mode, start_index)
        self.total_frames_offset = total_frames_offset
        self.sampling_contrastive = sampling_contrastive

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.sampling_contrastive is None:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        elif self.sampling_contrastive == 'uniform':
            avg_interval = (num_frames - ori_clip_len + 1)
        else:
            raise NotImplementedError

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames'] + self.total_frames_offset

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class RawFrameDecode_Custom(RawFrameDecode):

    def __init__(self, io_backend='disk', decoding_backend='cv2', extra_modalities=[], override_modality=None,
                 results_mapping=dict(), tempgrad_clip_len=None, **kwargs):
        super().__init__(io_backend, decoding_backend, **kwargs)
        self.override_modality = override_modality
        self.extra_modalities = extra_modalities
        self.results_mapping = results_mapping
        self.tempgrad_clip_len = tempgrad_clip_len

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        if self.override_modality is not None:
            modality = self.override_modality

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                break
        results['imgs'] = imgs

        if 'tempgrad' in self.extra_modalities:
            # custom imgs_diff
            imgs_diff = list()
            # last_frame = None
            filename_tmpl_diff = filename_tmpl.replace('img', 'tempgrad')
            # print(results['frame_inds'])

            if self.tempgrad_clip_len is not None:
                tempgrad_clip_len = self.tempgrad_clip_len
            else:
                tempgrad_clip_len = len(results['frame_inds'])
            for frame_idx in results['frame_inds'][:tempgrad_clip_len]:
                frame_idx += offset
                filepath = osp.join(directory, filename_tmpl_diff.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs_diff.append(cur_frame)

            results['imgs_diff'] = imgs_diff


        if 'flow_xym' in self.extra_modalities:
            # custom imgs_diff
            flow_xym = list()
            # last_frame = None
            filename_tmpl_diff = filename_tmpl.replace('img', 'flow_xym')
            # print(results['frame_inds'])


            flow_xym_clip_len = len(results['frame_inds'])
            for frame_idx in results['frame_inds'][:flow_xym_clip_len]:
                frame_idx += offset
                filepath = osp.join(directory, filename_tmpl_diff.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                flow_xym.append(cur_frame)

            results['imgs_flow_xym'] = flow_xym


        if len(self.results_mapping) > 0:
            for src, dst in self.results_mapping.items():
                results[dst] = results[src]

        results['original_shape'] = results['imgs'][0].shape[:2]
        results['img_shape'] = results['imgs'][0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results


# @PIPELINES.register_module()
# class Results_Mapping:
#     """Fuse lazy operations.
#
#     Fusion order:
#         crop -> resize -> flip
#
#     Required keys are "imgs", "img_shape" and "lazy", added or modified keys
#     are "imgs", "lazy".
#     Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
#     """
#     def __init__(self, mapping_dict=0):
#         self.tg_random_shift = tg_random_shift
#
#     def __call__(self, results):
#
#         imgs_diff = results['imgs_diff']
#
#         results['imgs_diff'] = imgs_diff
#         del results[]
#
#         return results

@PIPELINES.register_module()
class Fuse_WithDiff:
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __init__(self, tg_random_shift=0):
        self.tg_random_shift = tg_random_shift

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results['imgs']
        imgs_diff = results['imgs_diff']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]


        left_copy = copy.deepcopy(left)
        right_copy = copy.deepcopy(right)
        top_copy = copy.deepcopy(top)
        bottom_copy = copy.deepcopy(bottom)

        if self.tg_random_shift > 0:
            height = imgs_diff[0].shape[0]
            width = imgs_diff[0].shape[1]
            sigma = self.tg_random_shift
            shift_pixels = np.random.normal(0, sigma, size=4).round().astype(int)
            np.clip(shift_pixels, -2*sigma, 2*sigma, out=shift_pixels)

            center_h = (top + bottom) / 2
            center_w = (left + right) / 2

            h = bottom - top
            w = right - left

            left = (center_w - max(0.5 * w + shift_pixels[0], 2)).round().astype(int)
            right = (center_w + max(0.5 * w + shift_pixels[1], 2)).round().astype(int)

            top = (center_h - max(0.5 * h + shift_pixels[2], 2)).round().astype(int)
            bottom = (center_h + max(0.5 * h + shift_pixels[3], 2)).round().astype(int)


            # left = max(0, left + shift_pixels[0])
            # right = min(height, right + shift_pixels[1])
            # top = max(0, top + shift_pixels[2])
            # bottom = min(width, bottom + shift_pixels[3])
            left = max(0, left)
            # right = min(height, right)
            top = max(0, top)
            # bottom = min(width, bottom)


        imgs_diff = [img[top:bottom, left:right] for img in imgs_diff]
        # print(imgs_diff[0].shape)
        # resize
        if 0 in imgs_diff[0].shape:
            print(imgs_diff[0].shape)
            print(left, right, top, bottom)
            print(left_copy, right_copy, top_copy, bottom_copy)
            print(shift_pixels)
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']

        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]
        imgs_diff = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs_diff
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])
            for img in imgs_diff:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs'] = imgs
        results['imgs_diff'] = imgs_diff
        del results['lazy']

        return results


@PIPELINES.register_module()
class Fuse_OnlyDiff:
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs_diff = results['imgs_diff']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs_diff = [img[top:bottom, left:right] for img in imgs_diff]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']

        imgs_diff = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs_diff
        ]

        # flip
        if lazyop['flip']:
            for img in imgs_diff:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs_diff'] = imgs_diff
        del results['lazy']

        return results


@PIPELINES.register_module()
class Trans_to_RGB:
    def __init__(self, modality='Diff'):
        self.modality = modality

    def __call__(self, results):
        imgs_diff = np.array(results['imgs_diff'])
        if self.modality == 'Diff':
            imgs_diff = np.array(imgs_diff)
            imgs_diff = imgs_diff[1:] - imgs_diff[:-1]
            # print(type(imgs_diff).item())
            # print(imgs_diff)
            imgs_diff += 255.0
            imgs_diff /= 2.0
        else:
            raise NotImplemented
        results['imgs_diff'] = imgs_diff
        return results


@PIPELINES.register_module()
class Normalize_Diff(Normalize):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False, raw_to_diff=True, redist_to_rgb=False):
        super().__init__(mean, std, to_bgr, adjust_magnitude)
        self.raw_to_diff = raw_to_diff
        self.redist_to_rgb = redist_to_rgb

    def __call__(self, results):

        n = len(results['imgs_diff'])
        h, w, c = results['imgs_diff'][0].shape
        imgs = np.empty((n, h, w, c), dtype=np.float32)
        for i, img in enumerate(results['imgs_diff']):
            imgs[i] = img

        if self.redist_to_rgb:
            imgs = imgs[1:] - imgs[:-1]
            imgs += 255.0
            imgs /= 2.0

        for img in imgs:
            mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

        if self.raw_to_diff:
            imgs = imgs[1:] - imgs[:-1]

        # follow multi-view paper , first calucate diff then normalize
        results['imgs_diff'] = imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_bgr=self.to_bgr)
        return results


@PIPELINES.register_module()
class FormatShape_Diff:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, num_clips=None, clip_len=None, collapse=False):
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')
        self.num_clips = num_clips
        self.clip_len = clip_len

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs_diff'], np.ndarray):
            results['imgs_diff'] = np.array(results['imgs_diff'])
        # imgs_diff = results['imgs_diff'][1:] - results['imgs_diff'][:-1]
        imgs_diff = results['imgs_diff']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = self.num_clips if self.num_clips is not None else results['num_clips']
            clip_len = self.clip_len if self.clip_len is not None else results['clip_len']

            imgs_diff = imgs_diff.reshape((-1, num_clips, clip_len) + imgs_diff.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs_diff = np.transpose(imgs_diff, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs_diff = imgs_diff.reshape((-1,) + imgs_diff.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips

        results['imgs_diff'] = imgs_diff
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@PIPELINES.register_module()
class Reset_img_shape:
    def __init__(self):
        pass

    def __call__(self, results):
        results['img_shape'] = results['original_shape']

        return results


@PIPELINES.register_module()
class Normalize_Imgs2Diff(Normalize):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False, redist_to_rgb=True):
        super().__init__(mean, std, to_bgr, adjust_magnitude)
        self.redist_to_rgb = redist_to_rgb

    def __call__(self, results):
        n = len(results['imgs'])
        h, w, c = results['imgs'][0].shape
        imgs = np.empty((n, h, w, c), dtype=np.float32)
        for i, img in enumerate(results['imgs']):
            imgs[i] = img

        if self.redist_to_rgb:
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            if n == clip_len:
                imgs = imgs[1:] - imgs[:-1]
                imgs += 255.0
                imgs /= 2.0
            else:
                assert n > clip_len and n % clip_len == 0
                imgs = imgs[1:] - imgs[:-1]
                select_index = np.arange(n)
                select_index = np.delete(select_index, np.arange(clip_len - 1, n, clip_len))
                imgs = imgs[select_index]
                imgs += 255.0
                imgs /= 2.0

        for img in imgs:
            mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

        # follow multi-view paper , first calucate diff then normalize
        results['imgs'] = imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_bgr=self.to_bgr)
        return results


@PIPELINES.register_module()
class FormatShape_Imgs2Diff:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, collapse=False):
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        # imgs_diff = results['imgs_diff'][1:] - results['imgs_diff'][:-1]
        imgs_diff = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len'] - 1

            imgs_diff = imgs_diff.reshape((-1, num_clips, clip_len) + imgs_diff.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs_diff = np.transpose(imgs_diff, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs_diff = imgs_diff.reshape((-1,) + imgs_diff.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips

        results['imgs'] = imgs_diff
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str



@PIPELINES.register_module()
class DecordDecode_Custom:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets. Default: 'accurate'.
    """

    def __init__(self, mode='accurate', extra_modalities=[]):
        self.mode = mode
        assert mode in ['accurate', 'efficient']
        if extra_modalities != []:
            assert mode == 'accurate'
            self.extra_modalities = extra_modalities
    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']

        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
            if 'tempgrad' in self.extra_modalities:
                # custom imgs_diff
                # last_frame = None
                frame_inds_next = frame_inds + 1
                imgs_next = container.get_batch(frame_inds_next).asnumpy()
                imgs_next = list(imgs_next)
                imgs_diff = [((frame.astype(np.float32) - frame_next.astype(np.float32) + 255.0) / 2.0).astype(np.uint8) for frame, frame_next in zip(imgs, imgs_next)]
                results['imgs_diff'] = imgs_diff

        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str