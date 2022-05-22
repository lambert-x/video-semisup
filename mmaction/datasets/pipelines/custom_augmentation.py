import random
import warnings
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair
from .augmentations import Imgaug, RandomRescale, Resize
from ..builder import PIPELINES
from distutils.version import LooseVersion
from .formating import to_tensor
import cv2

@PIPELINES.register_module()
class Imgaug_Custom(Imgaug):
    """Imgaug augmentation.

    Adds custom transformations from imgaug library.
    Please visit `https://imgaug.readthedocs.io/en/latest/index.html`
    to get more information. Two demo configs could be found in tsn and i3d
    config folder.

    It's better to use uint8 images as inputs since imgaug works best with
    numpy dtype uint8 and isn't well tested with other dtypes. It should be
    noted that not all of the augmenters have the same input and output dtype,
    which may cause unexpected results.

    Required keys are "imgs", "img_shape"(if "gt_bboxes" is not None) and
    "modality", added or modified keys are "imgs", "img_shape", "gt_bboxes"
    and "proposals".

    It is worth mentioning that `Imgaug` will NOT create custom keys like
    "interpolation", "crop_bbox", "flip_direction", etc. So when using
    `Imgaug` along with other mmaction2 pipelines, we should pay more attention
    to required keys.

    Two steps to use `Imgaug` pipeline:
    1. Create initialization parameter `transforms`. There are three ways
        to create `transforms`.
        1) string: only support `default` for now.
            e.g. `transforms='default'`
        2) list[dict]: create a list of augmenters by a list of dicts, each
            dict corresponds to one augmenter. Every dict MUST contain a key
            named `type`. `type` should be a string(iaa.Augmenter's name) or
            an iaa.Augmenter subclass.
            e.g. `transforms=[dict(type='Rotate', rotate=(-20, 20))]`
            e.g. `transforms=[dict(type=iaa.Rotate, rotate=(-20, 20))]`
        3) iaa.Augmenter: create an imgaug.Augmenter object.
            e.g. `transforms=iaa.Rotate(rotate=(-20, 20))`
    2. Add `Imgaug` in dataset pipeline. It is recommended to insert imgaug
        pipeline before `Normalize`. A demo pipeline is listed as follows.
        ```
        pipeline = [
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
            ),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Imgaug', transforms='default'),
            # dict(type='Imgaug', transforms=[
            #     dict(type='Rotate', rotate=(-20, 20))
            # ]),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        ```

    Args:
        transforms (str | list[dict] | :obj:`iaa.Augmenter`): Three different
            ways to create imgaug augmenter.
    """

    def __init__(self, transforms, modality):
        super().__init__(transforms)
        self.modality_to_augment = modality

    def __call__(self, results):
        mod = self.modality_to_augment
        in_type = results[mod][0].dtype.type

        cur_aug = self.aug.to_deterministic()

        results[mod] = [
            cur_aug.augment_image(frame) for frame in results[mod]
        ]
        img_h, img_w, _ = results[mod][0].shape

        out_type = results[mod][0].dtype.type
        assert in_type == out_type, \
            ('Imgaug input dtype and output dtype are not the same. ',
             f'Convert from {in_type} to {out_type}')

        if 'gt_bboxes' in results:
            from imgaug.augmentables import bbs
            bbox_list = [
                bbs.BoundingBox(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                for bbox in results['gt_bboxes']
            ]
            bboxes = bbs.BoundingBoxesOnImage(
                bbox_list, shape=results['img_shape'])
            bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
            results['gt_bboxes'] = [[
                max(bbox.x1, 0),
                max(bbox.y1, 0),
                min(bbox.x2, img_w),
                min(bbox.y2, img_h)
            ] for bbox in bbox_aug.items]
            if 'proposals' in results:
                bbox_list = [
                    bbs.BoundingBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                    for bbox in results['proposals']
                ]
                bboxes = bbs.BoundingBoxesOnImage(
                    bbox_list, shape=results['img_shape'])
                bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
                results['proposals'] = [[
                    max(bbox.x1, 0),
                    max(bbox.y1, 0),
                    min(bbox.x2, img_w),
                    min(bbox.y2, img_h)
                ] for bbox in bbox_aug.items]

        results['img_shape'] = (img_h, img_w)

        return results


@PIPELINES.register_module()
class PytorchVideoTrans_Custom:
    """PytorchVideoTrans Augmentations, under pytorchvideo.transforms.

    Args:
        type (str): The name of the pytorchvideo transformation.
    """

    def __init__(self, aug_type, modality, **kwargs):
        try:
            import torch
            import pytorchvideo.transforms as ptv_trans
        except ImportError:
            raise RuntimeError('Install pytorchvideo to use PytorchVideoTrans')
        if LooseVersion(torch.__version__) < LooseVersion('1.8.0'):
            raise RuntimeError(
                'The version of PyTorch should be at least 1.8.0')
        type = aug_type
        trans = getattr(ptv_trans, type, None)
        assert trans, f'Transform {type} not in pytorchvideo'

        supported_pytorchvideo_trans = ('AugMix', 'RandAugment',
                                        'RandomResizedCrop', 'ShortSideScale',
                                        'RandomShortSideScale')
        assert type in supported_pytorchvideo_trans, \
            f'PytorchVideo Transform {type} is not supported in MMAction2'

        self.trans = trans(**kwargs)
        self.type = type
        self.modality = modality

    def __call__(self, results):
        assert self.modality in results

        assert 'gt_bboxes' not in results, \
            f'PytorchVideo {self.type} doesn\'t support bboxes yet.'
        assert 'proposals' not in results, \
            f'PytorchVideo {self.type} doesn\'t support bboxes yet.'

        if self.type in ('AugMix', 'RandAugment'):
            # list[ndarray(h, w, 3)] -> torch.tensor(t, c, h, w)
            imgs = [x.transpose(2, 0, 1) for x in results[self.modality]]
            imgs = to_tensor(np.stack(imgs))
        else:
            # list[ndarray(h, w, 3)] -> torch.tensor(c, t, h, w)
            # uint8 -> float32
            imgs = to_tensor((np.stack(results[self.modality]).transpose(3, 0, 1, 2) /
                              255.).astype(np.float32))

        imgs = self.trans(imgs).data.numpy()

        if self.type in ('AugMix', 'RandAugment'):
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(t, c, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x.transpose(1, 2, 0) for x in imgs]
        else:
            # float32 -> uint8
            imgs = imgs * 255
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(c, t, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x for x in imgs.transpose(1, 2, 3, 0)]

        results[self.modality] = imgs

        return results


@PIPELINES.register_module()
class RGB2GRAY:

    def __init__(self):
        super().__init__()

    def __call__(self, results):
        imgs = results['imgs']
        for i, img in enumerate(imgs):
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_gray = np.stack([img_gray for _ in range(3)], axis=2)
            results['imgs'][i] = img_gray
        return results
