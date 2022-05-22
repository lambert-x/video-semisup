import random

import numpy as np
from .compose import Compose
from ..builder import PIPELINES
import cv2
import inspect
import numpy as np
from PIL import Image, ImageFilter

import torch
from torchvision import transforms

from mmcv.utils import Registry, build_from_cfg


@PIPELINES.register_module()
class SampleImgs:
    def __init__(self, sample_number, mode='random'):
        self.sample_number = sample_number
        self.mode = mode

    def __call__(self, results):
        assert len(results['imgs']) >= self.sample_number
        if self.mode == 'random':
            results['imgs'] = random.sample(results['imgs'],
                                            k=self.sample_number)
            results['clip_len'] = self.sample_number
            return results
        else:
            raise NotImplementedError


# @PIPELINES.register_module()()
# class ColorJitter2D(object):
#     """Randomly change the brightness, contrast and saturation of an image.
#     Args:
#         brightness (float): How much to jitter brightness.
#             brightness_factor is chosen uniformly from
#             [max(0, 1 - brightness), 1 + brightness].
#         contrast (float): How much to jitter contrast.
#             contrast_factor is chosen uniformly from
#             [max(0, 1 - contrast), 1 + contrast].
#         saturation (float): How much to jitter saturation.
#             saturation_factor is chosen uniformly from
#             [max(0, 1 - saturation), 1 + saturation].
#     """
#
#     def __init__(self, brightness, contrast, saturation):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#
#     def __call__(self, results):
#         brightness_factor = random.uniform(0, self.brightness)
#         contrast_factor = random.uniform(0, self.contrast)
#         saturation_factor = random.uniform(0, self.saturation)
#         color_jittertransforms = [
#             dict(
#                 type='Brightness',
#                 magnitude=brightness_factor,
#                 prob=1.,
#                 random_negative_prob=0.5),
#             dict(
#                 type='Contrast',
#                 magnitude=contrast_factor,
#                 prob=1.,
#                 random_negative_prob=0.5),
#             dict(
#                 type='ColorTransform',
#                 magnitude=saturation_factor,
#                 prob=1.,
#                 random_negative_prob=0.5)
#         ]
#         random.shuffle(color_jittertransforms)
#         transform = Compose(color_jittertransforms)
#         return transform(results)
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(brightness={self.brightness}, '
#         repr_str += f'contrast={self.contrast}, '
#         repr_str += f'saturation={self.saturation})'
#         return repr_str


@PIPELINES.register_module()
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    """Randomly applied transformations.
    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class MocoV2_Transforms:
    """Randomly applied transformations.
    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self):
        augmentation = [
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.trans = transforms.Compose(augmentation)

    def __call__(self, results):
        imgs = results['imgs']
        imgs = torch.cat([self.trans(img) for img in imgs], dim=0)
        results['imgs'] = imgs
        return results
