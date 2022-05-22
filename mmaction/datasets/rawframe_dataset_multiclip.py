import copy
import numpy as np
import torch

from .pipelines import Compose
from .rawframe_dataset import RawframeDataset
from .builder import DATASETS


@DATASETS.register_module()
class RawframeDataset_Multiclip(RawframeDataset):
    """Rawframe dataset for action recognition.
       Unlabeled: return strong/weak augmented pair
       Only valid in training time, not test-time behavior

    Args:
        ann_file (str): Path to the annotation file.
        pipeline_weak (list[dict | callable]): A sequence of data transforms (shared augmentation).
        pipeline_strong (list[dict | callable]): A sequence of data transforms (strong augmentation).
        pipeline_format (list[dict | callable]): A sequence of data transforms (post-processing, for formating).
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int): Number of classes in the dataset. Default: None.
        modality (str): Modality of data. Support 'RGB' only.
        det_file (str): Path to the human box detection result file.
        cls_file (str): Path to the ImageNet classification result file.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 contrast_clip_num,
                 collect_keys,
                 pipeline_appearance=None,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB'):
        assert modality == 'RGB'
        super().__init__(ann_file, pipeline, data_prefix, test_mode, filename_tmpl, with_offset,
                         multi_class, num_classes, start_index, modality)
        if pipeline_appearance is not None:
            self.pipeline_appearance = Compose(pipeline_appearance)
        else:
            self.pipeline_appearance = None

        self.contrast_clip_num = contrast_clip_num
        self.collect_keys = collect_keys


    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        # output = dict()
        # Step1: Foward with the base labeled pipeline


        results_contrast_all = []
        for i in range(self.contrast_clip_num):
            results_single_clip = self.pipeline(copy.deepcopy(results))
            results_contrast_all.append(results_single_clip)


        # if self.pipeline_appearance is not None:
        #     results_appearance = self.pipeline_appearance(copy.deepcopy(results_main))
        #     output = self.pipeline_format(results_main)
        #     output['imgs_appearance'] = results_appearance['imgs']
        # else:
        #     output = self.pipeline_format(results_main)
        output = dict()
        for key in self.collect_keys:
            output[key] = torch.cat([result[key] for result in results_contrast_all], dim=0)

        return output

    def prepare_test_frames(self, idx):
        raise NotImplementedError

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        raise NotImplementedError
