import copy
import numpy as np
import torch

from .pipelines import Compose
from .rawframe_dataset import RawframeDataset
from .builder import DATASETS


@DATASETS.register_module()
class UnlabeledRawframeDataset_MultiView_Contrastive(RawframeDataset):
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
                 pipeline_weak,
                 pipeline_strong,
                 pipeline_format,
                 contrast_clip_num,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB', 
                 ):
        assert modality == 'RGB'
        super().__init__(ann_file, pipeline_weak, data_prefix, test_mode, filename_tmpl, with_offset,
                         multi_class, num_classes, start_index, modality)

        self.pipeline_weak = self.pipeline
        self.pipeline_strong = Compose(pipeline_strong)
        self.pipeline_format = Compose(pipeline_format)
        self.contrast_clip_num = contrast_clip_num

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index



        results_contrast_all = []
        for i in range(self.contrast_clip_num):
            # Step1: Sample frame and resizing
            result_single_clip = dict()
            results_weak = self.pipeline_weak(copy.deepcopy(results))
            # Step2: Strong augmentation
            results_strong = self.pipeline_strong(copy.deepcopy(results_weak))

            # Step3: Final formating
            results_weak = self.pipeline_format(results_weak)
            result_single_clip['label_unlabeled'] = results_weak['label']
            result_single_clip['imgs_weak'] = results_weak['imgs']
            result_single_clip['imgs_diff_weak'] = results_weak['imgs_diff']

            results_strong = self.pipeline_format(results_strong)
            result_single_clip['imgs_strong'] = results_strong['imgs']
            result_single_clip['imgs_diff_strong'] = results_strong['imgs_diff']
            results_contrast_all.append(result_single_clip)

        output = dict()
        for key in ['imgs_weak', 'imgs_diff_weak', 'imgs_strong', 'imgs_diff_strong', 'label_unlabeled']:
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
