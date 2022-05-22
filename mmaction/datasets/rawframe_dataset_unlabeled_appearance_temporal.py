import copy
import numpy as np
import torch

from .pipelines import Compose
from .rawframe_dataset import RawframeDataset
from .builder import DATASETS


@DATASETS.register_module()
class UnlabeledRawframeDataset_App_Temp(RawframeDataset):
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
                 pipeline_format_diff,
                 pipeline_appearance=None,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB', 
                 det_file=None,
                 cls_file=None):
        assert modality == 'RGB'
        super().__init__(ann_file, pipeline_weak, data_prefix, test_mode, filename_tmpl, with_offset,
                         multi_class, num_classes, start_index, modality)
        self.pipeline_weak = self.pipeline
        self.pipeline_strong = Compose(pipeline_strong)

        if pipeline_appearance is not None:
            self.pipeline_appearance = Compose(pipeline_appearance)
        else:
            self.pipeline_appearance = None

        self.pipeline_format = Compose(pipeline_format)
        self.pipeline_format_diff = Compose(pipeline_format_diff)


    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        output = dict()
        # Step1: Sample frame and resizing
        results_weak = self.pipeline_weak(results)
        # Step2: Strong augmentation
        results_strong = self.pipeline_strong(copy.deepcopy(results_weak))
        if self.pipeline_appearance is not None:
            results_appearance = self.pipeline_appearance(copy.deepcopy(results_weak))
            output['imgs_appearance'] = results_appearance['imgs']

        # Step3: Final formating
        results_weak = self.pipeline_format_diff(results_weak)
        output['label_unlabeled'] = results_weak['label']
        output['imgs_weak'] = results_weak['imgs']
        output['imgs_diff'] = results_weak['imgs_diff']

        results_strong = self.pipeline_format(results_strong)
        output['imgs_strong'] = results_strong['imgs']


        return output

    def prepare_test_frames(self, idx):
        raise NotImplementedError

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        raise NotImplementedError
