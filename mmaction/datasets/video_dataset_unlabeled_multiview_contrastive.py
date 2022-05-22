import os.path as osp
import copy

from .base import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
import torch


@DATASETS.register_module()
class UnlabeledVideoDataset_MultiView_Contrastive(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file,
                 pipeline_weak,
                 pipeline_strong,
                 pipeline_format,
                 contrast_clip_num,
                 start_index=0,
                 **kwargs):
        super().__init__(ann_file, pipeline_weak, start_index=start_index, **kwargs)

        self.pipeline_weak = self.pipeline
        self.pipeline_strong = Compose(pipeline_strong)
        self.pipeline_format = Compose(pipeline_format)
        self.contrast_clip_num = contrast_clip_num

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
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
