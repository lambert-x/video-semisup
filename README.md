# Learning from Temporal Gradient for Semi-supervised Action Recognition (CVPR 2022)

This repository includes the official pytorch implementation of the paper: [*Learning from Temporal Gradient for Semi-supervised Action Recognition*](https://arxiv.org/abs/2111.13241), CVPR 2022.



## Installation
The codes are built upon [MMAction2](https://github.com/open-mmlab/mmaction2). You are recommended to follow the official tutorials of [MMAction2](https://github.com/open-mmlab/mmaction2) to prepare the environment and datasets (without temporal gradient extraction). Please use the extraction script (e.g., [script for UCF101](https://github.com/lambert-x/video-semisup/blob/8ff44343bb34485f8ad08d50ca4d8de22e122c1d/tools/data/ucf101/extract_temporal_gradient_frames.sh)) if needed. 

### Create Environment

First of all, you can run the following scripts to prepare conda environment.

Install Miniconda (Optional)

```shell
bash install_miniconda.sh 
```

Conda create environment

```shell
bash create_mmact_env.sh
```

### Prepare Datasets

##### UCF101

```shell
bash prepare_ucf101.sh "number of your cpu threads"
```

##### Kinetics400

First link the `videos_train` and `videos_val` folders to `./data/kinetics400/`

```shell
bash prepare_k400.sh "number of your cpu threads"
```

### Run Experiments

For example,

```shell
bash exps/8gpu-rawframes-ucf101/our_method/exp3_ucf101_20percent_180e_align0123_1clip_weak_sameclip_ptv_new_loss_half.sh
```



## Citing our paper

If you use our code or paper in your research or wish to refer to our results, please use the following BibTeX entry.

```
@InProceedings{xiao2021learning,
  title={Learning from Temporal Gradient for Semi-supervised Action Recognition},
  author={Xiao, Junfei and Jing, Longlong and Zhang, Lin and He, Ju and She, Qi and Zhou, Zongwei and Yuille, Alan and Li, Yingwei},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2022}
}
```

## Acknowledgement

Code is built upon [MMAction2](https://github.com/open-mmlab/mmaction2) and [video-data-aug](https://github.com/vt-vl-lab/video-data-aug).
