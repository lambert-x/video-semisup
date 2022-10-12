# model settings
model = dict(
    type='Semi_MV_Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(warmup_epoch=30, fixmatch_threshold=0.3),
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'RawframeDataset'
dataset_type_labeled = 'RawframeDataset'
dataset_type_unlabeled = 'UnlabeledRawframeDataset_MultiView'
# dataset_type_appearance = 'RawframeDataset_withAPP'


data_root = 'data/ucf101/rawframes/'
data_root_val = 'data/ucf101/rawframes/'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
labeled_percentage = 10
ann_file_train_labeled = f'data/ucf101/videossl_splits/ucf101_train_{labeled_percentage}_percent_labeled_split_{split}_rawframes.txt'
ann_file_train_unlabeled = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames_Custom', clip_len=8, frame_interval=8, num_clips=1,
         total_frames_offset=-1),
    dict(type='RawFrameDecode_Custom', extra_modalities=['tempgrad']),
    dict(type='RandomRescale', scale_range=(256, 320), lazy=True),
    dict(type='RandomCrop', size=224, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Reset_img_shape'),
    dict(type='RandomRescale', scale_range=(256, 320), lazy=True),
    dict(type='RandomCrop', size=224, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse_OnlyDiff'),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalize_Diff', **img_norm_cfg, raw_to_diff=False, redist_to_rgb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FormatShape_Diff', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'imgs_diff'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'imgs_diff'])
]
# Get the frame and resize, shared by both weak and strong
train_pipeline_weak = [
    dict(type='SampleFrames_Custom', clip_len=8, frame_interval=8, num_clips=1,
         total_frames_offset=-1),
    dict(type='RawFrameDecode_Custom', extra_modalities=['tempgrad']),
    dict(type='RandomRescale', scale_range=(256, 320), lazy=True),
    dict(type='RandomCrop', size=224, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Reset_img_shape'),
    dict(type='RandomRescale', scale_range=(256, 320), lazy=True),
    dict(type='RandomCrop', size=224, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse_OnlyDiff'),
]
# Only used for strong augmentation
train_pipeline_strong = [
    dict(type='pytorchvideo.RandAugment',
         magnitude=10,
         sampling_type='uniform',
         sampling_hparas=dict(
             sampling_data_type="int",
             sampling_min=1)
         ),
    dict(type='Imgaug',
         transforms=
         [dict(
             type='Cutout',
             cval=0,
             nb_iterations=1,
             size=4.0 / 7.0,
             squared=True)]),
    dict(type='PytorchVideoTrans_Custom',
         aug_type='RandAugment',
         modality='imgs_diff',
         magnitude=10,
         sampling_type='uniform',
         sampling_hparas=dict(
             sampling_data_type="int",
             sampling_min=1)
         ),
    dict(
        type='Imgaug_Custom',
        transforms=
        [dict(
            type='Cutout',
            cval=0,
            nb_iterations=1,
            size=4.0 / 7.0,
            squared=True)],
        modality='imgs_diff'),
]
# Formating the input tensors, shared by both weak and strong
train_pipeline_format = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalize_Diff', **img_norm_cfg, raw_to_diff=False, redist_to_rgb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FormatShape_Diff', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'imgs_diff'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'imgs_diff'])
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='CenterCrop', crop_size=224, lazy=True),
    dict(type='Flip', flip_ratio=0, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,  # NOTE: Need to reduce batch size. 16 -> 5
    fixmatch_mu=4,
    workers_per_gpu=1,  # Default: 4
    train_dataloader=dict(drop_last=True, pin_memory=True),
    train_labeled=dict(
        type=dataset_type_labeled,
        ann_file=ann_file_train_labeled,
        data_prefix=data_root,
        start_index=0,
        pipeline=train_pipeline),
    train_unlabeled=dict(
        type=dataset_type_unlabeled,
        ann_file=ann_file_train_unlabeled,
        data_prefix=data_root,
        start_index=0,
        pipeline_weak=train_pipeline_weak,
        pipeline_strong=train_pipeline_strong,
        pipeline_format=train_pipeline_format),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        start_index=0,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        start_index=0,
        pipeline=test_pipeline,
        test_mode=True),
    precise_bn=dict(
        type=dataset_type,
        ann_file=ann_file_train_unlabeled,
        data_prefix=data_root,
        start_index=0,
        pipeline=val_pipeline),
    videos_per_gpu_precise_bn=5
)
# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr 0.2 is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_ratio=0.1,
                 warmup_by_epoch=True,
                 warmup_iters=30)

total_epochs = 240  # Might need to increase this number for different splits. Default: 180
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))  # Default: 5
log_config = dict(
    interval=20,  # Default: 20
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings=[p-5==[-v
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/r3d_r18_8x8x1_180e_ucf101_rgb_all_{labeled_percentage}percent/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
precise_bn = dict(num_iters=200, interval=5)
