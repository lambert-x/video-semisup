# model settings
model = dict(
    type='Semi_AppSup_TempSup_SimCLR_Crossclip_PTV_Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        depth=18,
        pretrained=None,
        pretrained2d=False,
        norm_eval=False,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
        act_cfg=dict(type='ReLU'),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 2, 2, 2),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    cls_head_temp=None,
    contrast_head_shared=dict(
            type='ContrastHead',
            out_channels=512,
            in_channels=512,
            hidden_channels=512,
            spatial_type='avg',
            loss_contrast=dict(type='SimCLRLoss_PTV_New', loss_weight=0.5),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            fc_layer_num=3,
            with_final_normalize=True,
            dropout_ratio=0,
            init_std=0.01),
    temp_backbone='same',
    temp_sup_head='same',
    train_cfg=dict(
        warmup_epoch=30,
        fixmatch_threshold=0.3,
        temp_align_indices=(0, 1, 2, 3),
        align_loss_func='Cosine',
        pseudo_label_metric='avg',
        crossclip_contrast_loss=['crossview_sameclip_sharedhead'],
        crossclip_contrast_range=['weak'],
    ),
    test_cfg=dict(average_clips='score'))

# dataset settings
dataset_type = 'RawframeDataset'
dataset_type_labeled = 'RawframeDataset_Contrastive'
dataset_type_unlabeled = 'UnlabeledRawframeDataset_MultiView_Contrastive'
# dataset_type_appearance = 'RawframeDataset_withAPP'


data_root = 'data/ucf101/rawframes/'
data_root_val = 'data/ucf101/rawframes/'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
labeled_percentage = 20
ann_file_train_labeled = f'data/ucf101/videossl_splits/ucf101_train_{labeled_percentage}_percent_labeled_split_{split}_rawframes.txt'
ann_file_train_unlabeled = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
# Human box detections
# det_file = f'data/ucf101/detections.npy'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames_Custom', clip_len=8, frame_interval=8, num_clips=1,
         total_frames_offset=-1),
    dict(type='RawFrameDecode_Custom',
         extra_modalities=['tempgrad'], decoding_backend='turbojpeg'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='RandomResizedCrop', lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse_WithDiff'),
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
    dict(type='RawFrameDecode_Custom',
         extra_modalities=['tempgrad'], decoding_backend='turbojpeg'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='RandomResizedCrop', lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse_WithDiff'),
]
# Only used for strong augmentation
train_pipeline_strong = [
    dict(type='Imgaug', transforms='default'),
    dict(type='Imgaug_Custom', transforms='default', modality='imgs_diff')
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
    dict(type='RawFrameDecode', decoding_backend='turbojpeg'),
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
    dict(type='RawFrameDecode', decoding_backend='turbojpeg'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=5,  # NOTE: Need to reduce batch size. 16 -> 5
    workers_per_gpu=4,  # Default: 4
    train_dataloader=dict(drop_last=True, pin_memory=True),
    train_labeled=dict(
        type=dataset_type_labeled,
        ann_file=ann_file_train_labeled,
        data_prefix=data_root,
        start_index=1,
        pipeline=train_pipeline,
        contrast_clip_num=1
        ),
    train_unlabeled=dict(
        type=dataset_type_unlabeled,
        ann_file=ann_file_train_unlabeled,
        data_prefix=data_root,
        start_index=1,
        pipeline_weak=train_pipeline_weak,
        pipeline_strong=train_pipeline_strong,
        pipeline_format=train_pipeline_format,
        contrast_clip_num=1
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        start_index=1,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        start_index=1,
        pipeline=test_pipeline,
        test_mode=True),
    precise_bn=dict(
        type=dataset_type,
        ann_file=ann_file_train_unlabeled,
        data_prefix=data_root,
        start_index=1,
        pipeline=val_pipeline),
    videos_per_gpu_precise_bn=5
)
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9,
    weight_decay=0.0001)  # this lr 0.2 is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_ratio=0.1,
                 warmup_by_epoch=True,
                 warmup_iters=30)
total_epochs = 180  # Might need to increase this number for different splits. Default: 180
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))  # Default: 5
log_config = dict(
    interval=20,  # Default: 20
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

precise_bn = dict(num_iters=200, interval=5,
                  bn_range=['backbone', 'cls_head'])

# runtime settings=[p-5==[-v
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/r3d_r18_8x8x1_180e_ucf101_rgb_all_{labeled_percentage}percent_align0123_L2_avg_allfixmatch/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False

