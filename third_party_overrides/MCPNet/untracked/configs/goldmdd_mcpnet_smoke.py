_base_ = [
    './_base_/default_runtime.py',
]

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

class_names = (
    'Building', 'Mining raft', 'Primary Forest', 'Heavy machinery',
    'Water bodies', 'Agricultural crop', 'Compact mounds', 'Gravel mounds',
    'Grass', 'Type1 regen', 'Type2 regen', 'Bare ground', 'Sluice', 'Vehicles'
)

palette = [
    [54, 144, 214], [255, 188, 121], [38, 115, 0], [138, 22, 84],
    [0, 112, 192], [106, 168, 79], [230, 145, 56], [204, 102, 0],
    [147, 196, 125], [182, 215, 168], [118, 165, 175], [191, 144, 0],
    [111, 168, 220], [153, 153, 153]
]

model = dict(
    type='EncoderDecoderisdnet',
    down_scale=4,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=[
        dict(
            type='RefineASPPHead',
            in_channels=2048,
            in_index=3,
            channels=128,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=14,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0)),
        dict(
            type='ISDHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            channels=128,
            num_classes=14,
            dropout_ratio=0.1,
            fusion_mode='raf',
            model_cls='ShallowNet',
            dims=[12, 24, 48, 96],
            depths=[1, 1, 2, 1],
            shallow_model_inchan=3,
            lap=False,
            consist=False,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
        ),
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

dataset_type = 'GoldMDDDataset'
data_root = '/deac/csc/yangGrp/cuij/GoldMDD/data-cropped'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(640, 640), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

common_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    reduce_zero_label=True,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        **common_dataset,
        data_prefix=dict(img_path='train/image', seg_map_path='train/label'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **common_dataset,
        data_prefix=dict(img_path='val/image', seg_map_path='val/label'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **common_dataset,
        data_prefix=dict(img_path='test/image', seg_map_path='test/label'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

optimizer = dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=5e-2)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=0, end=20, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10, max_keep_ckpts=2, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

randomness = dict(seed=3407)
