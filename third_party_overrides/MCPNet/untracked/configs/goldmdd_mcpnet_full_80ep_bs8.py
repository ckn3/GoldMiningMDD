_base_ = ['./goldmdd_mcpnet_smoke.py']

crop_size = (512, 512)

# GoldMDD-aligned augmentation (v2-style): scale jitter + crop + H/V flip + rotate + photometric
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(640, 640), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=12,
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(pipeline=test_pipeline),
)

# 65,798 train patches / bs=8 => 8,225 iters per epoch; 80 epochs => 658,000 iters.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=658000, val_interval=8225)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=0, end=658000, by_epoch=False)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=658000,
        max_keep_ckpts=2,
        save_best='mIoU',
        save_last=True,
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

randomness = dict(seed=3407)
