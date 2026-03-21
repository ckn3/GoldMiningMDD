_base_ = ['./goldmdd_mcpnet_full_80ep_bs8_weighted_ce.py']

crop_size = (128, 128)

data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=dict(size=crop_size))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(160, 160), ratio_range=(0.8, 1.2), keep_ratio=True),
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

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=train_pipeline, indices=list(range(8))))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=test_pipeline, indices=list(range(8))))

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=test_pipeline, indices=list(range(8))))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=1, val_interval=1000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1, max_keep_ckpts=1, save_best='mIoU'))
