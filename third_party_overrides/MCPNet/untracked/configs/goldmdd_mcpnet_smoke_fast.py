_base_ = ['./goldmdd_mcpnet_smoke.py']

crop_size = (128, 128)

data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=dict(size=crop_size),
)

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
    dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=0,
    dataset=dict(pipeline=test_pipeline))

test_dataloader = dict(
    num_workers=0,
    dataset=dict(pipeline=test_pipeline))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=1, val_interval=1000)
