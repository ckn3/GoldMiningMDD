# GoldMDD dataset config (cropped 512x512 patches).

dataset_type = 'CustomDataset'
data_root = '/deac/csc/yangGrp/cuij/GoldMDD/data-cropped'

goldmdd_classes = (
    'Building',
    'Mining raft',
    'Primary Forest',
    'Heavy machinery',
    'Water bodies',
    'Agricultural crop',
    'Compact mounds',
    'Gravel mounds',
    'Grass',
    'Type 1 natural regeneration',
    'Type 2 natural regeneration',
    'Bare ground',
    'Sluice',
    'Vehicles',
)

goldmdd_palette = [
    [138, 106, 61],
    [123, 235, 251],
    [176, 76, 24],
    [238, 146, 198],
    [79, 111, 111],
    [132, 208, 140],
    [35, 243, 227],
    [88, 84, 0],
    [141, 181, 29],
    [194, 22, 58],
    [247, 119, 87],
    [44, 216, 116],
    [97, 57, 145],
    [150, 154, 174],
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

custom_imports = dict(
    imports=['models.goldmdd_transforms'],
    allow_failed_imports=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='GoldMDDGaussianBlur', prob=0.2, sigma_min=0.3, sigma_max=1.2),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='GoldMDDRandomRotate90'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    _delete_=True,
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/image',
        ann_dir='train/label',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=goldmdd_classes,
        palette=goldmdd_palette,
        reduce_zero_label=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/label',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=goldmdd_classes,
        palette=goldmdd_palette,
        reduce_zero_label=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/image',
        ann_dir='test/label',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=goldmdd_classes,
        palette=goldmdd_palette,
        reduce_zero_label=True,
        pipeline=test_pipeline))
