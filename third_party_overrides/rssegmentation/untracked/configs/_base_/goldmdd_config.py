dataset = 'goldmdd'

dataset_config = dict(
    type='GoldMDD',
    data_root='/deac/csc/yangGrp/cuij/GoldMDD/data-cropped',
    train_mode=dict(
        transform=dict(
            RandomScale={'scale_list': [0.8, 1.2], 'mode': 'range'},
            RandomCrop={'size': 512, 'ignore_index': 14, 'nopad': False},
            RandomHorizontallyFlip=None,
            RandomVerticalFlip=None,
            RandomRotate90={'p': 1.0},
            RandomColorJitter={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.15, 'p': 0.8},
            RandomGaussianBlur={'p': 0.2},
        ),
        loader=dict(
            batch_size=8,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        ),
    ),
    val_mode=dict(
        transform=dict(),
        loader=dict(
            batch_size=8,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        ),
    ),
    test_mode=dict(
        transform=dict(),
        loader=dict(
            batch_size=8,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        ),
    ),
)

metric_cfg1 = dict(
    task='multiclass',
    average='micro',
    num_classes=15,
    ignore_index=14,
)

metric_cfg2 = dict(
    task='multiclass',
    average='none',
    num_classes=15,
    ignore_index=14,
)

eval_label_id_left = 0
eval_label_id_right = 14

class_name = [
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
    'Ignore',
]
