_base_ = './logcan_r50_goldmdd.py'

exp_name = "/deac/csc/yangGrp/cuij/GoldMDD/experiments/rsseg_logcan_r50_ce_dice_smoke"
epoch = 2

dataset_config = dict(
    data_root='/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-rsseg-smoke',
    train_mode=dict(
        loader=dict(
            batch_size=4,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        ),
    ),
    val_mode=dict(
        loader=dict(
            batch_size=4,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        ),
    ),
    test_mode=dict(
        loader=dict(
            batch_size=4,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        ),
    ),
)

optimizer_config = dict(
    scheduler=dict(
        max_epoch=epoch,
    ),
)
