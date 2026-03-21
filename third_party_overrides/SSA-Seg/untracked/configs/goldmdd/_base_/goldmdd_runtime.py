# GoldMDD runtime: 80 epochs equivalent to prior 658k-iter setup (bs=8).

runner = dict(type='IterBasedRunner', max_iters=658000)

# Keep only best + last checkpoint for long runs.
checkpoint_config = dict(
    by_epoch=False,
    interval=658001,
    save_last=True,
    max_keep_ckpts=2)

evaluation = dict(
    interval=8225,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    rule='greater')
