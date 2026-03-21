from config.goldmdd.abcnet import *  # noqa: F401,F403

# Fast smoke settings
max_epoch = 1
train_batch_size = 2
val_batch_size = 2
num_workers = 0

weights_name = "geoseg_abcnet_goldmdd_smoke"
weights_path = "/deac/csc/yangGrp/cuij/GoldMDD/experiments/{}".format(weights_name)
test_weights_name = weights_name
log_name = "goldmdd/{}".format(weights_name)

train_dataset = GoldMDDDataset(
    data_root="/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
    split="train",
    transform=train_aug,
    max_samples=2,
)
val_dataset = GoldMDDDataset(
    data_root="/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
    split="val",
    transform=val_aug,
    max_samples=2,
)
test_dataset = GoldMDDDataset(
    data_root="/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
    split="test",
    transform=val_aug,
    max_samples=2,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda ep: (1.0 - float(ep) / float(max_epoch)) ** 0.9
)
