from torch.utils.data import DataLoader
import torch

from geoseg.datasets.goldmdd_dataset import CLASSES, GoldMDDDataset, IGNORE_INDEX, train_aug, val_aug
from geoseg.losses import DiceLoss, FocalLoss, JointLoss
from tools.utils import Lookahead, process_model_params

# Protocol-aligned training hparams
max_epoch = 80
ignore_index = IGNORE_INDEX
train_batch_size = 8
val_batch_size = 8
num_workers = 8

lr = 2e-4
weight_decay = 5e-2
backbone_lr = 2e-5
backbone_weight_decay = 5e-2

num_classes = len(CLASSES)
classes = CLASSES

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 1
resume_ckpt_path = None

from geoseg.models.A2FPN import A2FPN

weights_name = "geoseg_a2fpn_goldmdd_b8_e80_poly_focal_dice"
weights_path = "/deac/csc/yangGrp/cuij/GoldMDD/experiments/{}".format(weights_name)
test_weights_name = weights_name
log_name = "goldmdd/{}".format(weights_name)

net = A2FPN(class_num=num_classes)
loss = JointLoss(
    FocalLoss(gamma=2.0, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

train_dataset = GoldMDDDataset(
    data_root="/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
    split="train",
    transform=train_aug,
)
val_dataset = GoldMDDDataset(
    data_root="/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
    split="val",
    transform=val_aug,
)
test_dataset = GoldMDDDataset(
    data_root="/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
    split="test",
    transform=val_aug,
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
