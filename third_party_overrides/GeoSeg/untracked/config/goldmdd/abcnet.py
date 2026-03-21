from torch.utils.data import DataLoader
import torch

from geoseg.datasets.goldmdd_dataset import CLASSES, GoldMDDDataset, IGNORE_INDEX, train_aug, val_aug
from geoseg.losses import DiceLoss, JointLoss, SoftCrossEntropyLoss
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

import torch.nn as nn

from geoseg.models.ABCNet import ABCNet

weights_name = "geoseg_abcnet_goldmdd_b8_e80_poly_ce_dice"
weights_path = "/deac/csc/yangGrp/cuij/GoldMDD/experiments/{}".format(weights_name)
test_weights_name = weights_name
log_name = "goldmdd/{}".format(weights_name)

class ABCNetLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and isinstance(logits, (tuple, list)) and len(logits) >= 3:
            logit_main, logit_aux1, logit_aux2 = logits[:3]
            return self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux1, labels) + 0.4 * self.aux_loss(logit_aux2, labels)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return self.main_loss(logits, labels)


net = ABCNet(n_classes=num_classes, pretrained=True)
loss = ABCNetLoss(ignore_index=ignore_index)
use_aux_loss = True

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
