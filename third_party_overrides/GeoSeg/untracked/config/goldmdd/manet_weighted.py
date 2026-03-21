from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from geoseg.datasets.goldmdd_dataset import CLASSES, GoldMDDDataset, IGNORE_INDEX, train_aug, val_aug
from geoseg.losses import DiceLoss
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

from geoseg.models.MANet import MANet

weights_name = "geoseg_manet_goldmdd_b8_e80_poly_weighted_ce_dice"
weights_path = "/deac/csc/yangGrp/cuij/GoldMDD/experiments/{}".format(weights_name)
test_weights_name = weights_name
log_name = "goldmdd/{}".format(weights_name)


class MANetWeightedLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        return self.ce(logits, labels) + self.dice(logits, labels)


# Same class-weight vector used by other GoldMDD weighted CE baselines (power=0.5).
CLASS_WEIGHTS = [
    0.30126821994781494,
    2.325228214263916,
    0.01622912287712097,
    3.28946852684021,
    0.02937043085694313,
    0.09195061028003693,
    0.04224363714456558,
    0.19870634377002716,
    0.10142096877098083,
    0.031047813594341278,
    0.017247214913368225,
    0.025090740993618965,
    5.463924407958984,
    2.0668039321899414,
]

net = MANet(num_classes=num_classes, backbone_name="resnet50", pretrained=True)
loss = MANetWeightedLoss(class_weights=CLASS_WEIGHTS, ignore_index=ignore_index)
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
