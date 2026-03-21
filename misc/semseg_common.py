#!/usr/bin/env python3
"""Shared dataset/loss/metrics utilities for GoldMDD semantic segmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


Image.MAX_IMAGE_PIXELS = None

IGNORE_INDEX = 255
NUM_FOREGROUND_CLASSES = 14  # labels 1..14
PATCH_SIZE = 512
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    label_path: Path


def build_split_samples(split_dir: Path) -> list[Sample]:
    img_dir = split_dir / "image"
    lbl_dir = split_dir / "label"
    image_map = {p.stem: p for p in img_dir.glob("*.jpg")}
    label_map = {p.stem: p for p in lbl_dir.glob("*.png")}
    common = sorted(image_map.keys() & label_map.keys())
    missing_img = sorted(label_map.keys() - image_map.keys())
    missing_lbl = sorted(image_map.keys() - label_map.keys())
    if missing_img or missing_lbl:
        raise RuntimeError(
            f"Unpaired files in {split_dir}: missing_img={len(missing_img)}, missing_lbl={len(missing_lbl)}"
        )
    return [Sample(stem=s, image_path=image_map[s], label_path=label_map[s]) for s in common]


def random_scale_to_patch(
    img_pil: Image.Image,
    label_pil: Image.Image,
    out_size: int = PATCH_SIZE,
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> tuple[Image.Image, Image.Image]:
    scale = random.uniform(*scale_range)
    scaled = max(1, int(round(out_size * scale)))
    if scaled != out_size:
        img_pil = img_pil.resize((scaled, scaled), resample=Image.BILINEAR)
        label_pil = label_pil.resize((scaled, scaled), resample=Image.NEAREST)

    if scaled > out_size:
        left = random.randint(0, scaled - out_size)
        top = random.randint(0, scaled - out_size)
        box = (left, top, left + out_size, top + out_size)
        img_pil = img_pil.crop(box)
        label_pil = label_pil.crop(box)
    elif scaled < out_size:
        img_canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
        lbl_canvas = Image.new(label_pil.mode, (out_size, out_size), 0)
        left = random.randint(0, out_size - scaled)
        top = random.randint(0, out_size - scaled)
        img_canvas.paste(img_pil, (left, top))
        lbl_canvas.paste(label_pil, (left, top))
        img_pil = img_canvas
        label_pil = lbl_canvas
    return img_pil, label_pil


def compute_train_class_weights(
    train_samples: list[Sample],
    power: float = 0.5,
) -> tuple[torch.Tensor, list[int]]:
    # Counts over foreground classes only (labels 1..14). Background 0 excluded.
    counts = np.zeros(NUM_FOREGROUND_CLASSES, dtype=np.int64)
    for s in train_samples:
        with Image.open(s.label_path) as lb:
            arr = np.array(lb, dtype=np.uint8, copy=False)
        binc = np.bincount(arr.reshape(-1), minlength=NUM_FOREGROUND_CLASSES + 1)
        counts += binc[1 : NUM_FOREGROUND_CLASSES + 1].astype(np.int64)

    counts_safe = counts.astype(np.float64).copy()
    nonzero = counts_safe > 0
    if not np.any(nonzero):
        raise RuntimeError("No foreground pixels found in train labels; cannot compute class weights.")
    if not np.all(nonzero):
        counts_safe[~nonzero] = counts_safe[nonzero].max()
    freqs = counts_safe / counts_safe.sum()
    weights = 1.0 / np.power(freqs, power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


class GoldMDDPatchDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool, aug_preset: str = "goldmdd_v1") -> None:
        self.samples = samples
        self.train = train
        self.aug_preset = aug_preset

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with Image.open(s.image_path) as im:
            img_pil = im.convert("RGB")
        with Image.open(s.label_path) as lb:
            label_pil = lb.copy()

        if self.train and self.aug_preset != "none":
            if self.aug_preset == "goldmdd_v1":
                if random.random() < 0.8:
                    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.0 + random.uniform(-0.20, 0.20))
                    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.0 + random.uniform(-0.20, 0.20))
                    img_pil = ImageEnhance.Color(img_pil).enhance(1.0 + random.uniform(-0.15, 0.15))
                if random.random() < 0.15:
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
            elif self.aug_preset == "goldmdd_v2":
                img_pil, label_pil = random_scale_to_patch(img_pil, label_pil, out_size=PATCH_SIZE, scale_range=(0.8, 1.2))
                if random.random() < 0.8:
                    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.0 + random.uniform(-0.20, 0.20))
                    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.0 + random.uniform(-0.20, 0.20))
                    img_pil = ImageEnhance.Color(img_pil).enhance(1.0 + random.uniform(-0.15, 0.15))
                if random.random() < 0.20:
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
            else:
                raise ValueError(f"Unknown aug preset: {self.aug_preset}")

        img = np.array(img_pil, dtype=np.uint8, copy=True)
        label = np.array(label_pil, dtype=np.uint8, copy=True)

        if img.shape[:2] != label.shape[:2]:
            raise RuntimeError(f"Size mismatch for {s.stem}: {img.shape[:2]} vs {label.shape[:2]}")

        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=1))
                label = np.ascontiguousarray(np.flip(label, axis=1))
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=0))
                label = np.ascontiguousarray(np.flip(label, axis=0))
            k = random.randint(0, 3)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
                label = np.ascontiguousarray(np.rot90(label, k, axes=(0, 1)))

        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        y = torch.from_numpy(label.astype(np.int64))
        y = torch.where(y == 0, torch.full_like(y, IGNORE_INDEX), y - 1)
        return x, y, s.stem


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


class DiceLossIgnoreBG(nn.Module):
    """Multiclass Dice over foreground classes only, with ignore_index support."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.new_tensor(0.0)

        target_safe = target.clone()
        target_safe[~valid] = 0
        onehot = F.one_hot(target_safe, num_classes=c).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        onehot = onehot * valid_f

        inter = (probs * onehot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class FocalLossIgnoreBG(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = IGNORE_INDEX, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.new_tensor(0.0)

        target_safe = target.clone()
        target_safe[~valid] = 0
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -torch.pow(1.0 - pt, self.gamma) * logpt

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss = loss * cw[target_safe]

        loss = loss[valid]
        if loss.numel() == 0:
            return logits.new_tensor(0.0)
        return loss.mean()


class CEDiceLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        loss_mode: str = "ce_dice",
        ce_class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.loss_mode = loss_mode
        ce_kwargs = {"ignore_index": IGNORE_INDEX}
        if ce_class_weights is not None:
            ce_kwargs["weight"] = ce_class_weights
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.focal = FocalLossIgnoreBG(gamma=focal_gamma, ignore_index=IGNORE_INDEX, class_weights=ce_class_weights)
        self.dice = DiceLossIgnoreBG(ignore_index=IGNORE_INDEX)
        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.loss_mode in {"ce_dice", "weighted_ce_dice"}:
            ce = self.ce(logits, target)
        elif self.loss_mode == "focal_dice":
            ce = self.focal(logits, target)
        else:  # pragma: no cover
            raise ValueError(self.loss_mode)
        dice = self.dice(logits, target)
        total = self.ce_w * ce + self.dice_w * dice
        return total, {"ce": float(ce.detach().item()), "dice": float(dice.detach().item())}


@torch.no_grad()
def update_confusion(conf: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> None:
    pred = logits.argmax(dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return
    t = target[valid].view(-1).to(torch.int64)
    p = pred[valid].view(-1).to(torch.int64)
    c = conf.shape[0]
    idx = t * c + p
    conf += torch.bincount(idx, minlength=c * c).reshape(c, c)


def compute_metrics_from_conf(
    conf: torch.Tensor,
) -> tuple[float, float, float, float, float, list[float], list[float], list[int]]:
    conf = conf.float()
    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom_iou = tp + fp + fn
    iou = torch.where(denom_iou > 0, tp / denom_iou, torch.full_like(denom_iou, float("nan")))
    denom_f1 = 2 * tp + fp + fn
    f1 = torch.where(denom_f1 > 0, (2 * tp) / denom_f1, torch.full_like(denom_f1, float("nan")))
    miou = torch.nanmean(iou).item()
    gt_pixels = conf.sum(1)
    present = gt_pixels > 0
    miou_present = torch.nanmean(iou[present]).item() if present.any() else float("nan")
    macro_f1 = torch.nanmean(f1).item()
    macro_f1_present = torch.nanmean(f1[present]).item() if present.any() else float("nan")
    oa_fg = (tp.sum() / conf.sum()).item() if conf.sum() > 0 else float("nan")
    return (
        miou,
        miou_present,
        macro_f1,
        macro_f1_present,
        oa_fg,
        [float(x) if torch.isfinite(x) else float("nan") for x in iou],
        [float(x) if torch.isfinite(x) else float("nan") for x in f1],
        [int(x.item()) for x in gt_pixels],
    )


def make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

