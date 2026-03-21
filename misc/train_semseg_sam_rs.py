#!/usr/bin/env python3
"""Train SAM_RS-style segmentation on GoldMDD with SAM-derived boundary/object priors.

Protocol alignment with other GoldMDD runs:
- data split: /GoldMDD/data-cropped/{train,val,test}
- epochs: default 80
- batch size: default 8
- augmentation: goldmdd_v2
- checkpoint rule: best by val_miou (plus last)

SAM_RS alignment:
- offline SAM prior generation (boundary/object)
- default loss mode: SEG+BDY+OBJ
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.segmentation import find_boundaries

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from semseg_common import (
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_FOREGROUND_CLASSES,
    PATCH_SIZE,
    build_split_samples,
    compute_metrics_from_conf,
    set_seed,
    update_confusion,
)


@dataclass(frozen=True)
class SamSample:
    stem: str
    image_path: Path
    label_path: Path
    boundary_path: Path
    object_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--priors-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-samrs-priors"))
    p.add_argument("--work-dir", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--run-name", type=str, default="sam_rs_unetformer_baseline1_augv2_seg_bdy_obj")
    p.add_argument(
        "--model",
        type=str,
        default="unetformer",
        choices=["unetformer", "ftunetformer", "abcnet", "cmtfnet"],
        help="SAM_RS official model variant",
    )

    p.add_argument("--ssrs-repo", type=Path, default=Path("/deac/csc/yangGrp/cuij/third_party/SSRS/SAM_RS"))
    p.add_argument("--segment-anything-path", type=Path, default=Path("/deac/csc/yangGrp/cuij/GeCo/sam-hq"))
    p.add_argument("--sam-checkpoint", type=Path, default=Path("/deac/csc/yangGrp/cuij/GeCo/MODEL_folder/sam_vit_h_4b8939.pth"))
    p.add_argument("--sam-model-type", type=str, default="vit_h")
    p.add_argument("--sam-device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--sam-min-area", type=int, default=50)
    p.add_argument("--sam-max-objects", type=int, default=50)
    p.add_argument("--sam-pred-iou-thresh", type=float, default=0.96)
    p.add_argument("--sam-box-nms-thresh", type=float, default=0.5)
    p.add_argument("--sam-crop-nms-thresh", type=float, default=0.5)
    p.add_argument("--prior-max-images", type=int, default=0, help="0=all")

    p.add_argument("--prepare-priors", action="store_true", default=True)
    p.add_argument("--skip-priors", dest="prepare_priors", action="store_false")
    p.add_argument("--prepare-only", action="store_true", help="Only generate priors, do not train")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--train-log-interval", type=int, default=200)
    p.add_argument("--val-log-interval", type=int, default=100)

    p.add_argument("--aug-preset", type=str, default="goldmdd_v2", choices=["goldmdd_v2", "none"])

    p.add_argument("--lambda-boundary", type=float, default=0.1)
    p.add_argument("--lambda-object", type=float, default=1.0)
    p.add_argument(
        "--loss-mode",
        type=str,
        default="seg_bdy_obj",
        choices=["seg", "seg_bdy", "seg_obj", "seg_bdy_obj"],
        help="SAM_RS default is seg_bdy_obj",
    )

    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    return p.parse_args()


def add_path(path: Path) -> None:
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_sam_generator(args: argparse.Namespace):
    add_path(args.segment_anything_path)
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        import segment_anything.automatic_mask_generator as amg_mod
    except Exception as e:
        raise RuntimeError(
            f"Failed to import segment_anything from {args.segment_anything_path}. "
            f"Set --segment-anything-path correctly. Error: {e}"
        )

    # Some clusters ship torchvision without compiled C++ ops (nms), which breaks
    # SAM's default batched_nms call. Replace with a torch-only fallback.
    def _nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep: list[int] = []
        while order.numel() > 0:
            i = int(order[0].item())
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            union = areas[i] + areas[rest] - inter
            iou = inter / union.clamp(min=1e-8)
            order = rest[iou <= iou_threshold]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _batched_nms_torch(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        max_coord = boxes.max()
        offsets = idxs.to(boxes) * (max_coord + 1)
        boxes_for_nms = boxes + offsets[:, None]
        return _nms_torch(boxes_for_nms, scores, iou_threshold)

    amg_mod.batched_nms = _batched_nms_torch

    ckpt = args.sam_checkpoint.expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

    sam_device = args.sam_device
    if sam_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available for SAM; fallback to CPU", flush=True)
        sam_device = "cpu"

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(ckpt))
    sam.to(device=sam_device)
    generator = SamAutomaticMaskGenerator(
        sam,
        crop_nms_thresh=args.sam_crop_nms_thresh,
        box_nms_thresh=args.sam_box_nms_thresh,
        pred_iou_thresh=args.sam_pred_iou_thresh,
    )
    return generator


def generate_sam_prior(image_rgb: np.ndarray, generator, min_area: int = 50, max_objects: int = 50) -> tuple[np.ndarray, np.ndarray]:
    masks = generator.generate(image_rgb)
    h, w = image_rgb.shape[:2]
    boundary = np.zeros((h, w), dtype=np.uint8)
    obj = np.zeros((h, w), dtype=np.uint8)

    if len(masks) == 0:
        return boundary, obj

    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    obj_id = 1
    for ann in sorted_masks:
        if ann["area"] < min_area:
            continue
        if obj_id > max_objects:
            break
        m = ann["segmentation"]
        obj[m] = obj_id
        obj_id += 1

    for ann in masks:
        m = ann["segmentation"]
        b = find_boundaries(m.astype(np.uint8), mode="thick")
        boundary[b] = 255

    # Keep object prior away from boundary pixels (same intent as upstream code)
    obj[boundary > 0] = 0
    return boundary, obj


def prepare_sam_priors(
    args: argparse.Namespace,
    split_samples: dict[str, list],
) -> None:
    samples: list[tuple[str, str, Path]] = []
    for split, items in split_samples.items():
        if split == "train" and args.limit_train > 0:
            items = items[: args.limit_train]
        elif split == "val" and args.limit_val > 0:
            items = items[: args.limit_val]
        elif split == "test" and args.limit_test > 0:
            items = items[: args.limit_test]
        for s in items:
            samples.append((split, s.stem, s.image_path))

    if args.prior_max_images > 0:
        samples = samples[: args.prior_max_images]

    total = len(samples)
    if total == 0:
        return

    need = 0
    for split, stem, _ in samples:
        b = args.priors_root / split / "boundary" / f"{stem}.png"
        o = args.priors_root / split / "object" / f"{stem}.png"
        if not (b.exists() and o.exists()):
            need += 1

    print(f"SAM prior check: total={total}, missing={need}", flush=True)
    if need == 0:
        return

    generator = load_sam_generator(args)

    done = 0
    for idx, (split, stem, image_path) in enumerate(samples, start=1):
        out_b = args.priors_root / split / "boundary" / f"{stem}.png"
        out_o = args.priors_root / split / "object" / f"{stem}.png"
        out_b.parent.mkdir(parents=True, exist_ok=True)
        out_o.parent.mkdir(parents=True, exist_ok=True)
        if out_b.exists() and out_o.exists():
            continue

        rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        b, o = generate_sam_prior(rgb, generator, min_area=args.sam_min_area, max_objects=args.sam_max_objects)
        Image.fromarray(b, mode="L").save(out_b)
        Image.fromarray(o, mode="L").save(out_o)
        done += 1

        if done % 50 == 0 or done == need:
            print(f"Generated SAM priors {done}/{need} (scanned {idx}/{total})", flush=True)


def random_scale_to_patch_multi(
    img_pil: Image.Image,
    maps_pil: list[Image.Image],
    out_size: int = PATCH_SIZE,
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> tuple[Image.Image, list[Image.Image]]:
    scale = random.uniform(*scale_range)
    scaled = max(1, int(round(out_size * scale)))
    if scaled != out_size:
        img_pil = img_pil.resize((scaled, scaled), resample=Image.BILINEAR)
        maps_pil = [m.resize((scaled, scaled), resample=Image.NEAREST) for m in maps_pil]

    if scaled > out_size:
        left = random.randint(0, scaled - out_size)
        top = random.randint(0, scaled - out_size)
        box = (left, top, left + out_size, top + out_size)
        img_pil = img_pil.crop(box)
        maps_pil = [m.crop(box) for m in maps_pil]
    elif scaled < out_size:
        img_canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
        map_canvases = [Image.new(m.mode, (out_size, out_size), 0) for m in maps_pil]
        left = random.randint(0, out_size - scaled)
        top = random.randint(0, out_size - scaled)
        img_canvas.paste(img_pil, (left, top))
        for c, m in zip(map_canvases, maps_pil):
            c.paste(m, (left, top))
        img_pil = img_canvas
        maps_pil = map_canvases

    return img_pil, maps_pil


def encode_label(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, IGNORE_INDEX, dtype=np.int64)
    fg = (mask >= 1) & (mask <= NUM_FOREGROUND_CLASSES)
    out[fg] = mask[fg].astype(np.int64) - 1
    return out


def object_process(obj: np.ndarray) -> np.ndarray:
    ids = np.unique(obj)
    out = obj.copy()
    new_id = 1
    for oid in ids:
        if oid == 0:
            continue
        out[obj == oid] = new_id
        new_id += 1
    return out


class GoldMDDSAMRSDataset(Dataset):
    def __init__(self, samples: list[SamSample], train: bool, aug_preset: str = "goldmdd_v2") -> None:
        self.samples = samples
        self.train = train
        self.aug_preset = aug_preset

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_pil = Image.open(s.image_path).convert("RGB")
        label_pil = Image.open(s.label_path).convert("L")
        bnd_pil = Image.open(s.boundary_path).convert("L")
        obj_pil = Image.open(s.object_path).convert("L")

        if self.train and self.aug_preset == "goldmdd_v2":
            img_pil, [label_pil, bnd_pil, obj_pil] = random_scale_to_patch_multi(
                img_pil, [label_pil, bnd_pil, obj_pil], out_size=PATCH_SIZE, scale_range=(0.8, 1.2)
            )
            if random.random() < 0.8:
                img_pil = ImageEnhance.Brightness(img_pil).enhance(1.0 + random.uniform(-0.20, 0.20))
                img_pil = ImageEnhance.Contrast(img_pil).enhance(1.0 + random.uniform(-0.20, 0.20))
                img_pil = ImageEnhance.Color(img_pil).enhance(1.0 + random.uniform(-0.15, 0.15))
            if random.random() < 0.20:
                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

        img = np.array(img_pil, dtype=np.uint8, copy=True)
        label = np.array(label_pil, dtype=np.uint8, copy=True)
        bnd = np.array(bnd_pil, dtype=np.uint8, copy=True)
        obj = np.array(obj_pil, dtype=np.uint8, copy=True)

        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=1))
                label = np.ascontiguousarray(np.flip(label, axis=1))
                bnd = np.ascontiguousarray(np.flip(bnd, axis=1))
                obj = np.ascontiguousarray(np.flip(obj, axis=1))
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=0))
                label = np.ascontiguousarray(np.flip(label, axis=0))
                bnd = np.ascontiguousarray(np.flip(bnd, axis=0))
                obj = np.ascontiguousarray(np.flip(obj, axis=0))
            k = random.randint(0, 3)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
                label = np.ascontiguousarray(np.rot90(label, k, axes=(0, 1)))
                bnd = np.ascontiguousarray(np.rot90(bnd, k, axes=(0, 1)))
                obj = np.ascontiguousarray(np.rot90(obj, k, axes=(0, 1)))

        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        y = torch.from_numpy(encode_label(label)).long()
        b = torch.from_numpy((bnd > 0).astype(np.int64))
        o = torch.from_numpy(object_process(obj).astype(np.int64))
        return x, b, o, y, s.stem


class ObjectConsistencyLoss(nn.Module):
    def __init__(self, max_object: int = 50):
        super().__init__()
        self.max_object = max_object

    def forward(self, pred: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        # pred: [N, C, H, W], gt_obj: [N, H, W]
        total = pred.new_tensor(0.0)
        terms = 0
        n = pred.shape[0]
        for bi in range(n):
            p = pred[bi : bi + 1]
            g = gt_obj[bi]
            num_object = min(int(g.max().item()) + 1, self.max_object)
            for oid in range(1, num_object):
                mask = (g == oid).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                num_point = mask.sum()
                if num_point <= 0:
                    continue
                avg_pool = mask / (num_point + 1.0)
                object_feature = p * avg_pool
                avg_feature = object_feature.sum(dim=(2, 3), keepdim=True).expand_as(p) * mask
                total = total + F.mse_loss(num_point * object_feature, avg_feature, reduction="mean")
                terms += 1
        if terms == 0:
            return pred.new_tensor(0.0)
        return total / float(terms)


class BoundaryF1Loss(nn.Module):
    def __init__(self, theta0: int = 3, theta: int = 5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred: torch.Tensor, gt_boundary: torch.Tensor) -> torch.Tensor:
        # pred: [N,C,H,W], gt_boundary: [N,H,W] in {0,1}
        n = pred.shape[0]
        class_map = pred.argmax(dim=1).float().unsqueeze(1)
        gt = gt_boundary.float().unsqueeze(1)

        gt_b = F.max_pool2d(1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b = gt_b - (1 - gt)

        pred_b = F.max_pool2d(1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b = pred_b - (1 - class_map)

        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        gt_b = gt_b.view(n, -1)
        pred_b = pred_b.view(n, -1)
        gt_b_ext = gt_b_ext.view(n, -1)
        pred_b_ext = pred_b_ext.view(n, -1)

        p = (pred_b * gt_b_ext).sum(dim=1) / (pred_b.sum(dim=1) + 1e-7)
        r = (pred_b_ext * gt_b).sum(dim=1) / (gt_b.sum(dim=1) + 1e-7)
        bf1 = 2 * p * r / (p + r + 1e-7)
        return torch.mean(1 - bf1)


def add_ssrs_repo(repo: Path) -> None:
    repo = repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"SSRS SAM_RS repo not found: {repo}")
    add_path(repo)


def build_model(ssrs_repo: Path, model_name: str) -> nn.Module:
    add_ssrs_repo(ssrs_repo)
    name = model_name.lower()
    if name == "unetformer":
        from model.UNetFormer import UNetFormer

        return UNetFormer(num_classes=NUM_FOREGROUND_CLASSES)
    if name == "ftunetformer":
        from model.FTUNetFormer import ft_unetformer

        # Avoid external-pretrain path dependency for stable cluster runs.
        model = ft_unetformer(pretrained=False, num_classes=NUM_FOREGROUND_CLASSES)
        # Upstream FTUNetFormer is built around 256x256 input resolution.
        setattr(model, "_goldmdd_resize_input", (256, 256))
        return model
    if name == "abcnet":
        from model.ABCNet import ABCNet

        return ABCNet(num_classes=NUM_FOREGROUND_CLASSES)
    if name == "cmtfnet":
        from model.CMTFNet.CMTFNet import CMTFNet

        return CMTFNet(num_classes=NUM_FOREGROUND_CLASSES)
    raise ValueError(f"Unknown model: {model_name}")


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    resize_to = getattr(model, "_goldmdd_resize_input", None)
    original_hw = x.shape[-2:]
    if resize_to is not None:
        x = F.interpolate(x, size=resize_to, mode="bilinear", align_corners=False)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if resize_to is not None and out.shape[-2:] != original_hw:
        out = F.interpolate(out, size=original_hw, mode="bilinear", align_corners=False)
    return out


def build_sam_samples(data_root: Path, priors_root: Path, split: str, limit: int = 0) -> list[SamSample]:
    base = build_split_samples(data_root / split)
    if limit > 0:
        base = base[:limit]
    out: list[SamSample] = []
    for s in base:
        b = priors_root / split / "boundary" / f"{s.stem}.png"
        o = priors_root / split / "object" / f"{s.stem}.png"
        if not b.exists() or not o.exists():
            raise FileNotFoundError(f"Missing SAM priors for {split}/{s.stem}: {b} {o}")
        out.append(SamSample(stem=s.stem, image_path=s.image_path, label_path=s.label_path, boundary_path=b, object_path=o))
    return out


def make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    bdy_criterion: BoundaryF1Loss,
    obj_criterion: ObjectConsistencyLoss,
    device: torch.device,
    amp: bool,
    args: argparse.Namespace,
    phase: str,
    log_interval: int,
) -> dict[str, object]:
    model.eval()
    conf = torch.zeros((NUM_FOREGROUND_CLASSES, NUM_FOREGROUND_CLASSES), dtype=torch.int64, device=device)

    loss_sum = 0.0
    ce_sum = 0.0
    bdy_sum = 0.0
    obj_sum = 0.0
    n_batches = 0

    t0 = time.time()
    num_batches = max(1, len(loader))
    skipped_nonfinite = 0
    for bi, (x, b, o, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        o = o.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = forward_logits(model, x)
            loss_ce = ce_criterion(logits, y)
            loss_bdy = bdy_criterion(logits, b)
        if args.loss_mode in {"seg_obj", "seg_bdy_obj"}:
            # Keep object-consistency loss in fp32 to avoid overflow on large objects.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                loss_obj = obj_criterion(logits.float(), o)
        else:
            loss_obj = logits.new_tensor(0.0)
        if args.loss_mode == "seg":
            loss = loss_ce
        elif args.loss_mode == "seg_bdy":
            loss = loss_ce + args.lambda_boundary * loss_bdy
        elif args.loss_mode == "seg_obj":
            loss = loss_ce + args.lambda_object * loss_obj
        else:
            loss = loss_ce + args.lambda_boundary * loss_bdy + args.lambda_object * loss_obj

        logits_finite = torch.isfinite(logits).all()
        finite = logits_finite and torch.isfinite(loss_ce) and torch.isfinite(loss_bdy) and torch.isfinite(loss_obj) and torch.isfinite(loss)
        if not finite:
            skipped_nonfinite += 1
            if skipped_nonfinite <= 5:
                print(
                    f"[{phase}] non-finite at batch {bi}; "
                    f"logits_finite={bool(logits_finite)} "
                    f"ce={float(loss_ce.detach().cpu()) if torch.isfinite(loss_ce) else 'nan/inf'} "
                    f"bdy={float(loss_bdy.detach().cpu()) if torch.isfinite(loss_bdy) else 'nan/inf'} "
                    f"obj={float(loss_obj.detach().cpu()) if torch.isfinite(loss_obj) else 'nan/inf'}; "
                    "skipping batch",
                    flush=True,
                )
            continue

        update_confusion(conf, logits, y)
        loss_sum += float(loss.item())
        ce_sum += float(loss_ce.item())
        bdy_sum += float(loss_bdy.item())
        obj_sum += float(loss_obj.item())
        n_batches += 1

        if log_interval > 0 and (bi % log_interval == 0 or bi == num_batches):
            dt = time.time() - t0
            it_s = dt / max(n_batches, 1)
            eta_s = max(num_batches - bi, 0) * it_s
            print(
                f"[{phase}] {bi}/{num_batches} loss={loss_sum/max(n_batches,1):.4f} "
                f"(ce={ce_sum/max(n_batches,1):.4f}, bdy={bdy_sum/max(n_batches,1):.4f}, obj={obj_sum/max(n_batches,1):.4f}) "
                f"{it_s:.3f}s/it eta={eta_s/60.0:.1f}m",
                flush=True,
            )

    (
        miou,
        miou_present,
        macro_f1,
        macro_f1_present,
        oa_fg,
        per_class_iou,
        per_class_f1,
        gt_pixels_per_class,
    ) = compute_metrics_from_conf(conf)

    return {
        "loss": loss_sum / max(n_batches, 1),
        "ce": ce_sum / max(n_batches, 1),
        "bdy": bdy_sum / max(n_batches, 1),
        "obj": obj_sum / max(n_batches, 1),
        "miou": miou,
        "miou_present": miou_present,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "oa_fg": oa_fg,
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
        "gt_pixels_per_class": gt_pixels_per_class,
        "skipped_nonfinite": int(skipped_nonfinite),
        "sec": time.time() - t0,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    ce_criterion: nn.Module,
    bdy_criterion: BoundaryF1Loss,
    obj_criterion: ObjectConsistencyLoss,
    device: torch.device,
    amp: bool,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    ce_sum = 0.0
    bdy_sum = 0.0
    obj_sum = 0.0
    n_batches = 0

    t0 = time.time()
    num_batches = max(1, len(loader))
    skipped_nonfinite = 0
    for bi, (x, b, o, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        o = o.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = forward_logits(model, x)
            loss_ce = ce_criterion(logits, y)
            loss_bdy = bdy_criterion(logits, b)
        if args.loss_mode in {"seg_obj", "seg_bdy_obj"}:
            # Keep object-consistency loss in fp32 to avoid overflow on large objects.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                loss_obj = obj_criterion(logits.float(), o)
        else:
            loss_obj = logits.new_tensor(0.0)
        if args.loss_mode == "seg":
            loss = loss_ce
        elif args.loss_mode == "seg_bdy":
            loss = loss_ce + args.lambda_boundary * loss_bdy
        elif args.loss_mode == "seg_obj":
            loss = loss_ce + args.lambda_object * loss_obj
        else:
            loss = loss_ce + args.lambda_boundary * loss_bdy + args.lambda_object * loss_obj

        logits_finite = torch.isfinite(logits).all()
        finite = logits_finite and torch.isfinite(loss_ce) and torch.isfinite(loss_bdy) and torch.isfinite(loss_obj) and torch.isfinite(loss)
        if not finite:
            skipped_nonfinite += 1
            if skipped_nonfinite <= 5:
                print(
                    f"[train e{epoch:03d}] non-finite at batch {bi}; "
                    f"logits_finite={bool(logits_finite)} "
                    f"ce={float(loss_ce.detach().cpu()) if torch.isfinite(loss_ce) else 'nan/inf'} "
                    f"bdy={float(loss_bdy.detach().cpu()) if torch.isfinite(loss_bdy) else 'nan/inf'} "
                    f"obj={float(loss_obj.detach().cpu()) if torch.isfinite(loss_obj) else 'nan/inf'}; "
                    "skipping optimizer step",
                    flush=True,
                )
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item())
        ce_sum += float(loss_ce.item())
        bdy_sum += float(loss_bdy.item())
        obj_sum += float(loss_obj.item())
        n_batches += 1

        if args.train_log_interval > 0 and (bi % args.train_log_interval == 0 or bi == num_batches):
            dt = time.time() - t0
            it_s = dt / max(n_batches, 1)
            eta_s = max(num_batches - bi, 0) * it_s
            print(
                f"[{epoch:03d}/{args.epochs}] train {bi}/{num_batches} "
                f"loss={loss_sum/max(n_batches,1):.4f} "
                f"(ce={ce_sum/max(n_batches,1):.4f}, bdy={bdy_sum/max(n_batches,1):.4f}, obj={obj_sum/max(n_batches,1):.4f}) "
                f"{it_s:.3f}s/it eta={eta_s/60.0:.1f}m",
                flush=True,
            )

    return {
        "loss": loss_sum / max(n_batches, 1),
        "ce": ce_sum / max(n_batches, 1),
        "bdy": bdy_sum / max(n_batches, 1),
        "obj": obj_sum / max(n_batches, 1),
        "skipped_nonfinite": float(skipped_nonfinite),
        "sec": time.time() - t0,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_miou: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_miou": best_miou,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.model.lower() == "abcnet" and args.amp:
        # ABCNet is numerically unstable with fp16 autocast in this pipeline.
        # Force fp32 training/eval so we don't silently skip most batches.
        print("[warn] ABCNet + SAM_RS: forcing --no-amp for numerical stability.", flush=True)
        args.amp = False

    split_base = {
        "train": build_split_samples(args.data_root / "train"),
        "val": build_split_samples(args.data_root / "val"),
        "test": build_split_samples(args.data_root / "test"),
    }

    if args.prepare_priors:
        prepare_sam_priors(args, split_base)
        if args.prepare_only:
            print("Done prior generation (prepare-only)", flush=True)
            return

    train_samples = build_sam_samples(args.data_root, args.priors_root, "train", limit=args.limit_train)
    val_samples = build_sam_samples(args.data_root, args.priors_root, "val", limit=args.limit_val)
    test_samples = build_sam_samples(args.data_root, args.priors_root, "test", limit=args.limit_test)

    train_ds = GoldMDDSAMRSDataset(train_samples, train=True, aug_preset=args.aug_preset)
    val_ds = GoldMDDSAMRSDataset(val_samples, train=False, aug_preset="none")
    test_ds = GoldMDDSAMRSDataset(test_samples, train=False, aug_preset="none")

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_model(args.ssrs_repo, args.model)
    model.to(device)

    n_visible = torch.cuda.device_count() if device.type == "cuda" else 0
    if device.type == "cuda" and n_visible > 1:
        model = nn.DataParallel(model)

    ce_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    bdy_criterion = BoundaryF1Loss()
    obj_criterion = ObjectConsistencyLoss(max_object=args.sam_max_objects)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = max(int(0.03 * total_steps), 100)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        t = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    run_dir = args.work_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    log_csv = run_dir / "train_log.csv"

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"Visible GPUs: {n_visible} | DataParallel: {isinstance(model, nn.DataParallel)}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Train/Val/Test patches: {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}", flush=True)
    print(
        f"Model: SAM_RS {args.model} (classes={NUM_FOREGROUND_CLASSES}, bg ignored) | "
        f"Loss mode: {args.loss_mode} (lambda_bdy={args.lambda_boundary}, lambda_obj={args.lambda_object})",
        flush=True,
    )

    best_miou_present = -1.0
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_ce",
                "train_bdy",
                "train_obj",
                "train_skipped_nonfinite",
                "train_sec",
                "val_loss",
                "val_ce",
                "val_bdy",
                "val_obj",
                "val_skipped_nonfinite",
                "val_miou",
                "val_miou_present",
                "val_macro_f1",
                "val_macro_f1_present",
                "val_oa_fg",
                "val_sec",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                ce_criterion,
                bdy_criterion,
                obj_criterion,
                device,
                args.amp,
                args,
                epoch,
            )

            for _ in range(len(train_loader)):
                scheduler.step()

            do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
            val_stats = {
                "loss": float("nan"),
                "ce": float("nan"),
                "bdy": float("nan"),
                "obj": float("nan"),
                "miou": float("nan"),
                "miou_present": float("nan"),
                "macro_f1": float("nan"),
                "macro_f1_present": float("nan"),
                "oa_fg": float("nan"),
                "sec": 0.0,
                "per_class_iou": [],
                "per_class_f1": [],
                "gt_pixels_per_class": [],
            }
            if do_val:
                val_stats = evaluate(
                    model,
                    val_loader,
                    ce_criterion,
                    bdy_criterion,
                    obj_criterion,
                    device,
                    args.amp,
                    args,
                    phase=f"val e{epoch:03d}",
                    log_interval=args.val_log_interval,
                )
                current = float(val_stats["miou_present"])
                if math.isfinite(current) and current > best_miou_present:
                    best_miou_present = current
                    save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best_miou_present, args)
                    (run_dir / "best_val_per_class_iou.json").write_text(
                        json.dumps(val_stats["per_class_iou"], indent=2), encoding="utf-8"
                    )

            if args.save_every > 0 and (epoch % args.save_every == 0):
                save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, best_miou_present, args)
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best_miou_present, args)

            lr_now = optimizer.param_groups[0]["lr"]
            w.writerow(
                [
                    epoch,
                    f"{lr_now:.8e}",
                    f"{train_stats['loss']:.6f}",
                    f"{train_stats['ce']:.6f}",
                    f"{train_stats['bdy']:.6f}",
                    f"{train_stats['obj']:.6f}",
                    f"{train_stats['skipped_nonfinite']:.0f}",
                    f"{train_stats['sec']:.2f}",
                    f"{val_stats['loss']:.6f}" if do_val else "",
                    f"{val_stats['ce']:.6f}" if do_val else "",
                    f"{val_stats['bdy']:.6f}" if do_val else "",
                    f"{val_stats['obj']:.6f}" if do_val else "",
                    f"{val_stats['skipped_nonfinite']:.0f}" if do_val else "",
                    f"{val_stats['miou']:.6f}" if do_val else "",
                    f"{val_stats['miou_present']:.6f}" if do_val else "",
                    f"{val_stats['macro_f1']:.6f}" if do_val else "",
                    f"{val_stats['macro_f1_present']:.6f}" if do_val else "",
                    f"{val_stats['oa_fg']:.6f}" if do_val else "",
                    f"{val_stats['sec']:.2f}" if do_val else "",
                ]
            )
            f.flush()

            msg = (
                f"[{epoch:03d}/{args.epochs}] lr={lr_now:.2e} train loss={train_stats['loss']:.4f} "
                f"(ce={train_stats['ce']:.4f}, bdy={train_stats['bdy']:.4f}, obj={train_stats['obj']:.4f})"
            )
            if do_val:
                msg += (
                    f" | val loss={val_stats['loss']:.4f} miou={val_stats['miou']:.4f} "
                    f"miou_present={val_stats['miou_present']:.4f} "
                    f"f1_present={val_stats['macro_f1_present']:.4f} oa_fg={val_stats['oa_fg']:.4f} "
                    f"(best_val_miou_present={best_miou_present:.4f})"
                )
            print(msg, flush=True)

    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')} for test eval", flush=True)

    test_stats = evaluate(
        model,
        test_loader,
        ce_criterion,
        bdy_criterion,
        obj_criterion,
        device,
        args.amp,
        args,
        phase="test",
        log_interval=args.val_log_interval,
    )

    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "miou": test_stats["miou"],
                "miou_present": test_stats["miou_present"],
                "macro_f1": test_stats["macro_f1"],
                "macro_f1_present": test_stats["macro_f1_present"],
                "oa_fg": test_stats["oa_fg"],
                "loss": test_stats["loss"],
                "ce": test_stats["ce"],
                "boundary": test_stats["bdy"],
                "object": test_stats["obj"],
                "per_class_iou": test_stats["per_class_iou"],
                "per_class_f1": test_stats["per_class_f1"],
                "gt_pixels_per_class": test_stats["gt_pixels_per_class"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"TEST: loss={test_stats['loss']:.4f}, miou={test_stats['miou']:.4f}, "
        f"miou_present={test_stats['miou_present']:.4f}, "
        f"f1_present={test_stats['macro_f1_present']:.4f}, oa_fg={test_stats['oa_fg']:.4f}. "
        f"Saved metrics to {run_dir / 'test_metrics.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
