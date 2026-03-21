#!/usr/bin/env python3
"""
GoldMDD SAM-family semantic segmentation fine-tuning.

Supported backbones:
- SAM 2.1
- SAM 3
- HQ-SAM (v1)

Protocol aligned with existing GoldMDD baselines:
- data-cropped split layout (train/val/test)
- aug-v2 for train only
- CE+Dice / weighted CE+Dice / focal+Dice
- background ignored (label 0 -> ignore index 255)
- best checkpoint selected by val_miou_present
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

from semseg_common import (
    CEDiceLoss,
    GoldMDDPatchDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_FOREGROUND_CLASSES,
    build_split_samples,
    compute_metrics_from_conf,
    compute_train_class_weights,
    make_loader,
    set_seed,
    update_confusion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--work-dir", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--run-name", type=str, default="sam2_1_bplus_baseline1_augv2_ce_dice")
    p.add_argument("--family", type=str, default="sam2_1", choices=["sam2_1", "sam3", "hq_sam"])

    # Common optimization
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument(
        "--loss-mode",
        type=str,
        default="ce_dice",
        choices=["ce_dice", "weighted_ce_dice", "focal_dice"],
    )
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--decoder-channels", type=int, default=256)
    p.add_argument(
        "--decoder-type",
        type=str,
        default="fpn_multiscale",
        choices=["fpn_multiscale", "tiny_single"],
        help="Semantic decoder on top of SAM features.",
    )
    p.add_argument("--freeze-backbone", action="store_true", default=False)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0, help="Periodic checkpoint interval; 0 disables.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--data-parallel", action="store_true", default=True)
    p.add_argument("--no-data-parallel", dest="data_parallel", action="store_false")
    p.add_argument("--train-log-interval", type=int, default=200)
    p.add_argument("--val-log-interval", type=int, default=200)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument(
        "--aug-preset",
        type=str,
        default="goldmdd_v2",
        choices=["goldmdd_v1", "goldmdd_v2", "none"],
    )

    # SAM 2.1
    p.add_argument("--sam2-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/third_party/sam2"))
    p.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--sam2-checkpoint", type=Path, default=None)
    p.add_argument("--sam2-image-size", type=int, default=512)

    # SAM 3
    p.add_argument("--sam3-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/third_party/sam3"))
    p.add_argument("--sam3-checkpoint", type=Path, default=None)
    p.add_argument("--sam3-load-from-hf", action="store_true", default=False)
    p.add_argument("--sam3-image-size", type=int, default=1008)

    # HQ-SAM
    p.add_argument("--hqsam-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/third_party/sam-hq"))
    p.add_argument("--hqsam-model-type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h", "vit_t"])
    p.add_argument("--hqsam-checkpoint", type=Path, default=None)

    return p.parse_args()


def _count_params(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        if isinstance(p, UninitializedParameter):
            continue
        try:
            n = int(p.numel())
        except (ValueError, RuntimeError):
            # Lazy params may still be uninitialized before the first forward.
            continue
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


class TinySegHead(nn.Module):
    def __init__(self, num_classes: int = NUM_FOREGROUND_CLASSES, mid_channels: int = 256) -> None:
        super().__init__()
        self.proj = nn.LazyConv2d(mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.1)
        self.cls = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(feat, list):
            feat = feat[-1]
        x = self.proj(feat)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return self.cls(x)


class MultiScaleFPNHead(nn.Module):
    """Lightweight FPN-style decoder for semantic segmentation."""

    def __init__(
        self,
        num_classes: int = NUM_FOREGROUND_CLASSES,
        mid_channels: int = 256,
        max_feature_levels: int = 4,
    ) -> None:
        super().__init__()
        self.max_feature_levels = int(max_feature_levels)
        self.lateral = nn.ModuleList(
            [nn.LazyConv2d(mid_channels, kernel_size=1, bias=False) for _ in range(self.max_feature_levels)]
        )
        self.lateral_bn = nn.ModuleList([nn.BatchNorm2d(mid_channels) for _ in range(self.max_feature_levels)])
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.max_feature_levels)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        self.cls = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, feats: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        if not feats:
            raise RuntimeError("MultiScaleFPNHead received no features.")

        # Keep the lowest-resolution branch and up to two higher-resolution branches.
        feats = feats[-self.max_feature_levels :]
        n_levels = len(feats)

        laterals: list[torch.Tensor] = []
        for i, f in enumerate(feats):
            x = self.lateral[i](f)
            x = self.lateral_bn[i](x)
            x = F.relu(x, inplace=True)
            laterals.append(x)

        # Top-down FPN fusion (feats are ordered high-res -> low-res).
        outs: list[torch.Tensor] = [laterals[-1]]
        cur = laterals[-1]
        for i in range(n_levels - 2, -1, -1):
            cur = F.interpolate(cur, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False) + laterals[i]
            outs.append(cur)
        outs = list(reversed(outs))

        for i in range(n_levels):
            outs[i] = self.smooth[i](outs[i])

        base_hw = outs[0].shape[-2:]
        fused = outs[0]
        for o in outs[1:]:
            if o.shape[-2:] != base_hw:
                o = F.interpolate(o, size=base_hw, mode="bilinear", align_corners=False)
            fused = fused + o
        fused = fused / float(n_levels)
        fused = self.fuse(fused)
        return self.cls(fused)


def _select_semantic_fpn_features(fpn_feats: list[torch.Tensor], max_levels: int = 3) -> list[torch.Tensor]:
    if not fpn_feats:
        raise RuntimeError("Empty backbone_fpn features.")
    # SAM orders FPN features from high-res to low-res; keep the coarsest levels used by SAM heads.
    return list(fpn_feats[-max_levels:])


class SAM21Backbone(nn.Module):
    def __init__(self, repo_root: Path, config_file: str, ckpt_path: Path | None, image_size: int) -> None:
        super().__init__()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from sam2.build_sam import build_sam2  # pylint: disable=import-error

        ckpt = str(ckpt_path) if ckpt_path is not None and ckpt_path.exists() else None
        overrides = [f"++model.image_size={int(image_size)}"]
        self.model = build_sam2(
            config_file=config_file,
            ckpt_path=ckpt,
            device="cpu",
            mode="train",
            hydra_overrides_extra=overrides,
            apply_postprocessing=False,
        )
        self.image_size = int(image_size)
        self.loaded_checkpoint = ckpt

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        out = self.model.forward_image(x)
        if "backbone_fpn" not in out or not out["backbone_fpn"]:
            raise RuntimeError("SAM2 backbone output missing 'backbone_fpn'.")
        return _select_semantic_fpn_features(out["backbone_fpn"], max_levels=3)


class SAM3Backbone(nn.Module):
    def __init__(
        self,
        repo_root: Path,
        ckpt_path: Path | None,
        load_from_hf: bool,
        image_size: int,
    ) -> None:
        super().__init__()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from sam3.model_builder import build_sam3_image_model  # pylint: disable=import-error

        ckpt = str(ckpt_path) if ckpt_path is not None and ckpt_path.exists() else None
        self.model = build_sam3_image_model(
            device="cpu",
            eval_mode=False,
            checkpoint_path=ckpt,
            load_from_HF=bool(load_from_hf),
            enable_segmentation=False,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.image_size = int(image_size)
        self.loaded_checkpoint = ckpt

    def _preprocess_for_sam3(self, x: torch.Tensor) -> torch.Tensor:
        # SAM3 default train configs use mean/std = 0.5.
        x01 = _denorm_imagenet(x)
        return (x01 - 0.5) / 0.5

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._preprocess_for_sam3(x)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        out = self.model.backbone.forward_image(x)
        if "backbone_fpn" not in out and "sam2_backbone_out" in out:
            out = out["sam2_backbone_out"]
        if "backbone_fpn" not in out or not out["backbone_fpn"]:
            raise RuntimeError("SAM3 backbone output missing 'backbone_fpn'.")
        return _select_semantic_fpn_features(out["backbone_fpn"], max_levels=3)


class HQSAMBackbone(nn.Module):
    def __init__(self, repo_root: Path, model_type: str, ckpt_path: Path | None) -> None:
        super().__init__()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from segment_anything import sam_model_registry  # pylint: disable=import-error

        ckpt = str(ckpt_path) if ckpt_path is not None and ckpt_path.exists() else None
        self.model = sam_model_registry[model_type](checkpoint=ckpt)
        self.image_size = int(self.model.image_encoder.img_size)
        self.loaded_checkpoint = ckpt

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = _denorm_imagenet(x) * 255.0
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = self.model.preprocess(x)
        enc = self.model.image_encoder(x)
        if isinstance(enc, tuple):
            feat = enc[0]
        elif isinstance(enc, dict):
            if "image_embeddings" in enc:
                feat = enc["image_embeddings"]
            elif "backbone_fpn" in enc and enc["backbone_fpn"]:
                return _select_semantic_fpn_features(enc["backbone_fpn"], max_levels=3)
            else:
                raise RuntimeError("Unsupported HQ-SAM encoder dict output.")
        else:
            feat = enc
        if not isinstance(feat, torch.Tensor):
            raise RuntimeError("Unsupported HQ-SAM feature type.")
        return [feat]


class SAMFamilySemantic(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        if args.family == "sam2_1":
            self.backbone = SAM21Backbone(
                repo_root=args.sam2_root,
                config_file=args.sam2_config,
                ckpt_path=args.sam2_checkpoint,
                image_size=args.sam2_image_size,
            )
            self.backbone_info = {
                "family": "sam2_1",
                "sam2_config": args.sam2_config,
                "sam2_checkpoint": self.backbone.loaded_checkpoint,
                "image_size": args.sam2_image_size,
            }
        elif args.family == "sam3":
            self.backbone = SAM3Backbone(
                repo_root=args.sam3_root,
                ckpt_path=args.sam3_checkpoint,
                load_from_hf=args.sam3_load_from_hf,
                image_size=args.sam3_image_size,
            )
            self.backbone_info = {
                "family": "sam3",
                "sam3_checkpoint": self.backbone.loaded_checkpoint,
                "sam3_load_from_hf": bool(args.sam3_load_from_hf),
                "image_size": args.sam3_image_size,
            }
        elif args.family == "hq_sam":
            self.backbone = HQSAMBackbone(
                repo_root=args.hqsam_root,
                model_type=args.hqsam_model_type,
                ckpt_path=args.hqsam_checkpoint,
            )
            self.backbone_info = {
                "family": "hq_sam",
                "hqsam_model_type": args.hqsam_model_type,
                "hqsam_checkpoint": self.backbone.loaded_checkpoint,
                "image_size": self.backbone.image_size,
            }
        else:  # pragma: no cover
            raise ValueError(args.family)

        if args.decoder_type == "tiny_single":
            self.decode_head = TinySegHead(num_classes=NUM_FOREGROUND_CLASSES, mid_channels=args.decoder_channels)
        else:
            self.decode_head = MultiScaleFPNHead(num_classes=NUM_FOREGROUND_CLASSES, mid_channels=args.decoder_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    criterion: CEDiceLoss,
    device: torch.device,
    amp: bool,
    epoch: int,
    epochs: int,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0
    n_batches = 0
    t0 = time.time()

    n_total = max(len(loader), 1)
    for i, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss, parts = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.detach().item())
        ce_sum += parts["ce"]
        dice_sum += parts["dice"]
        n_batches += 1

        if log_interval > 0 and (i % log_interval == 0 or i == n_total):
            dt = time.time() - t0
            it_s = dt / max(n_batches, 1)
            eta_s = max(n_total - i, 0) * it_s
            print(
                f"[{epoch:03d}/{epochs}] train {i}/{n_total} "
                f"loss={loss_sum/max(n_batches,1):.4f} "
                f"(ce={ce_sum/max(n_batches,1):.4f}, dice={dice_sum/max(n_batches,1):.4f}) "
                f"{it_s:.3f}s/it eta={eta_s/60.0:.1f}m",
                flush=True,
            )

    return {
        "loss": loss_sum / max(n_batches, 1),
        "ce": ce_sum / max(n_batches, 1),
        "dice": dice_sum / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: CEDiceLoss,
    device: torch.device,
    amp: bool,
    phase: str,
    log_interval: int,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0
    n_batches = 0
    conf = torch.zeros((NUM_FOREGROUND_CLASSES, NUM_FOREGROUND_CLASSES), dtype=torch.int64, device=device)
    t0 = time.time()

    n_total = max(len(loader), 1)
    for i, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss, parts = criterion(logits, y)
        loss_sum += float(loss.detach().item())
        ce_sum += parts["ce"]
        dice_sum += parts["dice"]
        n_batches += 1
        update_confusion(conf, logits, y)

        if log_interval > 0 and (i % log_interval == 0 or i == n_total):
            dt = time.time() - t0
            it_s = dt / max(n_batches, 1)
            eta_s = max(n_total - i, 0) * it_s
            print(
                f"[{phase}] {i}/{n_total} loss={loss_sum/max(n_batches,1):.4f} "
                f"(ce={ce_sum/max(n_batches,1):.4f}, dice={dice_sum/max(n_batches,1):.4f}) "
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
        "dice": dice_sum / max(n_batches, 1),
        "miou": miou,
        "miou_present": miou_present,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "oa_fg": oa_fg,
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
        "gt_pixels_per_class": gt_pixels_per_class,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_miou_present: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_miou_present": best_miou_present,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != "cuda":
        args.amp = False
    n_visible_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    train_samples = build_split_samples(args.data_root / "train")
    val_samples = build_split_samples(args.data_root / "val")
    test_samples = build_split_samples(args.data_root / "test")
    if args.limit_train > 0:
        train_samples = train_samples[: args.limit_train]
    if args.limit_val > 0:
        val_samples = val_samples[: args.limit_val]
    if args.limit_test > 0:
        test_samples = test_samples[: args.limit_test]

    train_ds = GoldMDDPatchDataset(train_samples, train=True, aug_preset=args.aug_preset)
    val_ds = GoldMDDPatchDataset(val_samples, train=False, aug_preset="none")
    test_ds = GoldMDDPatchDataset(test_samples, train=False, aug_preset="none")

    train_loader = make_loader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    run_dir = args.work_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = SAMFamilySemantic(args)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    model = model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]
    if args.data_parallel and device.type == "cuda" and n_visible_gpus > 1:
        model = nn.DataParallel(model)

    # Initialize lazy decode head params before optimizer/param accounting.
    try:
        x0, _, _ = next(iter(train_loader))
        x0 = x0[:1].to(device, non_blocking=True)
        model.eval()
        with torch.no_grad():
            _ = model(x0)
        model.train()
    except StopIteration:
        raise RuntimeError("Empty train split after filtering; cannot run training.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters after applying freeze settings.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    ce_class_weights = None
    if args.loss_mode == "weighted_ce_dice":
        ce_class_weights, class_counts = compute_train_class_weights(train_samples, power=args.class_weight_power)
        with (run_dir / "class_weights.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "power": args.class_weight_power,
                    "counts": class_counts,
                    "weights": [round(float(x), 8) for x in ce_class_weights.tolist()],
                },
                f,
                indent=2,
            )
        ce_class_weights = ce_class_weights.to(device)

    criterion = CEDiceLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        loss_mode=args.loss_mode,
        ce_class_weights=ce_class_weights,
        focal_gamma=args.focal_gamma,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    bare_model = unwrap_model(model)
    bb_total, bb_train = _count_params(bare_model.backbone)
    head_total, head_train = _count_params(bare_model.decode_head)
    total, trainable = _count_params(bare_model)

    args_dump = vars(args).copy()
    args_dump["backbone_info"] = bare_model.backbone_info
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(args_dump, f, indent=2, default=str)

    print(f"Device: {device}", flush=True)
    print(
        f"Visible GPUs: {n_visible_gpus} | DataParallel: {isinstance(model, nn.DataParallel)}\n"
        f"Run dir: {run_dir}\n"
        f"Train/Val/Test patches: {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}\n"
        f"Model family: {args.family} (classes={NUM_FOREGROUND_CLASSES}, bg ignored)\n"
        f"Decoder: {args.decoder_type}\n"
        f"Backbone info: {json.dumps(bare_model.backbone_info)}\n"
        f"Freeze backbone: {args.freeze_backbone}\n"
        f"Params (M): total={total/1e6:.3f}, trainable={trainable/1e6:.3f}, "
        f"backbone(trainable/total)={bb_train/1e6:.3f}/{bb_total/1e6:.3f}, "
        f"head(trainable/total)={head_train/1e6:.3f}/{head_total/1e6:.3f}\n"
        f"Loss mode: {args.loss_mode} (ce/focal={args.ce_weight}, dice={args.dice_weight})\n"
        f"Augmentation preset (train only): {args.aug_preset}",
        flush=True,
    )

    best_val_miou_present = -math.inf
    best_epoch = -1
    hist_rows: list[dict[str, object]] = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            amp=args.amp,
            epoch=epoch,
            epochs=args.epochs,
            log_interval=args.train_log_interval,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_dice": tr["dice"],
        }

        should_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if should_val:
            va = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp=args.amp,
                phase="val",
                log_interval=args.val_log_interval,
            )
            row.update(
                {
                    "val_loss": va["loss"],
                    "val_ce": va["ce"],
                    "val_dice": va["dice"],
                    "val_miou": va["miou"],
                    "val_miou_present": va["miou_present"],
                    "val_macro_f1": va["macro_f1"],
                    "val_macro_f1_present": va["macro_f1_present"],
                    "val_oa_fg": va["oa_fg"],
                }
            )
            if va["miou_present"] > best_val_miou_present:
                best_val_miou_present = float(va["miou_present"])
                best_epoch = epoch
                save_checkpoint(run_dir / "best.pth", model, optimizer, epoch, best_val_miou_present, args)

            print(
                f"[{epoch:03d}/{args.epochs}] "
                f"train_loss={tr['loss']:.4f} val_mIoU={va['miou']:.4f} val_mIoU_present={va['miou_present']:.4f}",
                flush=True,
            )
        else:
            print(f"[{epoch:03d}/{args.epochs}] train_loss={tr['loss']:.4f}", flush=True)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, best_val_miou_present, args)
        hist_rows.append(row)

    save_checkpoint(run_dir / "last.pth", model, optimizer, args.epochs, best_val_miou_present, args)

    best_path = run_dir / "best.pth"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(best_ckpt["model"], strict=True)

    te = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp=args.amp,
        phase="test",
        log_interval=args.val_log_interval,
    )

    with (run_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        if hist_rows:
            fieldnames = list(hist_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in hist_rows:
                w.writerow(r)

    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_miou_present": best_val_miou_present,
                "test_loss": te["loss"],
                "test_ce": te["ce"],
                "test_dice": te["dice"],
                "test_miou": te["miou"],
                "test_miou_present": te["miou_present"],
                "test_macro_f1": te["macro_f1"],
                "test_macro_f1_present": te["macro_f1_present"],
                "test_oa_fg": te["oa_fg"],
                "test_per_class_iou": te["per_class_iou"],
                "test_per_class_f1": te["per_class_f1"],
                "test_gt_pixels_per_class": te["gt_pixels_per_class"],
                "elapsed_sec": time.time() - t_start,
            },
            f,
            indent=2,
        )

    print(
        "Done.\n"
        f"Best epoch: {best_epoch} | best val_miou_present={best_val_miou_present:.4f}\n"
        f"Test mIoU={te['miou']:.4f} | Test mIoU_present={te['miou_present']:.4f} | "
        f"Test macro-F1={te['macro_f1']:.4f} | Test OA_fg={te['oa_fg']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
