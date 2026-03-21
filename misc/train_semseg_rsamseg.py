#!/usr/bin/env python3
"""
Train RSAM-Seg backbone for GoldMDD 14-class semantic segmentation.

This adapter keeps the shared GoldMDD protocol:
- same data split/layout and augmentation presets
- CE+Dice family losses with background ignored
- best checkpoint selected by val_miou_present
- unified val/test metrics (mIoU, macro-F1, OA_fg, per-class IoU/F1)

Original RSAM-Seg train.py is binary + DDP-only; this script provides
multiclass and single-GPU/DataParallel training.
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

from semseg_common import (
    CEDiceLoss,
    GoldMDDPatchDataset,
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
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/alqahtaniGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--work-dir", type=Path, default=Path("/deac/csc/alqahtaniGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--run-name", type=str, default="rsamseg_vit_b_baseline1_augv2_ce_dice")
    p.add_argument(
        "--rsam-root",
        type=Path,
        default=Path("/deac/csc/alqahtaniGrp/cuij/third_party/RSAM-Seg"),
        help="Path to cloned RSAM-Seg repo.",
    )
    p.add_argument(
        "--encoder-preset",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l"],
        help="Backbone preset from SAM-style ViT encoder.",
    )
    p.add_argument("--sam-checkpoint", type=Path, default=None, help="Optional SAM checkpoint for image encoder init.")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument(
        "--freeze-image-encoder",
        action="store_true",
        default=True,
        help="Match original RSAM-Seg: freeze image_encoder except prompt_generator params.",
    )
    p.add_argument("--no-freeze-image-encoder", dest="freeze_image_encoder", action="store_false")
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
    return p.parse_args()


def _build_encoder_cfg(preset: str, img_size: int = 512) -> dict:
    if preset == "vit_l":
        return {
            "img_size": img_size,
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attn_indexes": (5, 11, 17, 23),
            "out_chans": 256,
        }
    return {
        "img_size": img_size,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "use_rel_pos": True,
        "window_size": 14,
        "global_attn_indexes": (2, 5, 8, 11),
        "out_chans": 256,
    }


class RSAMSegSemantic(nn.Module):
    """RSAM image encoder + multiclass semantic head."""

    def __init__(self, rsam_root: Path, encoder_cfg: dict, num_classes: int = NUM_FOREGROUND_CLASSES) -> None:
        super().__init__()
        if str(rsam_root) not in sys.path:
            sys.path.insert(0, str(rsam_root))
        from models.sammodel import ImageEncoderViT  # pylint: disable=import-error

        self.image_encoder = ImageEncoderViT(
            img_size=encoder_cfg["img_size"],
            patch_size=encoder_cfg["patch_size"],
            in_chans=3,
            embed_dim=encoder_cfg["embed_dim"],
            depth=encoder_cfg["depth"],
            num_heads=encoder_cfg["num_heads"],
            mlp_ratio=encoder_cfg["mlp_ratio"],
            out_chans=encoder_cfg["out_chans"],
            qkv_bias=encoder_cfg["qkv_bias"],
            norm_layer=lambda c: nn.LayerNorm(c, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_cfg["use_rel_pos"],
            rel_pos_zero_init=True,
            window_size=encoder_cfg["window_size"],
            global_attn_indexes=encoder_cfg["global_attn_indexes"],
        )
        c = int(encoder_cfg["out_chans"])
        self.decode_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.image_encoder(x)
        logits = self.decode_head(feat)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def _extract_state_dict(ckpt_obj: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "network"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise RuntimeError("Unsupported checkpoint format")


def load_pretrained_encoder(model: RSAMSegSemantic, ckpt_path: Path) -> tuple[int, int]:
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(raw)
    enc_target = model.image_encoder.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        key = k[7:] if k.startswith("module.") else k
        candidates = [key]
        if key.startswith("image_encoder."):
            candidates.append(key[len("image_encoder.") :])
        for cand in candidates:
            if cand in enc_target and enc_target[cand].shape == v.shape:
                filtered[cand] = v
                break
    missing, unexpected = model.image_encoder.load_state_dict(filtered, strict=False)
    if unexpected:
        # Should be empty with filtered loading.
        raise RuntimeError(f"Unexpected keys while loading encoder: {unexpected[:8]}")
    return len(filtered), len(missing)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def apply_original_rsam_freeze_policy(model: RSAMSegSemantic) -> tuple[int, int]:
    """Freeze image encoder except prompt_generator parameters."""
    frozen_params = 0
    trainable_params = 0
    for name, param in model.image_encoder.named_parameters():
        keep_trainable = "prompt_generator" in name
        param.requires_grad_(keep_trainable)
        if keep_trainable:
            trainable_params += int(param.numel())
        else:
            frozen_params += int(param.numel())
    return frozen_params, trainable_params


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
                f"loss={loss_sum/max(n_batches,1):.4f} (ce={ce_sum/max(n_batches,1):.4f}, dice={dice_sum/max(n_batches,1):.4f}) "
                f"{it_s:.3f}s/it eta={eta_s/60.0:.1f}m",
                flush=True,
            )

    return {
        "loss": loss_sum / max(n_batches, 1),
        "ce": ce_sum / max(n_batches, 1),
        "dice": dice_sum / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: CEDiceLoss, device: torch.device, amp: bool, phase: str, log_interval: int) -> dict[str, object]:
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


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_miou: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_miou": best_miou,
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
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    encoder_cfg = _build_encoder_cfg(args.encoder_preset, img_size=512)
    model = RSAMSegSemantic(rsam_root=args.rsam_root, encoder_cfg=encoder_cfg, num_classes=NUM_FOREGROUND_CLASSES)
    if args.pretrained and args.sam_checkpoint is not None and args.sam_checkpoint.exists():
        loaded, missing = load_pretrained_encoder(model, args.sam_checkpoint)
        print(f"Loaded encoder checkpoint: {args.sam_checkpoint} | loaded={loaded} missing={missing}", flush=True)
    elif args.pretrained and args.sam_checkpoint is not None:
        print(f"WARNING: checkpoint not found, training from scratch: {args.sam_checkpoint}", flush=True)
    else:
        print("Pretrained encoder disabled; training from scratch.", flush=True)

    frozen_params = 0
    prompt_trainable_params = 0
    if args.freeze_image_encoder:
        frozen_params, prompt_trainable_params = apply_original_rsam_freeze_policy(model)
        if args.pretrained and args.sam_checkpoint is None:
            print("WARNING: freeze policy enabled but no SAM checkpoint provided.", flush=True)
    else:
        for param in model.image_encoder.parameters():
            param.requires_grad_(True)

    model = model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]
    if args.data_parallel and device.type == "cuda" and n_visible_gpus > 1:
        model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after applying freeze policy.")
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

    print(f"Device: {device}", flush=True)
    print(
        f"Visible GPUs: {n_visible_gpus} | DataParallel: {isinstance(model, nn.DataParallel)}\n"
        f"Run dir: {run_dir}\n"
        f"Train/Val/Test patches: {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}\n"
        f"Model: RSAM-Seg-{args.encoder_preset} (classes={NUM_FOREGROUND_CLASSES}, bg ignored)\n"
        f"Freeze policy: {'image_encoder frozen except prompt_generator' if args.freeze_image_encoder else 'no image_encoder freezing'}\n"
        f"Frozen encoder params: {frozen_params:,} | Prompt-generator trainable params: {prompt_trainable_params:,}\n"
        f"Loss mode: {args.loss_mode} (ce/focal weight={args.ce_weight}, dice weight={args.dice_weight})\n"
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
        best_ckpt = torch.load(best_path, map_location=device)
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
