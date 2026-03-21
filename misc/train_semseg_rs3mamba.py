#!/usr/bin/env python3
"""
Train semantic segmentation on GoldMDD cropped patches with RS3Mamba.

Assumptions for GoldMDD/data-cropped:
  split/image/*.jpg
  split/label/*.png

Labels use GoldMDD merged IDs: 0=background, 1..14=foreground classes.
This script trains a 14-class model (foreground only) and ignores background
in both CE and Dice losses by remapping:
  label 0   -> ignore_index (255)
  label 1-14 -> 0-13
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from semseg_common import (
    CEDiceLoss,
    GoldMDDPatchDataset,
    IGNORE_INDEX,
    NUM_FOREGROUND_CLASSES,
    build_split_samples,
    compute_metrics_from_conf,
    compute_train_class_weights,
    make_loader,
    set_seed,
    update_confusion,
)


def add_rs3mamba_repo_to_syspath(repo: Path) -> None:
    repo = repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"RS3Mamba repo not found: {repo}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--work-dir", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--rs3mamba-repo",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/third_party/SSRS/RS3Mamba"),
        help="Local clone of RS3Mamba from https://github.com/sstary/SSRS/tree/main/RS3Mamba",
    )
    p.add_argument(
        "--vmamba-pretrained",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/third_party/SSRS/RS3Mamba/pretrain/vmamba_tiny_e292.pth"),
        help="VMamba-Tiny pretrained checkpoint for RS3Mamba encoder. Set to '' to disable.",
    )
    p.add_argument("--epochs", type=int, default=60)
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
        help="Foreground loss composition. Dice is always included; CE/Focal ignores background.",
    )
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="For weighted_ce_dice: weight ~ 1 / (freq ^ power). 0.5 = inverse sqrt.",
    )
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0, help="Periodic epoch checkpoint interval; 0 disables (best/last are always saved)")
    p.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    p.add_argument("--data-parallel", action="store_true", default=True, help="Use nn.DataParallel when multiple GPUs are visible")
    p.add_argument("--no-data-parallel", dest="data_parallel", action="store_false")
    p.add_argument("--train-log-interval", type=int, default=200, help="Print train progress every N batches (0=off)")
    p.add_argument("--val-log-interval", type=int, default=200, help="Print val/test progress every N batches (0=off)")
    p.add_argument("--limit-train", type=int, default=0, help="Debug: limit number of train samples (0=all)")
    p.add_argument("--limit-val", type=int, default=0, help="Debug: limit number of val samples (0=all)")
    p.add_argument("--limit-test", type=int, default=0, help="Debug: limit number of test samples (0=all)")
    p.add_argument(
        "--aug-preset",
        type=str,
        default="goldmdd_v1",
        choices=["goldmdd_v1", "goldmdd_v2", "none"],
        help="Training augmentation preset; keep fixed across model comparisons.",
    )
    return p.parse_args()


def build_model(rs3mamba_repo: Path, vmamba_pretrained: Path | None = None) -> nn.Module:
    add_rs3mamba_repo_to_syspath(rs3mamba_repo)
    from model.RS3Mamba import RS3Mamba, load_pretrained_ckpt

    model = RS3Mamba(num_classes=NUM_FOREGROUND_CLASSES, pretrained=True)
    if vmamba_pretrained is not None:
        vmamba_pretrained = vmamba_pretrained.expanduser().resolve()
        if not vmamba_pretrained.exists():
            raise FileNotFoundError(f"--vmamba-pretrained not found: {vmamba_pretrained}")
        model = load_pretrained_ckpt(model, str(vmamba_pretrained))
        print(f"Loaded VMamba pretrained encoder: {vmamba_pretrained}", flush=True)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    criterion: CEDiceLoss,
    device: torch.device,
    amp: bool,
    epoch: int,
    epochs: int,
    log_interval: int = 0,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0
    n_batches = 0
    t0 = time.time()

    num_batches = max(len(loader), 1)
    for batch_idx, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
            if isinstance(logits, dict):  # torchvision style fallback, not expected for SMP
                logits = logits["out"]
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

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            dt = time.time() - t0
            avg_loss = loss_sum / max(n_batches, 1)
            avg_ce = ce_sum / max(n_batches, 1)
            avg_dice = dice_sum / max(n_batches, 1)
            it_s = dt / max(n_batches, 1)
            eta_s = max(num_batches - batch_idx, 0) * it_s
            print(
                f"[{epoch:03d}/{epochs}] train {batch_idx}/{num_batches} "
                f"loss={avg_loss:.4f} (ce={avg_ce:.4f}, dice={avg_dice:.4f}) "
                f"{it_s:.3f}s/it eta={eta_s/60.0:.1f}m",
                flush=True,
            )

    dt = time.time() - t0
    return {
        "loss": loss_sum / max(n_batches, 1),
        "ce": ce_sum / max(n_batches, 1),
        "dice": dice_sum / max(n_batches, 1),
        "sec": dt,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CEDiceLoss,
    device: torch.device,
    amp: bool,
    phase: str = "val",
    log_interval: int = 0,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0
    n_batches = 0
    conf = torch.zeros((NUM_FOREGROUND_CLASSES, NUM_FOREGROUND_CLASSES), dtype=torch.int64, device=device)
    t0 = time.time()

    num_batches = max(len(loader), 1)
    for batch_idx, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
            if isinstance(logits, dict):
                logits = logits["out"]
            loss, parts = criterion(logits, y)
        loss_sum += float(loss.detach().item())
        ce_sum += parts["ce"]
        dice_sum += parts["dice"]
        n_batches += 1
        update_confusion(conf, logits, y)

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            dt = time.time() - t0
            it_s = dt / max(n_batches, 1)
            eta_s = max(num_batches - batch_idx, 0) * it_s
            print(
                f"[{phase}] {batch_idx}/{num_batches} "
                f"loss={loss_sum / max(n_batches, 1):.4f} "
                f"(ce={ce_sum / max(n_batches, 1):.4f}, dice={dice_sum / max(n_batches, 1):.4f}) "
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
    dt = time.time() - t0
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
        "sec": dt,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_miou: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
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
    if isinstance(args.vmamba_pretrained, Path) and str(args.vmamba_pretrained) == "":
        args.vmamba_pretrained = None
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

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)

    model = build_model(args.rs3mamba_repo, args.vmamba_pretrained)
    model.to(device)
    if args.data_parallel and device.type == "cuda" and n_visible_gpus > 1:
        if args.compile:
            print("Disabling --compile because DataParallel is enabled", flush=True)
            args.compile = False
        model = nn.DataParallel(model)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]

    ce_class_weights = None
    train_class_counts: list[int] | None = None
    if args.loss_mode == "weighted_ce_dice":
        ce_class_weights, train_class_counts = compute_train_class_weights(train_samples, power=args.class_weight_power)
    criterion = CEDiceLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        loss_mode=args.loss_mode,
        ce_class_weights=ce_class_weights,
        focal_gamma=args.focal_gamma,
    )
    if ce_class_weights is not None:
        criterion = criterion.to(device)
    else:
        criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule over total steps (simple and robust baseline).
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = max(int(0.03 * total_steps), 100)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        t = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    run_name = args.run_name or "rs3mamba_r18_vmamba_tiny"
    run_dir = args.work_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_csv = run_dir / "train_log.csv"
    config_json = run_dir / "config.json"
    config_json.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Visible GPUs: {n_visible_gpus} | DataParallel: {isinstance(model, nn.DataParallel)}", flush=True)
    print(f"Run dir: {run_dir}")
    print(f"Train/Val/Test patches: {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}")
    print(f"Model: RS3Mamba (ResNet-18 + VMamba-Tiny, classes={NUM_FOREGROUND_CLASSES}, bg ignored)")
    print(f"Augmentation preset (train only): {args.aug_preset}", flush=True)
    print(f"Loss mode: {args.loss_mode} (ce/focal weight={args.ce_weight}, dice weight={args.dice_weight})", flush=True)
    if ce_class_weights is not None:
        cw_path = run_dir / "train_class_weights.json"
        cw_payload = {
            "power": args.class_weight_power,
            "train_class_counts_foreground_1_to_14": train_class_counts,
            "class_weights_for_model_targets_0_to_13": [float(x) for x in ce_class_weights.tolist()],
        }
        cw_path.write_text(json.dumps(cw_payload, indent=2), encoding="utf-8")
        print(f"Saved class weights to {cw_path}", flush=True)

    best_miou = -1.0
    global_step = 0
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_ce",
                "train_dice",
                "train_sec",
                "val_loss",
                "val_ce",
                "val_dice",
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
                criterion,
                device,
                args.amp,
                epoch=epoch,
                epochs=args.epochs,
                log_interval=args.train_log_interval,
            )
            for _ in range(len(train_loader)):
                scheduler.step()
                global_step += 1

            do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
            val_stats = {
                "loss": float("nan"),
                "ce": float("nan"),
                "dice": float("nan"),
                "miou": float("nan"),
                "miou_present": float("nan"),
                "macro_f1": float("nan"),
                "macro_f1_present": float("nan"),
                "oa_fg": float("nan"),
                "sec": 0.0,
            }
            if do_val:
                val_stats = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    args.amp,
                    phase=f"val e{epoch:03d}",
                    log_interval=args.val_log_interval,
                )
                if val_stats["miou"] > best_miou:
                    best_miou = float(val_stats["miou"])
                    save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best_miou, args)
                    (run_dir / "best_val_per_class_iou.json").write_text(
                        json.dumps(val_stats["per_class_iou"], indent=2), encoding="utf-8"
                    )

            if args.save_every > 0 and (epoch % args.save_every == 0):
                save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, best_miou, args)
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best_miou, args)

            lr_now = optimizer.param_groups[0]["lr"]
            w.writerow(
                [
                    epoch,
                    f"{lr_now:.8e}",
                    f"{train_stats['loss']:.6f}",
                    f"{train_stats['ce']:.6f}",
                    f"{train_stats['dice']:.6f}",
                    f"{train_stats['sec']:.2f}",
                    f"{val_stats['loss']:.6f}" if do_val else "",
                    f"{val_stats['ce']:.6f}" if do_val else "",
                    f"{val_stats['dice']:.6f}" if do_val else "",
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
                f"[{epoch:03d}/{args.epochs}] lr={lr_now:.2e} "
                f"train loss={train_stats['loss']:.4f} (ce={train_stats['ce']:.4f}, dice={train_stats['dice']:.4f}) "
            )
            if do_val:
                msg += (
                    f"| val loss={val_stats['loss']:.4f} "
                    f"miou={val_stats['miou']:.4f} "
                    f"miou_present={val_stats['miou_present']:.4f} "
                    f"f1_present={val_stats['macro_f1_present']:.4f} "
                    f"oa_fg={val_stats['oa_fg']:.4f} "
                    f"(best={best_miou:.4f})"
                )
            print(msg, flush=True)

    # Final test evaluation with best checkpoint if available.
    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')} for test eval", flush=True)
    test_stats = evaluate(model, test_loader, criterion, device, args.amp, phase="test", log_interval=args.val_log_interval)
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
                "dice": test_stats["dice"],
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
        f"f1_present={test_stats['macro_f1_present']:.4f}, "
        f"oa_fg={test_stats['oa_fg']:.4f}. "
        f"Saved metrics to {run_dir / 'test_metrics.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
