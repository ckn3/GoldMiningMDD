#!/usr/bin/env python3
"""
Train FarSeg / FarSeg++ on GoldMDD cropped patches.

Labels in GoldMDD:
  0 = background (ignored for training/metrics)
  1..14 = foreground classes

This script reuses GoldMDD split protocol and evaluates with the same
foreground-only metrics as other baselines (best checkpoint by val_miou_present).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

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


GOLDMDD_IGNORE_FOR_FARSEG = 14


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="farseg", choices=["farseg", "farsegpp"])
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--work-dir", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--farseg-repo", type=Path, default=Path("/deac/csc/yangGrp/cuij/third_party/FarSeg"))

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adamw", None])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--poly-power", type=float, default=0.9)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument(
        "--loss-mode",
        type=str,
        default="native_ce",
        choices=["native_ce", "ce_dice", "weighted_ce_dice", "focal_dice"],
        help="native_ce uses official FarSeg losses; others use GoldMDD CE/Focal+Dice protocol.",
    )
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="For weighted_ce_dice: weight ~ 1 / (freq ^ power). 0.5 = inverse sqrt.",
    )
    p.add_argument(
        "--farsegpp-backbone",
        type=str,
        default="mit_b2",
        choices=["mit_b2", "resnet50"],
        help="FarSeg++ backbone; default follows official FarSeg++ config (MiT-B2).",
    )

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--data-parallel", action="store_true", default=True)
    p.add_argument("--no-data-parallel", dest="data_parallel", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--aug-preset", type=str, default="goldmdd_v2", choices=["goldmdd_v1", "goldmdd_v2", "none"])

    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0, help="Periodic save interval (0 disables; best/last always kept)")
    p.add_argument("--train-log-interval", type=int, default=200)
    p.add_argument("--val-log-interval", type=int, default=200)

    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    return p.parse_args()


def _to_farseg_target(y: torch.Tensor) -> torch.Tensor:
    # semseg_common maps background->255; FarSeg configs use ignore_index=14.
    out = y.clone()
    out[out == IGNORE_INDEX] = GOLDMDD_IGNORE_FOR_FARSEG
    return out


def _sum_loss_dict(loss_dict: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    parts: dict[str, float] = {}
    total = None
    for k, v in loss_dict.items():
        if not k.endswith("loss"):
            continue
        if not torch.is_tensor(v):
            continue
        total = v if total is None else total + v
        parts[k] = float(v.detach().item())
    if total is None:
        raise RuntimeError(f"No loss terms found in model output keys: {list(loss_dict.keys())}")
    return total, parts


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward FarSeg backbone/head to raw logits regardless of model.train()."""
    m = unwrap_model(model)
    if not hasattr(m, "en") or not hasattr(m, "fpn") or not hasattr(m, "decoder"):
        raise RuntimeError("Custom CE/Dice is currently only supported for FarSeg architecture.")

    feat_list = m.en(x)
    fpn_feat_list = m.fpn(feat_list)
    if "scene_relation" in m.config:
        c5 = feat_list[-1]
        c6 = m.gap(c5)
        refined_fpn_feat_list = m.sr(c6, fpn_feat_list)
    else:
        refined_fpn_feat_list = fpn_feat_list

    final_feat = m.decoder(refined_fpn_feat_list)
    cls_pred = m.cls_pred_conv(final_feat)
    cls_pred = m.upsample4x_op(cls_pred)
    return cls_pred


def build_model(args: argparse.Namespace) -> nn.Module:
    import sys

    repo = args.farseg_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from simplecv.core.config import AttrDict

    if args.model == "farseg":
        from module.farseg import FarSeg
        from simplecv.module import fpn

        cfg = AttrDict.from_dict(
            dict(
                resnet_encoder=dict(
                    resnet_type="resnet50",
                    include_conv5=True,
                    batchnorm_trainable=True,
                    pretrained=bool(args.pretrained),
                    freeze_at=0,
                    output_stride=32,
                    with_cp=(False, False, False, False),
                    stem3_3x3=False,
                ),
                fpn=dict(
                    in_channels_list=(256, 512, 1024, 2048),
                    out_channels=256,
                    conv_block=fpn.default_conv_block,
                    top_blocks=None,
                ),
                scene_relation=dict(
                    in_channels=2048,
                    channel_list=(256, 256, 256, 256),
                    out_channels=256,
                    scale_aware_proj=True,
                ),
                decoder=dict(
                    in_channels=256,
                    out_channels=128,
                    in_feat_output_strides=(4, 8, 16, 32),
                    out_feat_output_stride=4,
                    norm_fn=nn.BatchNorm2d,
                    num_groups_gn=None,
                ),
                num_classes=NUM_FOREGROUND_CLASSES,
                loss=dict(
                    cls_weight=1.0,
                    ignore_index=GOLDMDD_IGNORE_FOR_FARSEG,
                ),
                annealing_softmax_focalloss=dict(
                    gamma=2.0,
                    max_step=10000,
                    annealing_type="cosine",
                ),
            )
        )
        return FarSeg(cfg)

    # farsegpp
    from module.farsegpp import FarSegPP

    if args.farsegpp_backbone == "mit_b2":
        backbone_cfg = dict(
            type="mit",
            name="mit_b2",
            # MiT encoder loads only when `pretrained` is a checkpoint path string.
            # Keep boolean to avoid hard dependency on an external local checkpoint path.
            pretrained=bool(args.pretrained),
            drop_path_rate=0.1,
        )
        ppm_in = 512
        ppm_pool = 128
        ppm_out = 128
        fpn_in = (64, 128, 320, 128)
        fs_scene = 512
    else:
        backbone_cfg = dict(
            type="resnet",
            in_channels=3,
            resnet_type="resnet50_v1c",
            batchnorm_trainable=True,
            pretrained=bool(args.pretrained),
            freeze_at=0,
            output_stride=32,
        )
        ppm_in = 2048
        ppm_pool = 512
        ppm_out = 512
        fpn_in = (256, 512, 1024, 512)
        fs_scene = 2048

    cfg = AttrDict.from_dict(
        dict(
            backbone=backbone_cfg,
            ppm=dict(
                in_channels=ppm_in,
                pool_channels=ppm_pool,
                out_channels=ppm_out,
                bins=(1, 2, 3, 6),
                bottleneck_conv="1x1",
                dropout=0.1,
            ),
            fpn=dict(
                in_channels_list=fpn_in,
                out_channels=256,
            ),
            fs_relation=dict(
                scene_embedding_channels=fs_scene,
                in_channels_list=(256, 256, 256, 256),
                out_channels=256,
                scale_aware_proj=True,
            ),
            decoder_arch="SegObjCascadeDecoder",
            obj_asy_decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 4, 8, 16, 32),
                out_feat_output_stride=4,
                classifier_config=dict(scale_factor=4.0, num_classes=1, kernel_size=3),
            ),
            asy_decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                classifier_config=dict(scale_factor=4.0, num_classes=NUM_FOREGROUND_CLASSES, kernel_size=3),
            ),
            loss=dict(
                objectness=dict(
                    log_objectness_iou_sigmoid=dict(),
                    dice=dict(),
                    bce=dict(),
                    ignore_index=GOLDMDD_IGNORE_FOR_FARSEG,
                    prefix="obj_",
                ),
                semantic=dict(
                    annealing_softmax_focal=dict(normalize=True, t_max=0),
                    log_objectness_iou=dict(),
                    ignore_index=GOLDMDD_IGNORE_FOR_FARSEG,
                ),
            ),
        )
    )
    return FarSegPP(cfg)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = args.optimizer
    if opt_name is None:
        opt_name = "sgd" if args.model == "farseg" else "adamw"

    if args.lr is None:
        lr = 7e-3 if args.model == "farseg" else 6e-5
    else:
        lr = float(args.lr)

    if args.weight_decay is None:
        weight_decay = 1e-4 if args.model == "farseg" else 1e-2
    else:
        weight_decay = float(args.weight_decay)

    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(opt_name)


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if args.model == "farseg":
        if args.loss_mode == "native_ce":
            return "farseg_r50_native"
        return f"farseg_r50_{args.loss_mode}"
    if args.farsegpp_backbone == "mit_b2":
        if args.loss_mode == "native_ce":
            return "farsegpp_mitb2_native"
        return f"farsegpp_mitb2_{args.loss_mode}"
    if args.loss_mode == "native_ce":
        return "farsegpp_r50_native"
    return f"farsegpp_r50_{args.loss_mode}"


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    criterion: CEDiceLoss | None,
    loss_mode: str,
    device: torch.device,
    amp: bool,
    epoch: int,
    epochs: int,
    log_interval: int = 0,
) -> dict[str, Any]:
    model.train()
    loss_sum = 0.0
    n_batches = 0
    part_sum: dict[str, float] = {}
    t0 = time.time()
    num_batches = max(1, len(loader))

    for bi, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            if loss_mode == "native_ce":
                y_farseg = _to_farseg_target(y)
                out = model(x, {"cls": y_farseg})
                loss, parts = _sum_loss_dict(out)
            else:
                if criterion is None:
                    raise RuntimeError("criterion is required for non-native loss modes")
                logits = _forward_logits(model, x)
                loss, parts = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        loss_sum += float(loss.detach().item())
        n_batches += 1
        for k, v in parts.items():
            part_sum[k] = part_sum.get(k, 0.0) + float(v)

        if log_interval > 0 and (bi % log_interval == 0 or bi == num_batches):
            dt = time.time() - t0
            it_s = dt / max(1, n_batches)
            eta_m = (num_batches - bi) * it_s / 60.0
            avg = loss_sum / max(1, n_batches)
            lr = optimizer.param_groups[0]["lr"]
            print(f"[{epoch:03d}/{epochs}] train {bi}/{num_batches} loss={avg:.4f} lr={lr:.3e} {it_s:.3f}s/it eta={eta_m:.1f}m", flush=True)

    avg_parts = {k: v / max(1, n_batches) for k, v in part_sum.items()}
    return {"loss": loss_sum / max(1, n_batches), "parts": avg_parts, "sec": time.time() - t0}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
    phase: str = "val",
    log_interval: int = 0,
) -> dict[str, Any]:
    model.eval()
    conf = torch.zeros((NUM_FOREGROUND_CLASSES, NUM_FOREGROUND_CLASSES), dtype=torch.int64, device=device)
    t0 = time.time()
    num_batches = max(1, len(loader))

    for bi, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            pred = model(x)
        update_confusion(conf, pred, y, ignore_index=IGNORE_INDEX)

        if log_interval > 0 and (bi % log_interval == 0 or bi == num_batches):
            dt = time.time() - t0
            it_s = dt / max(1, bi)
            eta_m = (num_batches - bi) * it_s / 60.0
            print(f"[{phase}] {bi}/{num_batches} {it_s:.3f}s/it eta={eta_m:.1f}m", flush=True)

    miou, miou_present, macro_f1, macro_f1_present, oa_fg, per_iou, per_f1, gt_pixels = compute_metrics_from_conf(conf)
    return {
        "miou": miou,
        "miou_present": miou_present,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "oa_fg": oa_fg,
        "per_class_iou": per_iou,
        "per_class_f1": per_f1,
        "gt_pixels_per_class": gt_pixels,
        "sec": time.time() - t0,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_miou: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_miou_present": best_miou,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = build_model(args).to(device)

    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if args.data_parallel and n_gpu > 1:
        model = nn.DataParallel(model)

    criterion: CEDiceLoss | None = None
    ce_class_weights = None
    train_class_counts: list[int] | None = None
    if args.loss_mode != "native_ce":
        if args.loss_mode == "weighted_ce_dice":
            ce_class_weights, train_class_counts = compute_train_class_weights(
                train_samples, power=args.class_weight_power
            )
        criterion = CEDiceLoss(
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            loss_mode=args.loss_mode,
            ce_class_weights=ce_class_weights,
            focal_gamma=args.focal_gamma,
        ).to(device)

    optimizer = build_optimizer(args, unwrap_model(model))
    total_iters = max(1, args.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda it: max((1.0 - float(it) / float(total_iters)) ** args.poly_power, 0.0),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    run_name = build_run_name(args)
    run_dir = args.work_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_csv = run_dir / "history.csv"
    best_ckpt = run_dir / "best.pth"
    last_ckpt = run_dir / "last.pth"

    print(f"Device: {device}", flush=True)
    print(f"Visible GPUs: {n_gpu} | DataParallel: {isinstance(model, nn.DataParallel)}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Train/Val/Test patches: {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}", flush=True)
    print(f"Model: {args.model} (classes={NUM_FOREGROUND_CLASSES}, bg ignored)", flush=True)
    print(f"Loss mode: {args.loss_mode}", flush=True)
    if ce_class_weights is not None:
        cw_path = run_dir / "train_class_weights.json"
        cw_payload = {
            "power": args.class_weight_power,
            "train_class_counts_foreground_1_to_14": train_class_counts,
            "class_weights_for_model_targets_0_to_13": [float(x) for x in ce_class_weights.tolist()],
        }
        cw_path.write_text(json.dumps(cw_payload, indent=2), encoding="utf-8")
        print(f"Saved class weights to {cw_path}", flush=True)

    best_val_miou_present = -1.0
    best_epoch = -1
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            args.loss_mode,
            device,
            args.amp,
            epoch,
            args.epochs,
            log_interval=args.train_log_interval,
        )

        val_stats = None
        if epoch % args.val_every == 0:
            val_stats = evaluate(
                model,
                val_loader,
                device,
                args.amp,
                phase="val",
                log_interval=args.val_log_interval,
            )
            if val_stats["miou_present"] > best_val_miou_present:
                best_val_miou_present = float(val_stats["miou_present"])
                best_epoch = epoch
                save_checkpoint(best_ckpt, model, optimizer, epoch, best_val_miou_present, args)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, best_val_miou_present, args)

        save_checkpoint(last_ckpt, model, optimizer, epoch, best_val_miou_present, args)

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_sec": train_stats["sec"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        for k, v in train_stats["parts"].items():
            row[f"train_{k}"] = v

        if val_stats is not None:
            row.update(
                {
                    "val_miou": val_stats["miou"],
                    "val_miou_present": val_stats["miou_present"],
                    "val_macro_f1": val_stats["macro_f1"],
                    "val_macro_f1_present": val_stats["macro_f1_present"],
                    "val_oa_fg": val_stats["oa_fg"],
                    "val_sec": val_stats["sec"],
                }
            )
            print(
                f"[{epoch:03d}/{args.epochs}] train_loss={train_stats['loss']:.4f} "
                f"val_mIoU_present={val_stats['miou_present']:.4f} val_mIoU={val_stats['miou']:.4f}",
                flush=True,
            )
        else:
            print(f"[{epoch:03d}/{args.epochs}] train_loss={train_stats['loss']:.4f}", flush=True)

        history_rows.append(row)
        keys = sorted({k for r in history_rows for k in r.keys()})
        with history_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in history_rows:
                w.writerow(r)

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)

    test_stats = evaluate(model, test_loader, device, args.amp, phase="test", log_interval=args.val_log_interval)

    summary = {
        "model": args.model,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_miou_present": best_val_miou_present,
        "test_miou": test_stats["miou"],
        "test_miou_present": test_stats["miou_present"],
        "test_macro_f1": test_stats["macro_f1"],
        "test_macro_f1_present": test_stats["macro_f1_present"],
        "test_oa_fg": test_stats["oa_fg"],
        "test_per_class_iou": test_stats["per_class_iou"],
        "test_per_class_f1": test_stats["per_class_f1"],
        "test_gt_pixels_per_class": test_stats["gt_pixels_per_class"],
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"BEST val_mIoU_present={best_val_miou_present:.4f} @ epoch {best_epoch} | "
        f"TEST mIoU_present={test_stats['miou_present']:.4f} mIoU={test_stats['miou']:.4f} "
        f"macroF1_present={test_stats['macro_f1_present']:.4f} OA_fg={test_stats['oa_fg']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
