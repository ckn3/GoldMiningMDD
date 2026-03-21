#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_semseg_sam_family import (
    CEDiceLoss,
    GoldMDDPatchDataset,
    SAMFamilySemantic,
    build_split_samples,
    compute_train_class_weights,
    make_loader,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAM-family best checkpoint on val split and export val_metrics_best.json.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args()


def _build_args_ns(args_json: dict) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in args_json.items():
        setattr(ns, k, v)
    # Cast known path-like fields.
    for k in ["data_root", "work_dir", "sam2_root", "sam2_checkpoint", "sam3_root", "sam3_checkpoint", "hqsam_root", "hqsam_checkpoint"]:
        if hasattr(ns, k):
            val = getattr(ns, k)
            if isinstance(val, str) and val:
                setattr(ns, k, Path(val))
            elif val is None:
                setattr(ns, k, None)
    return ns


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    args_path = run_dir / "args.json"
    best_path = run_dir / "best.pth"
    out_path = run_dir / "val_metrics_best.json"

    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json: {args_path}")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best_path}")

    run_args_raw = json.loads(args_path.read_text(encoding="utf-8"))
    run_args = _build_args_ns(run_args_raw)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    model = SAMFamilySemantic(run_args).to(device)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    val_samples = build_split_samples(Path(run_args.data_root) / "val")
    if int(getattr(run_args, "limit_val", 0)) > 0:
        val_samples = val_samples[: int(run_args.limit_val)]
    val_ds = GoldMDDPatchDataset(val_samples, train=False, aug_preset="none")
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    ce_class_weights = None
    if getattr(run_args, "loss_mode", "ce_dice") == "weighted_ce_dice":
        ce_class_weights, _ = compute_train_class_weights(
            build_split_samples(Path(run_args.data_root) / "train"),
            power=float(getattr(run_args, "class_weight_power", 0.5)),
        )
        ce_class_weights = ce_class_weights.to(device)

    criterion = CEDiceLoss(
        ce_weight=float(getattr(run_args, "ce_weight", 1.0)),
        dice_weight=float(getattr(run_args, "dice_weight", 1.0)),
        loss_mode=getattr(run_args, "loss_mode", "ce_dice"),
        ce_class_weights=ce_class_weights,
        focal_gamma=float(getattr(run_args, "focal_gamma", 2.0)),
    )

    stats = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        amp=amp,
        phase="val",
        log_interval=0,
    )

    out = {
        "loss": float(stats["loss"]),
        "ce": float(stats["ce"]),
        "dice": float(stats["dice"]),
        "miou": float(stats["miou"]),
        "miou_present": float(stats["miou_present"]),
        "macro_f1": float(stats["macro_f1"]),
        "macro_f1_present": float(stats["macro_f1_present"]),
        "oa_fg": float(stats["oa_fg"]),
        "per_class_iou": [float(x) for x in stats["per_class_iou"]],
        "per_class_f1": [float(x) for x in stats["per_class_f1"]],
        "gt_pixels_per_class": [int(x) for x in stats["gt_pixels_per_class"]],
        "source_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint": str(best_path),
        "split": "val",
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run": run_dir.name,
                "val_miou_present": out["miou_present"],
                "source_epoch": out["source_epoch"],
                "output": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
