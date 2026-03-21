#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from semseg_common import CEDiceLoss, GoldMDDPatchDataset, build_split_samples, make_loader
from train_semseg_rsamseg import RSAMSegSemantic, _build_encoder_cfg, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        default=[
            "rsamseg_vit_b_baseline2_augv2_weighted_ce_dice",
            "rsamseg_vit_b_baseline3_augv2_focal_dice",
        ],
    )
    p.add_argument("--exp-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-interval", type=int, default=200)
    return p.parse_args()


def evaluate_one(run_dir: Path, args: argparse.Namespace) -> None:
    cfg = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    data_root = Path(cfg["data_root"])
    rsam_root = Path(cfg["rsam_root"])
    encoder_preset = cfg.get("encoder_preset", "vit_b")
    loss_mode = cfg.get("loss_mode", "ce_dice")
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    ce_weight = float(cfg.get("ce_weight", 1.0))
    dice_weight = float(cfg.get("dice_weight", 1.0))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this evaluator.")

    model = RSAMSegSemantic(
        rsam_root=rsam_root,
        encoder_cfg=_build_encoder_cfg(encoder_preset),
    )
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    ce_class_weights = None
    if loss_mode == "weighted_ce_dice":
        cw_path = run_dir / "class_weights.json"
        if not cw_path.exists():
            raise FileNotFoundError(f"Missing class weights for weighted CE: {cw_path}")
        cw = json.loads(cw_path.read_text(encoding="utf-8"))["weights"]
        ce_class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

    criterion = CEDiceLoss(
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        loss_mode=loss_mode,
        ce_class_weights=ce_class_weights,
        focal_gamma=focal_gamma,
    )

    val_samples = build_split_samples(data_root / "val")
    val_ds = GoldMDDPatchDataset(val_samples, train=False, aug_preset="none")
    val_loader = make_loader(
        val_ds,
        batch_size=int(args.batch_size if args.batch_size > 0 else cfg.get("batch_size", 8)),
        num_workers=int(args.num_workers if args.num_workers >= 0 else cfg.get("num_workers", 8)),
        shuffle=False,
    )

    va = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        amp=bool(args.amp),
        phase="val",
        log_interval=int(args.log_interval),
    )

    out = {
        "miou": float(va["miou"]),
        "miou_present": float(va["miou_present"]),
        "macro_f1": float(va["macro_f1"]),
        "macro_f1_present": float(va["macro_f1_present"]),
        "oa_fg": float(va["oa_fg"]),
        "loss": float(va["loss"]),
        "ce": float(va["ce"]),
        "dice": float(va["dice"]),
        "per_class_iou": [float(x) for x in va["per_class_iou"]],
        "per_class_f1": [float(x) for x in va["per_class_f1"]],
        "gt_pixels_per_class": [int(x) for x in va["gt_pixels_per_class"]],
        "split": "val",
        "checkpoint": str((run_dir / "best.pth").resolve()),
    }
    (run_dir / "val_metrics_best.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"{run_dir.name}: val_miou_present={out['miou_present']:.4f} "
        f"val_macro_f1_present={out['macro_f1_present']:.4f} val_oa_fg={out['oa_fg']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    for run in args.runs:
        run_dir = (args.exp_root / run).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        evaluate_one(run_dir, args)


if __name__ == "__main__":
    main()
