#!/usr/bin/env python3
"""Eval-only pass for completed GoldMDD runs to backfill val/test metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT_MISC = Path("/deac/csc/yangGrp/cuij/GoldMDD/misc")
SMP_SCRIPT = ROOT_MISC / "train_semseg_smp.py"
SEG_SCRIPT = ROOT_MISC / "train_semseg_segformer.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size-override", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--include-runs", nargs="*", default=None)
    return p.parse_args()


def discover_runs(root: Path, include_runs: list[str] | None) -> list[Path]:
    runs: list[Path] = []
    allow = set(include_runs) if include_runs else None
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in {"logs", "diagnostics"}:
            continue
        if allow and d.name not in allow:
            continue
        if (d / "config.json").exists() and (d / "best.pt").exists():
            runs.append(d)
    return runs


def build_criterion(mod, cfg: dict, train_samples: list):
    ce_class_weights = None
    if cfg.get("loss_mode") == "weighted_ce_dice":
        ce_class_weights, _ = mod.compute_train_class_weights(train_samples, power=float(cfg.get("class_weight_power", 0.5)))
    return mod.CEDiceLoss(
        ce_weight=float(cfg.get("ce_weight", 1.0)),
        dice_weight=float(cfg.get("dice_weight", 1.0)),
        loss_mode=str(cfg.get("loss_mode", "ce_dice")),
        ce_class_weights=ce_class_weights,
        focal_gamma=float(cfg.get("focal_gamma", 2.0)),
    )


def eval_one_run(run_dir: Path, smp_mod, seg_mod, device: torch.device, batch_size_override: int, num_workers: int) -> None:
    cfg = json.load(open(run_dir / "config.json"))
    is_segformer = "model_name" in cfg
    mod = seg_mod if is_segformer else smp_mod

    data_root = Path(cfg["data_root"])
    train_samples = mod.build_split_samples(data_root / "train")
    val_samples = mod.build_split_samples(data_root / "val")
    test_samples = mod.build_split_samples(data_root / "test")

    val_ds = mod.GoldMDDPatchDataset(val_samples, train=False, aug_preset="none")
    test_ds = mod.GoldMDDPatchDataset(test_samples, train=False, aug_preset="none")
    bs = batch_size_override if batch_size_override > 0 else int(cfg.get("batch_size", 8))
    val_loader = mod.make_loader(val_ds, bs, num_workers, shuffle=False)
    test_loader = mod.make_loader(test_ds, bs, num_workers, shuffle=False)

    if is_segformer:
        model = mod.build_model(cfg["model_name"], pretrained=bool(cfg.get("pretrained", True)))
    else:
        model = mod.build_model(cfg["arch"], cfg["encoder"], cfg.get("encoder_weights"))
    model.to(device)

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    criterion = build_criterion(mod, cfg, train_samples).to(device)
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    print(f"[eval] {run_dir.name} ({'segformer' if is_segformer else 'smp'}) bs={bs}", flush=True)
    val_stats = mod.evaluate(model, val_loader, criterion, device, amp, phase=f"val:{run_dir.name}", log_interval=0)
    test_stats = mod.evaluate(model, test_loader, criterion, device, amp, phase=f"test:{run_dir.name}", log_interval=0)

    val_payload = {
        "miou": val_stats["miou"],
        "miou_present": val_stats["miou_present"],
        "macro_f1": val_stats["macro_f1"],
        "macro_f1_present": val_stats["macro_f1_present"],
        "oa_fg": val_stats["oa_fg"],
        "loss": val_stats["loss"],
        "ce": val_stats["ce"],
        "dice": val_stats["dice"],
        "per_class_iou": val_stats["per_class_iou"],
        "per_class_f1": val_stats["per_class_f1"],
        "gt_pixels_per_class": val_stats["gt_pixels_per_class"],
        "source_checkpoint": "best.pt",
        "source_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }
    test_payload = {
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
        "source_checkpoint": "best.pt",
        "source_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }

    (run_dir / "val_metrics_best.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")
    # Overwrite test_metrics.json to backfill OA/F1 in canonical location.
    (run_dir / "test_metrics.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
    print(
        f"  val: miou={val_stats['miou']:.4f} miou_present={val_stats['miou_present']:.4f} "
        f"f1_present={val_stats['macro_f1_present']:.4f} oa_fg={val_stats['oa_fg']:.4f}",
        flush=True,
    )
    print(
        f"  test: miou={test_stats['miou']:.4f} miou_present={test_stats['miou_present']:.4f} "
        f"f1_present={test_stats['macro_f1_present']:.4f} oa_fg={test_stats['oa_fg']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}", flush=True)
    smp_mod = load_module(SMP_SCRIPT, "goldmdd_train_smp")
    seg_mod = load_module(SEG_SCRIPT, "goldmdd_train_segformer")

    runs = discover_runs(args.experiments_root, args.include_runs)
    print(f"Discovered {len(runs)} runs", flush=True)
    for d in runs:
        eval_one_run(d, smp_mod, seg_mod, device, args.batch_size_override, args.num_workers)


if __name__ == "__main__":
    main()
