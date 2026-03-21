#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from semseg_common import IGNORE_INDEX, NUM_FOREGROUND_CLASSES, compute_metrics_from_conf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True, help="Path to rssegmentation repo root.")
    p.add_argument("--config", type=Path, required=True, help="Config file used for the run.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Best checkpoint path.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory to write unified metrics json.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def eval_loader(model, loader, device: torch.device, split_name: str) -> dict:
    conf = torch.zeros((NUM_FOREGROUND_CLASSES, NUM_FOREGROUND_CLASSES), dtype=torch.int64, device=device)
    model.eval()
    n_total = len(loader)
    for i, (imgs, masks, _) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs, True)
        pred = logits.argmax(dim=1)
        valid = (
            (masks != IGNORE_INDEX)
            & (masks >= 0)
            & (masks < NUM_FOREGROUND_CLASSES)
            & (pred >= 0)
            & (pred < NUM_FOREGROUND_CLASSES)
        )
        if valid.any():
            t = masks[valid].to(torch.int64)
            p = pred[valid].to(torch.int64)
            idx = t * NUM_FOREGROUND_CLASSES + p
            conf += torch.bincount(idx, minlength=NUM_FOREGROUND_CLASSES * NUM_FOREGROUND_CLASSES).reshape(
                NUM_FOREGROUND_CLASSES, NUM_FOREGROUND_CLASSES
            )
        if i % 200 == 0 or i == n_total:
            print(f"[{split_name}] {i}/{n_total}", flush=True)
    miou, miou_present, macro_f1, macro_f1_present, oa_fg, per_iou, per_f1, gt_pix = compute_metrics_from_conf(conf.cpu())
    return {
        "miou": float(miou),
        "miou_present": float(miou_present),
        "macro_f1": float(macro_f1),
        "macro_f1_present": float(macro_f1_present),
        "oa_fg": float(oa_fg),
        "loss": float("nan"),
        "ce": float("nan"),
        "dice": float("nan"),
        "per_class_iou": [float(x) for x in per_iou],
        "per_class_f1": [float(x) for x in per_f1],
        "gt_pixels_per_class": [int(x) for x in gt_pix],
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.config import Config  # pylint: disable=import-error
    from train import myTrain  # pylint: disable=import-error
    from rsseg.datasets.goldmdd_dataset import GoldMDD  # pylint: disable=import-error

    cfg = Config.fromfile(str(args.config.resolve()))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This evaluator is intended to run on CUDA for speed.")

    model = myTrain.load_from_checkpoint(str(args.checkpoint.resolve()), cfg=cfg).to(device)

    loader_bs = int(args.batch_size if args.batch_size > 0 else cfg.dataset_config.test_mode.loader.batch_size)
    loader_workers = int(args.workers if args.workers >= 0 else cfg.dataset_config.test_mode.loader.num_workers)

    # Build val/test loaders explicitly to avoid worker settings hidden in repo defaults.
    val_dataset = GoldMDD(
        data_root=cfg.dataset_config.data_root,
        mode="val",
        transform=cfg.dataset_config.val_mode.transform,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=loader_bs,
        num_workers=loader_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # Test loader for GoldMDD: explicitly use test split.
    test_dataset = GoldMDD(
        data_root=cfg.dataset_config.data_root,
        mode="test",
        transform=cfg.dataset_config.test_mode.transform,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=loader_bs,
        num_workers=loader_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    val_metrics = eval_loader(model, val_loader, device, split_name="val")
    test_metrics = eval_loader(model, test_loader, device, split_name="test")

    val_metrics.update(
        {
            "split": "val",
            "checkpoint": str(args.checkpoint.resolve()),
            "config": str(args.config.resolve()),
        }
    )
    test_metrics.update(
        {
            "split": "test",
            "checkpoint": str(args.checkpoint.resolve()),
            "config": str(args.config.resolve()),
        }
    )

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "val_metrics_best.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "val_miou_present": val_metrics["miou_present"],
                "test_miou_present": test_metrics["miou_present"],
                "test_oa_fg": test_metrics["oa_fg"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
