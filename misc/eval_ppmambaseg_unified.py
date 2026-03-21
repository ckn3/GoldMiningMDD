#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified val/test eval for PPMambaSeg GoldMDD run.")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/third_party/PPMambaSeg/GeoSeg"),
    )
    p.add_argument(
        "--config-path",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/third_party/PPMambaSeg/GeoSeg/config/goldmdd/ppmamba.py"),
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments/ppmambaseg_ppmamba_goldmdd_b8_e80_poly_ce_dice"),
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def compute_stats(conf: np.ndarray) -> dict:
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    gt_pixels = conf.sum(axis=1)
    iou = np.divide(tp, tp + fp + fn, out=np.full_like(tp, np.nan), where=(tp + fp + fn) > 0)
    f1 = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.full_like(tp, np.nan), where=(2.0 * tp + fp + fn) > 0)
    present = gt_pixels > 0
    return {
        "miou": float(np.nanmean(iou)),
        "miou_present": float(np.nanmean(iou[present])) if np.any(present) else float("nan"),
        "macro_f1": float(np.nanmean(f1)),
        "macro_f1_present": float(np.nanmean(f1[present])) if np.any(present) else float("nan"),
        "oa_fg": float(tp.sum() / gt_pixels.sum()) if gt_pixels.sum() > 0 else float("nan"),
        "per_class_iou": [float(x) if np.isfinite(x) else float("nan") for x in iou.tolist()],
        "per_class_f1": [float(x) if np.isfinite(x) else float("nan") for x in f1.tolist()],
        "gt_pixels_per_class": [int(x) for x in gt_pixels.tolist()],
    }


def evaluate_split(
    net: torch.nn.Module,
    dataset,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    device: torch.device,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
    )
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    net.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device, non_blocking=(device.type == "cuda"))
            gt = batch["gt_semantic_seg"].cpu().numpy().astype(np.int64)
            logits = net(img)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().astype(np.int64)
            for i in range(pred.shape[0]):
                valid = (gt[i] >= 0) & (gt[i] < num_classes)
                if not np.any(valid):
                    continue
                g = gt[i][valid]
                p = pred[i][valid]
                binc = np.bincount(g * num_classes + p, minlength=num_classes * num_classes)
                conf += binc.reshape(num_classes, num_classes)
    return compute_stats(conf)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    add_repo_to_path(args.repo_root)
    from tools.cfg import py2cfg  # type: ignore
    from train_supervision import Supervision_Train  # type: ignore

    cfg = py2cfg(args.config_path)
    run_dir = args.run_dir.resolve()
    ckpt_path = run_dir / f"{cfg.weights_name}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    source_epoch = ckpt_raw.get("epoch") if isinstance(ckpt_raw, dict) else None

    model = Supervision_Train.load_from_checkpoint(str(ckpt_path), config=cfg, map_location=device)
    net = model.net.to(device).eval()

    val_stats = evaluate_split(
        net=net,
        dataset=cfg.val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=int(cfg.num_classes),
        device=device,
    )
    test_stats = evaluate_split(
        net=net,
        dataset=cfg.test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=int(cfg.num_classes),
        device=device,
    )

    val_out = {
        **val_stats,
        "checkpoint": str(ckpt_path),
        "source_epoch": int(source_epoch) if source_epoch is not None else None,
        "config": str(args.config_path),
        "split": "val",
    }
    test_out = {
        **test_stats,
        "checkpoint": str(ckpt_path),
        "source_epoch": int(source_epoch) if source_epoch is not None else None,
        "config": str(args.config_path),
        "split": "test",
    }
    (run_dir / "val_metrics_best.json").write_text(json.dumps(val_out, indent=2), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(json.dumps(test_out, indent=2), encoding="utf-8")

    class_names = list(cfg.classes)
    best_val_iou = {str(n): float(v) for n, v in zip(class_names, val_stats["per_class_iou"])}
    (run_dir / "best_val_per_class_iou.json").write_text(json.dumps(best_val_iou, indent=2), encoding="utf-8")

    print(
        f"PPMamba eval done: val_miou_present={val_stats['miou_present']:.4f}, "
        f"test_miou_present={test_stats['miou_present']:.4f}"
    )


if __name__ == "__main__":
    main()
