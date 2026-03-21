#!/usr/bin/env python3
"""Evaluate mmseg GoldMDD runs on val/test split and write unified metrics json."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


IGNORE_INDEX = 255
NUM_CLASSES = 14


def compute_metrics_from_conf(conf: np.ndarray) -> dict:
    conf = conf.astype(np.float64, copy=False)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom_iou = tp + fp + fn
    iou = np.divide(tp, denom_iou, out=np.full_like(tp, np.nan), where=denom_iou > 0)
    denom_f1 = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom_f1, out=np.full_like(tp, np.nan), where=denom_f1 > 0)
    gt_pixels = conf.sum(axis=1)
    present = gt_pixels > 0

    miou = float(np.nanmean(iou))
    miou_present = float(np.nanmean(iou[present])) if np.any(present) else float("nan")
    macro_f1 = float(np.nanmean(f1))
    macro_f1_present = float(np.nanmean(f1[present])) if np.any(present) else float("nan")
    oa_fg = float(tp.sum() / conf.sum()) if conf.sum() > 0 else float("nan")

    return {
        "miou": miou,
        "miou_present": miou_present,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "oa_fg": oa_fg,
        "per_class_iou": [float(x) if np.isfinite(x) else float("nan") for x in iou],
        "per_class_f1": [float(x) if np.isfinite(x) else float("nan") for x in f1],
        "gt_pixels_per_class": [int(x) for x in gt_pixels.tolist()],
    }


def compute_metrics_from_areas(
    area_intersect: np.ndarray,
    area_union: np.ndarray,
    area_pred: np.ndarray,
    area_gt: np.ndarray,
) -> dict:
    area_intersect = area_intersect.astype(np.float64, copy=False)
    area_union = area_union.astype(np.float64, copy=False)
    area_pred = area_pred.astype(np.float64, copy=False)
    area_gt = area_gt.astype(np.float64, copy=False)
    iou = np.divide(
        area_intersect,
        area_union,
        out=np.full_like(area_intersect, np.nan),
        where=area_union > 0,
    )
    denom_f1 = area_pred + area_gt
    f1 = np.divide(
        2.0 * area_intersect,
        denom_f1,
        out=np.full_like(area_intersect, np.nan),
        where=denom_f1 > 0,
    )
    present = area_gt > 0
    miou = float(np.nanmean(iou))
    miou_present = float(np.nanmean(iou[present])) if np.any(present) else float("nan")
    macro_f1 = float(np.nanmean(f1))
    macro_f1_present = float(np.nanmean(f1[present])) if np.any(present) else float("nan")
    oa_fg = float(area_intersect.sum() / area_gt.sum()) if area_gt.sum() > 0 else float("nan")
    return {
        "miou": miou,
        "miou_present": miou_present,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "oa_fg": oa_fg,
        "per_class_iou": [float(x) if np.isfinite(x) else float("nan") for x in iou],
        "per_class_f1": [float(x) if np.isfinite(x) else float("nan") for x in f1],
        "gt_pixels_per_class": [int(x) for x in area_gt.tolist()],
    }


def pick_best_checkpoint(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
    if cands:
        return cands[-1]
    last = run_dir / "latest.pth"
    if last.exists():
        return last
    final_ckpt = run_dir / "iter_658000.pth"
    if final_ckpt.exists():
        return final_ckpt
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def eval_one(
    run_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    workers: int,
    repo_root: Path | None = None,
) -> dict:
    # SSA-Seg custom models are registered when importing repo-local `models`.
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))
        # Some repos (e.g., SSA-Seg) register custom models via a top-level `models` package.
        # Others (e.g., MCPNet) expose custom components through their own `mmseg` package.
        # Try importing `models` when present, but don't fail if it doesn't exist.
        if importlib.util.find_spec("models") is not None:
            import models  # noqa: F401

    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    cfg.data[split].test_mode = True
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = workers

    dataset = build_dataset(cfg.data[split])
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False,
    )

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    _ = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    # Use pre_eval to avoid materializing all prediction masks in RAM.
    pre_eval = single_gpu_test(model, data_loader, show=False, pre_eval=True)
    area_intersect = np.zeros((NUM_CLASSES,), dtype=np.float64)
    area_union = np.zeros((NUM_CLASSES,), dtype=np.float64)
    area_pred = np.zeros((NUM_CLASSES,), dtype=np.float64)
    area_gt = np.zeros((NUM_CLASSES,), dtype=np.float64)
    for item in pre_eval:
        ai, au, ap, ag = item
        area_intersect += np.asarray(ai)
        area_union += np.asarray(au)
        area_pred += np.asarray(ap)
        area_gt += np.asarray(ag)

    out = compute_metrics_from_areas(area_intersect, area_union, area_pred, area_gt)
    out.update(
        {
            "loss": float("nan"),
            "ce": float("nan"),
            "dice": float("nan"),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "split": split,
        }
    )
    out_file = run_dir / ("test_metrics.json" if split == "test" else "val_metrics_best.json")
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="Optional explicit config path. Defaults to *.py in run-dir.")
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg"),
        help="SSA-Seg repo root for custom model registration.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if args.config is None:
        cfgs = sorted(run_dir.glob("*.py"))
        if not cfgs:
            raise FileNotFoundError(f"No config .py found in {run_dir}")
        config = cfgs[0]
    else:
        config = args.config
    ckpt = args.checkpoint or pick_best_checkpoint(run_dir)
    metrics = eval_one(run_dir, config, ckpt, args.split, args.batch_size, args.workers, args.repo_root)
    print(json.dumps({k: metrics[k] for k in ["miou", "miou_present", "macro_f1_present", "oa_fg"]}, indent=2))


if __name__ == "__main__":
    main()
