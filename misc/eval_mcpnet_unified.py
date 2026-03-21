#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mmseg.apis import inference_model, init_model


IGNORE_INDEX = 255
NUM_CLASSES = 14


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--reduce-zero-label", action="store_true", default=True)
    return p.parse_args()


def pick_best_checkpoint(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("best_mIoU_iter_*.pth"))
    if cands:
        return cands[-1]
    p = run_dir / "iter_658000.pth"
    if p.exists():
        return p
    p = run_dir / "latest.pth"
    if p.exists():
        return p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


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


def _normalize_gt(gt_raw: np.ndarray, reduce_zero_label: bool) -> np.ndarray:
    gt = gt_raw.astype(np.int64, copy=True)
    if reduce_zero_label:
        # Match mmseg dataset behavior: background 0 -> ignore (255), foreground 1..N -> 0..N-1.
        bg = gt == 0
        gt[~bg] -= 1
        gt[bg] = IGNORE_INDEX
    return gt


def eval_split(model, split_root: Path, reduce_zero_label: bool) -> dict:
    img_dir = split_root / "image"
    lab_dir = split_root / "label"
    images = sorted(p for p in img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}")

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i, img_path in enumerate(images, start=1):
        lab_path = lab_dir / f"{img_path.stem}.png"
        if not lab_path.exists():
            continue
        pred_out = inference_model(model, str(img_path))
        pred = pred_out.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.int64, copy=False)
        gt_raw = np.asarray(Image.open(lab_path), dtype=np.int64)
        gt = _normalize_gt(gt_raw, reduce_zero_label=reduce_zero_label)

        valid = (gt >= 0) & (gt < NUM_CLASSES) & (gt != IGNORE_INDEX)
        if np.any(valid):
            gg = gt[valid]
            pp = pred[valid]
            binc = np.bincount(gg * NUM_CLASSES + pp, minlength=NUM_CLASSES * NUM_CLASSES)
            conf += binc.reshape(NUM_CLASSES, NUM_CLASSES)

        if i % 2000 == 0 or i == len(images):
            print(f"[{split_root.name}] {i}/{len(images)}", flush=True)

    return compute_metrics_from_conf(conf)


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint or pick_best_checkpoint(args.run_dir)
    model = init_model(str(args.config), str(ckpt), device=args.device)

    val_stats = eval_split(model, args.data_root / "val", reduce_zero_label=args.reduce_zero_label)
    test_stats = eval_split(model, args.data_root / "test", reduce_zero_label=args.reduce_zero_label)

    val_stats.update(
        {
            "loss": float("nan"),
            "ce": float("nan"),
            "dice": float("nan"),
            "checkpoint": str(ckpt),
            "config": str(args.config),
            "split": "val",
        }
    )
    test_stats.update(
        {
            "loss": float("nan"),
            "ce": float("nan"),
            "dice": float("nan"),
            "checkpoint": str(ckpt),
            "config": str(args.config),
            "split": "test",
        }
    )

    (args.run_dir / "val_metrics_best.json").write_text(json.dumps(val_stats, indent=2), encoding="utf-8")
    (args.run_dir / "test_metrics.json").write_text(json.dumps(test_stats, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "val_miou_present": val_stats["miou_present"],
                "test_miou_present": test_stats["miou_present"],
                "test_macro_f1_present": test_stats["macro_f1_present"],
                "test_oa_fg": test_stats["oa_fg"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
