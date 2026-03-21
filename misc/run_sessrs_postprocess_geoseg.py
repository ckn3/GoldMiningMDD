#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


RUN_TO_CONFIG = {
    "geoseg_unetformer_goldmdd_b8_e80_poly_ce_dice": "config/goldmdd/unetformer.py",
    "geoseg_unetformer_goldmdd_b8_e80_poly_weighted_ce_dice_auxce": "config/goldmdd/unetformer_weighted.py",
    "geoseg_unetformer_goldmdd_b8_e80_poly_focal_dice_auxfocal": "config/goldmdd/unetformer_focal.py",
    "geoseg_abcnet_goldmdd_b8_e80_poly_ce_dice": "config/goldmdd/abcnet.py",
    "geoseg_banet_goldmdd_b8_e80_poly_ce_dice": "config/goldmdd/banet.py",
    "geoseg_a2fpn_goldmdd_b8_e80_poly_ce_dice": "config/goldmdd/a2fpn.py",
    "geoseg_a2fpn_goldmdd_b8_e80_poly_weighted_ce_dice": "config/goldmdd/a2fpn_weighted.py",
    "geoseg_a2fpn_goldmdd_b8_e80_poly_focal_dice": "config/goldmdd/a2fpn_focal.py",
}


@dataclass
class JobArgs:
    pred_path: Path
    obj_path: Path
    out_path: Path
    t1: Sequence[float]
    t2: Sequence[float]
    modify_classes: Sequence[int]
    num_classes: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply SESSRS-style post-processing to GeoSeg runs on GoldMDD.")
    p.add_argument("--repo-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg"))
    p.add_argument("--experiments-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--sam-prior-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-samrs-priors"))
    p.add_argument("--runs", nargs="+", default=list(RUN_TO_CONFIG.keys()))
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None, help="Debug mode: limit images per split.")
    p.add_argument("--post-workers", type=int, default=8)
    p.add_argument("--t1", type=float, default=0.70, help="Default class-wise t1.")
    p.add_argument("--t2", type=float, default=0.75, help="Default class-wise t2.")
    p.add_argument("--modify-classes", nargs="+", type=int, default=None, help="0-based class ids; default all classes.")
    p.add_argument("--skip-inference", action="store_true", help="Use existing pre_p predictions.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = []
    for idx in range(1, n):
        comp = labels == idx
        if np.any(comp):
            out.append(comp)
    return out


def _resolve_overlaps(candidate_masks: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
    # candidate_masks: [C, H, W] bool
    counts = candidate_masks.sum(axis=0)
    out = raw_pred.copy()

    single = counts == 1
    if np.any(single):
        out[single] = np.argmax(candidate_masks[:, single], axis=0).astype(np.uint8)

    multi = counts > 1
    if np.any(multi):
        ys, xs = np.where(multi)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cand = np.where(candidate_masks[:, y, x])[0]
            rp = int(raw_pred[y, x])
            out[y, x] = np.uint8(rp if rp in cand else cand[0])
    return out


def sessrs_refine_from_object_map(
    raw_pred: np.ndarray,
    obj_map: np.ndarray,
    num_classes: int,
    t1: Sequence[float],
    t2: Sequence[float],
    modify_classes: Sequence[int],
) -> np.ndarray:
    modify_set = set(modify_classes)

    objects = []
    obj_ids = np.unique(obj_map)
    for oid in obj_ids.tolist():
        if oid == 0:
            continue
        omask = obj_map == oid
        area = int(omask.sum())
        if area == 0:
            continue
        counts = np.bincount(raw_pred[omask].ravel(), minlength=num_classes)
        semantic = int(np.argmax(counts))
        ratio = float(counts[semantic]) / float(area)
        if ratio >= t1[semantic]:
            objects.append({"mask": omask, "area": area, "semantic": semantic, "ratio": ratio})

    objects.sort(key=lambda x: x["area"], reverse=True)

    # De-intersect objects of different classes, same spirit as original SESSRS.
    for i, oi in enumerate(objects):
        ci = int(oi["semantic"])
        if ci not in modify_set:
            continue
        if oi["area"] <= 0:
            continue
        mask_i = oi["mask"]
        for j in range(i + 1, len(objects)):
            oj = objects[j]
            cj = int(oj["semantic"])
            if cj == ci or oj["area"] <= 0:
                continue
            inter = mask_i & oj["mask"]
            if not np.any(inter):
                continue
            overlap_on_j = float(inter.sum()) / float(oj["area"])
            if overlap_on_j > t2[ci] and float(oi["ratio"]) < float(oj["ratio"]):
                mask_i = mask_i & (~oj["mask"])
        oi["mask"] = mask_i
        oi["area"] = int(mask_i.sum())

    obj_by_class: dict[int, list[dict]] = {k: [] for k in range(num_classes)}
    for o in objects:
        if o["area"] > 0:
            obj_by_class[int(o["semantic"])].append(o)

    h, w = raw_pred.shape
    candidate_masks = np.zeros((num_classes, h, w), dtype=bool)

    for cls in range(num_classes):
        cls_pred = raw_pred == cls
        if cls not in modify_set:
            candidate_masks[cls] = cls_pred
            continue

        comps = _connected_components(cls_pred)
        cls_union = np.zeros((h, w), dtype=bool)
        for comp in comps:
            comp_area = int(comp.sum())
            if comp_area == 0:
                continue
            matched = []
            for obj in obj_by_class.get(cls, []):
                inter_area = int((comp & obj["mask"]).sum())
                if inter_area == 0:
                    continue
                os_ratio = float(inter_area) / float(obj["area"])
                op_ratio = float(inter_area) / float(comp_area)
                if (os_ratio > t1[cls]) or (op_ratio > t2[cls]):
                    matched.append(obj["mask"])

            new_comp = comp
            if 1 < len(matched) < 10:
                merged = np.logical_or.reduce(matched)
                if float((merged & comp).sum()) / float(comp_area) > t2[cls]:
                    new_comp = merged
            elif len(matched) == 1:
                m = matched[0]
                m_area = int(m.sum())
                if m_area > t2[cls] * float(comp_area) and m_area * t1[cls] < float(comp_area):
                    new_comp = m
            cls_union |= new_comp
        candidate_masks[cls] = cls_union

    return _resolve_overlaps(candidate_masks, raw_pred)


def _worker_process_one(job: JobArgs) -> str:
    pred = np.array(Image.open(job.pred_path), dtype=np.uint8)
    obj = np.array(Image.open(job.obj_path), dtype=np.uint8)
    refined = sessrs_refine_from_object_map(
        raw_pred=pred,
        obj_map=obj,
        num_classes=job.num_classes,
        t1=job.t1,
        t2=job.t2,
        modify_classes=job.modify_classes,
    )
    Image.fromarray(refined, mode="L").save(job.out_path)
    return job.out_path.name


def compute_metrics(pred_dir: Path, gt_dir: Path, num_classes: int, image_ids: Iterable[str]) -> dict:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    for img_id in image_ids:
        pred_path = pred_dir / f"{img_id}.png"
        gt_path = gt_dir / f"{img_id}.png"
        if not pred_path.exists():
            continue
        pred = np.array(Image.open(pred_path), dtype=np.uint8)
        gt_raw = np.array(Image.open(gt_path), dtype=np.uint8)
        # GoldMDD labels: 0=ignore, 1..14 are classes.
        valid = (gt_raw >= 1) & (gt_raw <= num_classes)
        if not np.any(valid):
            continue
        gt = gt_raw[valid].astype(np.int64) - 1
        pp = pred[valid].astype(np.int64)
        pp = np.clip(pp, 0, num_classes - 1)
        binc = np.bincount(gt * num_classes + pp, minlength=num_classes * num_classes)
        conf += binc.reshape(num_classes, num_classes)

    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    gt_pixels = conf.sum(axis=1)

    denom_iou = tp + fp + fn
    iou = np.divide(tp, denom_iou, out=np.full_like(tp, np.nan), where=denom_iou > 0)
    denom_f1 = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom_f1, out=np.full_like(tp, np.nan), where=denom_f1 > 0)

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


def infer_predictions(
    run_dir: Path,
    config_path: Path,
    split: str,
    args: argparse.Namespace,
) -> list[str]:
    import sys

    sys.path.insert(0, str(args.repo_root))
    from tools.cfg import py2cfg  # type: ignore
    from train_supervision import Supervision_Train  # type: ignore
    from geoseg.datasets.goldmdd_dataset import GoldMDDDataset, val_aug  # type: ignore

    cfg = py2cfg(config_path)
    num_classes = int(cfg.num_classes)
    ckpt = run_dir / f"{run_dir.name}.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = Supervision_Train.load_from_checkpoint(str(ckpt), config=cfg, map_location=device)
    net = model.net.to(device).eval()

    ds = GoldMDDDataset(
        data_root=str(args.data_root),
        split=split,
        transform=val_aug,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    out_pred_dir = run_dir / "sessrs_from_priors" / split / "pre_p"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    image_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{run_dir.name}][{split}] infer", leave=False):
            img = batch["img"].to(device, non_blocking=True)
            ids = list(batch["img_id"])
            logits = net(img)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for i, img_id in enumerate(ids):
                image_ids.append(img_id)
                out_path = out_pred_dir / f"{img_id}.png"
                if args.overwrite or (not out_path.exists()):
                    Image.fromarray(pred[i], mode="L").save(out_path)

    if num_classes != 14:
        raise RuntimeError(f"Unexpected num_classes={num_classes}; expected 14 for GoldMDD.")
    return image_ids


def run_postprocess_for_split(run_dir: Path, split: str, args: argparse.Namespace, image_ids: list[str]) -> None:
    pred_dir = run_dir / "sessrs_from_priors" / split / "pre_p"
    out_dir = run_dir / "sessrs_from_priors" / split / "se_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = args.data_root / split / "label"
    obj_dir = args.sam_prior_root / split / "object"
    if not obj_dir.exists():
        raise FileNotFoundError(f"SAM object prior dir not found: {obj_dir}")

    num_classes = 14
    modify_classes = args.modify_classes if args.modify_classes is not None else list(range(num_classes))
    t1 = [float(args.t1)] * num_classes
    t2 = [float(args.t2)] * num_classes

    jobs: list[JobArgs] = []
    for img_id in image_ids:
        pp = pred_dir / f"{img_id}.png"
        oo = obj_dir / f"{img_id}.png"
        if not pp.exists() or not oo.exists():
            continue
        outp = out_dir / f"{img_id}.png"
        if outp.exists() and (not args.overwrite):
            continue
        jobs.append(
            JobArgs(
                pred_path=pp,
                obj_path=oo,
                out_path=outp,
                t1=t1,
                t2=t2,
                modify_classes=modify_classes,
                num_classes=num_classes,
            )
        )

    if jobs:
        if args.post_workers > 1:
            with mp.Pool(processes=args.post_workers) as pool:
                list(tqdm(pool.imap_unordered(_worker_process_one, jobs), total=len(jobs), desc=f"[{run_dir.name}][{split}] SESSRS"))
        else:
            for j in tqdm(jobs, desc=f"[{run_dir.name}][{split}] SESSRS"):
                _worker_process_one(j)

    base_stats = compute_metrics(pred_dir=pred_dir, gt_dir=gt_dir, num_classes=num_classes, image_ids=image_ids)
    sessrs_stats = compute_metrics(pred_dir=out_dir, gt_dir=gt_dir, num_classes=num_classes, image_ids=image_ids)
    delta = {
        "delta_miou": float(sessrs_stats["miou"] - base_stats["miou"]),
        "delta_miou_present": float(sessrs_stats["miou_present"] - base_stats["miou_present"]),
        "delta_macro_f1": float(sessrs_stats["macro_f1"] - base_stats["macro_f1"]),
        "delta_macro_f1_present": float(sessrs_stats["macro_f1_present"] - base_stats["macro_f1_present"]),
        "delta_oa_fg": float(sessrs_stats["oa_fg"] - base_stats["oa_fg"]),
    }
    result = {
        "run": run_dir.name,
        "split": split,
        "t1": args.t1,
        "t2": args.t2,
        "modify_classes": modify_classes,
        "base_pred_metrics": base_stats,
        "sessrs_pred_metrics": sessrs_stats,
        "delta": delta,
    }
    (run_dir / "sessrs_from_priors" / f"{split}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{run_dir.name}][{split}] "
        f"base_mIoU_present={base_stats['miou_present']:.4f} -> "
        f"sessrs_mIoU_present={sessrs_stats['miou_present']:.4f} "
        f"(delta={delta['delta_miou_present']:+.4f})"
    )


def main() -> None:
    args = parse_args()
    runs = args.runs
    missing = [r for r in runs if r not in RUN_TO_CONFIG]
    if missing:
        raise ValueError(f"Unsupported run names: {missing}")

    for run_name in runs:
        run_dir = args.experiments_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"run dir not found: {run_dir}")
        config_rel = RUN_TO_CONFIG[run_name]
        config_path = args.repo_root / config_rel
        if not config_path.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        print(f"\n=== {run_name} ===")
        for split in args.splits:
            pred_dir = run_dir / "sessrs_from_priors" / split / "pre_p"
            if args.skip_inference and pred_dir.exists():
                image_ids = sorted([p.stem for p in pred_dir.glob("*.png")])
                if args.max_samples is not None:
                    image_ids = image_ids[: args.max_samples]
                if not image_ids:
                    raise RuntimeError(f"No predictions found under {pred_dir} with --skip-inference.")
            else:
                image_ids = infer_predictions(run_dir=run_dir, config_path=config_path, split=split, args=args)
            run_postprocess_for_split(run_dir=run_dir, split=split, args=args, image_ids=image_ids)


if __name__ == "__main__":
    main()
