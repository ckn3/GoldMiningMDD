#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from pycocotools.mask import area, decode, encode, iou, merge
from scipy.ndimage import find_objects, label as cc_label
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
    "geoseg_manet_goldmdd_b8_e80_poly_ce_dice": "config/goldmdd/manet.py",
    "geoseg_dcswin_small_goldmdd_b8_e80_poly_ce_dice": "config/goldmdd/dcswin.py",
    "geoseg_pyramidmamba_goldmdd_b8_e80_poly_ce_dice": "config/goldmdd/pyramidmamba.py",
}

T1_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
T2_GRID = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


@dataclass
class SessrsContext:
    num_classes: int
    p_dir: Path
    pre_p_info_dir: Path
    sam_label_info_dir: Path
    out_dir: Path
    fix_t1: dict[str, float]
    fix_t2: dict[str, float]
    modify_category: list[int]
    palette: list[int] | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official-style SESSRS post-process for GeoSeg GoldMDD runs.")
    p.add_argument("--repo-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg"))
    p.add_argument("--experiments-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--sam-prior-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-samrs-priors"))
    p.add_argument("--runs", nargs="+", default=list(RUN_TO_CONFIG.keys()))
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--post-workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--search-threshold-mode", type=float, default=0.7, help="Official threshold_mode (default 0.7).")
    p.add_argument("--search-classes", nargs="+", type=int, default=None, help="Class ids to search; default all.")
    p.add_argument(
        "--force-all-foreground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force SESSRS modify_category to all model class ids in prediction space (0..13 for GoldMDD).",
    )
    p.add_argument("--search-max-images", type=int, default=None, help="Limit val images during t1/t2 search.")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def single_mask_to_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def deintersect(mask1, mask2):
    if isinstance(mask1, dict) and isinstance(mask2, dict):
        mask1_bin = decode(mask1)
        mask2_bin = decode(mask2)
        if mask1_bin.shape != mask2_bin.shape:
            raise ValueError("mask shape mismatch in deintersect(dict)")
        return encode(mask1_bin & ~mask2_bin)
    if isinstance(mask1, np.ndarray) and isinstance(mask2, np.ndarray):
        if mask1.shape != mask2.shape:
            raise ValueError("mask shape mismatch in deintersect(array)")
        return mask1 & ~mask2
    raise ValueError("unsupported mask types in deintersect")


def segmentation_to_instance_json(pred_png: Path, out_json: Path) -> None:
    label_image = np.array(Image.open(pred_png), dtype=np.uint8)
    unique_labels = np.unique(label_image)
    data_list = []

    for cls_id in unique_labels.tolist():
        mask = (label_image == cls_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[:, 4:5][1:]
        bboxes = stats[:, :-1][1:]
        for i in range(1, num_labels):
            inst = labels == i
            ann = {
                "semantic": str(cls_id),
                "rles": single_mask_to_rle(inst),
                "bbox": bboxes[i - 1].tolist(),
                "size": int(sizes[i - 1][0]),
            }
            data_list.append(ann)

    out_json.write_text(json.dumps(data_list, indent=2), encoding="utf-8")


def object_map_to_sam_json(object_png: Path, out_json: Path) -> None:
    obj = np.array(Image.open(object_png), dtype=np.uint8)
    out = []
    for oid in np.unique(obj).tolist():
        if oid == 0:
            continue
        m = (obj == oid)
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]
        out.append(
            {
                "rles": single_mask_to_rle(m),
                "area": int(m.sum()),
                "bbox": bbox,
                "point_coords": [],
                "predicted_iou": 1.0,
                "stability_score": 1.0,
                "crop_box": [0, 0, int(obj.shape[1]), int(obj.shape[0])],
            }
        )
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")


def segmentation_to_instance(binary_img: np.ndarray) -> list[dict]:
    unique_labels = np.unique(binary_img)
    unique_labels = unique_labels[unique_labels != 0]
    data_list = []
    for label_id in unique_labels.tolist():
        mask = (binary_img == label_id).astype(np.uint8)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            inst = labels == i
            data_list.append({"rles": single_mask_to_rle(inst)})
    return data_list


def modify_b_w(array: np.ndarray, category: int) -> np.ndarray:
    padded = np.pad(array, pad_width=1, mode="constant", constant_values=254)
    labeled_array, _ = cc_label(padded == 255)
    slices = find_objects(labeled_array)

    for i, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        mask = labeled_array[slc] == i
        expanded_slc = (
            slice(max(slc[0].start - 1, 0), min(slc[0].stop + 1, padded.shape[0])),
            slice(max(slc[1].start - 1, 0), min(slc[1].stop + 1, padded.shape[1])),
        )
        neighborhood = padded[expanded_slc]
        neighborhood_mask = np.zeros_like(neighborhood, dtype=bool)
        yoff = slc[0].start - expanded_slc[0].start
        xoff = slc[1].start - expanded_slc[1].start
        neighborhood_mask[yoff:yoff + mask.shape[0], xoff:xoff + mask.shape[1]] = mask
        neighbors = neighborhood[(~neighborhood_mask) & (neighborhood != 255) & (neighborhood != 254)]
        fill_val = category
        if neighbors.size > 0:
            fill_val = int(np.bincount(neighbors).argmax())
        padded[slc][mask] = fill_val
    return padded[1:-1, 1:-1]


def modify(orig_mask_array: np.ndarray, modify_category_list: list[int]) -> np.ndarray:
    modified_category_list = []
    for category in modify_category_list:
        arr = orig_mask_array.copy()
        arr[arr == category] = 255
        modified_category_list.append(modify_b_w(arr, category))
    out = orig_mask_array.copy()
    for category, modified_category in zip(modify_category_list, modified_category_list):
        out[orig_mask_array == category] = modified_category[orig_mask_array == category]
    return out


def sessrs_one_image(path: str, ctx: SessrsContext) -> None:
    with open(ctx.pre_p_info_dir / f"{path[:-4]}.json", "r", encoding="utf-8") as f:
        label = json.load(f)
    with open(ctx.sam_label_info_dir / f"{path[:-4]}.json", "r", encoding="utf-8") as f:
        mask = json.load(f)

    label_segmentation = np.array(Image.open(ctx.p_dir / path), dtype=np.uint8)
    h, w = label_segmentation.shape

    Ak = {f"label{k}": [] for k in range(ctx.num_classes)}
    for m in mask:
        mask_l_segmentation = decode(m["rles"]).astype(np.bool_)
        areas = m["area"]
        counts = np.bincount(label_segmentation[mask_l_segmentation], minlength=ctx.num_classes)
        if counts.size == 0:
            m["semantic"] = None
            m["e_counts_ratio"] = None
            continue
        most_common_element = int(np.argmax(counts))
        e_counts = counts[most_common_element]
        if e_counts / areas > ctx.fix_t1[str(most_common_element)]:
            m["semantic"] = most_common_element
            m["e_counts_ratio"] = round(float(e_counts / areas), 3)
        else:
            m["semantic"] = None
            m["e_counts_ratio"] = None

    mask = sorted(mask, key=lambda x: x["area"], reverse=True)
    for i in range(len(mask) - 1):
        mask_i_semantic = mask[i]["semantic"]
        if mask_i_semantic in ctx.modify_category:
            for j in range(i + 1, len(mask)):
                mask_j_semantic = mask[j]["semantic"]
                if mask_i_semantic != mask_j_semantic and mask_j_semantic is not None:
                    inter_ratio = area(merge([mask[i]["rles"], mask[j]["rles"]], intersect=True)) / mask[j]["area"]
                    if inter_ratio > ctx.fix_t2[str(mask_i_semantic)] and mask[i]["e_counts_ratio"] < mask[j]["e_counts_ratio"]:
                        mask[i]["rles"] = deintersect(mask[i]["rles"], mask[j]["rles"])
                        mask[i]["area"] = area(mask[i]["rles"])
                else:
                    continue
        else:
            continue

    for m in mask:
        semantic = m["semantic"]
        if semantic is not None:
            Ak[f"label{semantic}"].append(m)

    Pk = {f"label{k}": [] for k in range(ctx.num_classes)}
    for l in label:
        for k in range(ctx.num_classes):
            if l["semantic"] == str(k):
                Pk[f"label{k}"].append(l)

    sessrs_Pk = {}
    for k in ctx.modify_category:
        flag = 0
        t1 = ctx.fix_t1[str(k)]
        t2 = ctx.fix_t2[str(k)]
        for p_k in Pk[f"label{k}"]:
            tmp = []
            for mask_S in Ak[f"label{k}"]:
                if area(mask_S["rles"]) > 200000:
                    tmp = []
                    continue
                Os = area(merge([p_k["rles"], mask_S["rles"]], intersect=True)) / mask_S["area"]
                Op = area(merge([p_k["rles"], mask_S["rles"]], intersect=True)) / p_k["size"]
                if Os > t1 or Op > t2:
                    tmp.append(mask_S["rles"])
            if len(tmp) > 1 and len(tmp) < 10:
                tmp_rles = merge(tmp)
                if area(merge([tmp_rles, p_k["rles"]], intersect=True)) / area(p_k["rles"]) > t2:
                    p_k["rles"] = tmp_rles
                    Pk[f"label{k}"][flag] = p_k
            elif len(tmp) == 1 and area(tmp[0]) > t2 * area(p_k["rles"]) and area(tmp[0]) * t1 < area(p_k["rles"]):
                p_k["rles"] = tmp[0]
                Pk[f"label{k}"][flag] = p_k
            flag += 1

        if len(Pk[f"label{k}"]) == 0:
            sessrs_Pk[f"{k}"] = None
        else:
            Pk_rles = [pk["rles"] for pk in Pk[f"label{k}"]]
            sessrs_Pk[f"{k}"] = merge(Pk_rles)

    intersect_list = []
    sessrs_index = ctx.modify_category
    for i in range(len(sessrs_index)):
        if sessrs_Pk[f"{sessrs_index[i]}"] is None:
            continue
        for j in range(i + 1, len(sessrs_index)):
            if sessrs_Pk[f"{sessrs_index[j]}"] is None:
                continue
            intersect = merge([sessrs_Pk[f"{sessrs_index[i]}"], sessrs_Pk[f"{sessrs_index[j]}"]], intersect=True)
            if area(intersect) == 0:
                continue
            intersect = decode(intersect)
            seg_ins_intersect = segmentation_to_instance(intersect)
            for inter in seg_ins_intersect:
                max_i_iou = 0
                max_j_iou = 0
                for pki in Pk[f"label{sessrs_index[i]}"]:
                    is_iou = iou([inter["rles"]], [pki["rles"]], [0])[0][0]
                    max_i_iou = max(max_i_iou, is_iou)
                for pkj in Pk[f"label{sessrs_index[j]}"]:
                    is_iou = iou([inter["rles"]], [pkj["rles"]], [0])[0][0]
                    max_j_iou = max(max_j_iou, is_iou)
                inter["semantic"] = sessrs_index[i] if max_i_iou > max_j_iou else sessrs_index[j]
                intersect_list.append(inter)

    image = np.ones((h, w), dtype=np.uint8) * 255
    flag = np.ones((h, w), dtype=np.uint8)
    for inter in intersect_list:
        semantic = int(inter["semantic"])
        seg = decode(inter["rles"])
        image[seg == 1] = (semantic * seg * flag)[seg == 1]
        flag[(flag & seg) == 1] = 0

    for k in ctx.modify_category:
        if sessrs_Pk[f"{k}"] is not None:
            seg = decode(sessrs_Pk[f"{k}"])
            image[flag * seg == 1] = (k * seg * flag)[flag * seg == 1]

    # Multiclass-safe fallback: keep original prediction labels for untouched pixels.
    # This prevents class-collapse behavior when many categories are selected.
    label_segmentation_modify = label_segmentation
    image[image == 255] = label_segmentation_modify[image == 255]

    # Strong multiclass guard (prediction-id space, e.g., 0..13):
    # 1) never introduce a class not present in the original prediction for this image,
    # 2) never drop an originally-present foreground class completely.
    base_classes = set(np.unique(label_segmentation_modify).tolist())
    out_classes = set(np.unique(image).tolist())
    for cls_id in out_classes - base_classes:
        mask_new = image == cls_id
        image[mask_new] = label_segmentation_modify[mask_new]
    for cls_id in sorted(base_classes):
        if not np.any(image == cls_id):
            image[label_segmentation_modify == cls_id] = cls_id

    if ctx.palette is not None:
        pil = Image.fromarray(np.uint8(image), "P")
        pil.putpalette(ctx.palette)
    else:
        # Save as grayscale label map to preserve raw class ids.
        pil = Image.fromarray(np.uint8(image), "L")
    pil.save(ctx.out_dir / path)


def run_sessrs_for_images(image_names: list[str], ctx: SessrsContext, workers: int) -> None:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        for n in tqdm(image_names, desc=f"SESSRS {ctx.out_dir.name}", leave=False):
            sessrs_one_image(n, ctx)
        return
    with mp.Pool(processes=workers) as pool:
        jobs = [(n, ctx) for n in image_names]
        list(tqdm(pool.starmap(sessrs_one_image, jobs), total=len(jobs), desc=f"SESSRS {ctx.out_dir.name}", leave=False))


def compute_miou_present(pred_dir: Path, gt_dir: Path, num_classes: int, image_names: Iterable[str]) -> float:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for name in image_names:
        p = pred_dir / name
        g = gt_dir / name
        if not p.exists() or not g.exists():
            continue
        pred = np.array(Image.open(p), dtype=np.uint8)
        gt_raw = np.array(Image.open(g), dtype=np.uint8)
        valid = (gt_raw >= 1) & (gt_raw <= num_classes)
        if not np.any(valid):
            continue
        gt = gt_raw[valid].astype(np.int64) - 1
        pp = pred[valid].astype(np.int64)
        pp = np.clip(pp, 0, num_classes - 1)
        binc = np.bincount(gt * num_classes + pp, minlength=num_classes * num_classes)
        conf += binc.reshape(num_classes, num_classes)
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0).astype(np.float64) - tp
    fn = conf.sum(axis=1).astype(np.float64) - tp
    iou = np.divide(tp, tp + fp + fn, out=np.full_like(tp, np.nan), where=(tp + fp + fn) > 0)
    present = conf.sum(axis=1) > 0
    return float(np.nanmean(iou[present])) if np.any(present) else float("nan")


def compute_metrics(pred_dir: Path, gt_dir: Path, num_classes: int, image_names: Iterable[str]) -> dict:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for name in image_names:
        p = pred_dir / name
        g = gt_dir / name
        if not p.exists() or not g.exists():
            continue
        pred = np.array(Image.open(p), dtype=np.uint8)
        gt_raw = np.array(Image.open(g), dtype=np.uint8)
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


def infer_predictions(
    run_dir: Path,
    config_path: Path,
    split: str,
    args: argparse.Namespace,
) -> list[str]:
    import sys

    sys.path.insert(0, str(args.repo_root))
    from geoseg.datasets.goldmdd_dataset import GoldMDDDataset, val_aug  # type: ignore
    from tools.cfg import py2cfg  # type: ignore
    from train_supervision import Supervision_Train  # type: ignore

    cfg = py2cfg(config_path)
    ckpt = run_dir / f"{run_dir.name}.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    num_classes = int(cfg.num_classes)
    if num_classes != 14:
        raise RuntimeError(f"Expected 14 classes for GoldMDD, got {num_classes}")

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

    out_pred_dir = run_dir / "sessrs_official" / split / "pre_p"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    image_names = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{run_dir.name}][{split}] infer", leave=False):
            img = batch["img"].to(device, non_blocking=True)
            ids = list(batch["img_id"])
            logits = net(img)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for i, img_id in enumerate(ids):
                name = f"{img_id}.png"
                image_names.append(name)
                out_path = out_pred_dir / name
                if args.overwrite or (not out_path.exists()):
                    Image.fromarray(pred[i], mode="L").save(out_path)
    return image_names


def prepare_jsons_for_split(run_dir: Path, split: str, image_names: list[str], args: argparse.Namespace) -> tuple[Path, Path, Path]:
    split_root = run_dir / "sessrs_official" / split
    pred_dir = split_root / "pre_p"
    pre_p_info_dir = split_root / "pre_p_info"
    sam_info_dir = split_root / "sam_label_info"
    pre_p_info_dir.mkdir(parents=True, exist_ok=True)
    sam_info_dir.mkdir(parents=True, exist_ok=True)

    obj_dir = args.sam_prior_root / split / "object"
    if not obj_dir.exists():
        raise FileNotFoundError(f"object prior dir missing: {obj_dir}")

    for name in tqdm(image_names, desc=f"[{run_dir.name}][{split}] pre_p_info", leave=False):
        out_json = pre_p_info_dir / f"{name[:-4]}.json"
        if args.overwrite or (not out_json.exists()):
            segmentation_to_instance_json(pred_dir / name, out_json)

    for name in tqdm(image_names, desc=f"[{run_dir.name}][{split}] sam_info", leave=False):
        out_json = sam_info_dir / f"{name[:-4]}.json"
        if args.overwrite or (not out_json.exists()):
            object_map_to_sam_json(obj_dir / name, out_json)

    return pred_dir, pre_p_info_dir, sam_info_dir


def search_best_t1_t2(
    run_dir: Path,
    val_image_names: list[str],
    pred_dir: Path,
    pre_p_info_dir: Path,
    sam_info_dir: Path,
    gt_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float], list[int], list[dict]]:
    num_classes = 14
    # Search is done in prediction-id space. For GoldMDD model outputs this is 0..13.
    search_classes = args.search_classes if args.search_classes is not None else list(range(num_classes))
    image_names = val_image_names
    if args.search_max_images is not None:
        image_names = image_names[: args.search_max_images]

    tmp_out = run_dir / "sessrs_official" / "val" / "se_mask_search"
    tmp_out.mkdir(parents=True, exist_ok=True)

    best_t1 = {str(i): 1.0 for i in range(num_classes)}
    best_t2 = {str(i): 1.0 for i in range(num_classes)}
    search_log = []

    for cls in search_classes:
        best_score = -1.0
        best_pair = (1.0, 1.0)
        for t1 in T1_GRID:
            for t2 in T2_GRID:
                fix_t1 = {str(i): 1.0 for i in range(num_classes)}
                fix_t2 = {str(i): 1.0 for i in range(num_classes)}
                fix_t1[str(cls)] = float(t1)
                fix_t2[str(cls)] = float(t2)
                ctx = SessrsContext(
                    num_classes=num_classes,
                    p_dir=pred_dir,
                    pre_p_info_dir=pre_p_info_dir,
                    sam_label_info_dir=sam_info_dir,
                    out_dir=tmp_out,
                    fix_t1=fix_t1,
                    fix_t2=fix_t2,
                    modify_category=[int(cls)],
                )
                run_sessrs_for_images(image_names=image_names, ctx=ctx, workers=args.post_workers)
                score = compute_miou_present(tmp_out, gt_dir, num_classes, image_names)
                search_log.append({"class": int(cls), "t1": float(t1), "t2": float(t2), "miou_present": float(score)})
                if score > best_score:
                    best_score = score
                    best_pair = (float(t1), float(t2))
        best_t1[str(cls)] = best_pair[0]
        best_t2[str(cls)] = best_pair[1]
        print(f"[{run_dir.name}] class={cls} best t1/t2={best_pair} miou_present={best_score:.4f}")

    if args.force_all_foreground:
        modify_category = list(range(num_classes))
    else:
        modify_category = [int(k) for k, v in best_t1.items() if float(v) <= float(args.search_threshold_mode)]
    return best_t1, best_t2, modify_category, search_log


def run_one(run_name: str, args: argparse.Namespace) -> None:
    run_dir = args.experiments_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    cfg = args.repo_root / RUN_TO_CONFIG[run_name]
    if not cfg.exists():
        raise FileNotFoundError(f"config missing: {cfg}")

    print(f"\n=== {run_name} ===")
    split_to_images = {}
    for split in args.splits:
        pred_dir = run_dir / "sessrs_official" / split / "pre_p"
        if args.skip_inference and pred_dir.exists():
            image_names = sorted([p.name for p in pred_dir.glob("*.png")])
            if args.max_samples is not None:
                image_names = image_names[: args.max_samples]
        else:
            image_names = infer_predictions(run_dir, cfg, split, args)
        split_to_images[split] = image_names
        prepare_jsons_for_split(run_dir, split, image_names, args)

    if "val" not in split_to_images:
        raise RuntimeError("Official threshold search requires val split.")

    val_root = run_dir / "sessrs_official" / "val"
    val_pred = val_root / "pre_p"
    val_pre_p_info = val_root / "pre_p_info"
    val_sam_info = val_root / "sam_label_info"
    gt_val = args.data_root / "val" / "label"

    best_t1, best_t2, modify_category, search_log = search_best_t1_t2(
        run_dir=run_dir,
        val_image_names=split_to_images["val"],
        pred_dir=val_pred,
        pre_p_info_dir=val_pre_p_info,
        sam_info_dir=val_sam_info,
        gt_dir=gt_val,
        args=args,
    )

    selection = {
        "run": run_name,
        "search_threshold_mode": float(args.search_threshold_mode),
        "fix_t1": best_t1,
        "fix_t2": best_t2,
        "modify_category": modify_category,
        "grid_t1": T1_GRID,
        "grid_t2": T2_GRID,
        "search_log": search_log,
    }
    (run_dir / "sessrs_official" / "selected_t1_t2.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    for split in args.splits:
        split_root = run_dir / "sessrs_official" / split
        ctx = SessrsContext(
            num_classes=14,
            p_dir=split_root / "pre_p",
            pre_p_info_dir=split_root / "pre_p_info",
            sam_label_info_dir=split_root / "sam_label_info",
            out_dir=split_root / "se_mask",
            fix_t1=best_t1,
            fix_t2=best_t2,
            modify_category=modify_category,
        )
        post_t0 = time.perf_counter()
        run_sessrs_for_images(split_to_images[split], ctx, args.post_workers)
        postprocess_seconds = time.perf_counter() - post_t0
        postprocess_ms_per_image = (
            (postprocess_seconds * 1000.0) / max(len(split_to_images[split]), 1)
        )

        gt_dir = args.data_root / split / "label"
        base_stats = compute_metrics(split_root / "pre_p", gt_dir, 14, split_to_images[split])
        sessrs_stats = compute_metrics(split_root / "se_mask", gt_dir, 14, split_to_images[split])
        delta = {
            "delta_miou": float(sessrs_stats["miou"] - base_stats["miou"]),
            "delta_miou_present": float(sessrs_stats["miou_present"] - base_stats["miou_present"]),
            "delta_macro_f1": float(sessrs_stats["macro_f1"] - base_stats["macro_f1"]),
            "delta_macro_f1_present": float(sessrs_stats["macro_f1_present"] - base_stats["macro_f1_present"]),
            "delta_oa_fg": float(sessrs_stats["oa_fg"] - base_stats["oa_fg"]),
        }
        out = {
            "run": run_name,
            "split": split,
            "selection_file": str(run_dir / "sessrs_official" / "selected_t1_t2.json"),
            "base_pred_metrics": base_stats,
            "sessrs_pred_metrics": sessrs_stats,
            "postprocess_seconds_total": float(postprocess_seconds),
            "postprocess_ms_per_image": float(postprocess_ms_per_image),
            "delta": delta,
        }
        (run_dir / "sessrs_official" / f"{split}_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(
            f"[{run_name}][{split}] "
            f"base_mIoU_present={base_stats['miou_present']:.4f} -> "
            f"sessrs_mIoU_present={sessrs_stats['miou_present']:.4f} "
            f"(delta={delta['delta_miou_present']:+.4f})"
        )


def main() -> None:
    args = parse_args()
    for r in args.runs:
        if r not in RUN_TO_CONFIG:
            raise ValueError(f"Unsupported run: {r}")
    for r in args.runs:
        run_one(r, args)


if __name__ == "__main__":
    main()
