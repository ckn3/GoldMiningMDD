#!/usr/bin/env python3
"""
Build GoldMDD object-detection patches from semantic labels.

Agreed rules:
- Background classes: {0, 3, 5, 9, 10, 11, 12}
- Foreground classes: [1,2,4,6,7,8,13,14] (8 classes)
- Patch size: 1024, stride: 512, cover-all windows
- Drop patch if class-0 ratio > 50%
- Per-class connected components (8-connectivity) -> bboxes
- Filter bbox if area < 64 or min(w, h) < 3
- Additional edge filter: if bbox touches patch edge and area < 128, drop
- Keep all remaining patches, including empty-label patches
- Export both COCO and YOLO labels
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

SPLITS = ("train", "val", "test")

# raw merged IDs in GoldMDD/README.md
FOREGROUND_RAW_IDS = [1, 2, 4, 6, 7, 8, 13, 14]
BACKGROUND_IDS = {0, 3, 5, 9, 10, 11, 12}
RAW_ID_TO_NAME = {
    1: "Building",
    2: "Mining raft",
    4: "Heavy machinery",
    5: "Water bodies",
    6: "Agricultural crop",
    7: "Compact mounds",
    8: "Gravel mounds",
    10: "Type 1 natural regeneration",
    11: "Type 2 natural regeneration",
    13: "Sluice",
    14: "Vehicles",
}

# COCO category IDs are 1-based contiguous
RAW_TO_COCO = {raw_id: idx + 1 for idx, raw_id in enumerate(FOREGROUND_RAW_IDS)}
COCO_TO_RAW = {v: k for k, v in RAW_TO_COCO.items()}

COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
]


@dataclass
class SplitState:
    images: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    next_image_id: int = 1
    next_ann_id: int = 1
    candidate_windows: int = 0
    kept_windows: int = 0
    dropped_zero_windows: int = 0
    num_boxes: int = 0


@dataclass
class SiteStats:
    split: str
    site: str
    candidate_windows: int
    kept_windows: int
    dropped_zero_windows: int
    num_boxes: int


@dataclass
class PreviewEntry:
    split: str
    site: str
    stem: str
    image_path: Path
    anns: list[dict]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    goldmdd = here.parents[1]
    parser = argparse.ArgumentParser(description="Build GoldMDD detection dataset from merged semantic labels.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=goldmdd / "data",
        help="GoldMDD semantic data root (contains train/val/test with image/label folders).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=goldmdd / "data-detection",
        help="Output detection dataset root.",
    )
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--drop-zero-ratio",
        type=float,
        default=0.5,
        help="Drop patch when class-0 ratio is strictly greater than this value.",
    )
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--min-side", type=int, default=3)
    parser.add_argument("--edge-min-area", type=int, default=128)
    parser.add_argument(
        "--merge-contained-thresh",
        type=float,
        default=0.8,
        help="Merge same-class boxes when overlap covers at least this fraction of the smaller box.",
    )
    parser.add_argument(
        "--merge-max-iters",
        type=int,
        default=5,
        help="Maximum iterative rounds for same-class containment-based merge.",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        help="Only process specific site(s). Can be repeated, e.g. --site Nayda --site Clavelito",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--clean", action="store_true", help="Delete output root before generation.")
    parser.add_argument(
        "--vis-count",
        type=int,
        default=24,
        help="Number of preview patches (with boxes) to render.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_starts(length: int, patch: int, stride: int) -> list[int]:
    if length < patch:
        raise ValueError(f"Image dimension {length} is smaller than patch size {patch}.")
    max_start = length - patch
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def iter_windows(width: int, height: int, patch: int, stride: int):
    ys = compute_starts(height, patch, stride)
    xs = compute_starts(width, patch, stride)
    for row_idx, y in enumerate(ys, start=1):
        for col_idx, x in enumerate(xs, start=1):
            yield row_idx, col_idx, x, y


def prepare_output_root(out_root: Path, clean: bool) -> None:
    if clean and out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)
    (out_root / "previews").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)


def list_sites(data_root: Path, selected_sites: set[str] | None) -> list[tuple[str, str, Path, Path]]:
    tasks: list[tuple[str, str, Path, Path]] = []
    for split in SPLITS:
        img_dir = data_root / split / "image"
        lbl_dir = data_root / split / "label"
        for img_path in sorted(img_dir.glob("*.png")):
            site = img_path.stem
            if selected_sites is not None and site not in selected_sites:
                continue
            lbl_path = lbl_dir / f"{site}.png"
            if not lbl_path.exists():
                raise FileNotFoundError(f"Missing label for {img_path}: {lbl_path}")
            tasks.append((split, site, img_path, lbl_path))
    return tasks


def bboxes_intersection_area(box_a: list[int], box_b: list[int]) -> int:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x2 = ax + aw
    a_y2 = ay + ah
    b_x2 = bx + bw
    b_y2 = by + bh
    inter_w = min(a_x2, b_x2) - max(ax, bx)
    inter_h = min(a_y2, b_y2) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0
    return int(inter_w * inter_h)


def should_merge_by_small_coverage(box_a: list[int], box_b: list[int], merge_contained_thresh: float) -> bool:
    inter = bboxes_intersection_area(box_a, box_b)
    if inter <= 0:
        return False
    area_a = int(box_a[2] * box_a[3])
    area_b = int(box_b[2] * box_b[3])
    small = max(1, min(area_a, area_b))
    return (float(inter) / float(small)) >= float(merge_contained_thresh)


def merge_same_class_overlaps(anns: list[dict], merge_contained_thresh: float, merge_max_iters: int) -> list[dict]:
    """Iteratively merge same-class boxes when the smaller box is mostly contained."""
    if len(anns) <= 1 or merge_max_iters <= 0:
        return anns

    by_cat: dict[int, list[dict]] = {}
    for ann in anns:
        cid = int(ann["category_id"])
        by_cat.setdefault(cid, []).append(ann)

    merged_all: list[dict] = []
    for cid in sorted(by_cat.keys()):
        current = list(by_cat[cid])

        for _ in range(int(merge_max_iters)):
            n = len(current)
            if n <= 1:
                break

            parent = list(range(n))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra = find(a)
                rb = find(b)
                if ra != rb:
                    parent[rb] = ra

            for i in range(n):
                for j in range(i + 1, n):
                    if should_merge_by_small_coverage(
                        current[i]["bbox"], current[j]["bbox"], merge_contained_thresh=merge_contained_thresh
                    ):
                        union(i, j)

            comps: dict[int, list[int]] = {}
            for i in range(n):
                r = find(i)
                comps.setdefault(r, []).append(i)

            changed = any(len(comp) > 1 for comp in comps.values())
            next_current: list[dict] = []
            for comp in comps.values():
                if len(comp) == 1:
                    next_current.append(current[comp[0]])
                    continue

                x1 = min(current[k]["bbox"][0] for k in comp)
                y1 = min(current[k]["bbox"][1] for k in comp)
                x2 = max(current[k]["bbox"][0] + current[k]["bbox"][2] for k in comp)
                y2 = max(current[k]["bbox"][1] + current[k]["bbox"][3] for k in comp)
                w = int(x2 - x1)
                h = int(y2 - y1)
                next_current.append(
                    {
                        "category_id": cid,
                        "raw_class_id": COCO_TO_RAW[cid],
                        "bbox": [int(x1), int(y1), w, h],
                        "area": int(w * h),
                        "iscrowd": 0,
                    }
                )

            current = next_current
            if not changed:
                break

        merged_all.extend(current)

    return merged_all


def extract_patch_annotations(
    lbl_patch: np.ndarray,
    patch_size: int,
    min_area: int,
    min_side: int,
    edge_min_area: int,
    merge_contained_thresh: float,
    merge_max_iters: int,
) -> list[dict]:
    out: list[dict] = []
    present_ids = [int(v) for v in np.unique(lbl_patch) if int(v) in RAW_TO_COCO]
    for raw_id in present_ids:
        mask = (lbl_patch == raw_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < min_area:
                continue
            if min(w, h) < min_side:
                continue

            touches_edge = (x == 0) or (y == 0) or ((x + w) >= patch_size) or ((y + h) >= patch_size)
            if touches_edge and area < edge_min_area:
                continue

            out.append(
                {
                    "category_id": RAW_TO_COCO[raw_id],
                    "raw_class_id": raw_id,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
            )
    return merge_same_class_overlaps(
        out, merge_contained_thresh=merge_contained_thresh, merge_max_iters=merge_max_iters
    )


def write_yolo_label(path: Path, anns: Iterable[dict], patch_size: int) -> None:
    lines: list[str] = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cls = int(ann["category_id"]) - 1  # YOLO class index is 0-based
        cx = (x + (w / 2.0)) / float(patch_size)
        cy = (y + (h / 2.0)) / float(patch_size)
        nw = w / float(patch_size)
        nh = h / float(patch_size)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_preview(entry: PreviewEntry, out_path: Path) -> None:
    color_by_coco = {idx + 1: COLORS[idx % len(COLORS)] for idx in range(len(FOREGROUND_RAW_IDS))}
    with Image.open(entry.image_path) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for ann in entry.anns:
        x, y, w, h = ann["bbox"]
        coco_id = int(ann["category_id"])
        raw_id = COCO_TO_RAW[coco_id]
        color = color_by_coco[coco_id]
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=color, width=2)
        label = f"{coco_id-1}:{RAW_ID_TO_NAME[raw_id]}"
        tx = x + 2
        ty = y + 2
        text_bbox = draw.textbbox((tx, ty), label)
        draw.rectangle(text_bbox, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color)
    canvas.save(out_path, format="JPEG", quality=92)


def build_categories() -> list[dict]:
    return [
        {
            "id": RAW_TO_COCO[raw_id],
            "name": RAW_ID_TO_NAME[raw_id],
            "supercategory": "mining_related",
            "raw_class_id": raw_id,
            "yolo_class_id": RAW_TO_COCO[raw_id] - 1,
        }
        for raw_id in FOREGROUND_RAW_IDS
    ]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    selected_sites = set(args.site) if args.site else None
    prepare_output_root(args.out_root, clean=args.clean)
    tasks = list_sites(args.data_root, selected_sites=selected_sites)
    if not tasks:
        raise RuntimeError("No sites matched the current filters.")

    split_states = {split: SplitState() for split in SPLITS}
    site_stats: list[SiteStats] = []
    preview_pool: list[PreviewEntry] = []

    for split, site, img_path, lbl_path in tasks:
        print(f"[{split}] Processing site: {site}", flush=True)
        state = split_states[split]
        site_candidate = 0
        site_kept = 0
        site_dropped_zero = 0
        site_boxes = 0

        with Image.open(img_path) as img, Image.open(lbl_path) as lbl:
            if img.size != lbl.size:
                raise RuntimeError(f"Image/label size mismatch for {site}: {img.size} vs {lbl.size}")
            width, height = img.size

            for row_idx, col_idx, x, y in iter_windows(width, height, args.patch_size, args.stride):
                site_candidate += 1
                state.candidate_windows += 1

                lbl_patch = np.asarray(lbl.crop((x, y, x + args.patch_size, y + args.patch_size)), dtype=np.uint8)
                zero_ratio = float((lbl_patch == 0).sum()) / float(args.patch_size * args.patch_size)
                if zero_ratio > args.drop_zero_ratio:
                    site_dropped_zero += 1
                    state.dropped_zero_windows += 1
                    continue

                anns = extract_patch_annotations(
                    lbl_patch=lbl_patch,
                    patch_size=args.patch_size,
                    min_area=args.min_area,
                    min_side=args.min_side,
                    edge_min_area=args.edge_min_area,
                    merge_contained_thresh=args.merge_contained_thresh,
                    merge_max_iters=args.merge_max_iters,
                )

                stem = f"{site}_{row_idx}_{col_idx}"
                out_img_path = args.out_root / split / "images" / f"{stem}.jpg"
                out_lbl_path = args.out_root / split / "labels" / f"{stem}.txt"

                img_patch = img.crop((x, y, x + args.patch_size, y + args.patch_size))
                img_patch.save(out_img_path, format="JPEG", quality=args.jpeg_quality)
                write_yolo_label(out_lbl_path, anns, patch_size=args.patch_size)

                image_id = state.next_image_id
                state.next_image_id += 1
                state.images.append(
                    {
                        "id": image_id,
                        "file_name": f"{split}/images/{stem}.jpg",
                        "width": args.patch_size,
                        "height": args.patch_size,
                        "site": site,
                        "row": row_idx,
                        "col": col_idx,
                    }
                )

                for ann in anns:
                    ann_id = state.next_ann_id
                    state.next_ann_id += 1
                    x0, y0, w, h = ann["bbox"]
                    state.annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": ann["category_id"],
                            "bbox": [float(x0), float(y0), float(w), float(h)],
                            "area": float(ann["area"]),
                            "iscrowd": 0,
                        }
                    )

                site_kept += 1
                state.kept_windows += 1
                site_boxes += len(anns)
                state.num_boxes += len(anns)

                if anns:
                    preview_pool.append(
                        PreviewEntry(
                            split=split,
                            site=site,
                            stem=stem,
                            image_path=out_img_path,
                            anns=anns,
                        )
                    )

                if site_candidate % 250 == 0:
                    print(
                        f"  [{split}/{site}] windows={site_candidate}, kept={site_kept}, "
                        f"dropped_zero={site_dropped_zero}, boxes={site_boxes}",
                        flush=True,
                    )

        site_stats.append(
            SiteStats(
                split=split,
                site=site,
                candidate_windows=site_candidate,
                kept_windows=site_kept,
                dropped_zero_windows=site_dropped_zero,
                num_boxes=site_boxes,
            )
        )
        print(
            f"[{split}] Done {site}: candidates={site_candidate}, kept={site_kept}, "
            f"dropped_zero={site_dropped_zero}, boxes={site_boxes}",
            flush=True,
        )

    categories = build_categories()
    info = {
        "description": "GoldMDD detection patches built from merged semantic labels",
        "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "drop_zero_ratio": args.drop_zero_ratio,
        "background_ids": sorted(BACKGROUND_IDS),
        "foreground_raw_ids": FOREGROUND_RAW_IDS,
        "merge_contained_thresh": args.merge_contained_thresh,
        "merge_max_iters": args.merge_max_iters,
    }

    for split in SPLITS:
        st = split_states[split]
        coco = {
            "info": info,
            "licenses": [],
            "images": st.images,
            "annotations": st.annotations,
            "categories": categories,
        }
        out_json = args.out_root / "annotations" / f"instances_{split}.json"
        out_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    mapping = {
        "background_ids": sorted(BACKGROUND_IDS),
        "foreground_raw_ids": FOREGROUND_RAW_IDS,
        "raw_to_coco": RAW_TO_COCO,
        "coco_to_raw": COCO_TO_RAW,
        "raw_id_to_name": RAW_ID_TO_NAME,
        "yolo_class_names": [RAW_ID_TO_NAME[raw_id] for raw_id in FOREGROUND_RAW_IDS],
    }
    (args.out_root / "annotations" / "class_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )

    yolo_yaml = (
        f"path: {args.out_root}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(FOREGROUND_RAW_IDS)}\n"
        "names:\n"
        + "\n".join(f"  {i}: {RAW_ID_TO_NAME[raw_id]}" for i, raw_id in enumerate(FOREGROUND_RAW_IDS))
        + "\n"
    )
    (args.out_root / "dataset_yolo.yaml").write_text(yolo_yaml, encoding="utf-8")

    site_csv = args.out_root / "site_summary.csv"
    with site_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "site", "candidate_windows", "kept_windows", "dropped_zero_windows", "num_boxes"])
        for row in sorted(site_stats, key=lambda x: (SPLITS.index(x.split), x.site)):
            w.writerow(
                [
                    row.split,
                    row.site,
                    row.candidate_windows,
                    row.kept_windows,
                    row.dropped_zero_windows,
                    row.num_boxes,
                ]
            )

    split_csv = args.out_root / "split_summary.csv"
    with split_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "candidate_windows", "kept_windows", "dropped_zero_windows", "num_boxes", "num_images"])
        for split in SPLITS:
            st = split_states[split]
            w.writerow(
                [
                    split,
                    st.candidate_windows,
                    st.kept_windows,
                    st.dropped_zero_windows,
                    st.num_boxes,
                    len(st.images),
                ]
            )

    preview_pool.sort(key=lambda x: len(x.anns), reverse=True)
    for i, entry in enumerate(preview_pool[: max(0, args.vis_count)]):
        out_preview = args.out_root / "previews" / f"{i+1:03d}_{entry.split}_{entry.stem}.jpg"
        draw_preview(entry, out_preview)

    print("\nGeneration finished.", flush=True)
    print(f"- Output root: {args.out_root}", flush=True)
    print(f"- Sites processed: {len(site_stats)}", flush=True)
    for split in SPLITS:
        st = split_states[split]
        print(
            f"- {split}: candidates={st.candidate_windows}, kept={st.kept_windows}, "
            f"dropped_zero={st.dropped_zero_windows}, images={len(st.images)}, boxes={st.num_boxes}",
            flush=True,
        )
    print(f"- Preview images: {min(len(preview_pool), max(0, args.vis_count))}", flush=True)


if __name__ == "__main__":
    main()
