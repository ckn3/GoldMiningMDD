#!/deac/opt/rocky9-noarch/anaconda3/bin/python
"""
Build a cropped GoldMDD patch dataset from train/val/test splits.

- Patch size: 512x512
- Stride: 256
- Drop patch if label background (ID=0) ratio > 80%
- Output patch naming: <site>_<row>_<col>.png (1-based row/col window index)

Also writes summary CSVs and updates GoldMDD/README.md with crop statistics.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


ROOT = Path("/deac/csc/alqahtaniGrp/cuij")
GOLDMDD = ROOT / "GoldMDD"
SRC_ROOT = GOLDMDD / "data"
OUT_ROOT = GOLDMDD / "data-cropped"
README_PATH = GOLDMDD / "README.md"

PATCH_SIZE = 512
STRIDE = 256
BACKGROUND_ID = 0
MAX_BG_RATIO = 0.80
IMAGE_EXT = ".jpg"
LABEL_EXT = ".png"
JPEG_QUALITY = 90
PNG_COMPRESS_LEVEL = 4


@dataclass
class SiteCropStats:
    split: str
    site: str
    source_width: int
    source_height: int
    candidate_windows: int
    kept_windows: int
    dropped_bg_windows: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4, help="Number of worker processes (site-level parallelism).")
    return p.parse_args()


def split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / split / "image", root / split / "label"


def iter_windows(width: int, height: int, patch: int, stride: int):
    row_idx = 0
    for y in range(0, height - patch + 1, stride):
        row_idx += 1
        col_idx = 0
        for x in range(0, width - patch + 1, stride):
            col_idx += 1
            yield row_idx, col_idx, x, y


def prepare_output_root(out_root: Path) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val", "test"):
        (out_root / split / "image").mkdir(parents=True, exist_ok=True)
        (out_root / split / "label").mkdir(parents=True, exist_ok=True)


def _process_site(task: tuple[str, str, str, str, str]) -> SiteCropStats:
    split, site, img_path_s, lbl_path_s, out_root_s = task
    img_path = Path(img_path_s)
    lbl_path = Path(lbl_path_s)
    out_img_dir = Path(out_root_s) / split / "image"
    out_lbl_dir = Path(out_root_s) / split / "label"

    if not lbl_path.exists():
        raise FileNotFoundError(f"Missing label for {img_path}: {lbl_path}")

    with Image.open(img_path) as img, Image.open(lbl_path) as lbl:
        if img.size != lbl.size:
            raise RuntimeError(f"Image/label size mismatch for {site}: {img.size} vs {lbl.size}")
        width, height = img.size

        candidates = 0
        kept = 0
        dropped_bg = 0

        for r, c, x, y in iter_windows(width, height, PATCH_SIZE, STRIDE):
            candidates += 1
            lbl_patch = lbl.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            lbl_arr = np.asarray(lbl_patch, dtype=np.uint8)
            bg_ratio = float((lbl_arr == BACKGROUND_ID).sum()) / float(PATCH_SIZE * PATCH_SIZE)
            if bg_ratio > MAX_BG_RATIO:
                dropped_bg += 1
                continue

            stem = f"{site}_{r}_{c}"
            img_patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            img_patch.save(out_img_dir / f"{stem}{IMAGE_EXT}", format="JPEG", quality=JPEG_QUALITY)
            Image.fromarray(lbl_arr, mode="L").save(
                out_lbl_dir / f"{stem}{LABEL_EXT}", format="PNG", compress_level=PNG_COMPRESS_LEVEL
            )
            kept += 1

    return SiteCropStats(
        split=split,
        site=site,
        source_width=width,
        source_height=height,
        candidate_windows=candidates,
        kept_windows=kept,
        dropped_bg_windows=dropped_bg,
    )


def build_crops(workers: int) -> tuple[list[SiteCropStats], dict[str, int]]:
    prepare_output_root(OUT_ROOT)

    site_stats: list[SiteCropStats] = []
    patch_counts = {"train": 0, "val": 0, "test": 0}

    tasks: list[tuple[str, str, str, str, str]] = []
    for split in ("train", "val", "test"):
        src_img_dir, src_lbl_dir = split_dirs(SRC_ROOT, split)
        for img_path in sorted(src_img_dir.glob("*.png")):
            site = img_path.stem
            lbl_path = src_lbl_dir / f"{site}.png"
            tasks.append((split, site, str(img_path), str(lbl_path), str(OUT_ROOT)))

    workers = max(1, min(int(workers), len(tasks)))
    print(f"Cropping with {workers} worker(s) across {len(tasks)} sites", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_site, t) for t in tasks]
        for fut in as_completed(futs):
            s = fut.result()
            site_stats.append(s)
            patch_counts[s.split] += s.kept_windows
            print(
                f"[{s.split}] {s.site}: candidates={s.candidate_windows}, kept={s.kept_windows}, dropped_bg={s.dropped_bg_windows}",
                flush=True,
            )

    # stable order in outputs
    split_rank = {"train": 0, "val": 1, "test": 2}
    site_stats.sort(key=lambda s: (split_rank[s.split], s.site.lower()))

    return site_stats, patch_counts


def write_summary_csvs(site_stats: list[SiteCropStats], patch_counts: dict[str, int]) -> None:
    site_csv = OUT_ROOT / "crop_summary_by_site.csv"
    with site_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "site",
                "source_size",
                "candidate_windows",
                "kept_windows",
                "dropped_bg_windows",
                "kept_ratio",
            ]
        )
        for s in site_stats:
            kept_ratio = (s.kept_windows / s.candidate_windows) if s.candidate_windows else 0.0
            w.writerow(
                [
                    s.split,
                    s.site,
                    f"{s.source_width}x{s.source_height}",
                    s.candidate_windows,
                    s.kept_windows,
                    s.dropped_bg_windows,
                    f"{kept_ratio:.6f}",
                ]
            )

    by_split = defaultdict(lambda: {"sites": 0, "candidates": 0, "kept": 0, "dropped_bg": 0})
    for s in site_stats:
        d = by_split[s.split]
        d["sites"] += 1
        d["candidates"] += s.candidate_windows
        d["kept"] += s.kept_windows
        d["dropped_bg"] += s.dropped_bg_windows

    split_csv = OUT_ROOT / "crop_summary_by_split.csv"
    with split_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "num_sites", "candidate_windows", "kept_windows", "dropped_bg_windows", "kept_ratio"])
        for split in ("train", "val", "test"):
            d = by_split[split]
            kept_ratio = (d["kept"] / d["candidates"]) if d["candidates"] else 0.0
            w.writerow([split, d["sites"], d["candidates"], d["kept"], d["dropped_bg"], f"{kept_ratio:.6f}"])


def build_cropped_readme_section(site_stats: list[SiteCropStats]) -> str:
    by_split = defaultdict(lambda: {"sites": 0, "candidates": 0, "kept": 0, "dropped_bg": 0})
    for s in site_stats:
        d = by_split[s.split]
        d["sites"] += 1
        d["candidates"] += s.candidate_windows
        d["kept"] += s.kept_windows
        d["dropped_bg"] += s.dropped_bg_windows

    total_candidates = sum(s.candidate_windows for s in site_stats)
    total_kept = sum(s.kept_windows for s in site_stats)
    total_dropped = sum(s.dropped_bg_windows for s in site_stats)

    lines: list[str] = []
    lines += ["", "## Cropped patch dataset (`data-cropped`)", ""]
    lines.append(f"- Output root: `GoldMDD/data-cropped`")
    lines.append(f"- Patch size: `{PATCH_SIZE}x{PATCH_SIZE}`")
    lines.append(f"- Stride: `{STRIDE}`")
    lines.append(
        f"- Filtering rule: drop a patch if background pixels in the merged label (`label=={BACKGROUND_ID}`) are `>{int(MAX_BG_RATIO*100)}%`."
    )
    lines.append("- Windowing rule: full windows only (no padding).")
    lines.append(
        "- Patch naming: matching basenames per pair, e.g. image `AcumulacionAaron2B_2_3.jpg` and label `AcumulacionAaron2B_2_3.png` (1-based row/col indices)."
    )
    lines.append("- Storage format: image patches = JPEG, label patches = PNG (lossless class IDs).")
    lines.append("- Folder structure: `train/image`, `train/label`, `val/image`, `val/label`, `test/image`, `test/label`.")
    lines.append("")
    lines.append("### Crop summary by split")
    lines.append("")
    lines.append("| Split | # Sites | Candidate windows | Kept patches | Dropped (>80% bg) | Kept ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for split in ("train", "val", "test"):
        d = by_split[split]
        kept_ratio = (d["kept"] / d["candidates"]) if d["candidates"] else 0.0
        lines.append(
            f"| {split} | {d['sites']} | {d['candidates']:,} | {d['kept']:,} | {d['dropped_bg']:,} | {kept_ratio:.3f} |"
        )
    total_ratio = (total_kept / total_candidates) if total_candidates else 0.0
    lines.append(
        f"| **Total** | **{len(site_stats)}** | **{total_candidates:,}** | **{total_kept:,}** | **{total_dropped:,}** | **{total_ratio:.3f}** |"
    )

    lines.append("")
    lines.append("### Crop summary by site")
    lines.append("")
    lines.append("| Split | Site | Source size (W x H) | Candidate windows | Kept patches | Dropped (>80% bg) | Kept ratio |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for s in site_stats:
        kept_ratio = (s.kept_windows / s.candidate_windows) if s.candidate_windows else 0.0
        lines.append(
            f"| {s.split} | {s.site} | {s.source_width}x{s.source_height} | {s.candidate_windows:,} | {s.kept_windows:,} | {s.dropped_bg_windows:,} | {kept_ratio:.3f} |"
        )

    lines.append("")
    lines.append("- Summary CSVs:")
    lines.append("  - `GoldMDD/data-cropped/crop_summary_by_split.csv`")
    lines.append("  - `GoldMDD/data-cropped/crop_summary_by_site.csv`")

    return "\n".join(lines).rstrip() + "\n"


def update_readme_with_crop_section(site_stats: list[SiteCropStats]) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    new_section = build_cropped_readme_section(site_stats)
    pattern = re.compile(r"\n## Cropped patch dataset \(`data-cropped`\)\n.*\Z", re.S)
    if pattern.search(text):
        text = pattern.sub(new_section.rstrip("\n"), text).rstrip() + "\n"
    else:
        text = text.rstrip() + "\n" + new_section
    README_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    site_stats, patch_counts = build_crops(workers=args.workers)
    write_summary_csvs(site_stats, patch_counts)
    update_readme_with_crop_section(site_stats)

    print("Done.")
    for split in ("train", "val", "test"):
        print(f"{split}: {patch_counts[split]:,} kept patches")
    print(f"Output: {OUT_ROOT}")
    print(f"README updated: {README_PATH}")


if __name__ == "__main__":
    main()
