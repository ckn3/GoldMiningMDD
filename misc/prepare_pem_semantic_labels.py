#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SRC = Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped")
DEFAULT_DST = Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-pem")
SPLITS = ("train", "val", "test")


def remap_mask(src_path: Path, dst_path: Path) -> None:
    arr = np.array(Image.open(src_path), dtype=np.uint8)
    invalid = arr > 14
    if np.any(invalid):
        bad = np.unique(arr[invalid]).tolist()
        raise ValueError(f"Unexpected label ids in {src_path}: {bad}")
    out = np.full(arr.shape, 255, dtype=np.uint8)
    fg = arr > 0
    out[fg] = arr[fg] - 1
    Image.fromarray(out, mode="L").save(dst_path)


def ensure_dir_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        current = Path(os.readlink(link_path))
        if current == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"{link_path} exists and is not a symlink")
    link_path.symlink_to(target_path, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GoldMDD semantic-seg labels for PEM.")
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST)
    parser.add_argument("--force", action="store_true", help="Regenerate existing remapped labels.")
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    for split in SPLITS:
        src_img_dir = src_root / split / "image"
        src_lbl_dir = src_root / split / "label"
        dst_split_dir = dst_root / split
        dst_lbl_dir = dst_split_dir / "label"
        dst_split_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        ensure_dir_symlink(dst_split_dir / "image", src_img_dir)

        for src_lbl in sorted(src_lbl_dir.glob("*.png")):
            total += 1
            dst_lbl = dst_lbl_dir / src_lbl.name
            if dst_lbl.exists() and not args.force:
                continue
            remap_mask(src_lbl, dst_lbl)
            written += 1

    print(f"Prepared PEM semantic labels under {dst_root}")
    print(f"Processed {total} label files; wrote {written} files")


if __name__ == "__main__":
    main()
