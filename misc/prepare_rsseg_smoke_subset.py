#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def link_subset(src_dir: Path, dst_dir: Path, n: int, seed: int) -> None:
    img_src = src_dir / "image"
    lbl_src = src_dir / "label"
    img_dst = dst_dir / "image"
    lbl_dst = dst_dir / "label"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in img_src.glob("*.jpg"))
    stems = [s for s in stems if (lbl_src / f"{s}.png").exists()]
    if n > len(stems):
        n = len(stems)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(stems, n))

    for stem in chosen:
        src_i = img_src / f"{stem}.jpg"
        src_l = lbl_src / f"{stem}.png"
        dst_i = img_dst / f"{stem}.jpg"
        dst_l = lbl_dst / f"{stem}.png"
        if dst_i.exists() or dst_l.exists():
            continue
        dst_i.symlink_to(src_i)
        dst_l.symlink_to(src_l)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    ap.add_argument("--dst-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-rsseg-smoke"))
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset and args.dst_root.exists():
        for p in sorted(args.dst_root.rglob("*"), reverse=True):
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()

    link_subset(args.src_root / "train", args.dst_root / "train", args.n_train, args.seed + 1)
    link_subset(args.src_root / "val", args.dst_root / "val", args.n_val, args.seed + 2)
    link_subset(args.src_root / "test", args.dst_root / "test", args.n_test, args.seed + 3)

    for split in ["train", "val", "test"]:
        ni = len(list((args.dst_root / split / "image").glob("*.jpg")))
        nl = len(list((args.dst_root / split / "label").glob("*.png")))
        print(f"{split}: image={ni}, label={nl}")


if __name__ == "__main__":
    main()
