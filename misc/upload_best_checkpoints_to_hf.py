#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def iter_manifest(manifest: Path):
    with manifest.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["relative_path"], int(row.get("size_bytes", 0) or 0)


def main():
    ap = argparse.ArgumentParser(description="Upload best checkpoints listed in CSV manifest to a private HF model repo")
    ap.add_argument("--repo-id", default="IRSC/ELDOR-checkpoints")
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--root", default="/deac/csc/yangGrp/cuij/GoldMDD/experiments")
    ap.add_argument(
        "--manifest",
        default="/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics/hf_best_checkpoints_manifest.csv",
    )
    ap.add_argument("--path-prefix", default="segmentation/checkpoints")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--sleep-sec", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    manifest = Path(args.manifest)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    print(f"[repo] ready: {args.repo_id}", flush=True)

    rows = list(iter_manifest(manifest))
    total = len(rows)
    total_bytes = sum(sz for _, sz in rows)
    print(f"[plan] files={total}, total={total_bytes / (1024**3):.2f} GB", flush=True)

    failures = []
    uploaded = 0
    uploaded_bytes = 0

    for i, (rel, sz) in enumerate(rows, start=1):
        src = root / rel
        if not src.exists():
            print(f"[{i}/{total}] MISSING {rel}", flush=True)
            failures.append((rel, "missing"))
            continue

        dst = f"{args.path_prefix}/{rel}"
        ok = False
        for k in range(1, args.max_retries + 1):
            try:
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=dst,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                )
                dt = time.time() - t0
                uploaded += 1
                uploaded_bytes += sz
                mbps = (sz / (1024**2)) / max(dt, 1e-6)
                print(
                    f"[{i}/{total}] OK {rel} ({sz / (1024**2):.1f} MB, {dt:.1f}s, {mbps:.1f} MB/s) "
                    f"uploaded={uploaded}/{total}",
                    flush=True,
                )
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{total}] RETRY {k}/{args.max_retries} {rel} :: {e}", flush=True)
                time.sleep(args.sleep_sec)
        if not ok:
            failures.append((rel, "upload_failed"))

    # upload metadata files
    extras = [
        (
            "/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics/summary.md",
            "segmentation/summary.md",
        ),
        (str(manifest), "segmentation/hf_best_checkpoints_manifest.csv"),
    ]
    for src, dst in extras:
        p = Path(src)
        if p.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=dst,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                )
                print(f"[meta] uploaded {dst}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[meta] failed {dst}: {e}", flush=True)

    print(
        f"[done] uploaded={uploaded}/{total}, bytes={uploaded_bytes / (1024**3):.2f} GB, "
        f"failed={len(failures)}",
        flush=True,
    )
    if failures:
        fail_path = root / "diagnostics" / "hf_upload_failures.txt"
        with fail_path.open("w") as f:
            for rel, why in failures:
                f.write(f"{why}\t{rel}\n")
        print(f"[done] failure list: {fail_path}", flush=True)


if __name__ == "__main__":
    main()
