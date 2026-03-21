#!/deac/opt/rocky9-noarch/anaconda3/bin/python
"""
Rebuild GoldMDD labels with merged classes, update README, and generate a site-class heatmap.

This script only writes under GoldMDD and reads source labels from GoldMining/Data/Label.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


ROOT = Path("/deac/csc/alqahtaniGrp/cuij")
GOLDMDD = ROOT / "GoldMDD"
GOLDMDD_DATA = GOLDMDD / "data"
GOLDMINING = ROOT / "GoldMining" / "Data"

SRC_LABEL_DIR = GOLDMINING / "Label"
TRAIN_LABEL_DIR = GOLDMDD_DATA / "train" / "label"
VAL_LABEL_DIR = GOLDMDD_DATA / "val" / "label"
TEST_LABEL_DIR = GOLDMDD_DATA / "test" / "label"
TRAIN_IMAGE_DIR = GOLDMDD_DATA / "train" / "image"
VAL_IMAGE_DIR = GOLDMDD_DATA / "val" / "image"
TEST_IMAGE_DIR = GOLDMDD_DATA / "test" / "image"

README_SRC = GOLDMDD / "README.md"
README_REF = GOLDMINING / "README.md"
README_OUT = GOLDMDD / "README.md"

HEATMAP_PNG = GOLDMDD / "site_class_pixel_counts_heatmap_merged.png"
HEATMAP_CSV = GOLDMDD / "site_class_pixel_counts_merged.csv"
SPLIT_DIST_PNG = GOLDMDD / "train_val_test_class_distribution_merged.png"
SPLIT_DIST_CSV = GOLDMDD / "train_val_test_class_distribution_merged.csv"

SITE_ORDER = [
    "AcumulacionAaron2B",
    "Anel",
    "Clavelito",
    "ElEngano",
    "Kotsimba",
    "Linda",
    "Los5Rebeldes",
    "Nayda",
    "Paolita1",
    "PlayaMirador",
    "SantaInesDosMil",
]
TRAIN_SITES = {"AcumulacionAaron2B", "Kotsimba", "Los5Rebeldes", "PlayaMirador"}
VAL_SITES = {"Clavelito", "Nayda"}
OUTPUT_SITE_RENAMES = {
    "Paolita1": "Paolita",
    "PlayaMirador": "PlayaMirador1",
}
PLAYA_SPLIT_TRAIN_NAME = "PlayaMirador1"
PLAYA_SPLIT_TOP_NAME = "PlayaMirador2"

# old canonical ID -> new canonical ID (0=background unchanged)
OLD_TO_NEW = {
    0: 0,
    1: 1,   # Urban area -> Building
    2: 2,   # Mining raft
    3: 3,   # Primary Forest
    4: 1,   # Mining camp -> Building
    5: 4,   # Front loader -> Heavy machinery
    6: 5,   # Water bodies
    7: 6,   # Agricultural crop
    8: 4,   # Excavator -> Heavy machinery
    9: 4,   # Heavy machinery
    10: 7,  # Compact mounds
    11: 8,  # Gravel mounds
    12: 9,  # Grass
    13: 10, # Type 1 natural regeneration
    14: 11, # Type 2 natural regeneration
    15: 12, # Bare ground
    16: 13, # Sluice
    17: 14, # Vehicles
    18: 14, # Small vehicles -> Vehicles
    19: 4,  # Dump truck -> Heavy machinery
}

# new canonical ID -> metadata
MERGED_CLASS_DEFS = OrderedDict(
    {
        1: {"name": "Building", "old_ids": [1, 4], "hex": "#8A6A3D"},
        2: {"name": "Mining raft", "old_ids": [2], "hex": "#7BEBFB"},
        3: {"name": "Primary Forest", "old_ids": [3], "hex": "#B04C18"},
        4: {"name": "Heavy machinery", "old_ids": [5, 8, 9, 19], "hex": "#EE92C6"},
        5: {"name": "Water bodies", "old_ids": [6], "hex": "#4F6F6F"},
        6: {"name": "Agricultural crop", "old_ids": [7], "hex": "#84D08C"},
        7: {"name": "Compact mounds", "old_ids": [10], "hex": "#23F3E3"},
        8: {"name": "Gravel mounds", "old_ids": [11], "hex": "#585400"},
        9: {"name": "Grass", "old_ids": [12], "hex": "#8DB51D"},
        10: {"name": "Type 1 natural regeneration", "old_ids": [13], "hex": "#C2163A"},
        11: {"name": "Type 2 natural regeneration", "old_ids": [14], "hex": "#F77757"},
        12: {"name": "Bare ground", "old_ids": [15], "hex": "#2CD874"},
        13: {"name": "Sluice", "old_ids": [16], "hex": "#613991"},
        14: {"name": "Vehicles", "old_ids": [17, 18], "hex": "#969AAE"},
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split-playamirador-top-to-holdout",
        action="store_true",
        help="Split GoldMDD PlayaMirador image/label into top (val) and bottom (train, keeps name PlayaMirador).",
    )
    return p.parse_args()


def split_for_site(site: str) -> str:
    if site == PLAYA_SPLIT_TOP_NAME:
        return "val"
    train_output_sites = {OUTPUT_SITE_RENAMES.get(s, s) for s in TRAIN_SITES}
    val_output_sites = {OUTPUT_SITE_RENAMES.get(s, s) for s in VAL_SITES}
    if site in train_output_sites:
        return "train"
    if site in val_output_sites:
        return "val"
    return "test"


def output_site_name(source_site: str) -> str:
    return OUTPUT_SITE_RENAMES.get(source_site, source_site)


def ordered_sample_names_from_disk() -> list[str]:
    """Return sample names in a stable order, including optional PlayaMirador2."""
    train_names = {p.stem for p in TRAIN_LABEL_DIR.glob("*.png")}
    val_names = {p.stem for p in VAL_LABEL_DIR.glob("*.png")}
    test_names = {p.stem for p in TEST_LABEL_DIR.glob("*.png")}
    all_names = train_names | val_names | test_names

    base_order: list[str] = []
    for s in SITE_ORDER:
        out_name = output_site_name(s)
        if out_name in all_names:
            base_order.append(out_name)
        # insert the split top half immediately after PlayaMirador1 if present
        if s == "PlayaMirador" and PLAYA_SPLIT_TOP_NAME in all_names:
            base_order.append(PLAYA_SPLIT_TOP_NAME)
    extras = sorted(all_names - set(base_order))
    base_order.extend(extras)

    split_rank = {"train": 0, "val": 1, "test": 2}
    order_index = {name: i for i, name in enumerate(base_order)}
    ordered = sorted(base_order, key=lambda s: (split_rank[split_for_site(s)], order_index.get(s, 9999)))
    return ordered


def parse_markdown_table_rows(readme_text: str, heading: str) -> list[list[str]]:
    lines = readme_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i + 1
            break
    if start is None:
        raise RuntimeError(f"Heading not found: {heading}")

    # find first table header row
    i = start
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    if i >= len(lines):
        raise RuntimeError(f"No table found after heading: {heading}")
    # skip header + separator
    i += 2

    rows: list[list[str]] = []
    while i < len(lines):
        line = lines[i]
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows


def parse_original_class_table() -> dict[int, dict[str, object]]:
    text = README_REF.read_text(encoding="utf-8")
    rows = parse_markdown_table_rows(text, "## Unified class mapping (global canonical IDs)")
    out: dict[int, dict[str, object]] = {}
    for cells in rows:
        cid = int(cells[0])
        out[cid] = {
            "name": cells[1],
            "hex": cells[3].strip("`"),
            "area_ha": float(cells[4]),
            "percent": float(cells[5]),
            "pixel_count": int(cells[6].replace(",", "")),
            "aliases": [a.strip() for a in cells[7].split(";") if a.strip()],
        }
    return out


def remap_goldmdd_labels() -> dict[str, dict[int, int]]:
    """Rebuild GoldMDD train/val/test labels from GoldMining labels using merged mapping.

    Returns per-site class pixel counts (new IDs including 0).
    """
    site_counts: dict[str, dict[int, int]] = {}

    for site in SITE_ORDER:
        src = SRC_LABEL_DIR / f"{site}.png"
        out_site = output_site_name(site)
        split = split_for_site(out_site)
        if split == "train":
            dst_dir = TRAIN_LABEL_DIR
        elif split == "val":
            dst_dir = VAL_LABEL_DIR
        else:
            dst_dir = TEST_LABEL_DIR
        dst = dst_dir / f"{out_site}.png"

        arr = np.array(Image.open(src), dtype=np.uint8)
        out = np.zeros_like(arr, dtype=np.uint8)
        vals = np.unique(arr)
        for v in vals:
            vv = int(v)
            if vv not in OLD_TO_NEW:
                raise RuntimeError(f"Unexpected label ID {vv} in {src}")
            out[arr == vv] = OLD_TO_NEW[vv]
        Image.fromarray(out, mode="L").save(dst, format="PNG")

        u, c = np.unique(out, return_counts=True)
        site_counts[out_site] = {int(k): int(v) for k, v in zip(u, c)}
        print(f"Remapped label: {site} -> {out_site} ({split})", flush=True)

    return site_counts


def split_playamirador_top_to_holdout() -> None:
    """Split PlayaMirador image/label into top half (val) and bottom half (train).

    Bottom half overwrites `PlayaMirador1.png` in train. Top half is written as `PlayaMirador2.png` in val.
    Idempotent: if train already contains the bottom half and val contains the top half, no action is taken.
    """
    train_img = TRAIN_IMAGE_DIR / f"{PLAYA_SPLIT_TRAIN_NAME}.png"
    train_lbl = TRAIN_LABEL_DIR / f"{PLAYA_SPLIT_TRAIN_NAME}.png"
    hold_img = VAL_IMAGE_DIR / f"{PLAYA_SPLIT_TOP_NAME}.png"
    hold_lbl = VAL_LABEL_DIR / f"{PLAYA_SPLIT_TOP_NAME}.png"

    if not train_img.exists() or not train_lbl.exists():
        raise FileNotFoundError(f"Expected train {PLAYA_SPLIT_TRAIN_NAME} image/label not found in GoldMDD")

    img = Image.open(train_img)
    lbl = Image.open(train_lbl)
    iw, ih = img.size
    lw, lh = lbl.size

    # Common state during reruns: image is already split (bottom half in train), but remap step rewrote
    # the train label as the full PlayaMirador label. In that case, only re-split the label using the
    # existing val image to preserve the prior image split.
    if (iw, ih) == (17624, 8883) and (lw, lh) == (17624, 17766):
        if not hold_img.exists() or not hold_lbl.parent.exists():
            raise RuntimeError(
                "PlayaMirador image is already split but val top image is missing; cannot recover full image split."
            )
        top_lbl = lbl.crop((0, 0, lw, lh // 2))
        bot_lbl = lbl.crop((0, lh // 2, lw, lh))
        top_lbl.save(hold_lbl, format="PNG")
        bot_lbl.save(train_lbl, format="PNG")
        print(
            f"Re-split only PlayaMirador labels for existing image split: train `{PLAYA_SPLIT_TRAIN_NAME}` and val `{PLAYA_SPLIT_TOP_NAME}`.",
            flush=True,
        )
        return

    if img.size != lbl.size:
        raise RuntimeError(f"PlayaMirador image/label size mismatch before split: {img.size} vs {lbl.size}")
    w, h = img.size
    if h % 2 != 0:
        raise RuntimeError(f"PlayaMirador height is odd ({h}); cannot split evenly into top/bottom")

    half = h // 2

    # If already split, train file should be bottom half and val top file should exist with same half height.
    if h == half and hold_img.exists() and hold_lbl.exists():
        # impossible branch (half == h only when h=0), keep for clarity
        return

    # Idempotence check: if train has already been overwritten to bottom-half size and val top exists.
    if h == 8883 and hold_img.exists() and hold_lbl.exists():
        hi = Image.open(hold_img)
        hl = Image.open(hold_lbl)
        if hi.size == (w, 8883) and hl.size == (w, 8883):
            print("PlayaMirador split already applied; skipping split step.", flush=True)
            return

    if h != 17766 and not hold_img.exists():
        # Guard against accidentally splitting an already modified file with unexpected height.
        raise RuntimeError(
            f"Unexpected PlayaMirador train height {h}; expected original 17766 before split or 8883 after split."
        )

    top_box = (0, 0, w, half)
    bot_box = (0, half, w, h)

    top_img = img.crop(top_box)
    bot_img = img.crop(bot_box)
    top_lbl = lbl.crop(top_box)
    bot_lbl = lbl.crop(bot_box)

    if top_img.size != bot_img.size or top_lbl.size != bot_lbl.size:
        raise RuntimeError("Top/bottom split produced inconsistent sizes")

    hold_img.parent.mkdir(parents=True, exist_ok=True)
    hold_lbl.parent.mkdir(parents=True, exist_ok=True)

    top_img.save(hold_img, format="PNG")
    top_lbl.save(hold_lbl, format="PNG")
    bot_img.save(train_img, format="PNG")
    bot_lbl.save(train_lbl, format="PNG")

    print(
        f"Split PlayaMirador into bottom(train keeps `{PLAYA_SPLIT_TRAIN_NAME}`) and top(val as `{PLAYA_SPLIT_TOP_NAME}`), size={w}x{half} each.",
        flush=True,
    )


def collect_site_counts_from_goldmdd(sample_order: list[str]) -> dict[str, dict[int, int]]:
    site_counts: dict[str, dict[int, int]] = {}
    for site in sample_order:
        split = split_for_site(site)
        if split == "train":
            lbl_path = TRAIN_LABEL_DIR / f"{site}.png"
        elif split == "val":
            lbl_path = VAL_LABEL_DIR / f"{site}.png"
        else:
            lbl_path = TEST_LABEL_DIR / f"{site}.png"
        arr = np.array(Image.open(lbl_path), dtype=np.uint8)
        u, c = np.unique(arr, return_counts=True)
        site_counts[site] = {int(k): int(v) for k, v in zip(u, c)}
    return site_counts


def build_merged_class_stats(site_counts: dict[str, dict[int, int]]) -> dict[int, dict[str, object]]:
    orig = parse_original_class_table()

    # Pixel counts from current remapped GoldMDD labels (authoritative for GoldMDD).
    pixel_counts = {cid: 0 for cid in MERGED_CLASS_DEFS}
    for site in site_counts:
        counts = site_counts[site]
        for cid in MERGED_CLASS_DEFS:
            pixel_counts[cid] += counts.get(cid, 0)

    # Areas and alias names merged from original 19-class table.
    merged: dict[int, dict[str, object]] = {}
    total_area = 0.0
    for cid, meta in MERGED_CLASS_DEFS.items():
        area = sum(float(orig[old]["area_ha"]) for old in meta["old_ids"])
        total_area += area
        aliases: list[str] = []
        for old in meta["old_ids"]:
            for a in orig[old]["aliases"]:
                if a not in aliases:
                    aliases.append(a)
        merged[cid] = {
            "name": meta["name"],
            "hex": meta["hex"],
            "area_ha": area,
            "pixel_count": pixel_counts[cid],
            "aliases": aliases,
        }
    for cid in merged:
        merged[cid]["percent"] = (merged[cid]["area_ha"] / total_area * 100.0) if total_area > 0 else 0.0
    return merged


def build_per_site_rows(site_counts: dict[str, dict[int, int]], sample_order: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site in sample_order:
        split = split_for_site(site)
        if split == "train":
            img_path = TRAIN_IMAGE_DIR / f"{site}.png"
            lbl_path = TRAIN_LABEL_DIR / f"{site}.png"
        elif split == "val":
            img_path = VAL_IMAGE_DIR / f"{site}.png"
            lbl_path = VAL_LABEL_DIR / f"{site}.png"
        else:
            img_path = TEST_IMAGE_DIR / f"{site}.png"
            lbl_path = TEST_LABEL_DIR / f"{site}.png"
        w, h = Image.open(img_path).size
        lw, lh = Image.open(lbl_path).size
        if (w, h) != (lw, lh):
            raise RuntimeError(f"Image/label size mismatch for {site}: image={w}x{h}, label={lw}x{lh}")
        counts = site_counts[site]
        present = [cid for cid in MERGED_CLASS_DEFS if counts.get(cid, 0) > 0]
        rows.append(
            {
                "site": site,
                "split": split,
                "ortho_png": f"{site}.png",
                "label_png": f"{site}.png",
                "size": f"{w}x{h}",
                "total_pixels": w * h,
                "class_ids": ",".join(str(cid) for cid in present),
                "class_names": "; ".join(MERGED_CLASS_DEFS[cid]["name"] for cid in present),
            }
        )
    return rows


def make_heatmap(site_counts: dict[str, dict[int, int]], sample_order: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_ids = list(MERGED_CLASS_DEFS.keys())
    class_names = [MERGED_CLASS_DEFS[cid]["name"] for cid in class_ids]
    mat = np.array([[site_counts[s].get(cid, 0) for cid in class_ids] for s in sample_order], dtype=np.float64)

    # Save CSV first (raw counts).
    with HEATMAP_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", *class_names])
        for site, row in zip(sample_order, mat.astype(np.int64)):
            writer.writerow([site, *row.tolist()])

    # Log-scale visualization for readability.
    vis = np.log10(mat + 1.0)
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=180)
    im = ax.imshow(vis, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ylabels = [f"{s} ({split_for_site(s)})" for s in sample_order]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Merged classes")
    ax.set_ylabel("Sites")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10(pixel count + 1)")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote heatmap: {HEATMAP_PNG}", flush=True)


def make_train_val_test_distribution_plot(site_counts: dict[str, dict[int, int]], sample_order: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_ids = list(MERGED_CLASS_DEFS.keys())
    class_names = [MERGED_CLASS_DEFS[cid]["name"] for cid in class_ids]

    split_names = ["train", "val", "test"]
    agg = {sp: {cid: 0 for cid in class_ids} for sp in split_names}
    for site in sample_order:
        split = split_for_site(site)
        counts = site_counts[site]
        for cid in class_ids:
            agg[split][cid] += counts.get(cid, 0)

    split_counts = {sp: np.array([agg[sp][cid] for cid in class_ids], dtype=np.int64) for sp in split_names}
    split_totals = {sp: int(split_counts[sp].sum()) for sp in split_names}
    split_pct = {
        sp: (split_counts[sp] / split_totals[sp]) if split_totals[sp] > 0 else np.zeros(len(class_ids), dtype=np.float64)
        for sp in split_names
    }

    SPLIT_DIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    ratio_train_val = np.log2((split_pct["train"] + 1e-12) / (split_pct["val"] + 1e-12))
    ratio_train_test = np.log2((split_pct["train"] + 1e-12) / (split_pct["test"] + 1e-12))

    with SPLIT_DIST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "train_pixels",
                "val_pixels",
                "test_pixels",
                "train_density_excluding_background",
                "val_density_excluding_background",
                "test_density_excluding_background",
                "log2_ratio_train_over_val",
                "log2_ratio_train_over_test",
            ]
        )
        for i, (cid, name) in enumerate(zip(class_ids, class_names)):
            writer.writerow(
                [
                    cid,
                    name,
                    int(split_counts["train"][i]),
                    int(split_counts["val"][i]),
                    int(split_counts["test"][i]),
                    f'{split_pct["train"][i]:.8f}',
                    f'{split_pct["val"][i]:.8f}',
                    f'{split_pct["test"][i]:.8f}',
                    f"{ratio_train_val[i]:.8f}",
                    f"{ratio_train_test[i]:.8f}",
                ]
            )

    x = np.arange(len(class_ids))
    w = 0.25
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 5.2), dpi=180, constrained_layout=True)
    colors = {"train": "#2C7FB8", "val": "#31A354", "test": "#D95F0E"}

    for offset, sp in zip([-w, 0.0, w], split_names):
        ax1.bar(x + offset, split_counts[sp], width=w, label=sp.capitalize(), color=colors[sp])
    ax1.set_yscale("log")
    ax1.set_ylabel("Pixel count (log scale)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelsize=10)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.set_title("GoldMDD Merged Class Distribution: Train vs Val vs Test", fontsize=14)
    fig.savefig(SPLIT_DIST_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote split distribution plot: {SPLIT_DIST_PNG}", flush=True)


def extract_spatial_section_prefix(current_readme: str) -> str:
    """Keep top bullets + spatial table + IoU note; drop later sections."""
    lines = current_readme.splitlines()
    out: list[str] = []
    seen_note = False
    for line in lines:
        if line.startswith("## Unified class mapping"):
            break
        out.append(line)
        if line.startswith("- Note: low IoU means the orthomosaic footprint is larger than the label footprint"):
            seen_note = True
    if not seen_note:
        raise RuntimeError("Could not find spatial table IoU note in GoldMDD README")
    # update canonical-label bullet text and add merged info if needed
    joined = "\n".join(out)
    joined = re.sub(
        r"- Canonical labels: .*",
        "- Canonical labels: 14 semantic classes (`1..14`) plus `0=Background` after class merging in GoldMDD.",
        joined,
    )
    joined = re.sub(
        r"- Output folders: .*",
        "- Output folders: `train/image`, `train/label`, `val/image`, `val/label`, `test/image`, `test/label` (copied from `GoldMining/Data/Orthomosaic` and `GoldMining/Data/Label`).",
        joined,
    )
    joined = re.sub(
        r"- Split rule in this dataset: .*",
        "- Split rule in this dataset: train = sites 1/5/7/10 (with `PlayaMirador1` = bottom half); val = sites 3/8 plus `PlayaMirador2` (top half); test = remaining sites.",
        joined,
    )
    joined = re.sub(
        r"\n- Note: `PlayaMirador` is additionally split in GoldMDD into .*?(?=\n##|\Z)",
        "",
        joined,
        flags=re.S,
    )
    joined = joined.replace("| Paolita1 |", "| Paolita |")
    playa_two_rows = (
        "| PlayaMirador1 | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.062030, -13.057446] | 17624x8883 | 0.6102 | 2022-03-10 | train |\n"
        "| PlayaMirador2 | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.057446, -13.052861] | 17624x8883 | 0.6102 | 2022-03-10 | val |"
    )
    # Replace any prior PlayaMirador row format with two explicit split rows.
    for old in [
        "| PlayaMirador | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.062030, -13.052861] | 17624x17766 | 0.6102 | 2022-03-10 | train |",
        "| PlayaMirador | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.062030, -13.052861] | 17624x17766 (source footprint) | 0.6102 | 2022-03-10 | split: bottom->train (`PlayaMirador1`), top->holdout (`PlayaMirador2`) |",
        "| PlayaMirador | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.062030, -13.052861] | 17624x17766 (source footprint) | 0.6102 | 2022-03-10 | split: bottom->train, top->holdout |",
        "| PlayaMirador | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.062030, -13.052861] | 17624x17766 (source footprint) | 0.6102 | 2022-03-10 | split: bottom->train (`PlayaMirador1`), top->test (`PlayaMirador2`) |",
        "| PlayaMirador | `PlayaMirador_corrected.tif` | `PlayaMirador_classified_corrected.tif` | EPSG:32719 | 22911x22394 | 17608x17608 | 0.056792, 0.056792 | 0.056840, 0.057298 | [-70.377536, -70.365474] | [-13.063390, -13.051829] | [-70.376072, -70.366793] | [-13.062030, -13.052861] | 17624x17766 (source footprint) | 0.6102 | 2022-03-10 | split: bottom->train (`PlayaMirador1`), top->val (`PlayaMirador2`) |",
    ]:
        joined = joined.replace(old, playa_two_rows)
    joined = joined.replace(
        "- Note: low IoU means the orthomosaic footprint is larger than the label footprint (not necessarily mismatched).",
        "- Note: low IoU means the orthomosaic footprint is larger than the label footprint (not necessarily mismatched).\n- Note: `PlayaMirador` is additionally split in GoldMDD into `PlayaMirador1` (bottom half, train) and `PlayaMirador2` (top half, val). The two rows above use the same source files and split the label latitude range at the midpoint.",
    )

    # Rewrite the spatial metadata split column from current split assignment.
    lines2 = joined.splitlines()
    in_spatial = False
    for i, line in enumerate(lines2):
        if line.strip() == "## Spatial metadata table (source files + aligned output)":
            in_spatial = True
            continue
        if in_spatial and line.startswith("## "):
            break
        if not in_spatial or not line.startswith("|"):
            continue
        if line.startswith("| Site |") or line.startswith("| --- "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 16:
            continue
        site = cells[0]
        if site in {"PlayaMirador1", "PlayaMirador2", "AcumulacionAaron2B", "Anel", "Clavelito", "ElEngano", "Kotsimba", "Linda", "Los5Rebeldes", "Nayda", "Paolita", "SantaInesDosMil"}:
            cells[-1] = split_for_site(site)
            lines2[i] = "| " + " | ".join(cells) + " |"
    joined = "\n".join(lines2)
    return joined.rstrip()


def build_readme(merged_stats: dict[int, dict[str, object]], per_site_rows: list[dict[str, object]]) -> str:
    prefix = extract_spatial_section_prefix(README_SRC.read_text(encoding="utf-8"))

    lines: list[str] = [prefix, "", "## Unified class mapping (GoldMDD merged classes)", ""]
    lines.append(
        "| Canonical ID | Class | Merged from original IDs | Color swatch | Color (HEX) | Area (ha) | Percentage (%) | Pixel count | Alias names seen in metadata |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    for cid, meta in MERGED_CLASS_DEFS.items():
        st = merged_stats[cid]
        swatch = (
            f'<span style="display:inline-block;width:14px;height:14px;background:{st["hex"]};border:1px solid #222;"></span>'
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(cid),
                    st["name"],
                    ",".join(str(x) for x in meta["old_ids"]),
                    swatch,
                    f'`{st["hex"]}`',
                    f'{st["area_ha"]:.4f}',
                    f'{st["percent"]:.2f}',
                    f'{int(st["pixel_count"]):,}',
                    "; ".join(st["aliases"]),
                ]
            )
            + " |"
        )

    lines += ["", "## Per-site classes using GoldMDD merged mapping", ""]
    lines.append(
        "| Site | Split | Output ortho PNG | Output label PNG | Output size (W x H px) | Output total pixels | Merged class IDs present | Merged class names present |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | --- | --- |")
    for row in per_site_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["site"],
                    row["split"],
                    f'`{row["ortho_png"]}`',
                    f'`{row["label_png"]}`',
                    row["size"],
                    f'{int(row["total_pixels"]):,}',
                    row["class_ids"],
                    row["class_names"],
                ]
            )
            + " |"
        )

    lines += ["", "## Generation workflow", ""]
    lines.append("- Source orthomosaic/label alignment and original canonical labels come from `GoldMining/Data`.")
    lines.append("- GoldMDD labels are remapped from the original 19-class canonical IDs into a 14-class merged scheme (stored only under `GoldMDD/data/*/label`).")
    lines.append("- Merge rules:")
    lines.append("  - `Heavy machinery` = original IDs 5 (Front loader), 8 (Excavator), 9 (Heavy machinery), 19 (Dump truck)")
    lines.append("  - `Vehicles` = original IDs 17 (Vehicles), 18 (Small vehicles)")
    lines.append("  - `Building` = original IDs 1 (Urban area), 4 (Mining camp)")
    lines.append(f"- Heatmap output: `{HEATMAP_PNG.relative_to(GOLDMDD.parent)}`")
    lines.append(f"- Heatmap CSV: `{HEATMAP_CSV.relative_to(GOLDMDD.parent)}`")
    lines.append(f"- Train/val/test distribution plot: `{SPLIT_DIST_PNG.relative_to(GOLDMDD.parent)}`")
    lines.append(f"- Train/val/test distribution CSV: `{SPLIT_DIST_CSV.relative_to(GOLDMDD.parent)}`")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()

    # Step 1: remap labels in GoldMDD
    print("Step 1/3: remap labels", flush=True)
    remap_goldmdd_labels()
    if args.split_playamirador_top_to_holdout:
        split_playamirador_top_to_holdout()
    sample_order = ordered_sample_names_from_disk()
    site_counts = collect_site_counts_from_goldmdd(sample_order)
    print(f"Active samples ({len(sample_order)}): {', '.join(sample_order)}", flush=True)

    # Step 2: build merged stats + per-site rows
    print("Step 2/3: rebuild stats and README content", flush=True)
    merged_stats = build_merged_class_stats(site_counts)
    per_site_rows = build_per_site_rows(site_counts, sample_order)

    # Step 3: outputs
    print("Step 3/3: write heatmap and README", flush=True)
    make_heatmap(site_counts, sample_order)
    make_train_val_test_distribution_plot(site_counts, sample_order)
    README_OUT.write_text(build_readme(merged_stats, per_site_rows), encoding="utf-8")

    print("Done.")
    print(f"Updated labels under: {TRAIN_LABEL_DIR}, {VAL_LABEL_DIR}, and {TEST_LABEL_DIR}")
    print(f"Updated README: {README_OUT}")
    print(f"Heatmap PNG: {HEATMAP_PNG}")
    print(f"Heatmap CSV: {HEATMAP_CSV}")
    print(f"Split distribution PNG: {SPLIT_DIST_PNG}")
    print(f"Split distribution CSV: {SPLIT_DIST_CSV}")


if __name__ == "__main__":
    main()
