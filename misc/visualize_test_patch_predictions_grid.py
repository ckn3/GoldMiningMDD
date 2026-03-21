#!/usr/bin/env python3
"""
Visualize random test patches with GT labels and predictions from multiple experiment best checkpoints.

Outputs one large grid image:
  columns = [RGB image, GT label, model1 pred, model2 pred, ...]
  rows    = random test samples
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError as e:  # pragma: no cover
    raise SystemExit("segmentation_models_pytorch is required") from e

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError as e:  # pragma: no cover
    raise SystemExit("transformers is required") from e

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib is required") from e


Image.MAX_IMAGE_PIXELS = None
NUM_FOREGROUND_CLASSES = 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# Background + 14 merged classes (from GoldMDD README)
PALETTE_HEX = [
    "#000000",  # 0 background
    "#8A6A3D",  # 1 Building
    "#7BEBFB",  # 2 Mining raft
    "#B04C18",  # 3 Primary Forest
    "#EE92C6",  # 4 Heavy machinery
    "#4F6F6F",  # 5 Water bodies
    "#84D08C",  # 6 Agricultural crop
    "#23F3E3",  # 7 Compact mounds
    "#585400",  # 8 Gravel mounds
    "#8DB51D",  # 9 Grass
    "#C2163A",  # 10 Type 1 natural regeneration
    "#F77757",  # 11 Type 2 natural regeneration
    "#2CD874",  # 12 Bare ground
    "#613991",  # 13 Sluice
    "#969AAE",  # 14 Vehicles
]

PALETTE = np.array(
    [[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in PALETTE_HEX],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    label_path: Path


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    config: dict
    kind: str  # "smp" or "segformer" or "efficientvit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/data-cropped"))
    p.add_argument("--experiments-root", type=Path, default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"))
    p.add_argument("--num-samples", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument(
        "--include-runs",
        nargs="*",
        default=None,
        help="Optional explicit run dir names. Default: auto-discover completed runs with best.pt/config.json.",
    )
    p.add_argument(
        "--exclude-prefixes",
        nargs="*",
        default=["smoke_", "logs"],
        help="Exclude runs whose names start with these prefixes.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics/test_random15_methods_grid.png"),
    )
    p.add_argument(
        "--sample-list-output",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics/test_random15_methods_grid_samples.txt"),
    )
    return p.parse_args()


def discover_runs(args: argparse.Namespace) -> list[RunSpec]:
    root = args.experiments_root
    runs: list[RunSpec] = []
    if args.include_runs:
        candidates = [root / n for n in args.include_runs]
    else:
        candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    for d in candidates:
        name = d.name
        if any(name.startswith(pref) for pref in args.exclude_prefixes):
            continue
        cfg_path = d / "config.json"
        best_path = d / "best.pt"
        if not (cfg_path.exists() and best_path.exists()):
            continue
        try:
            cfg = json.load(cfg_path.open())
        except Exception:
            continue
        if "efficientvit_repo" in cfg and "model_name" in cfg:
            kind = "efficientvit"
        elif "model_name" in cfg:
            kind = "segformer"
        elif "arch" in cfg and "encoder" in cfg:
            kind = "smp"
        else:
            kind = ""
        if not kind:
            continue
        runs.append(RunSpec(name=name, run_dir=d, config=cfg, kind=kind))
    return runs


def build_samples(test_root: Path) -> list[Sample]:
    img_dir = test_root / "test" / "image"
    lbl_dir = test_root / "test" / "label"
    image_map = {p.stem: p for p in img_dir.glob("*.jpg")}
    label_map = {p.stem: p for p in lbl_dir.glob("*.png")}
    common = sorted(image_map.keys() & label_map.keys())
    return [Sample(stem=s, image_path=image_map[s], label_path=label_map[s]) for s in common]


def load_rgb_and_label(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(sample.image_path) as im:
        img = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
    with Image.open(sample.label_path) as lb:
        lbl = np.array(lb, dtype=np.uint8, copy=True)
    return img, lbl


def preprocess_image(img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0).to(device)


def colorize_label(lbl: np.ndarray) -> np.ndarray:
    lbl_clip = np.clip(lbl.astype(np.int64), 0, len(PALETTE) - 1)
    return PALETTE[lbl_clip]


def build_smp_model(cfg: dict) -> nn.Module:
    enc_name = f"tu-{cfg['encoder']}"
    enc_weights = None if str(cfg.get("encoder_weights", "imagenet")).lower() == "none" else cfg.get("encoder_weights")
    common = dict(encoder_name=enc_name, encoder_weights=enc_weights, in_channels=3, classes=NUM_FOREGROUND_CLASSES)
    arch = cfg["arch"]
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)
    if arch == "fpn":
        return smp.FPN(**common)
    if arch == "unet":
        return smp.Unet(**common)
    raise ValueError(f"Unsupported SMP arch: {arch}")


def build_segformer_model(cfg: dict) -> nn.Module:
    return SegformerForSemanticSegmentation.from_pretrained(
        cfg["model_name"],
        num_labels=NUM_FOREGROUND_CLASSES,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )


def _load_module_from_path(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_efficientvit_model(cfg: dict) -> tuple[nn.Module, object]:
    misc_dir = Path(cfg.get("work_dir", "/deac/csc/yangGrp/cuij/GoldMDD/experiments")).parent / "misc"
    # Fallback to canonical misc path if work_dir doesn't map cleanly.
    if not misc_dir.exists():
        misc_dir = Path("/deac/csc/yangGrp/cuij/GoldMDD/misc")
    misc_dir_str = str(misc_dir)
    if misc_dir_str not in sys.path:
        sys.path.insert(0, misc_dir_str)
    mod = _load_module_from_path(misc_dir / "train_semseg_efficientvit.py", "goldmdd_train_efficientvit_viz")
    model = mod.build_model(
        cfg["model_name"],
        bool(cfg.get("pretrained", True)),
        Path(cfg["efficientvit_repo"]),
        Path(cfg["weights_cache_dir"]) if cfg.get("weights_cache_dir") else None,
        Path(cfg["pretrained_weight_file"]) if cfg.get("pretrained_weight_file") else None,
    )
    return model, mod


def load_checkpoint_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    # Retry because training may still be updating best.pt while we read.
    last_err: Exception | None = None
    for _ in range(3):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state, strict=True)
            return
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {last_err}")


def forward_logits(model: nn.Module, run: RunSpec, x: torch.Tensor) -> torch.Tensor:
    if run.kind == "segformer":
        out = model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
    if run.kind == "efficientvit":
        logits = model(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
    logits = model(x)
    if isinstance(logits, dict):
        logits = logits["out"]
    return logits


@torch.no_grad()
def predict_for_samples(run: RunSpec, samples: list[Sample], device: torch.device) -> list[np.ndarray]:
    if run.kind == "segformer":
        model = build_segformer_model(run.config)
    elif run.kind == "efficientvit":
        model, _ = build_efficientvit_model(run.config)
    else:
        model = build_smp_model(run.config)
    model.to(device).eval()
    load_checkpoint_weights(model, run.run_dir / "best.pt", device)

    preds: list[np.ndarray] = []
    for s in samples:
        img, _ = load_rgb_and_label(s)
        x = preprocess_image(img, device)
        logits = forward_logits(model, run, x)
        pred_fg = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # 0..13
        pred_lbl = pred_fg + 1  # map back to GoldMDD merged labels 1..14
        preds.append(pred_lbl)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds


def make_grid_figure(
    samples: list[Sample],
    rgbs: list[np.ndarray],
    gts: list[np.ndarray],
    pred_map: dict[str, list[np.ndarray]],
    out_path: Path,
) -> None:
    run_names = list(pred_map.keys())
    n_rows = len(samples)
    n_cols = 2 + len(run_names)

    fig_w = max(14, 2.6 * n_cols)
    fig_h = max(18, 2.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Image", "GT"] + run_names
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10)

    for r, sample in enumerate(samples):
        row_imgs = [rgbs[r], colorize_label(gts[r])] + [colorize_label(pred_map[name][r]) for name in run_names]
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(row_imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(sample.stem, fontsize=8, rotation=0, labelpad=55, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    runs = discover_runs(args)
    if not runs:
        raise SystemExit("No compatible runs with best.pt + config.json were found.")

    all_samples = build_samples(args.data_root)
    if len(all_samples) < args.num_samples:
        raise SystemExit(f"Requested {args.num_samples} samples but only {len(all_samples)} found.")

    rng = random.Random(args.seed)
    samples = rng.sample(all_samples, args.num_samples)
    samples = sorted(samples, key=lambda s: s.stem)

    rgbs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    for s in samples:
        img, lbl = load_rgb_and_label(s)
        rgbs.append(img)
        gts.append(lbl)

    print(f"Device: {device}")
    print(f"Selected {len(samples)} test patches")
    print("Runs:")
    for r in runs:
        print(f"  - {r.name} ({r.kind})")

    pred_map: dict[str, list[np.ndarray]] = {}
    for run in runs:
        print(f"[infer] {run.name}")
        try:
            pred_map[run.name] = predict_for_samples(run, samples, device)
        except Exception as e:
            print(f"  FAILED {run.name}: {e}")

    if not pred_map:
        raise SystemExit("All runs failed to infer.")

    make_grid_figure(samples, rgbs, gts, pred_map, args.output)
    args.sample_list_output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_list_output.write_text("\n".join(s.stem for s in samples) + "\n", encoding="utf-8")
    print(f"Saved figure: {args.output}")
    print(f"Saved sample list: {args.sample_list_output}")


if __name__ == "__main__":
    main()
