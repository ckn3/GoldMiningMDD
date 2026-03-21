#!/usr/bin/env python3
"""Compare multiple GoldMDD experiment runs and generate summary plots."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = [
    "Building",
    "Mining raft",
    "Primary Forest",
    "Heavy machinery",
    "Water bodies",
    "Agricultural crop",
    "Compact mounds",
    "Gravel mounds",
    "Grass",
    "Type1 regen",
    "Type2 regen",
    "Bare ground",
    "Sluice",
    "Vehicles",
]


@dataclass
class RunRecord:
    name: str
    family: str
    loss_variant: str
    log_df: pd.DataFrame
    test_metrics: dict
    val_metrics_best: dict | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics"),
    )
    return p.parse_args()


def infer_family(run_name: str) -> str:
    if run_name.startswith("efficientvit_"):
        return "EfficientViT-Seg-B2"
    if run_name.startswith("segformer_"):
        return "SegFormer-B2"
    return "DeepLabV3+/ConvNeXt-Tiny"


def infer_loss_variant(run_name: str) -> str:
    if "weighted_ce_dice" in run_name:
        return "weighted_ce_dice"
    if "focal_dice" in run_name:
        return "focal_dice"
    return "ce_dice"


def discover_runs(root: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in {"logs", "diagnostics"}:
            continue
        cfg = d / "config.json"
        best = d / "best.pt"
        log_csv = d / "train_log.csv"
        test_json = d / "test_metrics.json"
        if not (cfg.exists() and best.exists() and log_csv.exists() and test_json.exists()):
            continue
        log_df = pd.read_csv(log_csv)
        with open(test_json) as f:
            test_metrics = json.load(f)
        val_metrics_best = None
        val_best_json = d / "val_metrics_best.json"
        if val_best_json.exists():
            with open(val_best_json) as f:
                val_metrics_best = json.load(f)
        runs.append(
            RunRecord(
                name=d.name,
                family=infer_family(d.name),
                loss_variant=infer_loss_variant(d.name),
                log_df=log_df,
                test_metrics=test_metrics,
                val_metrics_best=val_metrics_best,
            )
        )
    return runs


def build_summary_df(runs: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in runs:
        df = r.log_df.copy()
        best_idx = int(df["val_miou"].idxmax())
        best_present_idx = int(df["val_miou_present"].idxmax())
        per_class_iou = np.array(r.test_metrics.get("per_class_iou", []), dtype=float)
        gt_pixels = np.array(r.test_metrics.get("gt_pixels_per_class", []), dtype=float)
        present = gt_pixels > 0
        per_class_f1 = r.test_metrics.get("per_class_f1")
        if per_class_f1 is None and per_class_iou.size > 0:
            per_class_f1_arr = (2 * per_class_iou) / (1 + per_class_iou)
            per_class_f1_arr[~np.isfinite(per_class_iou)] = np.nan
        else:
            per_class_f1_arr = np.array(per_class_f1, dtype=float) if per_class_f1 is not None else np.array([], dtype=float)
        macro_f1 = (
            float(np.nanmean(per_class_f1_arr)) if per_class_f1_arr.size else float("nan")
        )
        macro_f1_present = (
            float(np.nanmean(per_class_f1_arr[present])) if (per_class_f1_arr.size and present.any()) else float("nan")
        )
        rows.append(
            {
                "run": r.name,
                "family": r.family,
                "loss_variant": r.loss_variant,
                "epochs": int(len(df)),
                "best_val_miou": float(df["val_miou"].max()),
                "best_val_miou_epoch": int(df.loc[best_idx, "epoch"]),
                "best_val_miou_present": float(df["val_miou_present"].max()),
                "best_val_miou_present_epoch": int(df.loc[best_present_idx, "epoch"]),
                "final_val_miou": float(df["val_miou"].iloc[-1]),
                "final_val_miou_present": float(df["val_miou_present"].iloc[-1]),
                "final_train_loss": float(df["train_loss"].iloc[-1]),
                "final_val_loss": float(df["val_loss"].iloc[-1]),
                "min_val_loss": float(df["val_loss"].min()),
                "val_overfit_drop_from_best": float(df["val_miou"].max() - df["val_miou"].iloc[-1]),
                "test_loss": float(r.test_metrics["loss"]),
                "test_miou": float(r.test_metrics["miou"]),
                "test_miou_present": float(r.test_metrics.get("miou_present", np.nan)),
                "test_macro_f1": float(r.test_metrics.get("macro_f1", macro_f1)),
                "test_macro_f1_present": float(r.test_metrics.get("macro_f1_present", macro_f1_present)),
                "test_oa_fg": float(r.test_metrics.get("oa_fg", np.nan)),
                "test_ce": float(r.test_metrics["ce"]),
                "test_dice": float(r.test_metrics["dice"]),
                "val_bestckpt_miou": float(r.val_metrics_best.get("miou", np.nan)) if r.val_metrics_best else float("nan"),
                "val_bestckpt_miou_present": float(r.val_metrics_best.get("miou_present", np.nan)) if r.val_metrics_best else float("nan"),
                "val_bestckpt_macro_f1_present": float(r.val_metrics_best.get("macro_f1_present", np.nan)) if r.val_metrics_best else float("nan"),
                "val_bestckpt_oa_fg": float(r.val_metrics_best.get("oa_fg", np.nan)) if r.val_metrics_best else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    sort_order = {
        "baseline1_augv2_ce_dice": 0,
        "baseline2_augv2_weighted_ce_dice": 1,
        "baseline3_augv2_focal_dice": 2,
        "segformer_b2_baseline1_augv2_ce_dice": 3,
        "segformer_b2_baseline2_augv2_weighted_ce_dice": 4,
        "segformer_b2_baseline3_augv2_focal_dice": 5,
        "efficientvit_b2_baseline1_augv2_ce_dice": 6,
        "efficientvit_b2_baseline2_augv2_weighted_ce_dice": 7,
        "efficientvit_b2_baseline3_augv2_focal_dice": 8,
    }
    out["sort_key"] = out["run"].map(sort_order).fillna(999).astype(int)
    out = out.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return out


def plot_training_curves(runs: list[RunRecord], out_path: Path) -> None:
    color_map = {
        "baseline1_augv2_ce_dice": "#1f77b4",
        "baseline2_augv2_weighted_ce_dice": "#ff7f0e",
        "baseline3_augv2_focal_dice": "#2ca02c",
        "segformer_b2_baseline1_augv2_ce_dice": "#d62728",
        "segformer_b2_baseline2_augv2_weighted_ce_dice": "#9467bd",
        "segformer_b2_baseline3_augv2_focal_dice": "#8c564b",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    metric_specs = [
        ("train_loss", "Train loss"),
        ("val_loss", "Val loss"),
        ("val_miou", "Val mIoU"),
        ("val_miou_present", "Val mIoU (present classes only)"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        for r in runs:
            df = r.log_df
            ax.plot(
                df["epoch"],
                df[metric],
                label=r.name,
                linewidth=1.8,
                alpha=0.95,
                color=color_map.get(r.name),
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=1, loc="upper right")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    specs = [
        ("best_val_miou", "Best val mIoU"),
        ("best_val_miou_present", "Best val mIoU (present)"),
        ("test_miou", "Test mIoU"),
        ("test_miou_present", "Test mIoU (present)"),
    ]
    x = np.arange(len(summary_df))
    labels = summary_df["run"].tolist()
    for ax, (col, title) in zip(axes.flat, specs):
        vals = summary_df[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=["#4c78a8" if "segformer" not in n else "#e45756" for n in labels])
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.18 if np.isfinite(vals).any() else 1)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_class_iou_heatmap(runs: list[RunRecord], out_path: Path, split_dist_csv: Path | None = None) -> None:
    run_names = [r.name for r in runs]
    mat_full = np.array([r.test_metrics["per_class_iou"] for r in runs], dtype=float)
    gt_pixels = np.array(runs[0].test_metrics.get("gt_pixels_per_class", [0] * len(CLASS_NAMES)), dtype=float)
    present_cols = np.where(gt_pixels > 0)[0]
    mat = mat_full[:, present_cols]
    class_names = [CLASS_NAMES[i] for i in present_cols]
    fig_h = 6 + 0.35 * len(runs)
    fig, axes = plt.subplots(2, 1, figsize=(16, fig_h), constrained_layout=True, height_ratios=[3.3, 1.7])
    ax = axes[0]
    disp = np.ma.masked_invalid(mat)
    im = ax.imshow(disp, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(run_names)))
    ax.set_yticklabels(run_names, fontsize=9)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_title("Test per-class IoU by run (classes present in test only)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="white")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("IoU")

    ax2 = axes[1]
    if split_dist_csv and split_dist_csv.exists():
        ddf = pd.read_csv(split_dist_csv)
        ddf = ddf.sort_values("class_id")
        ddf = ddf[ddf["test_pixels"] > 0].reset_index(drop=True)
        x = np.arange(len(ddf))
        w = 0.26
        ax2.bar(x - w, ddf["train_density_excluding_background"], width=w, label="train", alpha=0.8)
        ax2.bar(x, ddf["val_density_excluding_background"], width=w, label="val", alpha=0.8)
        ax2.bar(x + w, ddf["test_density_excluding_background"], width=w, label="test", alpha=0.8)
        ax2.set_ylabel("Density (non-bg)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(ddf["class_name"].tolist(), rotation=35, ha="right", fontsize=8)
        ax2.legend(loc="upper right", ncol=3, fontsize=8)
        ax2.set_title("Class density by split (context)")
        ax2.grid(True, axis="y", alpha=0.25)
        ax2.set_yscale("log")
    else:
        gt = np.array(runs[0].test_metrics.get("gt_pixels_per_class", [0] * 14), dtype=float)
        x = np.arange(len(gt))
        ax2.bar(x, gt + 1, color="#6baed6")
        ax2.set_yscale("log")
        ax2.set_xticks(x)
        ax2.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("GT pixels (+1, log)")
        ax2.set_title("Test GT class pixels")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_per_class_best_table(runs: list[RunRecord]) -> pd.DataFrame:
    gt = np.array(runs[0].test_metrics.get("gt_pixels_per_class", [0] * len(CLASS_NAMES)), dtype=float)
    mat = np.array([r.test_metrics["per_class_iou"] for r in runs], dtype=float)
    rows = []
    for j, cname in enumerate(CLASS_NAMES):
        if gt[j] <= 0:
            continue
        col = mat[:, j]
        best_i = int(np.nanargmax(col))
        rows.append(
            {
                "class_id": j + 1,
                "class_name": cname,
                "test_gt_pixels": int(gt[j]),
                "best_run": runs[best_i].name,
                "best_iou": float(col[best_i]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_md(
    summary_df: pd.DataFrame,
    runs: list[RunRecord],
    out_path: Path,
    split_dist_csv: Path | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Experiment Suite Summary")
    lines.append("")
    lines.append("## Metric Notes")
    lines.append("")
    lines.append("- `val_miou`: mean IoU over classes with non-NaN IoU (`denom > 0`), so false positives on absent classes can reduce it.")
    lines.append("- `val_miou_present`: mean IoU over classes with GT pixels in the validation set (`gt_pixels > 0`).")
    lines.append("- Validation uses the full validation split each epoch (no subsampling in these runs).")
    lines.append("")

    rank_cols = [
        "run",
        "family",
        "loss_variant",
        "test_miou_present",
        "test_macro_f1_present",
        "test_oa_fg",
        "test_miou",
        "best_val_miou",
        "best_val_miou_present",
    ]
    ranking = summary_df.sort_values("test_miou_present", ascending=False)[rank_cols].copy()
    for c in ["test_miou_present", "test_miou", "best_val_miou", "best_val_miou_present"]:
        ranking[c] = ranking[c].map(lambda x: f"{x:.4f}")
    lines.append("## Ranking (by test_mIoU_present)")
    lines.append("")
    lines.append(ranking.to_markdown(index=False))
    lines.append("")

    detail_cols = [
        "run", "best_val_miou_epoch", "best_val_miou", "best_val_miou_present_epoch",
        "best_val_miou_present", "final_val_miou", "final_val_miou_present",
        "val_bestckpt_macro_f1_present", "val_bestckpt_oa_fg",
        "final_train_loss", "final_val_loss", "val_overfit_drop_from_best"
    ]
    details = summary_df[detail_cols].copy()
    for c in details.columns:
        if c == "run" or c.endswith("_epoch"):
            continue
        details[c] = details[c].map(lambda x: f"{x:.4f}")
    lines.append("## Training / Validation Summary")
    lines.append("")
    lines.append(details.to_markdown(index=False))
    lines.append("")

    family = summary_df.groupby("family")[["test_miou", "test_miou_present", "test_macro_f1_present", "test_oa_fg", "best_val_miou", "best_val_miou_present"]].mean().reset_index()
    for c in family.columns:
        if c != "family":
            family[c] = family[c].map(lambda x: f"{x:.4f}")
    lines.append("## Family Averages")
    lines.append("")
    lines.append(family.to_markdown(index=False))
    lines.append("")

    loss = summary_df.groupby("loss_variant")[["test_miou", "test_miou_present", "test_macro_f1_present", "test_oa_fg", "best_val_miou", "best_val_miou_present"]].mean().reset_index()
    for c in loss.columns:
        if c != "loss_variant":
            loss[c] = loss[c].map(lambda x: f"{x:.4f}")
    lines.append("## Loss Variant Averages")
    lines.append("")
    lines.append(loss.to_markdown(index=False))
    lines.append("")

    per_class_best = build_per_class_best_table(runs)
    if not per_class_best.empty:
        pc = per_class_best.copy()
        pc["best_iou"] = pc["best_iou"].map(lambda x: f"{x:.4f}")
        lines.append("## Per-Class Best Run on Test (classes present in test only)")
        lines.append("")
        lines.append(pc.to_markdown(index=False))
        lines.append("")

    if split_dist_csv and split_dist_csv.exists():
        ddf = pd.read_csv(split_dist_csv).sort_values("class_id")
        val_absent = ddf.loc[ddf["val_pixels"] == 0, ["class_id", "class_name"]]
        test_absent = ddf.loc[ddf["test_pixels"] == 0, ["class_id", "class_name"]]
        lines.append("## Split Coverage Notes")
        lines.append("")
        if len(val_absent):
            lines.append("Validation classes with zero GT pixels:")
            lines.append("")
            lines.append(val_absent.to_markdown(index=False))
            lines.append("")
        if len(test_absent):
            lines.append("Test classes with zero GT pixels:")
            lines.append("")
            lines.append(test_absent.to_markdown(index=False))
            lines.append("")

    lines.append("## Additional Metric Availability")
    lines.append("")
    lines.append("- `test_macro_f1(_present)` is reported here (derived from `per_class_iou` if not explicitly stored in older runs).")
    lines.append("- `test_oa_fg` requires confusion-count aggregation; older runs may show `NaN` until re-evaluated with updated trainers.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.experiments_root)
    if not runs:
        raise SystemExit("No completed runs found.")

    summary_df = build_summary_df(runs)
    summary_csv = args.output_dir / "experiment_suite_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_per_class_iou_heatmap(
        runs,
        args.output_dir / "experiment_suite_per_class_iou_heatmap.png",
        split_dist_csv=Path("/deac/csc/yangGrp/cuij/GoldMDD/train_val_test_class_distribution_merged.csv"),
    )
    write_summary_md(
        summary_df,
        runs,
        args.output_dir / "summary.md",
        split_dist_csv=Path("/deac/csc/yangGrp/cuij/GoldMDD/train_val_test_class_distribution_merged.csv"),
    )

    # Remove older suite plots the user no longer wants.
    for old_name in [
        "experiment_suite_training_curves.png",
        "experiment_suite_metrics_bars.png",
        "experiment_suite_family_loss_matrix.png",
    ]:
        old_path = args.output_dir / old_name
        if old_path.exists():
            old_path.unlink()

    best_row = summary_df.loc[summary_df["test_miou_present"].idxmax()].to_dict()
    out_json = args.output_dir / "experiment_suite_analysis.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "num_runs": int(len(summary_df)),
                "best_by_test_miou_present": best_row,
                "ranking_test_miou_present": summary_df.sort_values("test_miou_present", ascending=False)[
                    ["run", "test_miou_present", "test_miou", "test_macro_f1_present", "test_oa_fg", "best_val_miou", "best_val_miou_present"]
                ].to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plots under: {args.output_dir}")
    print(f"Best run (test mIoU_present): {best_row['run']} = {best_row['test_miou_present']:.4f}")


if __name__ == "__main__":
    main()
