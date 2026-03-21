#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog


TRAIN_PATCH_COUNT = 65798


def load_train_module(repo_root: Path):
    train_net = repo_root / "train_net.py"
    spec = importlib.util.spec_from_file_location(f"{repo_root.name}_train_net", train_net)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(repo_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def iou_to_f1(iou_value: float) -> float:
    return (2.0 * iou_value) / (1.0 + iou_value) if iou_value >= 0.0 else float("nan")


def parse_training_metrics(metrics_path: Path, iters_per_epoch: int):
    train_entries = []
    eval_entries = []
    if not metrics_path.exists():
        return {}, []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "sem_seg/mIoU" in entry:
                eval_entries.append(entry)
            if "total_loss" in entry:
                train_entries.append(entry)

    summary = {}
    if train_entries:
        summary["final_train_loss"] = float(train_entries[-1]["total_loss"])

    if eval_entries:
        best = max(eval_entries, key=lambda x: float(x.get("sem_seg/mIoU", float("-inf"))))
        final = eval_entries[-1]

        def event_to_stats(entry):
            ious = []
            for key, value in entry.items():
                if key.startswith("sem_seg/IoU-"):
                    if value == value:
                        ious.append(float(value) / 100.0)
            macro_f1 = float("nan")
            if ious:
                macro_f1 = sum(iou_to_f1(v) for v in ious) / len(ious)
            return {
                "miou": float(entry["sem_seg/mIoU"]) / 100.0,
                "miou_present": float(entry["sem_seg/mIoU"]) / 100.0,
                "macro_f1_present": macro_f1,
                "oa_fg": float(entry["sem_seg/pACC"]) / 100.0,
                "epoch": max(1, round(float(entry.get("iteration", 0)) / iters_per_epoch)),
            }

        best_stats = event_to_stats(best)
        final_stats = event_to_stats(final)
        summary.update(
            {
                "best_val_miou": best_stats["miou"],
                "best_val_miou_present": best_stats["miou_present"],
                "best_val_miou_epoch": best_stats["epoch"],
                "best_val_miou_present_epoch": best_stats["epoch"],
                "final_val_miou": final_stats["miou"],
                "final_val_miou_present": final_stats["miou_present"],
                "val_bestckpt_miou": best_stats["miou"],
                "val_bestckpt_miou_present": best_stats["miou_present"],
                "val_bestckpt_macro_f1_present": best_stats["macro_f1_present"],
                "val_bestckpt_oa_fg": best_stats["oa_fg"],
                "val_overfit_drop_from_best": best_stats["miou"] - final_stats["miou"],
            }
        )
    return summary, eval_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--loss-variant", default="native_set_prediction")
    parser.add_argument("--iters-per-epoch", type=int, default=8225)
    parser.add_argument("--dataset-test", default="goldmdd_sem_seg_test")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GOLDMDD_PEM_ROOT", "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-pem")
    os.environ.setdefault("GOLDMDD_DATA_CROPPED_ROOT", "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped")

    module = load_train_module(repo_root)
    train_args = SimpleNamespace(
        config_file=str(Path(args.config_file).resolve()),
        opts=[
            "MODEL.WEIGHTS",
            str(Path(args.weights).resolve()),
            "OUTPUT_DIR",
            str(output_dir),
            "DATASETS.TEST",
            f"('{args.dataset_test}',)",
        ],
        eval_only=True,
        resume=False,
        num_gpus=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
    )
    cfg = module.setup(train_args)
    model = module.Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    results = module.Trainer.test(cfg, model)

    sem_seg = results.get("sem_seg", {})
    meta = MetadataCatalog.get(args.dataset_test)
    class_names = list(meta.stuff_classes)

    per_class_iou = {}
    present_ious = []
    for class_name in class_names:
        key = f"IoU-{class_name}"
        raw_value = sem_seg.get(key, float("nan"))
        value = float(raw_value) / 100.0 if raw_value == raw_value else float("nan")
        per_class_iou[class_name] = value
        if value == value:
            present_ious.append(value)

    macro_f1_present = float("nan")
    if present_ious:
        macro_f1_present = sum(iou_to_f1(v) for v in present_ious) / len(present_ious)

    training_summary, _ = parse_training_metrics(Path(args.metrics_path), args.iters_per_epoch)

    unified = {
        "run": args.run_name,
        "family": args.family,
        "loss_variant": args.loss_variant,
        "epochs": 80,
        "test_miou": float(sem_seg.get("mIoU", float("nan"))) / 100.0,
        "test_miou_present": sum(present_ious) / len(present_ious) if present_ious else float("nan"),
        "test_macro_f1_present": macro_f1_present,
        "test_oa_fg": float(sem_seg.get("pACC", float("nan"))) / 100.0,
        "test_fw_iou": float(sem_seg.get("fwIoU", float("nan"))) / 100.0,
        "test_macc": float(sem_seg.get("mACC", float("nan"))) / 100.0,
        "per_class_iou": per_class_iou,
        "raw_detectron2_sem_seg": sem_seg,
    }
    unified.update(training_summary)

    out_json = output_dir / "test_metrics_unified.json"
    out_json.write_text(json.dumps(unified, indent=2))
    print(json.dumps(unified, indent=2))


if __name__ == "__main__":
    main()
