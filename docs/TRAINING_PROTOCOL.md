# Unified Training/Evaluation Protocol

- Dataset root for training: `GoldMDD/data-cropped` (train/val/test patch split).
- Patch size and split prep: `512x512`, stride `256`, invalid/background-heavy patches filtered at dataset build time.
- Label space: 14 foreground classes; background pixels are excluded from supervised loss and mIoU-present computations.
- Default unified setup (unless explicitly marked as native):
  - epochs: `80`
  - batch size: `8`
  - optimizer/lr schedule: model wrapper default aligned to each framework script
  - augmentation preset: `goldmdd_v2`
  - loss family: one of `ce+dice`, `weighted_ce+dice`, `focal+dice`
  - best checkpoint selection: by `best_val_miou_present`
- Evaluation:
  - validation and test unified exports are produced by `misc/eval_*_unified.py` scripts
  - suite sync/aggregation is handled by `misc/sync_experiment_tables.py`
- Efficiency metrics (Params/GFLOPs/Latency/Peak VRAM):
  - generated with `misc/benchmark_semseg_model.py`
  - current latency suite is standardized on L40S runs via `slurm/run_goldmdd_latency_l40s.sh`.
