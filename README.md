# GoldMDD Multi-Model Benchmark Code

Code-only release for GoldMDD experiments (wrappers, configs, scripts, evaluation, and experiment summaries).

## What is included
- `misc/`: training/eval/analysis scripts used in this benchmark.
- `slurm/`: submission scripts for cluster jobs.
- `envs/`: exported conda environment files.
- `third_party_overrides/`: local patches/untracked additions on top of cloned method repos.
- `results/`: synchronized experiment summary tables.
- `assets/`: key dataset/analysis figures.

## Unified protocol
- See `docs/TRAINING_PROTOCOL.md`.
- All unified runs use the same patch split (`data-cropped`) and matched evaluation/sync pipeline.

## Method sources and where code comes from
- See `docs/METHOD_SOURCES.md`.
- Unmodified third-party code is **not vendored** here; only our wrappers and overrides are included.

## Key results (best run per model)

| Model | Backbone | Loss | Venue | Test mIoU (present) | Test Macro-F1 (present) | Test OA_fg | Params (M) | GFLOPs | Latency ms (1x3x512x512) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| PPMambaSeg | swsl-ResNet-18 | weighted_ce+dice | GRSL2025 | 0.3854 | 0.5298 | 0.6816 | 21.7049 | 45.9905 | 11.2748 |
| SAM2.1 | Hiera-B+ (full finetune, msfpn) | focal+dice | ICLR2025 | 0.2980 | 0.4155 | 0.6870 | 83.8976 | 191.8167 | 10.4669 |
| PyramidMamba | Swin-Base | ce+dice | JAG2025 | 0.3985 | 0.5360 | 0.6833 | 125.1077 | 217.7548 | 29.2066 |
| HQ-SAM | ViT-B + HQ decoder (full finetune, msfpn) | weighted_ce+dice | NeurIPS2023 | 0.2538 | 0.3711 | 0.6150 | 97.8294 | 983.1302 | 189.3055 |
| DC-Swin | Swin-Small | ce+dice | TGRS2022 | 0.2971 | 0.4173 | 0.6584 | 66.9503 | 144.3925 | 12.9504 |
| MANet | ResNet-50 | focal+dice | TGRS2022 | 0.3999 | 0.5431 | 0.6848 | 35.8629 | 109.6158 | 4.7845 |
| A2FPN | ResNet-18 | weighted_ce+dice | nan | 0.3720 | 0.5107 | 0.7094 | 12.1620 | 27.1366 | 3.8670 |
| ABCNet | ResNet-18 | ce+dice+aux_ce | nan | 0.3145 | 0.4302 | 0.6831 | 13.9645 | 32.3860 | 4.0397 |
| Afformer | AFFormer-Base | ce+dice | nan | 0.3047 | 0.4362 | 0.6389 | 2.9690 | 8.5730 | 7.4704 |
| BANet | ResT-Lite | ce+dice | nan | 0.2926 | 0.4147 | 0.6535 | 12.8608 | 31.3805 | 4.6832 |
| CGRSeg | EfficientFormerV2-B | ce+dice | nan | 0.2679 | 0.3961 | 0.5844 | 19.0799 | 7.5003 | 15.2859 |
| DOCNet | HRNet-W32 | ce+aux_ce (native) | nan | 0.3147 | 0.4398 | 0.6785 | 39.1269 | 395.3173 | 22.5021 |
| DeepLabV3+ | ConvNeXt-Tiny | ce+dice | nan | 0.3895 | 0.5260 | 0.7189 | 29.3108 | 75.9139 | 4.4970 |
| EfficientViT-Seg | EfficientViT-B2 | ce+dice | nan | 0.3799 | 0.5065 | 0.7258 | 15.2802 | 18.3156 | 6.4212 |
| FarSeg | ResNet-50 | ce (native) | nan | 0.3564 | 0.4726 | 0.7130 | 31.3698 | 94.1161 | 3.9675 |
| FarSeg++ | MiT-B2 | ce (native) | nan | 0.3062 | 0.4358 | 0.6669 | 32.5566 | 95.0793 | 8.3478 |
| LoGCAN | ResNet-50 | ce+aux_ce (native) | nan | 0.3108 | 0.4081 | 0.7474 | 30.9157 | 99.2253 | 6.0530 |
| LoGCAN++ | RepViT-M2.3 | ce+aux_ce (native) | nan | 0.2264 | 0.3066 | 0.6353 | 25.1927 | 74.3696 | 17.1870 |
| MCPNet | ResNet-50 | ce (native) | nan | 0.3293 | 0.4489 | 0.7103 | 45.1516 | 110.9866 | 7.1530 |
| MF-Mamba | HRNet-W18 | ce+dice | nan | 0.3001 | 0.4242 | 0.6376 | 11.2729 | 38.9439 | 20.5415 |
| Mask2Former | ResNet-50 | set_matching_ce+mask+dice | nan | 0.2985 | 0.4285 | 0.6561 | 44.0064 | 133.2907 | 17.4630 |
| OCRNet | HRNet-W48 | ce+dice | nan | 0.2722 | 0.3954 | 0.5735 | 70.3653 | 325.3542 | 61.4944 |
| PEM | ResNet-50 | set_matching_ce+mask+dice | nan | 0.2789 | 0.4011 | 0.6502 | 35.5313 | 60.6003 | 11.5152 |
| RS3Mamba | ResNet-18 + VMamba-Tiny | weighted_ce+dice | nan | 0.3068 | 0.4280 | 0.6519 | 43.3254 | 78.5912 | 11.6012 |
| RSAM-Seg | SAM-ViT-B (frozen encoder) | weighted_ce+dice | nan | 0.3696 | 0.5085 | 0.6978 | 98.5875 | 247.0546 | 15.1369 |
| SACANet | HRNet-W32 | ce+aux_ce (native) | nan | 0.3294 | 0.4557 | 0.6573 | 30.2704 | 115.9042 | 20.6124 |
| SAM_RS | UNetFormer + SAM priors | seg+bdy+obj (native) | nan | 0.3241 | 0.4452 | 0.6839 | 11.6880 | - | 3.2453 |
| SESSRS | UNetFormer (ce+dice) | t1/t2 search + postprocess | nan | 0.3958 | 0.5167 | 0.7279 | 11.7259 | 23.5509 | 6.9183 |
| SeaFormer | SeaFormer-Base | ce+dice | nan | 0.3117 | 0.4408 | 0.6392 | 8.5838 | 3.4741 | 12.4666 |
| SegFormer | MiT-B2 | weighted_ce+dice | nan | 0.4010 | 0.5297 | 0.7163 | 27.3574 | 121.9349 | 8.3250 |
| SegNeXt | MSCAN-Tiny | ce+dice | nan | 0.2682 | 0.3884 | 0.6065 | 4.2285 | 12.6449 | 9.2612 |
| UNetFormer | ResNet-18 | ce+dice+aux_ce | nan | 0.3941 | 0.5152 | 0.7276 | 11.7259 | 23.5509 | 5.8413 |
| UPerNet | Swin-Tiny | ce+dice | nan | 0.3371 | 0.4651 | 0.6729 | 59.8371 | 472.1168 | 22.6344 |

Full suite tables and per-class metrics:
- `results/summary.md`
- `results/experiment_suite_summary.csv`
- `results/val_per_class_iou_completed_models_present_only.csv`
- `results/test_per_class_iou_completed_models_present_only.csv`

## Dataset visuals
![Site distribution](assets/site_distribution_osm.png)
![Site class heatmap](assets/site_class_pixel_counts_heatmap_merged.png)
![Train/Val/Test distribution](assets/train_val_test_class_distribution_merged.png)

## Repro notes
- External repositories and local commits are listed in `docs/THIRD_PARTY_LOCKS.md`.
- Apply local patches in `third_party_overrides/` when reproducing method-specific integrations.
