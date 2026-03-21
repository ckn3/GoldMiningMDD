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
- Unified runs target consistent training controls (80 epochs, batch size 8, augmentation preset `goldmdd_v2`) unless explicitly marked as native.

## Complete method coverage

Total models tracked: **33**

| Model | Status | Venue | Upstream project | Upstream repo | Local entrypoint |
|---|---|---|---|---|---|
| DeepLabV3+ | completed | ECCV2018 | SMP + torchvision | https://github.com/qubvel-org/segmentation_models.pytorch | `misc/train_semseg_smp.py` |
| UPerNet | completed | ECCV2018 | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=upernet_swin_tiny`) |
| FarSeg | completed | CVPR2020 | FarSeg | https://github.com/Z-Zheng/FarSeg.git | `misc/train_semseg_farseg.py --model farseg` |
| OCRNet | completed | ECCV2020 | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=ocrnet_hr48`) |
| ABCNet | completed | ISPRSJPRS2021 | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/abcnet.py` |
| BANet | completed | RS2021 | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/banet.py` |
| SegFormer | completed | NeurIPS2021 | Transformers | https://github.com/huggingface/transformers | `misc/train_semseg_segformer.py` |
| A2FPN | completed | IJRS2022 | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/a2fpn.py` |
| DC-Swin | completed | TGRS2022 | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/dcswin.py` |
| MANet | completed | TGRS2022 | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/manet.py` |
| Mask2Former | completed | CVPR2022 | Mask2Former | https://github.com/facebookresearch/Mask2Former.git | `slurm/submit_goldmdd_mask2former_single.slurm` |
| SegNeXt | completed | NeurIPS2022 | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=segnext_tiny`) |
| UNetFormer | completed | ISPRSJPRS2022 | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/unetformer.py` |
| Afformer | completed | AAAI2023 | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=afformer_base`) |
| EfficientViT-Seg | completed | ICCV2023 | EfficientViT | https://github.com/mit-han-lab/efficientvit.git | `misc/train_semseg_efficientvit.py` |
| FarSeg++ | completed | TGRS2023 | FarSeg | https://github.com/Z-Zheng/FarSeg.git | `misc/train_semseg_farseg.py --model farsegpp` |
| HQ-SAM | completed | NeurIPS2023 | SAM-HQ | https://github.com/SysCV/sam-hq | `misc/train_semseg_sam_family.py --model-family hq_sam` |
| LoGCAN | completed | ICASSP2023 | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=logcan`) |
| SACANet | completed | ICME2023 | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=sacanet`) |
| SeaFormer | completed | ICLR2023 | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=seaformer_base`) |
| CGRSeg | completed | ECCV2024 | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=cgrseg_b`) |
| DOCNet | completed | GRSL2024 | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=docnet`) |
| PEM | completed | CVPR2024 | PEM | https://github.com/NiccoloCavagnero/PEM.git | `slurm/submit_goldmdd_pem_single.slurm` |
| RS3Mamba | completed | GRSL2024 | SSRS | https://github.com/sstary/SSRS.git | `misc/train_semseg_rs3mamba.py` |
| SAM_RS | completed | TGRS2024 | SSRS | https://github.com/sstary/SSRS.git | `misc/train_semseg_sam_rs.py` |
| LoGCAN++ | completed | TGRS2025 | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=logcanplus`) |
| MCPNet | completed | TGRS2025 | MCPNet | https://github.com/fsqy-zhang/MCPNet.git | `third_party/MCPNet/tools/train.py -c third_party/MCPNet/configs/goldmdd_mcpnet_full_80ep_bs8*.py` |
| MF-Mamba | completed | TGRS2025 | MF-Mamba | https://github.com/Mango-Mars/MF-Mamba.git | `misc/train_semseg_mfmamba.py` |
| PPMambaSeg | running | GRSL2025 | PPMambaSeg | https://github.com/Jerrymo59/PPMambaSeg.git | `slurm/submit_goldmdd_ppmambaseg_single.slurm` |
| PyramidMamba | completed | JAG2025 | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/pyramidmamba.py` |
| RSAM-Seg | completed | TGRS2025 | RSAM-Seg | https://github.com/Chief-byte/RSAM-Seg | `misc/train_semseg_rsamseg.py` |
| SAM2.1 | completed | ICLR2025 | SAM2 | https://github.com/facebookresearch/sam2 | `misc/train_semseg_sam_family.py --model-family sam2_1` |
| SESSRS | completed | TGRS2025 | SESSRS | https://github.com/qycools/SESSRS.git | `misc/run_sessrs_postprocess_geoseg_official.py` |

## Key results (best completed run per model)

| Model | Venue | Backbone | Loss | Test mIoU (present) | Test Macro-F1 (present) | Test OA_fg | Params (M) | GFLOPs | Latency ms (1x3x512x512) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A2FPN | IJRS2022 | ResNet-18 | weighted_ce+dice | 0.3720 | 0.5107 | 0.7094 | 12.1620 | 27.1366 | 3.8670 |
| ABCNet | ISPRSJPRS2021 | ResNet-18 | ce+dice+aux_ce | 0.3145 | 0.4302 | 0.6831 | 13.9645 | 32.3860 | 4.0397 |
| Afformer | AAAI2023 | AFFormer-Base | ce+dice | 0.3047 | 0.4362 | 0.6389 | 2.9690 | 8.5730 | 7.4704 |
| BANet | RS2021 | ResT-Lite | ce+dice | 0.2926 | 0.4147 | 0.6535 | 12.8608 | 31.3805 | 4.6832 |
| CGRSeg | ECCV2024 | EfficientFormerV2-B | ce+dice | 0.2679 | 0.3961 | 0.5844 | 19.0799 | 7.5003 | 15.2859 |
| DC-Swin | TGRS2022 | Swin-Small | ce+dice | 0.2971 | 0.4173 | 0.6584 | 66.9503 | 144.3925 | 12.9504 |
| DOCNet | GRSL2024 | HRNet-W32 | ce+aux_ce (native) | 0.3147 | 0.4398 | 0.6785 | 39.1269 | 395.3173 | 22.5021 |
| DeepLabV3+ | ECCV2018 | ConvNeXt-Tiny | ce+dice | 0.3895 | 0.5260 | 0.7189 | 29.3108 | 75.9139 | 4.4970 |
| EfficientViT-Seg | ICCV2023 | EfficientViT-B2 | ce+dice | 0.3799 | 0.5065 | 0.7258 | 15.2802 | 18.3156 | 6.4212 |
| FarSeg | CVPR2020 | ResNet-50 | ce (native) | 0.3564 | 0.4726 | 0.7130 | 31.3698 | 94.1161 | 3.9675 |
| FarSeg++ | TGRS2023 | MiT-B2 | ce (native) | 0.3062 | 0.4358 | 0.6669 | 32.5566 | 95.0793 | 8.3478 |
| HQ-SAM | NeurIPS2023 | ViT-B + HQ decoder (full finetune, msfpn) | weighted_ce+dice | 0.2538 | 0.3711 | 0.6150 | 97.8294 | 983.1302 | 189.3055 |
| LoGCAN | ICASSP2023 | ResNet-50 | ce+aux_ce (native) | 0.3108 | 0.4081 | 0.7474 | 30.9157 | 99.2253 | 6.0530 |
| LoGCAN++ | TGRS2025 | RepViT-M2.3 | ce+aux_ce (native) | 0.2264 | 0.3066 | 0.6353 | 25.1927 | 74.3696 | 17.1870 |
| MANet | TGRS2022 | ResNet-50 | focal+dice | 0.3999 | 0.5431 | 0.6848 | 35.8629 | 109.6158 | 4.7845 |
| MCPNet | TGRS2025 | ResNet-50 | ce (native) | 0.3293 | 0.4489 | 0.7103 | 45.1516 | 110.9866 | 7.1530 |
| MF-Mamba | TGRS2025 | HRNet-W18 | ce+dice | 0.3001 | 0.4242 | 0.6376 | 11.2729 | 38.9439 | 20.5415 |
| Mask2Former | CVPR2022 | ResNet-50 | set_matching_ce+mask+dice | 0.2985 | 0.4285 | 0.6561 | 44.0064 | 133.2907 | 17.4630 |
| OCRNet | ECCV2020 | HRNet-W48 | ce+dice | 0.2722 | 0.3954 | 0.5735 | 70.3653 | 325.3542 | 61.4944 |
| PEM | CVPR2024 | ResNet-50 | set_matching_ce+mask+dice | 0.2789 | 0.4011 | 0.6502 | 35.5313 | 60.6003 | 11.5152 |
| PPMambaSeg | GRSL2025 | swsl-ResNet-18 | weighted_ce+dice | 0.3854 | 0.5298 | 0.6816 | 21.7049 | 45.9905 | 11.2748 |
| PyramidMamba | JAG2025 | Swin-Base | ce+dice | 0.3985 | 0.5360 | 0.6833 | 125.1077 | 217.7548 | 29.2066 |
| RS3Mamba | GRSL2024 | ResNet-18 + VMamba-Tiny | weighted_ce+dice | 0.3068 | 0.4280 | 0.6519 | 43.3254 | 78.5912 | 11.6012 |
| RSAM-Seg | TGRS2025 | SAM-ViT-B (frozen encoder) | weighted_ce+dice | 0.3696 | 0.5085 | 0.6978 | 98.5875 | 247.0546 | 15.1369 |
| SACANet | ICME2023 | HRNet-W32 | ce+aux_ce (native) | 0.3294 | 0.4557 | 0.6573 | 30.2704 | 115.9042 | 20.6124 |
| SAM2.1 | ICLR2025 | Hiera-B+ (full finetune, msfpn) | focal+dice | 0.2980 | 0.4155 | 0.6870 | 83.8976 | 191.8167 | 10.4669 |
| SAM_RS | TGRS2024 | UNetFormer + SAM priors | seg+bdy+obj (native) | 0.3241 | 0.4452 | 0.6839 | 11.6880 | - | 3.2453 |
| SESSRS | TGRS2025 | UNetFormer (ce+dice) | t1/t2 search + postprocess | 0.3958 | 0.5167 | 0.7279 | 11.7259 | 23.5509 | 6.9183 |
| SeaFormer | ICLR2023 | SeaFormer-Base | ce+dice | 0.3117 | 0.4408 | 0.6392 | 8.5838 | 3.4741 | 12.4666 |
| SegFormer | NeurIPS2021 | MiT-B2 | weighted_ce+dice | 0.4010 | 0.5297 | 0.7163 | 27.3574 | 121.9349 | 8.3250 |
| SegNeXt | NeurIPS2022 | MSCAN-Tiny | ce+dice | 0.2682 | 0.3884 | 0.6065 | 4.2285 | 12.6449 | 9.2612 |
| UNetFormer | ISPRSJPRS2022 | ResNet-18 | ce+dice+aux_ce | 0.3941 | 0.5152 | 0.7276 | 11.7259 | 23.5509 | 5.8413 |
| UPerNet | ECCV2018 | Swin-Tiny | ce+dice | 0.3371 | 0.4651 | 0.6729 | 59.8371 | 472.1168 | 22.6344 |

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
- Apply local patches with `scripts/apply_overrides.sh`.
