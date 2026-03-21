# GoldMDD Multi-Model Benchmark Code

Code-only release for GoldMDD experiments (wrappers, configs, scripts, evaluation, and experiment summaries).

## What is included
- `misc/`: training/eval/analysis scripts used in this benchmark.
- `envs/`: exported conda environment files.
- `third_party_overrides/`: local patches/untracked additions on top of cloned method repos.
- `results/`: synchronized experiment summary tables.
- `assets/`: key dataset/analysis figures.

## Unified protocol
- See `docs/TRAINING_PROTOCOL.md`.
- Unified runs target consistent training controls (80 epochs, batch size 8, augmentation preset `goldmdd_v2`) unless explicitly marked as native.

## Complete method list
- Total models: **33**

| Model | Status | Venue | Local path | Official GitHub repo |
|---|---|---|---|---|
| DeepLabV3+ | completed | ECCV2018 | `misc/train_semseg_smp.py` | https://github.com/qubvel-org/segmentation_models.pytorch |
| UPerNet | completed | ECCV2018 | `third_party/SSA-Seg/train.py + third_party/SSA-Seg/configs/goldmdd/upernet_swin_tiny_goldmdd.py` | https://github.com/xwmaxwma/SSA-Seg.git |
| FarSeg | completed | CVPR2020 | `misc/train_semseg_farseg.py --model farseg` | https://github.com/Z-Zheng/FarSeg.git |
| OCRNet | completed | ECCV2020 | `third_party/SSA-Seg/train.py + third_party/SSA-Seg/configs/goldmdd/ocrnet_hr48_goldmdd.py` | https://github.com/xwmaxwma/SSA-Seg.git |
| ABCNet | completed | ISPRSJPRS2021 | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/abcnet.py` | https://github.com/WangLibo1995/GeoSeg.git |
| BANet | completed | RS2021 | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/banet.py` | https://github.com/WangLibo1995/GeoSeg.git |
| SegFormer | completed | NeurIPS2021 | `misc/train_semseg_segformer.py` | https://github.com/huggingface/transformers |
| A2FPN | completed | IJRS2022 | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/a2fpn.py` | https://github.com/WangLibo1995/GeoSeg.git |
| DC-Swin | completed | TGRS2022 | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/dcswin.py` | https://github.com/WangLibo1995/GeoSeg.git |
| MANet | completed | TGRS2022 | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/manet.py` | https://github.com/WangLibo1995/GeoSeg.git |
| Mask2Former | completed | CVPR2022 | `third_party/Mask2Former/train_net.py + third_party/Mask2Former/configs/goldmdd/semantic-segmentation/*.yaml` | https://github.com/facebookresearch/Mask2Former.git |
| SegNeXt | completed | NeurIPS2022 | `third_party/SSA-Seg/train.py + third_party/SSA-Seg/configs/goldmdd/segnext_tiny_goldmdd.py` | https://github.com/xwmaxwma/SSA-Seg.git |
| UNetFormer | completed | ISPRSJPRS2022 | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/unetformer.py` | https://github.com/WangLibo1995/GeoSeg.git |
| Afformer | completed | AAAI2023 | `third_party/SSA-Seg/train.py + third_party/SSA-Seg/configs/goldmdd/afformer_base_goldmdd.py` | https://github.com/xwmaxwma/SSA-Seg.git |
| EfficientViT-Seg | completed | ICCV2023 | `misc/train_semseg_efficientvit.py` | https://github.com/mit-han-lab/efficientvit.git |
| FarSeg++ | completed | TGRS2023 | `misc/train_semseg_farseg.py --model farsegpp` | https://github.com/Z-Zheng/FarSeg.git |
| HQ-SAM | completed | NeurIPS2023 | `misc/train_semseg_sam_family.py --model-family hq_sam` | https://github.com/SysCV/sam-hq |
| LoGCAN | completed | ICASSP2023 | `third_party/rssegmentation/train.py + third_party/rssegmentation/configs/goldmdd/logcan_r50_goldmdd.py` | https://github.com/xwmaxwma/rssegmentation.git |
| SACANet | completed | ICME2023 | `third_party/rssegmentation/train.py + third_party/rssegmentation/configs/goldmdd/sacanet_hrnetw32_goldmdd.py` | https://github.com/xwmaxwma/rssegmentation.git |
| SeaFormer | completed | ICLR2023 | `third_party/SSA-Seg/train.py + third_party/SSA-Seg/configs/goldmdd/seaformer_base_goldmdd.py` | https://github.com/xwmaxwma/SSA-Seg.git |
| CGRSeg | completed | ECCV2024 | `third_party/SSA-Seg/train.py + third_party/SSA-Seg/configs/goldmdd/cgrseg_b_goldmdd.py` | https://github.com/xwmaxwma/SSA-Seg.git |
| DOCNet | completed | GRSL2024 | `third_party/rssegmentation/train.py + third_party/rssegmentation/configs/goldmdd/docnet_hrnetw32_goldmdd.py` | https://github.com/xwmaxwma/rssegmentation.git |
| PEM | completed | CVPR2024 | `third_party/PEM/train_net.py + third_party/PEM/configs/goldmdd/semantic-segmentation/*.yaml` | https://github.com/NiccoloCavagnero/PEM.git |
| RS3Mamba | completed | GRSL2024 | `misc/train_semseg_rs3mamba.py` | https://github.com/sstary/SSRS.git |
| SAM_RS | completed | TGRS2024 | `misc/train_semseg_sam_rs.py` | https://github.com/sstary/SSRS.git |
| LoGCAN++ | completed | TGRS2025 | `third_party/rssegmentation/train.py + third_party/rssegmentation/configs/goldmdd/logcanplus_repvitm23_goldmdd.py` | https://github.com/xwmaxwma/rssegmentation.git |
| MCPNet | completed | TGRS2025 | `third_party/MCPNet/tools/train.py + third_party/MCPNet/configs/goldmdd_*.py` | https://github.com/fsqy-zhang/MCPNet.git |
| MF-Mamba | completed | TGRS2025 | `misc/train_semseg_mfmamba.py` | https://github.com/Mango-Mars/MF-Mamba.git |
| PPMambaSeg | running | GRSL2025 | `third_party/PPMambaSeg/GeoSeg/train_supervision.py + third_party/PPMambaSeg/GeoSeg/config/goldmdd/*.py` | https://github.com/Jerrymo59/PPMambaSeg.git |
| PyramidMamba | completed | JAG2025 | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/pyramidmamba.py` | https://github.com/WangLibo1995/GeoSeg.git |
| RSAM-Seg | completed | TGRS2025 | `misc/train_semseg_rsamseg.py` | https://github.com/Chief-byte/RSAM-Seg |
| SAM2.1 | completed | ICLR2025 | `misc/train_semseg_sam_family.py --model-family sam2_1` | https://github.com/facebookresearch/sam2 |
| SESSRS | completed | TGRS2025 | `misc/run_sessrs_postprocess_geoseg_official.py` | https://github.com/qycools/SESSRS.git |

## Complete results table (all runs)
- Total runs: **75**

| Status | Model | Backbone | Loss | Venue | Test mIoU (present) | Test Macro-F1 (present) | Test OA_fg | Best Val mIoU (present) | Params (M) | GFLOPs | Latency ms | Peak VRAM (GB) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| completed | DeepLabV3+ | ConvNeXt-Tiny | ce+dice | ECCV2018 | 0.3895 | 0.5260 | 0.7189 | 0.3545 | 29.3108 | 75.9139 | 4.4970 | 0.2086 |
| completed | DeepLabV3+ | ConvNeXt-Tiny | focal+dice | ECCV2018 | 0.3735 | 0.5068 | 0.7048 | 0.3581 | 29.3108 | 75.9139 | 4.4970 | 0.2086 |
| completed | DeepLabV3+ | ConvNeXt-Tiny | weighted_ce+dice | ECCV2018 | 0.3766 | 0.5193 | 0.6840 | 0.3750 | 29.3108 | 75.9139 | 4.4970 | 0.2086 |
| completed | DeepLabV3+ | ResNet-50 | ce+dice | ECCV2018 | 0.3662 | 0.4897 | 0.7092 | 0.3379 | 26.6809 | 73.5922 | 2.8963 | 0.2300 |
| completed | DeepLabV3+ | ResNet-50 | focal+dice | ECCV2018 | 0.3515 | 0.4819 | 0.7000 | 0.3461 | 26.6809 | 73.5922 | 2.8963 | 0.2300 |
| completed | DeepLabV3+ | ResNet-50 | weighted_ce+dice | ECCV2018 | 0.3438 | 0.4755 | 0.6951 | 0.3427 | 26.6809 | 73.5922 | 2.8963 | 0.2300 |
| completed | UPerNet | Swin-Tiny | ce+dice | ECCV2018 | 0.3371 | 0.4651 | 0.6729 | 0.2873 | 59.8371 | 472.1168 | 22.6344 | 0.5306 |
| completed | FarSeg | ResNet-50 | ce (native) | CVPR2020 | 0.3564 | 0.4726 | 0.7130 | 0.2989 | 31.3698 | 94.1161 | 3.9675 | 0.2414 |
| completed | OCRNet | HRNet-W48 | ce+dice | ECCV2020 | 0.2722 | 0.3954 | 0.5735 | 0.2954 | 70.3653 | 325.3542 | 61.4944 | 0.5052 |
| completed | ABCNet | ResNet-18 | ce+dice+aux_ce | ISPRSJPRS2021 | 0.3145 | 0.4302 | 0.6831 | 0.3070 | 13.9645 | 32.3860 | 4.0397 | 0.1004 |
| completed | BANet | ResT-Lite | ce+dice | RS2021 | 0.2926 | 0.4147 | 0.6535 | 0.2992 | 12.8608 | 31.3805 | 4.6832 | 0.1026 |
| completed | SegFormer | MiT-B2 | ce+dice | NeurIPS2021 | 0.3222 | 0.4594 | 0.6484 | 0.3480 | 27.3574 | 121.9349 | 8.3250 | 0.5432 |
| completed | SegFormer | MiT-B2 | focal+dice | NeurIPS2021 | 0.3332 | 0.4760 | 0.6385 | 0.3501 | 27.3574 | 121.9349 | 8.3250 | 0.5432 |
| completed | SegFormer | MiT-B2 | weighted_ce+dice | NeurIPS2021 | 0.4010 | 0.5297 | 0.7163 | 0.3423 | 27.3574 | 121.9349 | 8.3250 | 0.5432 |
| completed | A2FPN | ResNet-18 | ce+dice | IJRS2022 | 0.3688 | 0.4834 | 0.7335 | 0.3085 | 12.1620 | 27.1366 | 3.8670 | 0.2150 |
| completed | A2FPN | ResNet-18 | focal+dice | IJRS2022 | 0.3363 | 0.4602 | 0.6659 | 0.3027 | 12.1620 | 27.1366 | 3.8670 | 0.2150 |
| completed | A2FPN | ResNet-18 | weighted_ce+dice | IJRS2022 | 0.3720 | 0.5107 | 0.7094 | 0.3187 | 12.1620 | 27.1366 | 3.8670 | 0.2150 |
| completed | DC-Swin | Swin-Small | ce+dice | TGRS2022 | 0.2971 | 0.4173 | 0.6584 | 0.2884 | 66.9503 | 144.3925 | 12.9504 | 0.3561 |
| completed | MANet | ResNet-50 | ce+dice | TGRS2022 | 0.3711 | 0.4922 | 0.6862 | 0.3147 | 35.8629 | 109.6158 | 4.7845 | 0.3940 |
| completed | MANet | ResNet-50 | focal+dice | TGRS2022 | 0.3999 | 0.5431 | 0.6848 | 0.3246 | 35.8629 | 109.6158 | 4.7845 | 0.3940 |
| completed | MANet | ResNet-50 | weighted_ce+dice | TGRS2022 | 0.3828 | 0.5228 | 0.6759 | 0.3316 | 35.8629 | 109.6158 | 4.7845 | 0.3940 |
| completed | Mask2Former | ResNet-50 | set_matching_ce+mask+dice | CVPR2022 | 0.2985 | 0.4285 | 0.6561 | 0.3033 | 44.0064 | 133.2907 | 17.4630 | 0.4213 |
| completed | SegNeXt | MSCAN-Tiny | ce+dice | NeurIPS2022 | 0.2682 | 0.3884 | 0.6065 | 0.2896 | 4.2285 | 12.6449 | 9.2612 | 0.1673 |
| completed | UNetFormer | ResNet-18 | ce+dice+aux_ce | ISPRSJPRS2022 | 0.3941 | 0.5152 | 0.7276 | 0.3388 | 11.7259 | 23.5509 | 5.8413 | 0.0899 |
| completed | UNetFormer | ResNet-18 | focal+dice+aux_focal | ISPRSJPRS2022 | 0.3765 | 0.4989 | 0.7182 | 0.3314 | 11.7259 | 23.5509 | 5.8413 | 0.0899 |
| completed | UNetFormer | ResNet-18 | weighted_ce+dice+aux_ce | ISPRSJPRS2022 | 0.3566 | 0.4931 | 0.6811 | 0.3275 | 11.7259 | 23.5509 | 5.8413 | 0.0899 |
| completed | Afformer | AFFormer-Base | ce+dice | AAAI2023 | 0.3047 | 0.4362 | 0.6389 | 0.3153 | 2.9690 | 8.5730 | 7.4704 | 0.1691 |
| completed | EfficientViT-Seg | EfficientViT-B2 | ce+dice | ICCV2023 | 0.3799 | 0.5065 | 0.7258 | 0.3444 | 15.2802 | 18.3156 | 6.4212 | 0.1213 |
| completed | EfficientViT-Seg | EfficientViT-B2 | focal+dice | ICCV2023 | 0.3693 | 0.5086 | 0.6781 | 0.3790 | 15.2802 | 18.3156 | 6.4212 | 0.1213 |
| completed | EfficientViT-Seg | EfficientViT-B2 | weighted_ce+dice | ICCV2023 | 0.3790 | 0.5181 | 0.7154 | 0.3840 | 15.2802 | 18.3156 | 6.4212 | 0.1213 |
| completed | FarSeg++ | MiT-B2 | ce (native) | TGRS2023 | 0.3062 | 0.4358 | 0.6669 | 0.3284 | 32.5566 | 95.0793 | 8.3478 | 0.2784 |
| completed | HQ-SAM | ViT-B + HQ decoder (full finetune, msfpn) | ce+dice | NeurIPS2023 | 0.2485 | 0.3558 | 0.6390 | 0.2226 | 97.8294 | 983.1302 | 189.3055 | 2.7571 |
| completed | HQ-SAM | ViT-B + HQ decoder (full finetune, msfpn) | focal+dice | NeurIPS2023 | 0.2503 | 0.3561 | 0.6539 | 0.2248 | 97.8294 | 983.1302 | 189.3055 | 2.7571 |
| completed | HQ-SAM | ViT-B + HQ decoder (full finetune, msfpn) | weighted_ce+dice | NeurIPS2023 | 0.2538 | 0.3711 | 0.6150 | 0.2237 | 97.8294 | 983.1302 | 189.3055 | 2.7571 |
| completed | LoGCAN | ResNet-50 | ce+aux_ce (native) | ICASSP2023 | 0.3108 | 0.4081 | 0.7474 | 0.2951 | 30.9157 | 99.2253 | 6.0530 | 0.2298 |
| completed | SACANet | HRNet-W32 | ce+aux_ce (native) | ICME2023 | 0.3294 | 0.4557 | 0.6573 | 0.3215 | 30.2704 | 115.9042 | 20.6124 | 0.3073 |
| completed | SeaFormer | SeaFormer-Base | ce+dice | ICLR2023 | 0.3117 | 0.4408 | 0.6392 | 0.3116 | 8.5838 | 3.4741 | 12.4666 | 0.1700 |
| completed | CGRSeg | EfficientFormerV2-B | ce+dice | ECCV2024 | 0.2679 | 0.3961 | 0.5844 | 0.3054 | 19.0799 | 7.5003 | 15.2859 | 0.2569 |
| completed | DOCNet | HRNet-W32 | ce+aux_ce (native) | GRSL2024 | 0.3147 | 0.4398 | 0.6785 | 0.2772 | 39.1269 | 395.3173 | 22.5021 | 0.4263 |
| completed | PEM | ResNet-50 | set_matching_ce+mask+dice | CVPR2024 | 0.2789 | 0.4011 | 0.6502 | 0.2549 | 35.5313 | 60.6003 | 11.5152 | 0.3881 |
| completed | RS3Mamba | ResNet-18 + VMamba-Tiny | ce+dice | GRSL2024 | 0.2385 | 0.3080 | 0.7257 | 0.1559 | 43.3254 | 78.5912 | 11.6012 | 0.4624 |
| completed | RS3Mamba | ResNet-18 + VMamba-Tiny | focal+dice | GRSL2024 | 0.2399 | 0.3125 | 0.7251 | 0.1910 | 43.3254 | 78.5912 | 11.6012 | 0.4624 |
| completed | RS3Mamba | ResNet-18 + VMamba-Tiny | weighted_ce+dice | GRSL2024 | 0.3068 | 0.4280 | 0.6519 | 0.2313 | 43.3254 | 78.5912 | 11.6012 | 0.4624 |
| completed | SAM_RS | ABCNet + SAM priors | seg+bdy+obj (native) | TGRS2024 | 0.2964 | 0.4098 | 0.6573 | 0.3104 | 13.9645 | - | 2.6472 | 0.1014 |
| completed | SAM_RS | CMTFNet + SAM priors | seg+bdy+obj (native) | TGRS2024 | 0.2916 | 0.4084 | 0.6598 | 0.2909 | 30.0727 | - | 6.3132 | 0.3345 |
| completed | SAM_RS | FTUNetFormer + SAM priors | seg+bdy+obj (native) | TGRS2024 | 0.2922 | 0.4094 | 0.6871 | 0.2859 | 96.1376 | - | 14.9292 | 0.4374 |
| completed | SAM_RS | UNetFormer + SAM priors | seg+bdy+obj (native) | TGRS2024 | 0.3241 | 0.4452 | 0.6839 | 0.2971 | 11.6880 | - | 3.2453 | 0.0874 |
| completed | LoGCAN++ | RepViT-M2.3 | ce+aux_ce (native) | TGRS2025 | 0.2264 | 0.3066 | 0.6353 | 0.2264 | 25.1927 | 74.3696 | 17.1870 | 0.2225 |
| completed | MCPNet | ResNet-50 | ce (native) | TGRS2025 | 0.3293 | 0.4489 | 0.7103 | 0.3063 | 45.1516 | 110.9866 | 7.1530 | 0.3528 |
| completed | MCPNet | ResNet-50 | ce+dice | TGRS2025 | 0.3056 | 0.4267 | 0.6680 | 0.3051 | 45.1516 | 110.9866 | 7.1530 | 0.3528 |
| completed | MCPNet | ResNet-50 | focal+dice | TGRS2025 | 0.3233 | 0.4448 | 0.6898 | 0.3027 | 45.1516 | 110.9866 | 7.1530 | 0.3528 |
| completed | MCPNet | ResNet-50 | weighted_ce+dice | TGRS2025 | 0.3193 | 0.4552 | 0.6405 | 0.3181 | 45.1516 | 110.9866 | 7.1530 | 0.3528 |
| completed | MF-Mamba | HRNet-W18 | ce+dice | TGRS2025 | 0.3001 | 0.4242 | 0.6376 | 0.3039 | 11.2729 | 38.9439 | 20.5415 | 0.1326 |
| completed | PPMambaSeg | swsl-ResNet-18 | ce+dice | GRSL2025 | 0.3520 | 0.4780 | 0.6683 | 0.3362 | 21.7049 | 45.9905 | 11.2748 | 0.3103 |
| running | PPMambaSeg | swsl-ResNet-18 | focal+dice | GRSL2025 | - | - | - | - | 21.7049 | 45.9905 | 11.2748 | 0.3103 |
| completed | PPMambaSeg | swsl-ResNet-18 | weighted_ce+dice | GRSL2025 | 0.3854 | 0.5298 | 0.6816 | 0.3466 | 21.7049 | 45.9905 | 11.2748 | 0.3103 |
| completed | PyramidMamba | Swin-Base | ce+dice | JAG2025 | 0.3985 | 0.5360 | 0.6833 | 0.3703 | 125.1077 | 217.7548 | 29.2066 | 0.6582 |
| completed | RSAM-Seg | SAM-ViT-B (frozen encoder) | ce+dice | TGRS2025 | 0.3263 | 0.4472 | 0.6978 | 0.2959 | 98.5875 | 247.0546 | 15.1369 | 0.6103 |
| completed | RSAM-Seg | SAM-ViT-B (frozen encoder) | focal+dice | TGRS2025 | 0.3450 | 0.4642 | 0.7430 | 0.2841 | 98.5875 | 247.0546 | 15.1369 | 0.6103 |
| completed | RSAM-Seg | SAM-ViT-B (frozen encoder) | weighted_ce+dice | TGRS2025 | 0.3696 | 0.5085 | 0.6978 | 0.3333 | 98.5875 | 247.0546 | 15.1369 | 0.6103 |
| completed | SAM2.1 | Hiera-B+ (frozen backbone, msfpn) | ce+dice | ICLR2025 | 0.2422 | 0.3510 | 0.6193 | 0.2242 | 83.8976 | 191.8167 | 10.4669 | 0.5898 |
| completed | SAM2.1 | Hiera-B+ (frozen backbone, msfpn) | focal+dice | ICLR2025 | 0.2351 | 0.3438 | 0.6158 | 0.2288 | 83.8976 | 191.8167 | 10.4669 | 0.5898 |
| completed | SAM2.1 | Hiera-B+ (frozen backbone, msfpn) | weighted_ce+dice | ICLR2025 | 0.2207 | 0.3235 | 0.5938 | 0.2326 | 83.8976 | 191.8167 | 10.4669 | 0.5898 |
| completed | SAM2.1 | Hiera-B+ (full finetune, msfpn) | ce+dice | ICLR2025 | 0.2885 | 0.4089 | 0.6562 | 0.2903 | 83.8976 | 191.8167 | 10.4669 | 0.5898 |
| completed | SAM2.1 | Hiera-B+ (full finetune, msfpn) | focal+dice | ICLR2025 | 0.2980 | 0.4155 | 0.6870 | 0.2906 | 83.8976 | 191.8167 | 10.4669 | 0.5898 |
| completed | SAM2.1 | Hiera-B+ (full finetune, msfpn) | weighted_ce+dice | ICLR2025 | 0.2875 | 0.4058 | 0.6769 | 0.2933 | 83.8976 | 191.8167 | 10.4669 | 0.5898 |
| completed | SESSRS | A2FPN (ce+dice) | t1/t2 search + postprocess | TGRS2025 | 0.3702 | 0.4848 | 0.7338 | 0.3094 | 12.1620 | 27.1366 | 5.3153 | 0.2150 |
| completed | SESSRS | A2FPN (focal) | t1/t2 search + postprocess | TGRS2025 | 0.3374 | 0.4613 | 0.6663 | 0.3035 | 12.1620 | 27.1366 | 5.9873 | 0.2150 |
| completed | SESSRS | A2FPN (weighted) | t1/t2 search + postprocess | TGRS2025 | 0.3745 | 0.5139 | 0.7098 | 0.3214 | 12.1620 | 27.1366 | 5.4877 | 0.2150 |
| completed | SESSRS | ABCNet (ce+dice+aux) | t1/t2 search + postprocess | TGRS2025 | 0.3154 | 0.4311 | 0.6835 | 0.3078 | 13.9645 | 32.3860 | 82.7830 | 0.1004 |
| completed | SESSRS | BANet (ce+dice) | t1/t2 search + postprocess | TGRS2025 | 0.2937 | 0.4161 | 0.6536 | 0.3006 | 12.8608 | 31.3805 | 12.8519 | 0.1029 |
| completed | SESSRS | MANet (ce+dice) | t1/t2 search + postprocess | TGRS2025 | 0.3604 | 0.4820 | 0.6775 | 0.3162 | 35.8629 | 109.6158 | 12.4341 | 0.3940 |
| completed | SESSRS | UNetFormer (ce+dice) | t1/t2 search + postprocess | TGRS2025 | 0.3958 | 0.5167 | 0.7279 | 0.3399 | 11.7259 | 23.5509 | 6.9183 | 0.0876 |
| completed | SESSRS | UNetFormer (focal) | t1/t2 search + postprocess | TGRS2025 | 0.3873 | 0.5091 | 0.7195 | 0.3406 | 11.7259 | 23.5509 | 7.1679 | 0.0876 |
| completed | SESSRS | UNetFormer (weighted) | t1/t2 search + postprocess | TGRS2025 | 0.3578 | 0.4943 | 0.6816 | 0.3286 | 11.7259 | 23.5509 | 7.3700 | 0.0876 |

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
