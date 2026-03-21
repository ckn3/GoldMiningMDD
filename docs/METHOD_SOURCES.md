# Method Sources and Local Integration

This repository does not vendor full third-party projects. It keeps wrappers, configs, and patches required to reproduce GoldMDD experiments.

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
