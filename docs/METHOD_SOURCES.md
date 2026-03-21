# Method Sources and Local Integration

This repo does not vendor full third-party projects. It stores wrappers + patches needed for GoldMDD runs.

| Model | Upstream project | Upstream repo | Local entrypoint (this repo) |
|---|---|---|---|
| A2FPN | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/a2fpn.py` |
| ABCNet | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/abcnet.py` |
| Afformer | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=afformer_base`) |
| BANet | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/banet.py` |
| CGRSeg | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=cgrseg_b`) |
| DC-Swin | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/dcswin.py` |
| DOCNet | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=docnet`) |
| DeepLabV3+ | SMP + torchvision | https://github.com/qubvel-org/segmentation_models.pytorch | `misc/train_semseg_smp.py` |
| EfficientViT-Seg | EfficientViT | https://github.com/mit-han-lab/efficientvit.git | `misc/train_semseg_efficientvit.py` |
| FarSeg | FarSeg | https://github.com/Z-Zheng/FarSeg.git | `misc/train_semseg_farseg.py --model farseg` |
| FarSeg++ | FarSeg | https://github.com/Z-Zheng/FarSeg.git | `misc/train_semseg_farseg.py --model farsegpp` |
| HQ-SAM | SAM-HQ | https://github.com/SysCV/sam-hq | `misc/train_semseg_sam_family.py --model-family hq_sam` |
| LoGCAN | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=logcan`) |
| LoGCAN++ | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=logcanplus`) |
| MANet | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/manet.py` |
| MCPNet | MCPNet | https://github.com/fsqy-zhang/MCPNet.git | `third_party/MCPNet/tools/train.py -c third_party/MCPNet/configs/goldmdd_mcpnet_full_80ep_bs8*.py` |
| MF-Mamba | MF-Mamba | https://github.com/Mango-Mars/MF-Mamba.git | `misc/train_semseg_mfmamba.py` |
| Mask2Former | Mask2Former | https://github.com/facebookresearch/Mask2Former.git | `slurm/submit_goldmdd_mask2former_single.slurm` |
| OCRNet | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=ocrnet_hr48`) |
| PEM | PEM | https://github.com/NiccoloCavagnero/PEM.git | `slurm/submit_goldmdd_pem_single.slurm` |
| PPMambaSeg | PPMambaSeg | https://github.com/Jerrymo59/PPMambaSeg.git | `slurm/submit_goldmdd_ppmambaseg_single.slurm` |
| PyramidMamba | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/pyramidmamba.py` |
| RS3Mamba | SSRS | https://github.com/sstary/SSRS.git | `misc/train_semseg_rs3mamba.py` |
| RSAM-Seg | RSAM-Seg | https://github.com/Chief-byte/RSAM-Seg | `misc/train_semseg_rsamseg.py` |
| SACANet | rssegmentation | https://github.com/xwmaxwma/rssegmentation.git | `slurm/submit_goldmdd_rsseg_single.slurm` (`METHOD=sacanet`) |
| SAM2.1 | SAM2 | https://github.com/facebookresearch/sam2 | `misc/train_semseg_sam_family.py --model-family sam2_1` |
| SAM_RS | SSRS | https://github.com/sstary/SSRS.git | `misc/train_semseg_sam_rs.py` |
| SESSRS | SESSRS | https://github.com/qycools/SESSRS.git | `misc/run_sessrs_postprocess_geoseg_official.py` |
| SeaFormer | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=seaformer_base`) |
| SegFormer | Transformers | https://github.com/huggingface/transformers | `misc/train_semseg_segformer.py` |
| SegNeXt | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=segnext_tiny`) |
| UNetFormer | GeoSeg | https://github.com/WangLibo1995/GeoSeg.git | `third_party/GeoSeg/train_supervision.py -c third_party/GeoSeg/config/goldmdd/unetformer.py` |
| UPerNet | SSA-Seg | https://github.com/xwmaxwma/SSA-Seg.git | `slurm/submit_goldmdd_ssaseg_single.slurm` (`MODEL=upernet_swin_tiny`) |
