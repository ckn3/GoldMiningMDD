# Segmentation Confidence-Derived Multi-Label AP

This file follows the method grouping, row order, backbone names, and loss naming used in `experiments/results_seg_mlc.tex`. The original area-ratio mAP from that tex table is kept as `Area mAP`; the new ranking metric is reported as `Max/Proxy mAP`.

Coverage: 93/93 rows have recomputed confidence-based AP. Rows without available confidence outputs are kept in order and marked `missing_max_result`. Score types include `max`, `query-score proxy`, and `sessrs_proxy_max`.

## mAP Summary

### General Segmentation Models

| Model        | Backbone            | Loss             | Status             | Score | Area mAP | Max/Proxy mAP |
| ------------ | ------------------- | ---------------- | ------------------ | ----- | -------- | ------------- |
| DeepLabV3+   | ConvNeXt-Tiny       | CE+Dice          | computed           | max   | 0.6138   | 0.6230        |
| DeepLabV3+   | ConvNeXt-Tiny       | Focal+Dice       | computed           | max   | 0.5926   | 0.6179        |
| DeepLabV3+   | ConvNeXt-Tiny       | WCE+Dice         | computed           | max   | 0.5937   | 0.6107        |
| DeepLabV3+   | ResNet-50           | CE+Dice          | computed           | max   | 0.5582   | 0.5776        |
| DeepLabV3+   | ResNet-50           | Focal+Dice       | computed           | max   | 0.5796   | 0.6120        |
| DeepLabV3+   | ResNet-50           | WCE+Dice         | computed           | max   | 0.5584   | 0.5664        |
| UPerNet      | Swin-Tiny           | CE+Dice          | computed           | max   | 0.5011   | 0.5252        |
| OCRNet       | HRNet-W48           | CE+Dice          | computed           | max   | 0.4633   | 0.4867        |
| BiSeNetv2 | Custom Bilateral | CE+Dice | computed | max | 0.5487 | 0.5377 |
| BiSeNetv2 | Custom Bilateral | Focal+Dice | computed | max | 0.5278 | 0.5263 |
| BiSeNetv2 | Custom Bilateral | WCE+Dice | computed | max | 0.5440 | 0.5369 |
| SegFormer    | MiT-B2              | CE+Dice          | computed           | max   | 0.5235   | 0.5382        |
| SegFormer    | MiT-B2              | Focal+Dice       | computed           | max   | 0.5691   | 0.5857        |
| SegFormer    | MiT-B2              | WCE+Dice         | computed           | max   | 0.6167   | 0.6576        |
| STDC2 | STDCNet | CE+Dice | computed | max | 0.5054 | 0.5016 |
| STDC2 | STDCNet | Focal+Dice | computed | max | 0.5061 | 0.5010 |
| STDC2 | STDCNet | WCE+Dice | computed | max | 0.5441 | 0.5756 |
| Mask2Former | ResNet-50 | Set CE+Mask+Dice | computed | query-score proxy | 0.5094 | 0.5297 |
| SegNeXt      | MSCAN-Tiny          | CE+Dice          | computed           | max   | 0.4682   | 0.4756        |
| DDRNet | DDRNet-23-slim | Focal+Dice | computed | max | 0.4659 | 0.5081 |
| Afformer     | AFFormer-Base       | CE+Dice          | computed           | max   | 0.5154   | 0.5297        |
| EfficientViT | EfficientViT-B2     | CE+Dice          | computed           | max   | 0.5625   | 0.6002        |
| EfficientViT | EfficientViT-B2     | Focal+Dice       | computed           | max   | 0.5776   | 0.6100        |
| EfficientViT | EfficientViT-B2     | WCE+Dice         | computed           | max   | 0.5904   | 0.6228        |
| SeaFormer    | SeaFormer-Base      | CE+Dice          | computed           | max   | 0.4942   | 0.5171        |
| PIDNet | PIDNet-M | CE+Dice | computed | max | 0.5203 | 0.5254 |
| PIDNet | PIDNet-M | Focal+Dice | computed | max | 0.4289 | 0.4298 |
| PIDNet | PIDNet-M | WCE+Dice | computed | max | 0.5248 | 0.5254 |
| CGRSeg       | EfficientFormerV2-B | CE+Dice          | computed           | max   | 0.4820   | 0.4990        |
| PEM | ResNet-50 | Set CE+Mask+Dice | computed | query-score proxy | 0.4868 | 0.5069 |
| VMamba | VMamba-Tiny | CE+Dice | computed | max | 0.4378 | 0.4648 |

| MP-Former | Mask2Former Swin-L | Focal+Dice | computed | query-score proxy | 0.3311 | 0.3122 |
### Remote-Sensing-Specific Methods

| Model        | Backbone    | Loss             | Status             | Score | Area mAP | Max/Proxy mAP |
| ------------ | ----------- | ---------------- | ------------------ | ----- | -------- | ------------- |
| FarSeg       | ResNet-50   | CE               | computed           | max   | 0.5263   | 0.5576        |
| FarSeg       | ResNet-50   | CE+Dice          | computed           | max   | 0.5562   | 0.5752        |
| FarSeg       | ResNet-50   | Focal+Dice       | computed           | max   | 0.5582   | 0.5773        |
| FarSeg       | ResNet-50   | WCE+Dice         | computed           | max   | 0.5762   | 0.6159        |
| BANet        | ResT-Lite   | CE+Dice          | computed           | max   | 0.4664   | 0.4863        |
| ABCNet       | ResNet-18   | CE+Dice$^{*}$    | computed           | max   | 0.4916   | 0.5150        |
| MANet        | ResNet-50   | CE+Dice          | computed           | max   | 0.5693   | 0.5556        |
| MANet        | ResNet-50   | Focal+Dice       | computed           | max   | 0.6643   | 0.6170        |
| MANet        | ResNet-50   | WCE+Dice         | computed           | max   | 0.6559   | 0.6817        |
| UNetFormer   | ResNet-18   | CE+Dice$^{*}$    | computed           | max   | 0.5648   | 0.5913        |
| UNetFormer   | ResNet-18   | Focal+Dice$^{*}$ | computed           | max   | 0.5613   | 0.5807        |
| UNetFormer   | ResNet-18   | WCE+Dice$^{*}$   | computed           | max   | 0.5826   | 0.6119        |
| DC-Swin      | Swin-Small  | CE+Dice          | computed           | max   | 0.4726   | 0.4953        |
| A2-FPN       | ResNet-18   | CE+Dice          | computed           | max   | 0.5239   | 0.5592        |
| A2-FPN       | ResNet-18   | Focal+Dice       | computed           | max   | 0.5219   | 0.5414        |
| A2-FPN       | ResNet-18   | WCE+Dice         | computed           | max   | 0.5982   | 0.6176        |
| LoGCAN       | ResNet-50   | CE$^{*}$         | computed           | max   | 0.5335   | 0.5499        |
| FarSeg++     | MiT-B2      | CE               | computed           | max   | 0.4864   | 0.4957        |
| SACANet      | HRNet-W32   | CE$^{*}$         | computed           | max   | 0.5281   | 0.5525        |
| DOCNet       | HRNet-W32   | CE$^{*}$         | computed           | max   | 0.5145   | 0.5317        |
| PPMambaSeg   | SWSL        | CE+Dice          | computed           | max   | 0.5402   | 0.5615        |
| PPMambaSeg   | SWSL        | Focal+Dice       | computed           | max   | 0.5624   | 0.5828        |
| PPMambaSeg   | SWSL        | WCE+Dice         | computed           | max   | 0.6220   | 0.6408        |
| RS3Mamba     | VMamba-Tiny | CE+Dice          | computed           | max   | 0.4047   | 0.4235        |
| RS3Mamba     | VMamba-Tiny | Focal+Dice       | computed           | max   | 0.4554   | 0.4709        |
| RS3Mamba     | VMamba-Tiny | WCE+Dice         | computed           | max   | 0.4716   | 0.5263        |
| PyramidMamba | Swin-Base   | CE+Dice          | computed           | max   | 0.6050   | 0.6301        |
| PyramidMamba | Swin-Base   | Focal+Dice       | computed           | max   | 0.5994   | 0.6365        |
| PyramidMamba | Swin-Base   | WCE+Dice         | computed           | max   | 0.6749   | 0.7024        |
| LoGCAN++     | RepViT-M2.3 | CE$^{*}$         | computed           | max   | 0.4190   | 0.4524        |
| MF-Mamba     | HRNet-W18   | CE+Dice          | computed           | max   | 0.4807   | 0.5137        |
| MCPNet | ResNet-50 | CE+Dice | computed | max | 0.4649 | 0.5063 |
| MCPNet | ResNet-50 | Focal+Dice | computed | max | 0.4849 | 0.5267 |
| MCPNet | ResNet-50 | WCE+Dice | computed | max | 0.5141 | 0.5545 |

### Vision Foundation Model Related Methods

| Model    | Backbone                   | Loss        | Status   | Score            | Area mAP | Max/Proxy mAP |
| -------- | -------------------------- | ----------- | -------- | ---------------- | -------- | ------------- |
| HQ-SAM   | ViT-B + HQ Decoder         | CE+Dice     | computed | max              | 0.4749   | 0.4697        |
| HQ-SAM   | ViT-B + HQ Decoder         | Focal+Dice  | computed | max              | 0.4821   | 0.4710        |
| HQ-SAM   | ViT-B + HQ Decoder         | WCE+Dice    | computed | max              | 0.4625   | 0.4693        |
| SAM_RS   | ABCNet + SAM Priors        | Seg+Bdy+Obj | computed | max              | 0.4666   | 0.4938        |
| SAM_RS   | CMTFNet + SAM Priors       | Seg+Bdy+Obj | computed | max              | 0.4615   | 0.4867        |
| SAM_RS   | FTUNetFormer + SAM Priors  | Seg+Bdy+Obj | computed | max              | 0.4528   | 0.4757        |
| SAM_RS   | UNetFormer + SAM Priors    | Seg+Bdy+Obj | computed | max              | 0.4944   | 0.5091        |
| SAM2.1   | Hiera-B+ (Frozen, MSFPN)   | CE+Dice     | computed | max              | 0.4621   | 0.4668        |
| SAM2.1   | Hiera-B+ (Frozen, MSFPN)   | Focal+Dice  | computed | max              | 0.4541   | 0.4648        |
| SAM2.1   | Hiera-B+ (Frozen, MSFPN)   | WCE+Dice    | computed | max              | 0.4107   | 0.4665        |
| SAM2.1   | Hiera-B+ (Full FT, MSFPN)  | CE+Dice     | computed | max              | 0.4595   | 0.4929        |
| SAM2.1   | Hiera-B+ (Full FT, MSFPN)  | Focal+Dice  | computed | max              | 0.4640   | 0.5124        |
| SAM2.1   | Hiera-B+ (Full FT, MSFPN)  | WCE+Dice    | computed | max              | 0.4850   | 0.5113        |
| RSAM-Seg | SAM-ViT-B (Frozen Encoder) | CE+Dice     | computed | max              | 0.5046   | 0.5349        |
| RSAM-Seg | SAM-ViT-B (Frozen Encoder) | Focal+Dice  | computed | max              | 0.5454   | 0.5705        |
| RSAM-Seg | SAM-ViT-B (Frozen Encoder) | WCE+Dice    | computed | max              | 0.5869   | 0.6206        |
| SESSRS | A2-FPN (CE+Dice) | Postprocess | computed | sessrs_proxy_max | 0.5240 | 0.5592 |
| SESSRS | A2-FPN (Focal+Dice) | Postprocess | computed | sessrs_proxy_max | 0.5218 | 0.5414 |
| SESSRS | A2-FPN (WCE+Dice) | Postprocess | computed | sessrs_proxy_max | 0.5983 | 0.6176 |
| SESSRS | ABCNet (CE+Dice)$^{*}$ | Postprocess | computed | sessrs_proxy_max | 0.4916 | 0.5150 |
| SESSRS | BANet (CE+Dice) | Postprocess | computed | sessrs_proxy_max | 0.4664 | 0.4863 |
| SESSRS | MANet (CE+Dice) | Postprocess | computed | sessrs_proxy_max | 0.5695 | 0.5556 |
| SESSRS | MANet (Focal+Dice) | Postprocess | computed | sessrs_proxy_max | 0.6645 | 0.6170 |
| SESSRS | MANet (WCE+Dice) | Postprocess | computed | sessrs_proxy_max | 0.6557 | 0.6817 |
| SESSRS | UNetFormer (CE+Dice) | Postprocess | computed | sessrs_proxy_max | 0.5646 | 0.5913 |
| SESSRS | UNetFormer (Focal+Dice) | Postprocess | computed | sessrs_proxy_max | 0.5612 | 0.5807 |
| SESSRS | UNetFormer (WCE+Dice) | Postprocess | computed | sessrs_proxy_max | 0.5828 | 0.6120 |

## Per-Class AP

Per-class AP is reported for the ten classes present in the test split, using the same class abbreviations as the paper tables.

### General Segmentation Models

| Model        | Backbone            | Loss             | Score | mAP    | BU     | MR     | PF     | WB     | AC     | GM     | T1R    | T2R    | BG     | SL     |
| ------------ | ------------------- | ---------------- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| DeepLabV3+   | ConvNeXt-Tiny       | CE+Dice          | max   | 0.6230 | 0.5605 | 0.4701 | 0.8847 | 0.9299 | 0.4738 | 0.7446 | 0.7435 | 0.5157 | 0.8092 | 0.0980 |
| DeepLabV3+   | ConvNeXt-Tiny       | Focal+Dice       | max   | 0.6179 | 0.6140 | 0.4814 | 0.8796 | 0.9305 | 0.4805 | 0.6994 | 0.7469 | 0.5092 | 0.8130 | 0.0247 |
| DeepLabV3+   | ConvNeXt-Tiny       | WCE+Dice         | max   | 0.6107 | 0.6399 | 0.3737 | 0.8786 | 0.9266 | 0.3730 | 0.7248 | 0.7488 | 0.4562 | 0.7829 | 0.2023 |
| DeepLabV3+   | ResNet-50           | CE+Dice          | max   | 0.5776 | 0.7742 | 0.0189 | 0.8792 | 0.9309 | 0.4392 | 0.7893 | 0.7381 | 0.4932 | 0.7031 | 0.0100 |
| DeepLabV3+   | ResNet-50           | Focal+Dice       | max   | 0.6120 | 0.6631 | 0.3782 | 0.8915 | 0.9342 | 0.4640 | 0.7316 | 0.7507 | 0.5020 | 0.7952 | 0.0100 |
| DeepLabV3+   | ResNet-50           | WCE+Dice         | max   | 0.5664 | 0.7206 | 0.3840 | 0.8781 | 0.9187 | 0.1457 | 0.7374 | 0.7482 | 0.4850 | 0.6364 | 0.0100 |
| UPerNet      | Swin-Tiny           | CE+Dice          | max   | 0.5252 | 0.4622 | 0.1962 | 0.8641 | 0.9184 | 0.3475 | 0.5842 | 0.7145 | 0.4247 | 0.7305 | 0.0100 |
| OCRNet       | HRNet-W48           | CE+Dice          | max   | 0.4867 | 0.4815 | 0.0189 | 0.8225 | 0.8599 | 0.3158 | 0.6986 | 0.7299 | 0.4226 | 0.5076 | 0.0100 |
| BiSeNetv2 | Custom Bilateral | CE+Dice | max | 0.5377 | 0.4594 | 0.0189 | 0.8778 | 0.9257 | 0.4250 | 0.7560 | 0.7736 | 0.4815 | 0.6496 | 0.0100 |
| BiSeNetv2 | Custom Bilateral | Focal+Dice | max | 0.5263 | 0.3515 | 0.0189 | 0.9016 | 0.9179 | 0.4481 | 0.6983 | 0.7525 | 0.5067 | 0.6578 | 0.0100 |
| BiSeNetv2 | Custom Bilateral | WCE+Dice | max | 0.5369 | 0.3610 | 0.0189 | 0.8482 | 0.9442 | 0.4472 | 0.7794 | 0.7533 | 0.4421 | 0.7650 | 0.0100 |
| SegFormer    | MiT-B2              | CE+Dice          | max   | 0.5382 | 0.5163 | 0.3023 | 0.8575 | 0.8678 | 0.2783 | 0.7244 | 0.7231 | 0.4819 | 0.5599 | 0.0701 |
| SegFormer    | MiT-B2              | Focal+Dice       | max   | 0.5857 | 0.5922 | 0.4603 | 0.8903 | 0.8403 | 0.5325 | 0.7161 | 0.7384 | 0.4723 | 0.5539 | 0.0605 |
| SegFormer    | MiT-B2              | WCE+Dice         | max   | 0.6576 | 0.8137 | 0.5563 | 0.8925 | 0.9333 | 0.5681 | 0.8198 | 0.7717 | 0.4194 | 0.7908 | 0.0100 |
| STDC2 | STDCNet | CE+Dice | max | 0.5016 | 0.2907 | 0.0189 | 0.8943 | 0.9225 | 0.3793 | 0.5378 | 0.6640 | 0.4966 | 0.8024 | 0.0100 |
| STDC2 | STDCNet | Focal+Dice | max | 0.5010 | 0.2553 | 0.0189 | 0.8924 | 0.9105 | 0.3562 | 0.6966 | 0.7334 | 0.4529 | 0.6834 | 0.0100 |
| STDC2 | STDCNet | WCE+Dice | max | 0.5756 | 0.5716 | 0.2593 | 0.8718 | 0.9157 | 0.4984 | 0.6962 | 0.7486 | 0.4659 | 0.7181 | 0.0100 |
| Mask2Former | ResNet-50 | Set CE+Mask+Dice | query-score proxy | 0.5297 | 0.4482 | 0.3164 | 0.7934 | 0.7863 | 0.3046 | 0.6250 | 0.6906 | 0.5706 | 0.7516 | 0.0100 |
| SegNeXt      | MSCAN-Tiny          | CE+Dice          | max   | 0.4756 | 0.4830 | 0.1331 | 0.8519 | 0.8595 | 0.1045 | 0.5783 | 0.7317 | 0.4265 | 0.5776 | 0.0100 |
| DDRNet | DDRNet-23-slim | Focal+Dice | max | 0.5081 | 0.4808 | 0.0189 | 0.8657 | 0.8715 | 0.3883 | 0.6356 | 0.7235 | 0.4485 | 0.6381 | 0.0100 |
| Afformer     | AFFormer-Base       | CE+Dice          | max   | 0.5297 | 0.5744 | 0.2345 | 0.8592 | 0.8407 | 0.3401 | 0.6210 | 0.7575 | 0.4570 | 0.5805 | 0.0319 |
| EfficientViT | EfficientViT-B2     | CE+Dice          | max   | 0.6002 | 0.6018 | 0.2333 | 0.8910 | 0.9257 | 0.5477 | 0.7617 | 0.7343 | 0.4750 | 0.8210 | 0.0100 |
| EfficientViT | EfficientViT-B2     | Focal+Dice       | max   | 0.6100 | 0.5371 | 0.4208 | 0.8769 | 0.9121 | 0.4961 | 0.7526 | 0.7441 | 0.4659 | 0.7921 | 0.1021 |
| EfficientViT | EfficientViT-B2     | WCE+Dice         | max   | 0.6228 | 0.6490 | 0.3473 | 0.8843 | 0.9280 | 0.4724 | 0.7022 | 0.7340 | 0.5165 | 0.8111 | 0.1827 |
| SeaFormer    | SeaFormer-Base      | CE+Dice          | max   | 0.5171 | 0.5452 | 0.1100 | 0.8709 | 0.8387 | 0.4316 | 0.6091 | 0.7631 | 0.4593 | 0.5326 | 0.0100 |
| PIDNet | PIDNet-M | CE+Dice | max | 0.5254 | 0.4811 | 0.0189 | 0.8766 | 0.9101 | 0.4882 | 0.5863 | 0.7391 | 0.4270 | 0.7172 | 0.0100 |
| PIDNet | PIDNet-M | Focal+Dice | max | 0.4298 | 0.1607 | 0.0189 | 0.8815 | 0.8879 | 0.0578 | 0.3648 | 0.7354 | 0.4331 | 0.7475 | 0.0100 |
| PIDNet | PIDNet-M | WCE+Dice | max | 0.5254 | 0.3210 | 0.3226 | 0.8719 | 0.9172 | 0.3865 | 0.5997 | 0.6427 | 0.4775 | 0.7047 | 0.0100 |
| CGRSeg       | EfficientFormerV2-B | CE+Dice          | max   | 0.4990 | 0.5869 | 0.2319 | 0.8501 | 0.8265 | 0.2199 | 0.5555 | 0.7172 | 0.4265 | 0.5447 | 0.0307 |
| PEM | ResNet-50 | Set CE+Mask+Dice | query-score proxy | 0.5069 | 0.3560 | 0.1290 | 0.8110 | 0.7546 | 0.3815 | 0.5961 | 0.6932 | 0.5735 | 0.7643 | 0.0100 |
| VMamba | VMamba-Tiny | CE+Dice | max | 0.4648 | 0.3841 | 0.0189 | 0.8790 | 0.8191 | 0.1858 | 0.6347 | 0.6951 | 0.4674 | 0.5540 | 0.0100 |

| MP-Former | Mask2Former Swin-L | Focal+Dice | query-score proxy | 0.3122 | 0.0150 | 0.0189 | 0.8288 | 0.6130 | 0.0452 | 0.2435 | 0.5106 | 0.4081 | 0.4289 | 0.0100 |
### Remote-Sensing-Specific Methods

| Model        | Backbone    | Loss             | Score | mAP    | BU     | MR     | PF     | WB     | AC     | GM     | T1R    | T2R    | BG     | SL     |
| ------------ | ----------- | ---------------- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| FarSeg       | ResNet-50   | CE               | max   | 0.5576 | 0.5435 | 0.0189 | 0.8746 | 0.9459 | 0.3968 | 0.7467 | 0.7406 | 0.4529 | 0.8462 | 0.0100 |
| FarSeg       | ResNet-50   | CE+Dice          | max   | 0.5752 | 0.5892 | 0.0189 | 0.8490 | 0.9422 | 0.5241 | 0.7795 | 0.7598 | 0.4413 | 0.8379 | 0.0100 |
| FarSeg       | ResNet-50   | Focal+Dice       | max   | 0.5773 | 0.7438 | 0.0189 | 0.9053 | 0.9516 | 0.2878 | 0.7579 | 0.7630 | 0.4816 | 0.8528 | 0.0100 |
| FarSeg       | ResNet-50   | WCE+Dice         | max   | 0.6159 | 0.6762 | 0.3389 | 0.8606 | 0.9338 | 0.5183 | 0.7662 | 0.6375 | 0.4348 | 0.6965 | 0.2962 |
| BANet        | ResT-Lite   | CE+Dice          | max   | 0.4863 | 0.5043 | 0.0185 | 0.8800 | 0.8655 | 0.2897 | 0.5945 | 0.6983 | 0.4473 | 0.5548 | 0.0100 |
| ABCNet       | ResNet-18   | CE+Dice$^{*}$    | max   | 0.5150 | 0.5336 | 0.0189 | 0.8696 | 0.9077 | 0.2597 | 0.7413 | 0.6850 | 0.4197 | 0.7041 | 0.0100 |
| MANet        | ResNet-50   | CE+Dice          | max   | 0.5556 | 0.7770 | 0.0181 | 0.8558 | 0.9490 | 0.5034 | 0.5916 | 0.7336 | 0.4234 | 0.6941 | 0.0100 |
| MANet        | ResNet-50   | Focal+Dice       | max   | 0.6170 | 0.7816 | 0.2292 | 0.8944 | 0.9397 | 0.5425 | 0.7294 | 0.7422 | 0.4348 | 0.6382 | 0.2377 |
| MANet        | ResNet-50   | WCE+Dice         | max   | 0.6817 | 0.8797 | 0.7104 | 0.8894 | 0.9461 | 0.5387 | 0.8221 | 0.7289 | 0.4043 | 0.6655 | 0.2316 |
| UNetFormer   | ResNet-18   | CE+Dice$^{*}$    | max   | 0.5913 | 0.7615 | 0.0189 | 0.9018 | 0.9415 | 0.4988 | 0.7970 | 0.7621 | 0.4917 | 0.7298 | 0.0100 |
| UNetFormer   | ResNet-18   | Focal+Dice$^{*}$ | max   | 0.5807 | 0.7688 | 0.0189 | 0.8928 | 0.9451 | 0.4542 | 0.7449 | 0.7203 | 0.5087 | 0.7437 | 0.0100 |
| UNetFormer   | ResNet-18   | WCE+Dice$^{*}$   | max   | 0.6119 | 0.8290 | 0.4190 | 0.8935 | 0.9273 | 0.4719 | 0.7618 | 0.7550 | 0.4596 | 0.5921 | 0.0100 |
| DC-Swin      | Swin-Small  | CE+Dice          | max   | 0.4953 | 0.4915 | 0.0189 | 0.8828 | 0.9204 | 0.2186 | 0.6074 | 0.7343 | 0.4530 | 0.6165 | 0.0100 |
| A2-FPN       | ResNet-18   | CE+Dice          | max   | 0.5592 | 0.6169 | 0.0189 | 0.8789 | 0.9370 | 0.3888 | 0.7417 | 0.6752 | 0.4866 | 0.8384 | 0.0100 |
| A2-FPN       | ResNet-18   | Focal+Dice       | max   | 0.5414 | 0.7121 | 0.0189 | 0.8722 | 0.9017 | 0.3364 | 0.7490 | 0.7505 | 0.4352 | 0.6282 | 0.0100 |
| A2-FPN       | ResNet-18   | WCE+Dice         | max   | 0.6176 | 0.6652 | 0.3397 | 0.8789 | 0.9183 | 0.4605 | 0.7435 | 0.7266 | 0.4551 | 0.6861 | 0.3019 |
| LoGCAN       | ResNet-50   | CE$^{*}$         | max   | 0.5499 | 0.5547 | 0.2676 | 0.9053 | 0.9331 | 0.4860 | 0.4014 | 0.7218 | 0.5253 | 0.6934 | 0.0100 |
| FarSeg++     | MiT-B2      | CE               | max   | 0.4957 | 0.4085 | 0.3274 | 0.8789 | 0.8719 | 0.1357 | 0.5440 | 0.7032 | 0.4743 | 0.6031 | 0.0099 |
| SACANet      | HRNet-W32   | CE$^{*}$         | max   | 0.5525 | 0.5654 | 0.0937 | 0.8755 | 0.8899 | 0.5717 | 0.6314 | 0.7664 | 0.4211 | 0.7002 | 0.0100 |
| DOCNet       | HRNet-W32   | CE$^{*}$         | max   | 0.5317 | 0.5083 | 0.2296 | 0.8815 | 0.9130 | 0.3333 | 0.6102 | 0.7011 | 0.4370 | 0.6935 | 0.0100 |
| PPMambaSeg   | SWSL        | CE+Dice          | max   | 0.5615 | 0.6559 | 0.0189 | 0.8838 | 0.8978 | 0.4776 | 0.8051 | 0.7090 | 0.5149 | 0.6416 | 0.0100 |
| PPMambaSeg   | SWSL        | Focal+Dice       | max   | 0.5828 | 0.7406 | 0.0189 | 0.8977 | 0.9258 | 0.5313 | 0.8103 | 0.7524 | 0.4300 | 0.7112 | 0.0100 |
| PPMambaSeg   | SWSL        | WCE+Dice         | max   | 0.6408 | 0.7717 | 0.3993 | 0.8787 | 0.9093 | 0.4862 | 0.7686 | 0.7121 | 0.4693 | 0.6474 | 0.3652 |
| RS3Mamba     | VMamba-Tiny | CE+Dice          | max   | 0.4235 | 0.0150 | 0.0189 | 0.9011 | 0.9291 | 0.0453 | 0.2438 | 0.7405 | 0.5423 | 0.7889 | 0.0100 |
| RS3Mamba     | VMamba-Tiny | Focal+Dice       | max   | 0.4709 | 0.0150 | 0.0189 | 0.8981 | 0.9312 | 0.3472 | 0.4767 | 0.7502 | 0.5131 | 0.7485 | 0.0100 |
| RS3Mamba     | VMamba-Tiny | WCE+Dice         | max   | 0.5263 | 0.4916 | 0.0189 | 0.8792 | 0.9219 | 0.5172 | 0.5694 | 0.7669 | 0.4904 | 0.5971 | 0.0100 |
| PyramidMamba | Swin-Base   | CE+Dice          | max   | 0.6301 | 0.7195 | 0.4744 | 0.8822 | 0.9089 | 0.4577 | 0.7697 | 0.7673 | 0.4868 | 0.8241 | 0.0100 |
| PyramidMamba | Swin-Base   | Focal+Dice       | max   | 0.6365 | 0.8647 | 0.4124 | 0.8799 | 0.9025 | 0.5748 | 0.7770 | 0.7689 | 0.4686 | 0.7060 | 0.0100 |
| PyramidMamba | Swin-Base   | WCE+Dice         | max   | 0.7024 | 0.8734 | 0.5746 | 0.8683 | 0.9185 | 0.5646 | 0.8256 | 0.7681 | 0.5418 | 0.7480 | 0.3406 |
| LoGCAN++     | RepViT-M2.3 | CE$^{*}$         | max   | 0.4524 | 0.3992 | 0.0829 | 0.8489 | 0.8472 | 0.1764 | 0.4763 | 0.6906 | 0.4567 | 0.5363 | 0.0100 |
| MF-Mamba     | HRNet-W18   | CE+Dice          | max   | 0.5137 | 0.5553 | 0.0189 | 0.8765 | 0.8671 | 0.3626 | 0.6620 | 0.7263 | 0.4751 | 0.5832 | 0.0100 |
| MCPNet | ResNet-50 | CE+Dice | max | 0.5063 | 0.4156 | 0.0186 | 0.8820 | 0.8715 | 0.3612 | 0.6907 | 0.7297 | 0.5011 | 0.5824 | 0.0100 |
| MCPNet | ResNet-50 | Focal+Dice | max | 0.5267 | 0.4573 | 0.0189 | 0.8901 | 0.9069 | 0.4804 | 0.5895 | 0.7350 | 0.5057 | 0.6699 | 0.0136 |
| MCPNet | ResNet-50 | WCE+Dice | max | 0.5545 | 0.3707 | 0.3980 | 0.8808 | 0.8533 | 0.5057 | 0.6440 | 0.7542 | 0.5084 | 0.6202 | 0.0100 |

### Vision Foundation Model Related Methods

| Model    | Backbone                   | Loss        | Score            | mAP    | BU     | MR     | PF     | WB     | AC     | GM     | T1R    | T2R    | BG     | SL     |
| -------- | -------------------------- | ----------- | ---------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| HQ-SAM   | ViT-B + HQ Decoder         | CE+Dice     | max              | 0.4697 | 0.4272 | 0.0977 | 0.8905 | 0.9218 | 0.1728 | 0.5084 | 0.7092 | 0.4227 | 0.5372 | 0.0100 |
| HQ-SAM   | ViT-B + HQ Decoder         | Focal+Dice  | max              | 0.4710 | 0.5042 | 0.1114 | 0.8973 | 0.9182 | 0.1216 | 0.4533 | 0.7032 | 0.4449 | 0.5463 | 0.0100 |
| HQ-SAM   | ViT-B + HQ Decoder         | WCE+Dice    | max              | 0.4693 | 0.4210 | 0.1291 | 0.8922 | 0.9088 | 0.1921 | 0.5121 | 0.7033 | 0.4122 | 0.5130 | 0.0095 |
| SAM_RS   | ABCNet + SAM Priors        | Seg+Bdy+Obj | max              | 0.4938 | 0.4019 | 0.0189 | 0.8645 | 0.9125 | 0.2399 | 0.6537 | 0.6969 | 0.4391 | 0.7011 | 0.0100 |
| SAM_RS   | CMTFNet + SAM Priors       | Seg+Bdy+Obj | max              | 0.4867 | 0.5065 | 0.0189 | 0.8661 | 0.8742 | 0.1383 | 0.6519 | 0.7126 | 0.4589 | 0.6297 | 0.0100 |
| SAM_RS   | FTUNetFormer + SAM Priors  | Seg+Bdy+Obj | max              | 0.4757 | 0.4173 | 0.0189 | 0.8821 | 0.8749 | 0.1570 | 0.6086 | 0.6996 | 0.5340 | 0.5551 | 0.0100 |
| SAM_RS   | UNetFormer + SAM Priors    | Seg+Bdy+Obj | max              | 0.5091 | 0.4374 | 0.0189 | 0.8913 | 0.9069 | 0.2495 | 0.7081 | 0.7474 | 0.4871 | 0.6345 | 0.0100 |
| SAM2.1   | Hiera-B+ (Frozen, MSFPN)   | CE+Dice     | max              | 0.4668 | 0.5237 | 0.1335 | 0.8854 | 0.8844 | 0.1010 | 0.5053 | 0.7059 | 0.4033 | 0.5151 | 0.0100 |
| SAM2.1   | Hiera-B+ (Frozen, MSFPN)   | Focal+Dice  | max              | 0.4648 | 0.5176 | 0.1467 | 0.8940 | 0.8874 | 0.0643 | 0.5239 | 0.7078 | 0.3942 | 0.5023 | 0.0100 |
| SAM2.1   | Hiera-B+ (Frozen, MSFPN)   | WCE+Dice    | max              | 0.4665 | 0.4531 | 0.2150 | 0.8880 | 0.8756 | 0.0649 | 0.5488 | 0.7134 | 0.3982 | 0.4952 | 0.0126 |
| SAM2.1   | Hiera-B+ (Full FT, MSFPN)  | CE+Dice     | max              | 0.4929 | 0.5516 | 0.0189 | 0.8773 | 0.8599 | 0.2326 | 0.5910 | 0.7171 | 0.4537 | 0.6164 | 0.0100 |
| SAM2.1   | Hiera-B+ (Full FT, MSFPN)  | Focal+Dice  | max              | 0.5124 | 0.6102 | 0.0189 | 0.8771 | 0.8750 | 0.2271 | 0.6144 | 0.7275 | 0.4858 | 0.6777 | 0.0100 |
| SAM2.1   | Hiera-B+ (Full FT, MSFPN)  | WCE+Dice    | max              | 0.5113 | 0.3644 | 0.3615 | 0.8940 | 0.8916 | 0.2359 | 0.5707 | 0.7070 | 0.4720 | 0.6060 | 0.0100 |
| RSAM-Seg | SAM-ViT-B (Frozen Encoder) | CE+Dice     | max              | 0.5349 | 0.5917 | 0.0189 | 0.8828 | 0.9084 | 0.3755 | 0.6160 | 0.7402 | 0.4752 | 0.7300 | 0.0100 |
| RSAM-Seg | SAM-ViT-B (Frozen Encoder) | Focal+Dice  | max              | 0.5705 | 0.6099 | 0.2017 | 0.9024 | 0.9202 | 0.3439 | 0.6827 | 0.7598 | 0.5294 | 0.7449 | 0.0100 |
| RSAM-Seg | SAM-ViT-B (Frozen Encoder) | WCE+Dice    | max              | 0.6206 | 0.6679 | 0.4979 | 0.8896 | 0.9172 | 0.4805 | 0.7221 | 0.7459 | 0.4666 | 0.6769 | 0.1417 |
| SESSRS | A2-FPN (CE+Dice) | Postprocess | sessrs_proxy_max | 0.5592 | 0.6169 | 0.0189 | 0.8789 | 0.9370 | 0.3888 | 0.7417 | 0.6752 | 0.4866 | 0.8384 | 0.0100 |
| SESSRS | A2-FPN (Focal+Dice) | Postprocess | sessrs_proxy_max | 0.5414 | 0.7121 | 0.0189 | 0.8722 | 0.9017 | 0.3364 | 0.7490 | 0.7505 | 0.4353 | 0.6282 | 0.0100 |
| SESSRS | A2-FPN (WCE+Dice) | Postprocess | sessrs_proxy_max | 0.6176 | 0.6652 | 0.3398 | 0.8789 | 0.9183 | 0.4605 | 0.7433 | 0.7266 | 0.4550 | 0.6861 | 0.3019 |
| SESSRS | ABCNet (CE+Dice)$^{*}$ | Postprocess | sessrs_proxy_max | 0.5150 | 0.5336 | 0.0189 | 0.8696 | 0.9078 | 0.2597 | 0.7413 | 0.6850 | 0.4196 | 0.7041 | 0.0100 |
| SESSRS | BANet (CE+Dice) | Postprocess | sessrs_proxy_max | 0.4863 | 0.5043 | 0.0185 | 0.8800 | 0.8655 | 0.2898 | 0.5945 | 0.6983 | 0.4473 | 0.5548 | 0.0100 |
| SESSRS | MANet (CE+Dice) | Postprocess | sessrs_proxy_max | 0.5556 | 0.7770 | 0.0181 | 0.8558 | 0.9489 | 0.5034 | 0.5917 | 0.7334 | 0.4235 | 0.6941 | 0.0100 |
| SESSRS | MANet (Focal+Dice) | Postprocess | sessrs_proxy_max | 0.6170 | 0.7816 | 0.2291 | 0.8943 | 0.9397 | 0.5424 | 0.7294 | 0.7421 | 0.4348 | 0.6382 | 0.2379 |
| SESSRS | MANet (WCE+Dice) | Postprocess | sessrs_proxy_max | 0.6817 | 0.8797 | 0.7103 | 0.8894 | 0.9463 | 0.5389 | 0.8220 | 0.7288 | 0.4043 | 0.6655 | 0.2316 |
| SESSRS | UNetFormer (CE+Dice) | Postprocess | sessrs_proxy_max | 0.5913 | 0.7615 | 0.0189 | 0.9018 | 0.9415 | 0.4988 | 0.7970 | 0.7620 | 0.4917 | 0.7298 | 0.0100 |
| SESSRS | UNetFormer (Focal+Dice) | Postprocess | sessrs_proxy_max | 0.5807 | 0.7688 | 0.0189 | 0.8928 | 0.9451 | 0.4542 | 0.7449 | 0.7202 | 0.5087 | 0.7437 | 0.0100 |
| SESSRS | UNetFormer (WCE+Dice) | Postprocess | sessrs_proxy_max | 0.6120 | 0.8290 | 0.4201 | 0.8934 | 0.9273 | 0.4719 | 0.7618 | 0.7550 | 0.4596 | 0.5921 | 0.0100 |

## Rows Without Max/Proxy AP

No rows remain without max/proxy AP.

## SESSRS Proxy Implementation

For regular segmentation models, the confidence score for class `c` in image `i` is computed from the model softmax output. Let `P_i(p,c)` be the softmax probability for class `c` at pixel `p`, and let `argmax_i(p)` be the final predicted class at that pixel. The max-confidence score is:

```text
s_max(i,c) = max_{p: argmax_i(p)=c} P_i(p,c), with score 0 if class c is not predicted.
```

The area-ratio baseline remains:

```text
s_area(i,c) = #{p: argmax_i(p)=c} / #{valid pixels in image i}.
```

For SESSRS, the reported segmentation output is produced after the official `t1/t2` search and post-processing. That final post-processed mask does not have its own calibrated logits. To avoid pretending that post-processing logits exist, SESSRS rows use `sessrs_proxy_max`:

```text
M_i,c = {p: postprocessed SESSRS mask predicts class c}
s_proxy(i,c) = max_{p in M_i,c} P_base_i(p,c), with score 0 if M_i,c is empty.
```

Here `P_base_i(p,c)` is the softmax probability from the underlying base segmentation model before SESSRS post-processing. The hard mask used for area-ratio mAP is still the final post-processed SESSRS mask. Therefore `sessrs_proxy_max` should be read as an approximate ranking confidence tied to the final SESSRS mask, not as a true confidence produced by the post-processing stage itself.

## Additional Confidence Implementations for Standalone/HuggingFace Repos

The following rows were recomputed from the repositories/checkpoints that were not covered by the earlier local max-confidence sweep:

- `Mask2Former` and `PEM` use Detectron2's final semantic score map, which combines query class scores and mask probabilities. The reported `query-score proxy` is the max value of that final class score map over pixels whose final `argmax` prediction is class `c`; if no pixel is predicted as `c`, the score is 0.
- `BiSeNetv2`, `STDC2`, `DDRNet`, `PIDNet`, and `VMamba` use the native dense segmentation output from their corresponding implementation repositories. For each image and class, the score is `max softmax P(class c)` over valid pixels whose final `argmax` prediction is class `c`; if no pixel is predicted as `c`, the score is 0.
- `MP-Former` was implemented with HuggingFace `Mask2FormerForUniversalSegmentation` and `facebook/mask2former-swin-large-cityscapes-semantic`, fine-tuned on ELDOR. Since this is query-based, the closest comparable confidence is the final semantic query-mask score map, computed from query class probabilities and mask probabilities. The reported `query-score proxy` is the max value of that final class score map over pixels whose final `argmax` prediction is class `c`; if no pixel is predicted as `c`, the score is 0.
- These scores are used only for ranking-based AP/mAP. The hard multi-label presence decision remains derived from the final predicted segmentation mask.
