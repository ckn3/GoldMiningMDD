# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABCMeta
from typing import List, Tuple

from torch import Tensor

from mmseg.utils import ConfigType, SampleList
from .decode_head import BaseDecodeHead


class RefineBaseDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    """Decode head base that also exposes intermediate features.

    Subclasses are expected to return ``(seg_logits, feature_list)`` from
    ``forward``. This matches MCPNet's two-stage decode + refine segmentor.
    """

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, list[Tensor]]:
        raise NotImplementedError

    def forward_train(self, inputs: List[Tensor], batch_data_samples: SampleList,
                      train_cfg: ConfigType | None = None):
        seg_logits, fm_middle = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses, fm_middle

    def forward_test(self, inputs: List[Tensor], *args, **kwargs):
        return self.forward(inputs)
