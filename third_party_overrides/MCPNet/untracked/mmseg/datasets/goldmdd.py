# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class GoldMDDDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'Building', 'Mining raft', 'Primary Forest', 'Heavy machinery',
            'Water bodies', 'Agricultural crop', 'Compact mounds',
            'Gravel mounds', 'Grass', 'Type1 regen', 'Type2 regen',
            'Bare ground', 'Sluice', 'Vehicles'
        ),
        palette=[
            [54, 144, 214], [255, 188, 121], [38, 115, 0], [138, 22, 84],
            [0, 112, 192], [106, 168, 79], [230, 145, 56], [204, 102, 0],
            [147, 196, 125], [182, 215, 168], [118, 165, 175], [191, 144, 0],
            [111, 168, 220], [153, 153, 153]
        ],
    )

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
