from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .base_dataset import BaseDataset


class GoldMDD(BaseDataset):
    """GoldMDD cropped patches dataset.

    Expected layout:
      data_root/
        train/{image/*.jpg,label/*.png}
        val/{image/*.jpg,label/*.png}
        test/{image/*.jpg,label/*.png}

    Label convention in source PNG:
      0   -> background/ignore
      1-14 -> semantic classes

    Model target convention:
      0-13 -> semantic classes
      14   -> ignore_index
    """

    def __init__(
        self,
        data_root: str = "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
        mode: str = "train",
        transform=None,
        img_dir: str = "image",
        mask_dir: str = "label",
        img_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        ignore_index: int = 14,
        **kwargs,
    ):
        super(GoldMDD, self).__init__(transform)
        self.mode = mode
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.ignore_index = ignore_index

        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode={mode}")
        self.data_root = str(Path(data_root) / mode)
        self.file_paths = self.get_path(self.data_root, img_dir, mask_dir)

        self.num_classes = 14

    def load_img_and_mask(self, index):
        img_id = self.file_paths[index]
        img_name = os.path.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = os.path.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        mask_np = np.array(mask, dtype=np.uint8, copy=True)
        # Map: 0->ignore, 1..14 -> 0..13
        mask_np[mask_np == 0] = self.ignore_index + 1
        mask_np = mask_np - 1
        mask = Image.fromarray(mask_np).convert("L")
        return [img, mask, img_id]
