import os
import os.path as osp
import random
from typing import Optional

import albumentations as albu
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .transform import Compose, RandomCrop, RandomScale


CLASSES = (
    "Building",
    "Mining raft",
    "Primary Forest",
    "Heavy machinery",
    "Water bodies",
    "Agricultural crop",
    "Compact mounds",
    "Gravel mounds",
    "Grass",
    "Type 1 natural regeneration",
    "Type 2 natural regeneration",
    "Bare ground",
    "Sluice",
    "Vehicles",
)

PALETTE = [
    [138, 106, 61],
    [123, 235, 251],
    [176, 76, 24],
    [238, 146, 198],
    [79, 111, 111],
    [132, 208, 140],
    [35, 243, 227],
    [88, 84, 0],
    [141, 181, 29],
    [194, 22, 58],
    [247, 119, 87],
    [44, 216, 116],
    [97, 57, 145],
    [150, 154, 174],
]

IGNORE_INDEX = len(CLASSES)
PATCH_SIZE = 512


def _encode_mask(mask: np.ndarray) -> np.ndarray:
    """Map GoldMDD labels: 0->ignore, 1..14 -> 0..13, others->ignore."""
    out = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    fg = (mask >= 1) & (mask <= len(CLASSES))
    out[fg] = (mask[fg] - 1).astype(np.uint8)
    return out


def _apply_color_jitter(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-0.20, 0.20))
    img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-0.20, 0.20))
    img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-0.15, 0.15))
    return img


def train_aug(img: Image.Image, mask: Image.Image):
    # Align with GoldMDD aug-v2 used in other baselines.
    crop_aug = Compose([
        RandomScale(scale_list=[0.8, 1.2], mode="range"),
        RandomCrop(size=PATCH_SIZE, ignore_index=0, nopad=False),
    ])
    img, mask = crop_aug(img, mask)

    if random.random() < 0.8:
        img = _apply_color_jitter(img)
    if random.random() < 0.2:
        sigma = random.uniform(0.3, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    img_np = np.array(img, dtype=np.uint8)
    mask_np = np.array(mask, dtype=np.uint8)

    if random.random() < 0.5:
        img_np = np.ascontiguousarray(np.flip(img_np, axis=1))
        mask_np = np.ascontiguousarray(np.flip(mask_np, axis=1))
    if random.random() < 0.5:
        img_np = np.ascontiguousarray(np.flip(img_np, axis=0))
        mask_np = np.ascontiguousarray(np.flip(mask_np, axis=0))

    k = random.randint(0, 3)
    if k:
        img_np = np.ascontiguousarray(np.rot90(img_np, k, axes=(0, 1)))
        mask_np = np.ascontiguousarray(np.rot90(mask_np, k, axes=(0, 1)))

    aug = albu.Normalize()(image=img_np.copy(), mask=mask_np.copy())
    img_np, mask_np = aug["image"], aug["mask"]
    mask_np = _encode_mask(mask_np)
    return img_np, mask_np


def val_aug(img: Image.Image, mask: Image.Image):
    img_np = np.array(img, dtype=np.uint8)
    mask_np = np.array(mask, dtype=np.uint8)
    aug = albu.Normalize()(image=img_np.copy(), mask=mask_np.copy())
    img_np, mask_np = aug["image"], aug["mask"]
    mask_np = _encode_mask(mask_np)
    return img_np, mask_np


class GoldMDDDataset(Dataset):
    def __init__(
        self,
        data_root: str = "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped",
        split: str = "train",
        img_dir: str = "image",
        mask_dir: str = "label",
        img_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        transform=val_aug,
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_ids = self._get_img_ids()
        if max_samples is not None:
            self.img_ids = self.img_ids[: int(max_samples)]

    def _get_img_ids(self):
        split_mask_dir = osp.join(self.data_root, self.split, self.mask_dir)
        split_img_dir = osp.join(self.data_root, self.split, self.img_dir)
        if not osp.isdir(split_mask_dir):
            raise RuntimeError(f"Mask dir not found: {split_mask_dir}")
        if not osp.isdir(split_img_dir):
            raise RuntimeError(f"Image dir not found: {split_img_dir}")
        mask_names = [n for n in os.listdir(split_mask_dir) if n.endswith(self.mask_suffix)]
        img_ids = []
        for n in sorted(mask_names):
            stem = n[: -len(self.mask_suffix)]
            if osp.exists(osp.join(split_img_dir, stem + self.img_suffix)):
                img_ids.append(stem)
        if not img_ids:
            raise RuntimeError(f"No image/label pairs found in {self.split}")
        return img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root, self.split, self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root, self.split, self.mask_dir, img_id + self.mask_suffix)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = np.array(img, dtype=np.uint8)
            mask = _encode_mask(np.array(mask, dtype=np.uint8))
            img = albu.Normalize()(image=img)["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return {"img_id": img_id, "img": img, "gt_semantic_seg": mask}
