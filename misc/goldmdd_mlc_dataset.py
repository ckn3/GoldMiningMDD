"""
GoldMDD Multi-Label Classification Dataset
Derives binary image-level labels from segmentation GT masks.
Follows multilabel_protocol.py exactly.
Single source — imported by all MLC model trainers.
"""
from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import yaml

NUM_CLASSES  = 14
IGNORE_INDEX = 255
CLASS_NAMES  = [
    "Building", "Mining raft", "Primary Forest", "Heavy machinery",
    "Water bodies", "Agricultural crop", "Compact mounds", "Gravel mounds",
    "Grass", "Type1 regen", "Type2 regen", "Bare ground", "Sluice", "Vehicles",
]

# GT labels: 0=background(ignore), 1-14=foreground → 0-13 in model space
# For MLC: class c is present if label (c+1) appears in any pixel of the mask


def get_transforms(split: str) -> A.Compose:
    """goldmdd_v2 augmentation — same policy as segmentation."""
    if split == 'train':
        return A.Compose([
            A.RandomScale(scale_limit=0.2, p=1.0),
            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=IGNORE_INDEX),
            A.RandomCrop(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.2),
            A.GaussianBlur(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ], additional_targets={'mask': 'mask'})


def mask_to_label(mask: np.ndarray) -> np.ndarray:
    """
    Convert segmentation mask to binary MLC label vector.
    GT labels 1-14 → classes 0-13.
    Background (0) and ignore (255) are excluded.
    Returns float32 array of shape (14,).
    """
    label = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        if np.any(mask == (c + 1)):  # GT label c+1 → class c
            label[c] = 1.0
    return label


def mask_to_score(mask: np.ndarray) -> np.ndarray:
    """
    Confidence score per class = fraction of valid pixels predicted as class c.
    Used for mAP computation.
    """
    valid = (mask != 0) & (mask != IGNORE_INDEX)
    denom = int(valid.sum())
    score = np.zeros(NUM_CLASSES, dtype=np.float32)
    if denom > 0:
        for c in range(NUM_CLASSES):
            score[c] = np.sum(mask[valid] == (c + 1)) / denom
    return score


class GoldMDDMLC(Dataset):
    """
    GoldMDD Multi-Label Classification Dataset.
    Args:
        data_root: path to data-cropped directory
        split:     'train' | 'val' | 'test'
        transform: optional albumentations transform (overrides default)
    """
    def __init__(self, data_root: str, split: str,
                 transform: A.Compose | None = None):
        super().__init__()
        self.data_root = Path(data_root)
        self.split     = split
        self.transform = transform or get_transforms(split)

        img_dir  = self.data_root / split / 'image'
        mask_dir = self.data_root / split / 'label'

        assert img_dir.exists(),  f"Image dir not found: {img_dir}"
        assert mask_dir.exists(), f"Label dir not found: {mask_dir}"

        self.samples = []
        for img_path in sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg')):
            mask_path = mask_dir / (img_path.stem + '.png')
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        assert len(self.samples) > 0, f"No samples found in {img_dir}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        img  = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        # Apply augmentation
        result = self.transform(image=img, mask=mask)
        img    = result['image']
        mask   = result['mask']

        # Convert image to tensor [C, H, W]
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # Derive binary label vector
        label = mask_to_label(mask)
        label_t = torch.from_numpy(label)

        return img_t, label_t

    def get_class_freq(self) -> np.ndarray:
        """Compute class frequencies for weighted loss."""
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _, mask_path in self.samples:
            mask = np.array(Image.open(mask_path))
            for c in range(NUM_CLASSES):
                if np.any(mask == (c + 1)):
                    counts[c] += 1
        return counts / len(self.samples)


def build_dataloader(data_root: str, split: str,
                     batch_size: int = 8,
                     num_workers: int = 4,
                     transform: A.Compose | None = None):
    """Build DataLoader for GoldMDD MLC."""
    dataset = GoldMDDMLC(data_root, split, transform)
    shuffle = (split == 'train')
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    return loader, dataset
