"""
Shared MLC evaluator for all GoldMDD MLC models.
Uses multilabel_protocol.py as single source of truth.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
from typing import Callable
import sys

sys.path.insert(0, str(Path(__file__).parent))
from multilabel_protocol import MultilabelAccumulator, CANONICAL_CLASSES_14

NUM_CLASSES  = 14
IGNORE_INDEX = 255


def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             forward_fn: Callable | None = None,
             use_sigmoid: bool = True) -> dict:
    """
    Evaluate MLC model using multilabel_protocol.
    Args:
        model:      MLC model (outputs logits [B, 14])
        loader:     DataLoader returning (images, labels)
        device:     torch device
        forward_fn: optional custom forward fn(model, images) -> logits
    Returns:
        dict with all protocol metrics
    """
    model.eval()
    acc = MultilabelAccumulator(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_np = labels.numpy()  # [B, 14] binary

            if forward_fn:
                logits = forward_fn(model, images)
            else:
                logits = model(images)

            # Convert logits → binary predictions
            # use_sigmoid=True for MLC models (logit outputs)
            # use_sigmoid=False for CLIP models (probability outputs)
            if use_sigmoid:
                probs = torch.sigmoid(logits).cpu().numpy()  # [B, 14]
            else:
                probs = logits.cpu().numpy()                 # [B, 14] already probs
            preds = (probs >= 0.5).astype(np.uint8)          # [B, 14]

            # Build fake segmentation maps for accumulator compatibility
            # Protocol expects per-image class presence, not seg maps
            B = labels_np.shape[0]
            for b in range(B):
                # GT presence from label vector
                gt_present   = labels_np[b].astype(np.uint8)
                pred_present = preds[b]
                score        = probs[b]

                # Directly update accumulator internal state
                acc._gt_presence.append(gt_present)
                acc._pred_presence.append(pred_present)
                acc._scores.append(score.astype(np.float64))
                acc._valid_pixels.append(1)

    return acc.finalize(class_names=CANONICAL_CLASSES_14)


def save_results(metrics: dict, out_path: str,
                 model_name: str, split: str) -> None:
    """Save evaluation results to JSON."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model_name,
        'split': split,
        'results': {split: metrics}
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {out_path}")
