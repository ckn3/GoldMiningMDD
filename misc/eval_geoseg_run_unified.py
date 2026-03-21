#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, default=Path('/deac/csc/yangGrp/cuij/third_party/GeoSeg'))
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def compute_metrics_from_conf(conf: np.ndarray) -> dict:
    conf = conf.astype(np.float64, copy=False)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom_iou = tp + fp + fn
    iou = np.divide(tp, denom_iou, out=np.full_like(tp, np.nan), where=denom_iou > 0)
    denom_f1 = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom_f1, out=np.full_like(tp, np.nan), where=denom_f1 > 0)
    gt = conf.sum(axis=1)
    present = gt > 0
    return {
        'miou': float(np.nanmean(iou)),
        'miou_present': float(np.nanmean(iou[present])) if np.any(present) else float('nan'),
        'macro_f1': float(np.nanmean(f1)),
        'macro_f1_present': float(np.nanmean(f1[present])) if np.any(present) else float('nan'),
        'oa_fg': float(tp.sum() / gt.sum()) if gt.sum() > 0 else float('nan'),
        'per_class_iou': [float(x) if np.isfinite(x) else float('nan') for x in iou.tolist()],
        'per_class_f1': [float(x) if np.isfinite(x) else float('nan') for x in f1.tolist()],
        'gt_pixels_per_class': [int(x) for x in gt.tolist()],
    }


def eval_split(net: torch.nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> dict:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    net.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch['img'].to(device, non_blocking=True)
            gt = batch['gt_semantic_seg'].cpu().numpy()
            logits = net(img)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            # gt uses ignore index == num_classes in GeoSeg goldmdd dataset
            for g, p in zip(gt, pred):
                valid = (g >= 0) & (g < num_classes)
                if not np.any(valid):
                    continue
                gg = g[valid].astype(np.int64, copy=False)
                pp = p[valid].astype(np.int64, copy=False)
                binc = np.bincount(gg * num_classes + pp, minlength=num_classes * num_classes)
                conf += binc.reshape(num_classes, num_classes)
    return compute_metrics_from_conf(conf)


def _extract_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and 'state_dict' in raw and isinstance(raw['state_dict'], dict):
        sd = raw['state_dict']
    elif isinstance(raw, dict):
        sd = raw
    else:
        raise TypeError('Unsupported checkpoint format')

    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith('module.net.'):
            nk = nk[len('module.net.'):]
        elif nk.startswith('net.'):
            nk = nk[len('net.'):]
        elif nk.startswith('module.'):
            nk = nk[len('module.'):]
        out[nk] = v
    return out


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root))
    from tools.cfg import py2cfg  # type: ignore

    cfg = py2cfg(args.config)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    ckpt = args.checkpoint
    if ckpt is None:
        preferred = args.run_dir / f"{args.run_dir.name}.ckpt"
        if preferred.exists():
            ckpt = preferred
        else:
            ckpt = args.run_dir / 'last.ckpt'
    if not ckpt.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt}')

    raw = torch.load(str(ckpt), map_location='cpu')
    net = cfg.net.to(device)
    sd = _extract_state_dict(raw)
    incompat = net.load_state_dict(sd, strict=False)
    if len(incompat.unexpected_keys) > 0:
        print(f"[warn] unexpected keys: {len(incompat.unexpected_keys)}")
    if len(incompat.missing_keys) > 0:
        print(f"[warn] missing keys: {len(incompat.missing_keys)}")

    val_loader = DataLoader(cfg.val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=False)
    test_loader = DataLoader(cfg.test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=False)

    val_stats = eval_split(net, val_loader, cfg.num_classes, device)
    test_stats = eval_split(net, test_loader, cfg.num_classes, device)

    val_stats.update({
        'loss': float('nan'),
        'ce': float('nan'),
        'dice': float('nan'),
        'source_checkpoint': str(ckpt),
        'config': str(args.config),
        'split': 'val',
    })
    test_stats.update({
        'loss': float('nan'),
        'ce': float('nan'),
        'dice': float('nan'),
        'source_checkpoint': str(ckpt),
        'config': str(args.config),
        'split': 'test',
    })

    (args.run_dir / 'val_metrics_best.json').write_text(json.dumps(val_stats, indent=2), encoding='utf-8')
    (args.run_dir / 'test_metrics.json').write_text(json.dumps(test_stats, indent=2), encoding='utf-8')

    print(json.dumps({'val_miou_present': val_stats['miou_present'], 'test_miou_present': test_stats['miou_present']}, indent=2))


if __name__ == '__main__':
    main()
