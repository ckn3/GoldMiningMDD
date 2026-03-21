#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--model', choices=['farseg', 'farsegpp'], default='farsegpp')
    p.add_argument('--farsegpp-backbone', default='mit_b2', choices=['mit_b2', 'resnet50'])
    p.add_argument('--data-root', type=Path, default=Path('/deac/csc/yangGrp/cuij/GoldMDD/data-cropped'))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--farseg-repo', type=Path, default=Path('/deac/csc/yangGrp/cuij/third_party/FarSeg'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import train_semseg_farseg as mod

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    train_args = argparse.Namespace(
        model=args.model,
        farseg_repo=args.farseg_repo,
        pretrained=True,
        farsegpp_backbone=args.farsegpp_backbone,
        optimizer=None,
        lr=None,
        weight_decay=None,
        momentum=0.9,
    )

    model = mod.build_model(train_args).to(device)
    ckpt = torch.load(args.run_dir / 'best.pth', map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    val_samples = mod.build_split_samples(args.data_root / 'val')
    test_samples = mod.build_split_samples(args.data_root / 'test')
    val_ds = mod.GoldMDDPatchDataset(val_samples, train=False, aug_preset='none')
    test_ds = mod.GoldMDDPatchDataset(test_samples, train=False, aug_preset='none')
    val_loader = mod.make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    test_loader = mod.make_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)

    val_stats = mod.evaluate(model, val_loader, device, args.amp, phase='val', log_interval=200)
    test_stats = mod.evaluate(model, test_loader, device, args.amp, phase='test', log_interval=200)

    def pack(stats: dict, split: str) -> dict:
        return {
            'miou': stats['miou'],
            'miou_present': stats['miou_present'],
            'macro_f1': stats['macro_f1'],
            'macro_f1_present': stats['macro_f1_present'],
            'oa_fg': stats['oa_fg'],
            'loss': float('nan'),
            'ce': float('nan'),
            'dice': float('nan'),
            'per_class_iou': stats['per_class_iou'],
            'per_class_f1': stats['per_class_f1'],
            'gt_pixels_per_class': stats['gt_pixels_per_class'],
            'source_checkpoint': 'best.pth',
            'source_epoch': ckpt.get('epoch') if isinstance(ckpt, dict) else None,
            'split': split,
        }

    (args.run_dir / 'val_metrics_best.json').write_text(json.dumps(pack(val_stats, 'val'), indent=2), encoding='utf-8')
    (args.run_dir / 'test_metrics.json').write_text(json.dumps(pack(test_stats, 'test'), indent=2), encoding='utf-8')

    # Keep summary.json aligned with unified test metrics.
    summary_path = args.run_dir / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary.update(
            {
                'test_miou': test_stats['miou'],
                'test_miou_present': test_stats['miou_present'],
                'test_macro_f1': test_stats['macro_f1'],
                'test_macro_f1_present': test_stats['macro_f1_present'],
                'test_oa_fg': test_stats['oa_fg'],
                'test_per_class_iou': test_stats['per_class_iou'],
                'test_per_class_f1': test_stats['per_class_f1'],
                'test_gt_pixels_per_class': test_stats['gt_pixels_per_class'],
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps({'val_miou_present': val_stats['miou_present'], 'test_miou_present': test_stats['miou_present']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
