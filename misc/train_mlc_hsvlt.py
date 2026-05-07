#!/usr/bin/env python3
"""
HSVLT trainer for GoldMDD MLC.
Uses ConvNeXt backbone + cross-attention transformer + ASL loss.
Paper: NaturalKnight/HSVLT, ACMMM2023
"""
import sys, os, argparse, time
from pathlib import Path

MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
REPO     = MLC_ROOT / 'repos/HSVLT'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import logging
import numpy as np

from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from asl_loss import AsymmetricLoss
from train_mlc_base import build_optimizer, build_scheduler, load_protocol

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def build_hsvlt_model():
    """Build HSVLT with ConvNeXt backbone via timm (no pretrained file needed)."""
    import argparse as ap
    from models import hsvlt as hsvlt_module  # trigger registration
    from models.factory import create_model

    cfg = ap.Namespace(
        embed_type='random',   # torch.eye(num_classes) — no GloVe needed
        embed_path=None,
        num_classes=NUM_CLASSES,
        num_heads=8,
    )

    class HSVLTWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            # Build HSVLT with timm ConvNeXt instead of checkpoint file
            from models.hsvlt import HSVLT
            self.model = HSVLT(
                pretrained=None,       # skip file loading
                cfg=cfg,
                depths=[3, 3, 27, 3],
                dims=[96, 192, 384, 768],
            )
            # Load ConvNeXt weights from timm
            import timm
            convnext = timm.create_model(
                'convnext_small.fb_in22k_ft_in1k_384',
                pretrained=True
            )
            # Copy weights to HSVLT backbone
            self._init_from_timm(convnext)

        def _init_from_timm(self, timm_model):
            """Copy ConvNeXt weights from timm to HSVLT backbone."""
            try:
                hsvlt_sd = self.model.state_dict()
                timm_sd  = timm_model.state_dict()
                # Map timm keys to HSVLT keys
                # timm: stem.* -> HSVLT: downsample_layers.0.*
                # timm: stages.N.blocks.M.* -> HSVLT: stages.N.M.*
                remap = {}
                for k in timm_sd:
                    if k.startswith('stem.'):
                        remap[k] = k.replace('stem.', 'downsample_layers.0.')
                    elif k.startswith('stages.'):
                        parts = k.split('.')
                        if len(parts) > 3 and parts[2] == 'blocks':
                            remap[k] = 'stages.' + parts[1] + '.' + '.'.join(parts[3:])
                        else:
                            remap[k] = k
                    elif k.startswith('downsample_layers.'):
                        remap[k] = k
                new_sd = dict(hsvlt_sd)
                matched = 0
                for tk, hk in remap.items():
                    if hk in hsvlt_sd and timm_sd[tk].shape == hsvlt_sd[hk].shape:
                        new_sd[hk] = timm_sd[tk]
                        matched += 1
                self.model.load_state_dict(new_sd, strict=False)
                logger.info(f"ConvNeXt weights: {matched}/{len(hsvlt_sd)} matched")
            except Exception as e:
                logger.warning(f"Weight init failed: {e}")
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, dict):
                return out['logits']
            if isinstance(out, tuple):
                return out[0]  # (logits, attention_weights) -> logits only
            return out

    return HSVLTWrapper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to GoldMDD data-cropped directory. '
                             'Overrides protocol.yaml if set.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
    cfg    = load_protocol()

    model_name = 'hsvlt'
    out_dir    = Path(cfg['output']['runs_dir']) / model_name
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(out_dir / 'train.log'))

    model = build_hsvlt_model().to(device)
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"HSVLT (ConvNeXt+CrossAttn): {total:.2f}M total, {trainable:.2f}M trainable")

    data_root  = cfg['dataset']['data_root']
    batch_size = cfg['training']['batch_size']
    train_loader, _ = build_dataloader(data_root, 'train', batch_size, 4)
    val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, 4)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    # HSVLT: large model (154M) needs lower LR to prevent NaN
    import copy
    hsvlt_cfg = copy.deepcopy(cfg)
    hsvlt_cfg['training']['lr'] = 1e-5
    optimizer = build_optimizer(model, hsvlt_cfg)
    scheduler = build_scheduler(optimizer, hsvlt_cfg, len(train_loader))
    scaler    = GradScaler()

    epochs = cfg['training']['epochs']
    best_map = 0.0; start_ep = 1

    if args.resume and (out_dir / 'checkpoints/last.pt').exists():
        ck = torch.load(out_dir / 'checkpoints/last.pt', map_location=device)
        model.load_state_dict(ck['model'])
        start_ep = ck['epoch'] + 1
        best_map = ck.get('map', 0.0)
        logger.info(f"Resumed from epoch {ck['epoch']}")

    csv_path = out_dir / 'train_log.csv'
    if not args.resume:
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,val_map,val_cf1,val_macro_f1\n')

    for epoch in range(start_ep, epochs + 1):
        model.train()
        total_loss = 0.0; t0 = time.time()
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()

        train_loss  = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        logger.info(f"[hsvlt] Epoch {epoch:03d}/{epochs} | "
                    f"loss={train_loss:.4f} mAP={val_metrics['map']:.4f} "
                    f"CF1={val_metrics['cf1']:.4f} macro_F1={val_metrics['macro_f1']:.4f} | "
                    f"{elapsed:.1f}s")
        with open(csv_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_metrics['map']:.4f},"
                    f"{val_metrics['cf1']:.4f},{val_metrics['macro_f1']:.4f}\n")

        torch.save({'epoch': epoch, 'model': model.state_dict(), 'map': val_metrics['map']},
                   out_dir / 'checkpoints/last.pt')
        if val_metrics['map'] > best_map:
            best_map = val_metrics['map']
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'map': best_map, 'metrics': val_metrics},
                       out_dir / 'checkpoints/best.pt')
            logger.info(f"  ★ New best mAP={best_map:.4f}")

    logger.info(f"Training complete. Best mAP={best_map:.4f}")
    test_loader, _ = build_dataloader(data_root, 'test', batch_size, 4)
    ck = torch.load(out_dir / 'checkpoints/best.pt', map_location=device)
    model.load_state_dict(ck['model'])
    test_metrics = evaluate(model, test_loader, device)
    save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                 model_name, 'test')
    logger.info(f"Test mAP={test_metrics['map']:.4f}")


if __name__ == '__main__':
    main()
