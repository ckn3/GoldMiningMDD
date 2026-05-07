#!/usr/bin/env python3
"""C-Tran trainer for GoldMDD MLC."""
import sys, os, argparse
from pathlib import Path

MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
REPO     = MLC_ROOT / 'repos/C-Tran'
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import numpy as np
import yaml, logging, time, json
from torch.cuda.amp import GradScaler, autocast

from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from multilabel_protocol import CANONICAL_CLASSES_14
import torch.nn.functional as F
from train_mlc_base import build_optimizer, build_scheduler, load_protocol

from models import CTranModel

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


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

    model_name = 'ctran'
    out_dir    = Path(cfg['output']['runs_dir']) / model_name
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / 'train.log')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

    # Build C-Tran model
    # use_lmt=False: no label masking (we have full labels)
    # layers=3, heads=4 — default settings
    model = CTranModel(
        num_labels=NUM_CLASSES,
        use_lmt=False,
        pos_emb=False,   # disabled: pos_encoding is a Tensor not callable in this version
        layers=3,
        heads=4,
        dropout=0.1,
        int_loss=0,
        no_x_features=False,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"C-Tran: {total:.2f}M total, {trainable:.2f}M trainable")

    data_root  = cfg['dataset']['data_root']
    batch_size = cfg['training']['batch_size']
    train_loader, _ = build_dataloader(data_root, 'train', batch_size, 4)
    val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, 4)

    # C-Tran original loss: BCE
    criterion = lambda logits, labels: F.binary_cross_entropy_with_logits(logits, labels)
    # C-Tran needs lower LR due to large transformer architecture
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,  # lower than protocol 2e-4 to prevent NaN
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler  = build_scheduler(optimizer, cfg, len(train_loader))
    scaler     = GradScaler()

    epochs   = cfg['training']['epochs']
    best_map = 0.0
    start_ep = 1

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
        total_loss = 0.0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # C-Tran forward: needs label_input as extra arg
            label_input = model.label_input.to(device)
            optimizer.zero_grad()
            with autocast():
                # mask_in: all -1 (unknown) — no partial labels
                mask_in = torch.ones(imgs.size(0), NUM_CLASSES).to(device) * -1
                logits = model(imgs, mask_in)[0]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        train_loss  = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device,
            forward_fn=lambda m, x: m(x, torch.ones(x.size(0), NUM_CLASSES).to(x.device) * -1)[0])
        elapsed = time.time() - t0

        logger.info(
            f"[ctran] Epoch {epoch:03d}/{epochs} | "
            f"loss={train_loss:.4f} mAP={val_metrics['map']:.4f} "
            f"CF1={val_metrics['cf1']:.4f} macro_F1={val_metrics['macro_f1']:.4f} | "
            f"{elapsed:.1f}s")

        with open(csv_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_metrics['map']:.4f},"
                    f"{val_metrics['cf1']:.4f},{val_metrics['macro_f1']:.4f}\n")

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'map': val_metrics['map']},
                   out_dir / 'checkpoints/last.pt')

        if val_metrics['map'] > best_map:
            best_map = val_metrics['map']
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'map': best_map, 'metrics': val_metrics},
                       out_dir / 'checkpoints/best.pt')
            logger.info(f"  ★ New best mAP={best_map:.4f}")

    logger.info(f"Training complete. Best mAP={best_map:.4f}")

    # Test evaluation
    test_loader, _ = build_dataloader(data_root, 'test', batch_size, 4)
    ck = torch.load(out_dir / 'checkpoints/best.pt', map_location=device)
    model.load_state_dict(ck['model'])
    test_metrics = evaluate(model, test_loader, device,
        forward_fn=lambda m, x: m(x, torch.ones(x.size(0), NUM_CLASSES).to(x.device) * -1)[0])
    save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                 model_name, 'test')
    logger.info(f"Test mAP={test_metrics['map']:.4f}")





if __name__ == '__main__':
    main()
