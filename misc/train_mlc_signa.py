#!/usr/bin/env python3
"""
SIGNA trainer for GoldMDD MLC.
Uses ResNet50 + GCN (torch_geometric) + multi-head attention + BCE loss.
Paper: kyle-one/SIGNA, IJRS2024
"""
import sys, os, argparse, time
from pathlib import Path

MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
REPO     = MLC_ROOT / 'repos/SIGNA'

# Must set global args BEFORE importing SIGNA.py
import argparse as _ap
_signa_parser = _ap.ArgumentParser()
_signa_parser.add_argument("--WORK_DIR", default='/tmp/signa_work/', type=str)
_signa_parser.add_argument("--MODEL", default='resnet50', type=str)
_signa_parser.add_argument("--AT_SIZE", default=14, type=int)
_signa_parser.add_argument("--LAYER", default='4', type=str)
_signa_parser.add_argument("--HEADS_NUM", default=4, type=int)
_signa_parser.add_argument("--num_classes", default=14, type=int)
import sys as _sys
_sys.argv = [_sys.argv[0]]  # clear argv before parse
_args_signa, _ = _signa_parser.parse_known_args()

# Inject into SIGNA module namespace
sys.path.insert(0, str(REPO))
import builtins as _builtins

# Patch: SIGNA uses module-level 'args' — inject before import
import types
_signa_mod = types.ModuleType('__signa_args__')
_signa_mod.args = _args_signa

# Override args globally for SIGNA
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import logging
import torchvision.models as tv

from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from train_mlc_base import build_optimizer, build_scheduler, load_protocol

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def build_signa_model():
    """Build SIGNA GCNResnet with ResNet50 backbone."""
    # Import SIGNA with args already set as global
    import importlib.util
    spec = importlib.util.spec_from_file_location("signa_module", str(REPO / 'SIGNA.py'))
    signa_mod = importlib.util.module_from_spec(spec)
    signa_mod.args = _args_signa  # inject args
    spec.loader.exec_module(signa_mod)

    resnet50 = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
    model = signa_mod.GCNResnet(
        model=resnet50,
        num_classes=NUM_CLASSES,
        in_channel=300,
        t=0.4,
        adj_file=str(MLC_ROOT / 'protocols/goldmdd_adj.pkl'),
        graph_type='GAT',
        res='True',
    )
    return model


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

    model_name = 'signa'
    out_dir    = Path(cfg['output']['runs_dir']) / model_name
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(out_dir / 'train.log'))

    model = build_signa_model().to(device)
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"SIGNA (ResNet50+GCN): {total:.2f}M total, {trainable:.2f}M trainable")

    data_root  = cfg['dataset']['data_root']
    batch_size = cfg['training']['batch_size']
    train_loader, _ = build_dataloader(data_root, 'train', batch_size, 4)
    val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, 4)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler()

    # Load word embeddings for GCN input
    import pickle as _pkl
    with open(str(MLC_ROOT / 'protocols/goldmdd_word_emb.pkl'), 'rb') as _f:
        _word_emb = _pkl.load(_f)
    word_emb_t = torch.from_numpy(_word_emb).float().to(device).unsqueeze(0)  # [1,14,300]

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
                logits = model(imgs, [word_emb_t])
                if isinstance(logits, tuple): logits = logits[0]
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()

        train_loss  = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device,
            forward_fn=lambda m, x: m(x, [word_emb_t])[0] if isinstance(m(x, [word_emb_t]), tuple) else m(x, [word_emb_t]))
        elapsed = time.time() - t0
        logger.info(f"[signa] Epoch {epoch:03d}/{epochs} | "
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
    test_loader, _ = build_dataloader(data_root, 'test', batch_size, 4)
    ck = torch.load(out_dir / 'checkpoints/best.pt', map_location=device)
    model.load_state_dict(ck['model'])
    test_metrics = evaluate(model, test_loader, device,
        forward_fn=lambda m, x: m(x, [word_emb_t])[0] if isinstance(m(x, [word_emb_t]), tuple) else m(x, [word_emb_t]))
    save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                 model_name, 'test')
    logger.info(f"Test mAP={test_metrics['map']:.4f}")


if __name__ == '__main__':
    main()
