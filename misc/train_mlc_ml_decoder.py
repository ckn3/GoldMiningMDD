#!/usr/bin/env python3
"""ML-Decoder trainer for GoldMDD MLC."""
import sys, os, argparse, time
from pathlib import Path

MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
REPO     = MLC_ROOT / 'repos/ML_Decoder'
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import logging, argparse

from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from asl_loss import AsymmetricLoss
from train_mlc_base import build_optimizer, build_scheduler, load_protocol

from src_files.models import create_model
from src_files.ml_decoder.ml_decoder import add_ml_decoder_head

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

    model_name = 'ml_decoder'
    out_dir    = Path(cfg['output']['runs_dir']) / model_name
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)

    logging.getLogger().addHandler(
        logging.FileHandler(out_dir / 'train.log'))

    # Build ResNet-101 backbone + ML-Decoder head
    import torchvision.models as tv
    import torch.nn as nn

    class MLDecoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            r = tv.resnet101(weights=tv.ResNet101_Weights.IMAGENET1K_V1)
            self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                      r.layer1, r.layer2, r.layer3, r.layer4)
            self.num_features = 2048
            self.num_classes  = NUM_CLASSES
            self.global_pool  = nn.Identity()
            self.head = None  # will be replaced by add_ml_decoder_head
        def forward(self, x):
            x = self.body(x)          # [B, 2048, H, W]
            return self.head(x)       # [B, num_classes]

    model = MLDecoderModel()
    model = add_ml_decoder_head(model, num_classes=NUM_CLASSES,
                                 num_of_groups=-1, decoder_embedding=768, zsl=0)
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"ML-Decoder (TResNet-L): {total:.2f}M total, {trainable:.2f}M trainable")

    data_root  = cfg['dataset']['data_root']
    batch_size = cfg['training']['batch_size']
    train_loader, _ = build_dataloader(data_root, 'train', batch_size, 4)
    val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, 4)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
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
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()

        train_loss  = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        logger.info(f"[ml_decoder] Epoch {epoch:03d}/{epochs} | "
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
    test_metrics = evaluate(model, test_loader, device)
    save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                 model_name, 'test')
    logger.info(f"Test mAP={test_metrics['map']:.4f}")

if __name__ == '__main__':
    main()
