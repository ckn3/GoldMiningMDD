"""
Base MLC trainer for GoldMDD.
Each model imports this and overrides build_model() only.
Protocol settings read from protocols/protocol.yaml.
"""
from __future__ import annotations
import argparse, logging, time, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
sys.path.insert(0, str(ROOT / 'utils' / 'dataset'))
sys.path.insert(0, str(ROOT / 'utils' / 'metrics'))

from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from multilabel_protocol import CANONICAL_CLASSES_14

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def load_protocol() -> dict:
    with open(ROOT / 'protocols/protocol.yaml') as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    epochs       = cfg['training']['epochs']
    warmup_ep    = cfg['training']['warmup_epochs']
    total_steps  = epochs * steps_per_epoch
    warmup_steps = warmup_ep * steps_per_epoch
    min_lr       = cfg['training']['lr'] * 0.01

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / cfg['training']['lr'] + \
               0.5 * (1 - min_lr / cfg['training']['lr']) * \
               (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class BaseMLCTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.protocol   = load_protocol()
        self.cfg        = self.protocol

    def build_model(self) -> nn.Module:
        raise NotImplementedError("Subclass must implement build_model()")

    def run(self, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg    = self.cfg
        out_dir = Path(cfg['output']['runs_dir']) / self.model_name
        (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (out_dir / 'results').mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_path = out_dir / 'train.log'
        csv_path = out_dir / 'train_log.csv'
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(fh)

        # Build model
        model = self.build_model().to(device)
        total     = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model: {self.model_name} | {total:.1f}M total, {trainable:.1f}M trainable")

        # Data
        data_root  = args.data_root if args.data_root else cfg['dataset']['data_root']
        batch_size = cfg['training']['batch_size']
        train_loader, _ = build_dataloader(data_root, 'train', batch_size, num_workers=4)
        val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, num_workers=4)

        # Loss — ASL by default
        from asl_loss import AsymmetricLoss
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)

        optimizer  = build_optimizer(model, cfg)
        scheduler  = build_scheduler(optimizer, cfg, len(train_loader))
        scaler     = torch.cuda.amp.GradScaler()

        epochs   = cfg['training']['epochs']
        best_map = 0.0
        start_ep = 1

        # Resume
        if args.resume:
            ckpt_path = out_dir / 'checkpoints/last.pt'
            if ckpt_path.exists():
                ck = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ck['model'])
                start_ep = ck['epoch'] + 1
                best_map = ck.get('map', 0.0)
                logger.info(f"Resumed from epoch {ck['epoch']}")

        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,val_map,val_cf1,val_macro_f1\n')

        for epoch in range(start_ep, epochs + 1):
            # Train
            model.train()
            total_loss = 0.0
            t0 = time.time()
            for imgs, labels in train_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)

            # Validate
            val_metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            logger.info(
                f"[{self.model_name}] Epoch {epoch:03d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"mAP={val_metrics['map']:.4f} "
                f"CF1={val_metrics['cf1']:.4f} "
                f"macro_F1={val_metrics['macro_f1']:.4f} | "
                f"{elapsed:.1f}s")

            with open(csv_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},"
                        f"{val_metrics['map']:.4f},"
                        f"{val_metrics['cf1']:.4f},"
                        f"{val_metrics['macro_f1']:.4f}\n")

            # Save last
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'map': val_metrics['map']},
                       out_dir / 'checkpoints/last.pt')

            # Save best
            if val_metrics['map'] > best_map:
                best_map = val_metrics['map']
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'map': best_map, 'metrics': val_metrics},
                           out_dir / 'checkpoints/best.pt')
                logger.info(f"  ★ New best mAP={best_map:.4f} → saved best.pt")

        logger.info(f"Training complete. Best mAP={best_map:.4f}")

        # Final test evaluation
        logger.info("Running test evaluation...")
        test_loader, _ = build_dataloader(data_root, 'test', batch_size, num_workers=4)
        ck = torch.load(out_dir / 'checkpoints/best.pt', map_location=device)
        model.load_state_dict(ck['model'])
        test_metrics = evaluate(model, test_loader, device)
        save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                     self.model_name, 'test')
        logger.info(f"Test mAP={test_metrics['map']:.4f}")


def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to GoldMDD data-cropped directory. '
                             'Overrides protocol.yaml dataset.data_root if set.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', default='0')
    return parser
