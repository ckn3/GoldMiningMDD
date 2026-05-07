#!/usr/bin/env python3
"""Q2L trainer for GoldMDD MLC.
Paper: Query2Label: A Simple Transformer Way to Multi-Label Classification (NeurIPS2021)
Backbone: ResNet-101 + Transformer decoder, Loss: ASL
"""
import sys, os, time, argparse as _ap
from pathlib import Path

# Path setup: assumes this script lives in misc/ of the GoldMiningMDD repo
# and the MLC third-party repos are cloned under a sibling directory.
# Set GOLDMDD_MLC_REPO env var to point to the multi-label-classification root,
# or pass --mlc-root argument.
MLC_ROOT = Path(os.environ.get(
    'GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'
))
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))
sys.path.insert(0, str(MLC_ROOT / 'repos/Q2L'))
sys.path.insert(0, str(MLC_ROOT / 'repos/Q2L/lib'))

import torch
import torch.nn as nn
import logging
from train_mlc_base import BaseMLCTrainer, build_optimizer, build_scheduler, get_base_args
from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from asl_loss import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')


class Q2LTrainer(BaseMLCTrainer):
    def build_model(self):
        import argparse
        from models.query2label import build_q2l
        args = argparse.Namespace(
            backbone='resnet101',
            hidden_dim=2048,
            nheads=4,
            enc_layers=1,
            dec_layers=2,
            dim_feedforward=8192,
            dropout=0.1,
            pre_norm=False,
            position_embedding='sine',
            num_class=NUM_CLASSES,
            img_size=512,
            keep_input_proj=False,
            keep_other_self_attn_dec=False,
            keep_first_self_attn_dec=False,
            pretrained=True,
            train_backbone=True,
            return_interm_layers=False,
            dilation=False,
        )
        model = build_q2l(args)
        logger.info("Q2L built ✅")
        return model

    def run(self, args):
        device = torch.device('cuda')
        cfg    = self.cfg
        out_dir = Path(cfg['output']['runs_dir']) / self.model_name
        (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (out_dir / 'results').mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(out_dir / 'train.log')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)

        model = self.build_model().to(device)
        data_root  = cfg['dataset']['data_root']
        batch_size = cfg['training']['batch_size']
        train_loader, _ = build_dataloader(data_root, 'train', batch_size, 4)
        val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, 4)

        criterion = torch.nn.MultiLabelSoftMarginLoss()  # BCE — more stable than ASL for Q2L
        # Override lr for transformer model — lower to prevent NaN
        import copy
        cfg_override = copy.deepcopy(cfg)
        cfg_override['training']['lr'] = 2e-5
        optimizer = build_optimizer(model, cfg_override)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))
        scaler    = GradScaler()

        epochs = cfg['training']['epochs']
        best_map = 0.0
        start_epoch = 1
        if args.resume:
            last = out_dir / 'checkpoints/last.pt'
            if last.exists():
                ck = torch.load(last, map_location=device)
                model.load_state_dict(ck['model'])
                start_epoch = ck['epoch'] + 1
                best_map = ck.get('map', 0.0)
                logger.info(f"Resumed from epoch {ck['epoch']} mAP={best_map:.4f}")

        with open(out_dir / 'train_log.csv', 'w') as f:
            f.write('epoch,train_loss,val_map,val_cf1,val_macro_f1\n')

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_loss = 0.0
            t0 = time.time()
            for imgs, labels in train_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step()
                total_loss += loss.item()

            val_metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            logger.info(f"Epoch {epoch:03d}/{epochs} | loss={total_loss/len(train_loader):.4f} | "
                        f"mAP={val_metrics['map']:.4f} CF1={val_metrics['cf1']:.4f} | {elapsed:.1f}s")

            with open(out_dir / 'train_log.csv', 'a') as f:
                f.write(f"{epoch},{total_loss/len(train_loader):.4f},"
                        f"{val_metrics['map']:.4f},{val_metrics['cf1']:.4f},"
                        f"{val_metrics['macro_f1']:.4f}\n")

            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'map': val_metrics['map']}, out_dir / 'checkpoints/last.pt')

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
                     self.model_name, 'test')
        logger.info(f"Test mAP={test_metrics['map']:.4f}")


def main():
    args = get_base_args().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    Q2LTrainer('q2l').run(args)

if __name__ == '__main__':
    main()
