"""
DualCoOp trainer for GoldMDD MLC.
Uses CLIP RN-101 + dual context optimization (positive/negative prompts).
Paper: sunxm2357/DualCoOp, NeurIPS2022
"""
import sys, os, argparse, time
from pathlib import Path

MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
REPO     = MLC_ROOT / 'repos/DualCoOp'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import logging

from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from asl_loss import AsymmetricLoss
from train_mlc_base import build_optimizer, build_scheduler, load_protocol

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Building", "Mining raft", "Primary Forest", "Heavy machinery",
    "Water bodies", "Agricultural crop", "Compact mounds", "Gravel mounds",
    "Grass", "Type1 regen", "Type2 regen", "Bare ground", "Sluice", "Vehicles"
]


def build_dualcoop_cfg():
    """Build nested cfg object for DualCoOp."""
    class Namespace:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, Namespace(**v))
                else:
                    setattr(self, k, v)

    cfg = Namespace(
        MODEL=Namespace(
            BACKBONE=Namespace(NAME='RN101')
        ),
        TRAINER=Namespace(
            COOP_MLC=Namespace(
                N_CTX_POS=16,
                N_CTX_NEG=16,
                POSITIVE_PROMPT_INIT='',
                NEGATIVE_PROMPT_INIT='',
                CSC=False,        # class-specific context
                LS=0.0,           # label smoothing
            ),
            FINETUNE_BACKBONE=False,
            FINETUNE_ATTN=False,
        ),
        USE_CUDA=True,
        DATASET=Namespace(NAME='goldmdd'),
        INPUT=Namespace(SIZE=(224,224)),
        DATALOADER=Namespace(
            TRAIN_X=Namespace(BATCH_SIZE=8),
        ),
        OPTIM=Namespace(
            LR_SCHEDULER='cosine',
            MAX_EPOCH=80,
        ),
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
    cfg_proto = load_protocol()

    model_name = 'dualcoop'
    out_dir    = Path(cfg_proto['output']['runs_dir']) / model_name
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(out_dir / 'train.log'))

    # Build DualCoOp model
    from models.model_builder import build_model
    cfg = build_dualcoop_cfg()
    model_args = argparse.Namespace(prefix='', evaluate=False)
    model, arch_name = build_model(cfg, model_args, CLASS_NAMES)
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"DualCoOp (CLIP RN-101): {total:.2f}M total, {trainable:.2f}M trainable")

    data_root  = cfg_proto['dataset']['data_root']
    batch_size = cfg_proto['training']['batch_size']
    train_loader, _ = build_dataloader(data_root, 'train', batch_size, 4)
    val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, 4)

    # DualCoOp uses ASL loss
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    optimizer = build_optimizer(model, cfg_proto)
    scheduler = build_scheduler(optimizer, cfg_proto, len(train_loader))
    scaler    = GradScaler()

    epochs   = cfg_proto['training']['epochs']
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
                output = model(imgs)
                # DualCoOp returns [B, 2, C] — take positive logits
                if output.dim() == 3:
                    logits = output[:, 0, :]  # positive channel
                else:
                    logits = output
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()

        train_loss  = total_loss / len(train_loader)

        # Evaluate — DualCoOp returns [B,2,C], extract positive logits
        def dualcoop_fwd(m, x):
            out = m(x)
            return out[:, 0, :] if out.dim() == 3 else out

        val_metrics = evaluate(model, val_loader, device,
                               forward_fn=dualcoop_fwd)
        elapsed = time.time() - t0
        logger.info(f"[dualcoop] Epoch {epoch:03d}/{epochs} | "
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
                            forward_fn=dualcoop_fwd)
    save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                 model_name, 'test')
    logger.info(f"Test mAP={test_metrics['map']:.4f}")


if __name__ == '__main__':
    main()
