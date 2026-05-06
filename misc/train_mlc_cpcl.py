"""
CPCL trainer for GoldMDD MLC.
Uses ResNet + GCN (Graph Convolution) with co-occurrence adjacency matrix.
Paper: FT-ZHOU-ZZZ/CPCL, TCSVT2022
"""
import sys, os, argparse, time, pickle
from pathlib import Path

MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
REPO     = MLC_ROOT / 'repos/CPCL/MS-COCO'
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


class CPCLDatasetWrapper(torch.utils.data.Dataset):
    """Wraps GoldMDDMLC to also return word embeddings for GCN."""
    def __init__(self, base_dataset, word_emb):
        self.base = base_dataset
        self.word_emb = torch.from_numpy(word_emb).float()  # [C, 300]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, self.word_emb


def build_cpcl_model():
    from models import get_model
    import torchvision.models as tv

    # Load word embeddings and adj matrix
    with open(str(MLC_ROOT / 'protocols/goldmdd_word_emb.pkl'), 'rb') as f:
        word_emb = pickle.load(f)  # [14, 300]

    # Use ResNet-101 as backbone
    resnet = tv.resnet101(weights=tv.ResNet101_Weights.IMAGENET1K_V1)

    model = get_model(
        num_classes=NUM_CLASSES,
        t=0.4,
        pretrained=True,
        adj_file=str(MLC_ROOT / 'protocols/goldmdd_adj.pkl'),
        in_channel=300,
    )
    # Replace backbone with pretrained ResNet-101
    model.features = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
    )
    return model, word_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
    cfg    = load_protocol()

    model_name = 'cpcl'
    out_dir    = Path(cfg['output']['runs_dir']) / model_name
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)

    logging.getLogger().addHandler(logging.FileHandler(out_dir / 'train.log'))

    model, word_emb = build_cpcl_model()
    model = model.to(device)
    word_emb_t = torch.from_numpy(word_emb).float().to(device)  # [14, 300]

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"CPCL (ResNet-101+GCN): {total:.2f}M total, {trainable:.2f}M trainable")

    data_root  = cfg['dataset']['data_root']
    batch_size = cfg['training']['batch_size']

    # Build datasets with word embeddings
    from goldmdd_mlc import GoldMDDMLC
    train_ds = GoldMDDMLC(data_root, 'train')
    val_ds   = GoldMDDMLC(data_root, 'val')
    test_ds  = GoldMDDMLC(data_root, 'test')

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # CPCL original loss: classification + triplet
    from loss import MyLoss
    criterion = MyLoss(margin=20.0, beta=0.5)

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

    # word_emb input for GCN: [1, C, 300] repeated per batch
    inp = word_emb_t  # [14, 300] — model does inp=inp[0] internally

    for epoch in range(start_ep, epochs + 1):
        model.train()
        total_loss = 0.0; t0 = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                output, decoupling_dist, coupling_dist = model(imgs, [inp])
                loss = criterion(labels, output, decoupling_dist, coupling_dist)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()

        train_loss  = total_loss / len(train_loader)
        # Evaluate with standard forward (image only)
        val_metrics = evaluate(model, val_loader, device,
            forward_fn=lambda m, x: m(x, [inp])[0])
        elapsed = time.time() - t0

        logger.info(f"[cpcl] Epoch {epoch:03d}/{epochs} | "
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
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    ck = torch.load(out_dir / 'checkpoints/best.pt', map_location=device)
    model.load_state_dict(ck['model'])
    test_metrics = evaluate(model, test_loader, device,
        forward_fn=lambda m, x: m(x, [inp])[0])
    save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                 model_name, 'test')
    logger.info(f"Test mAP={test_metrics['map']:.4f}")


if __name__ == '__main__':
    main()
