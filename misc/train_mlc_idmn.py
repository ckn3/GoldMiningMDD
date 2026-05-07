#!/usr/bin/env python3
"""IDMN trainer for GoldMDD MLC.
Paper: IDMN — Instance-Dependent Multi-Label Noise (JSTARS2024)
Backbone: ResNet-50, Loss: SAT (Self-Adaptive Training)
Special: Loss requires sample index; dataloader returns (image, label, index)
"""
import sys, os, time, argparse
from pathlib import Path
MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))
sys.path.insert(0, str(MLC_ROOT / 'repos/IDMN'))

import torch
import torch.nn as nn
import numpy as np
import logging
from train_mlc_base import load_protocol, build_scheduler, get_base_args
from evaluate_mlc import evaluate, save_results
from goldmdd_mlc import NUM_CLASSES
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')


class GoldMDDWithIndex(torch.utils.data.Dataset):
    """Wraps GoldMDD dataset to return (image, label, index) for IDMN loss."""
    def __init__(self, data_root, split, transform=None):
        import albumentations as A
        import cv2
        from pathlib import Path as P
        import numpy as np
        from PIL import Image

        self.img_dir  = P(data_root) / split / 'image'
        self.mask_dir = P(data_root) / split / 'label'
        self.samples  = []
        for p in sorted(self.img_dir.glob('*.jpg')) + \
                 sorted(self.img_dir.glob('*.png')):
            mp = self.mask_dir / (p.stem + '.png')
            if mp.exists():
                self.samples.append((p, mp))

        # Standard MLC augmentation
        if split == 'train':
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=512, width=512),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        import cv2, numpy as np
        img_path, mask_path = self.samples[idx]
        img  = cv2.imread(str(img_path))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.array(__import__('PIL').Image.open(mask_path))

        aug  = self.transform(image=img)
        img  = torch.from_numpy(aug['image'].transpose(2,0,1)).float()

        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for c in range(NUM_CLASSES):
            if np.any(mask == (c + 1)):
                label[c] = 1.0

        return img, label, idx  # return index for SAT loss


class NoIndexDataset(torch.utils.data.Dataset):
    """Wraps GoldMDDWithIndex to return (image, label) only — for standard evaluate()."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label, _ = self.dataset[idx]
        return img, label


def main():
    args = get_base_args().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')

    cfg     = load_protocol()
    model_name = 'idmn'
    out_dir = Path(cfg['output']['runs_dir']) / model_name
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / 'train.log')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

    data_root  = cfg['dataset']['data_root']
    batch_size = cfg['training']['batch_size']
    epochs     = cfg['training']['epochs']

    # Datasets with index
    train_ds = GoldMDDWithIndex(data_root, 'train')
    val_ds   = GoldMDDWithIndex(data_root, 'val')
    test_ds  = GoldMDDWithIndex(data_root, 'test')

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # No-index loaders for standard evaluate()
    val_loader_noindex = torch.utils.data.DataLoader(
        NoIndexDataset(val_ds), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    test_loader_noindex = torch.utils.data.DataLoader(
        NoIndexDataset(test_ds), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # Model
    import torchvision.models as tv
    from models import ImageClassifier
    model = ImageClassifier(num_classes=NUM_CLASSES).to(device)
    logger.info(f"IDMN (ResNet-50) built ✅")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Params: {params:.1f}M")

    # SAT Loss — needs num_train
    sat_args = argparse.Namespace(
        scheme='SAT',
        device=device,
        num_train=len(train_ds),
        num_classes=NUM_CLASSES,
        Es=10,           # start self-adaptive training after epoch 10
        lam1=1.0,
    )
    from losses import get_criterion
    criterion = get_criterion(sat_args)

    # Optimizer — Adam with cosine schedule (as per paper)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * epochs)
    scaler = GradScaler()

    best_map = 0.0
    start_epoch = 1
    if args.resume:
        last = out_dir / 'checkpoints/last.pt'
        if last.exists():
            ck = torch.load(last, map_location=device)
            model.load_state_dict(ck['model'])
            start_epoch = ck['epoch'] + 1
            best_map    = ck.get('map', 0.0)
            logger.info(f'Resumed from epoch {ck["epoch"]} mAP={best_map:.4f}')
    with open(out_dir / 'train_log.csv', 'w') as f:
        f.write('epoch,train_loss,val_map,val_cf1,val_macro_f1\n')

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for imgs, labels, indices in train_loader:
            imgs    = imgs.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels, indices)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            total_loss += loss.item()

        criterion.end_of_epoch()

        # Validate using standard evaluator — use wrapper loader
        val_metrics = evaluate(model, val_loader_noindex, device,
                               forward_fn=lambda m, x: m(x))
        elapsed = time.time() - t0
        train_loss = total_loss / len(train_loader)

        logger.info(f"Epoch {epoch:03d}/{epochs} | loss={train_loss:.4f} | "
                    f"mAP={val_metrics['map']:.4f} CF1={val_metrics['cf1']:.4f} | {elapsed:.1f}s")

        with open(out_dir / 'train_log.csv', 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_metrics['map']:.4f},"
                    f"{val_metrics['cf1']:.4f},{val_metrics['macro_f1']:.4f}\n")

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'map': val_metrics['map']}, out_dir / 'checkpoints/last.pt')

        if val_metrics['map'] > best_map:
            best_map = val_metrics['map']
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'map': best_map, 'metrics': val_metrics},
                       out_dir / 'checkpoints/best.pt')
            logger.info(f"  ★ New best mAP={best_map:.4f}")

    logger.info(f"Training complete. Best mAP={best_map:.4f}")

    # Test evaluation
    ck = torch.load(out_dir / 'checkpoints/best.pt', map_location=device)
    model.load_state_dict(ck['model'])
    test_metrics = evaluate(model, test_loader_noindex, device,
                            forward_fn=lambda m, x: m(x))
    save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                 model_name, 'test')
    logger.info(f"Test mAP={test_metrics['map']:.4f}")


if __name__ == '__main__':
    main()
