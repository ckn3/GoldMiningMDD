"""ML-GCN trainer for GoldMDD MLC.
Paper: Multi-Label Image Recognition with Graph Convolutional Networks (CVPR2019)
Backbone: ResNet-101 + GCN, Loss: BCE, Word embeddings + adjacency matrix required.
"""
import sys, os, pickle
from pathlib import Path
MLC_ROOT = Path(os.environ.get('GOLDMDD_MLC_REPO',
    Path(__file__).resolve().parents[2] / 'multi-label-classification'))
sys.path.insert(0, str(MLC_ROOT / 'utils/dataset'))
sys.path.insert(0, str(MLC_ROOT / 'utils/metrics'))
sys.path.insert(0, str(MLC_ROOT / 'utils'))
sys.path.insert(0, str(MLC_ROOT / 'repos/ML-GCN'))

import torch
import torch.nn as nn
import torchvision.models as tv
import logging
from train_mlc_base import BaseMLCTrainer, load_protocol, build_optimizer, build_scheduler, get_base_args
from goldmdd_mlc import build_dataloader, NUM_CLASSES
from evaluate_mlc import evaluate, save_results
from asl_loss import AsymmetricLoss
import time
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')


class MLGCNTrainer(BaseMLCTrainer):
    def build_model(self):
        from models import GCNResnet
        resnet = tv.resnet101(weights=tv.ResNet101_Weights.IMAGENET1K_V1)
        adj_file = str(MLC_ROOT / 'protocols/goldmdd_adj.pkl')
        word_emb = str(MLC_ROOT / 'protocols/goldmdd_word_emb.pkl')
        model = GCNResnet(model=resnet, num_classes=NUM_CLASSES,
                          in_channel=300, t=0.4, adj_file=adj_file)
        # Load word embeddings
        with open(word_emb, 'rb') as f:
            inp = pickle.load(f)
        model.inp = torch.from_numpy(inp).float()
        logger.info("ML-GCN built ✅")
        return model

    def run(self, args):
        # Override forward to pass word embeddings
        device = torch.device('cuda')
        cfg    = self.cfg
        out_dir = Path(cfg['output']['runs_dir']) / self.model_name
        (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (out_dir / 'results').mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(out_dir / 'train.log')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)

        model = self.build_model().to(device)
        inp   = model.inp.to(device)

        data_root  = cfg['dataset']['data_root']
        batch_size = cfg['training']['batch_size']
        train_loader, _ = build_dataloader(data_root, 'train', batch_size, 4)
        val_loader,   _ = build_dataloader(data_root, 'val',   batch_size, 4)

        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))
        scaler    = GradScaler()

        epochs   = cfg['training']['epochs']
        best_map = 0.0

        with open(out_dir / 'train_log.csv', 'w') as f:
            f.write('epoch,train_loss,val_map,val_cf1,val_macro_f1\n')

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            t0 = time.time()
            for imgs, labels in train_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs, inp.unsqueeze(0).expand(imgs.size(0), -1, -1))
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                scheduler.step()
                total_loss += loss.item()

            val_metrics = evaluate(model, val_loader, device,
                                   forward_fn=lambda m, x: m(x, inp.unsqueeze(0).expand(x.size(0), -1, -1)))
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
        test_metrics = evaluate(model, test_loader, device,
                                forward_fn=lambda m, x: m(x, inp.unsqueeze(0).expand(x.size(0), -1, -1)))
        save_results(test_metrics, str(out_dir / 'results/multilabel_results.json'),
                     self.model_name, 'test')
        logger.info(f"Test mAP={test_metrics['map']:.4f}")


def main():
    args = get_base_args().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    MLGCNTrainer('ml_gcn').run(args)

if __name__ == '__main__':
    main()
