#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_MISC = Path('/deac/csc/yangGrp/cuij/GoldMDD/misc')
SMP_SCRIPT = ROOT_MISC / 'train_semseg_smp.py'
SEGFORMER_SCRIPT = ROOT_MISC / 'train_semseg_segformer.py'
EFFICIENTVIT_SCRIPT = ROOT_MISC / 'train_semseg_efficientvit.py'

CLASS_NAMES = [
    'Building', 'Mining raft', 'Primary Forest', 'Heavy machinery', 'Water bodies',
    'Agricultural crop', 'Compact mounds', 'Gravel mounds', 'Grass',
    'Type1 regen', 'Type2 regen', 'Bare ground', 'Sluice', 'Vehicles'
]


@dataclass
class RunInfo:
    name: str
    family: str
    run_dir: Path
    config: dict


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--experiments-root', type=Path, default=Path('/deac/csc/yangGrp/cuij/GoldMDD/experiments'))
    p.add_argument('--diagnostics-dir', type=Path, default=Path('/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics'))
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--use-family-bests', action='store_true', default=True)
    p.add_argument('--runs', nargs='*', default=None)
    return p.parse_args()


def infer_family(name: str) -> str | None:
    if name.startswith('baseline'):
        return 'DeepLabV3+/ConvNeXt-Tiny'
    if name.startswith('segformer_'):
        return 'SegFormer-B2'
    if name.startswith('efficientvit_'):
        return 'EfficientViT-Seg-B2'
    return None


def discover_runs(root: Path) -> list[RunInfo]:
    out: list[RunInfo] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in {'logs', 'diagnostics'}:
            continue
        fam = infer_family(d.name)
        if fam is None:
            continue
        cfg_path = d / 'config.json'
        best_path = d / 'best.pt'
        if not (cfg_path.exists() and best_path.exists()):
            continue
        out.append(RunInfo(d.name, fam, d, json.loads(cfg_path.read_text())))
    return out


def pick_family_bests(runs: list[RunInfo], summary_csv: Path) -> list[RunInfo]:
    sdf = pd.read_csv(summary_csv)
    chosen = []
    for fam in ['DeepLabV3+/ConvNeXt-Tiny', 'SegFormer-B2', 'EfficientViT-Seg-B2']:
        sub = sdf[sdf['family'] == fam]
        if sub.empty:
            continue
        row = sub.sort_values('test_miou_present', ascending=False).iloc[0]
        name = row['run']
        match = next((r for r in runs if r.name == name), None)
        if match is not None:
            chosen.append(match)
    return chosen


def build_model_and_forward(run: RunInfo, smp_mod, seg_mod, eff_mod):
    cfg = run.config
    if run.family == 'DeepLabV3+/ConvNeXt-Tiny':
        model = smp_mod.build_model(cfg['arch'], cfg['encoder'], cfg.get('encoder_weights'))
        forward = smp_mod.forward_logits if hasattr(smp_mod, 'forward_logits') else None
        if forward is None:
            def forward(m, x):
                out = m(x)
                if isinstance(out, dict):
                    out = out['out']
                return out
        return model, forward
    if run.family == 'SegFormer-B2':
        model = seg_mod.build_model(cfg['model_name'], bool(cfg.get('pretrained', True)))
        return model, seg_mod.forward_logits
    if run.family == 'EfficientViT-Seg-B2':
        model = eff_mod.build_model(
            cfg['model_name'],
            bool(cfg.get('pretrained', True)),
            Path(cfg['efficientvit_repo']),
            Path(cfg['weights_cache_dir']) if cfg.get('weights_cache_dir') else None,
            Path(cfg['pretrained_weight_file']) if cfg.get('pretrained_weight_file') else None,
        )
        return model, eff_mod.forward_logits
    raise ValueError(run.family)


def compute_confusion_for_run(run: RunInfo, semseg_common, smp_mod, seg_mod, eff_mod, device: torch.device, batch_size: int, num_workers: int) -> tuple[np.ndarray, dict]:
    cache_path = run.run_dir / 'test_confusion_best.npz'
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        conf = z['conf']
        meta = json.loads(str(z['meta'].tolist()))
        return conf, meta

    samples = semseg_common.build_split_samples(Path(run.config['data_root']) / 'test')
    ds = semseg_common.GoldMDDPatchDataset(samples, train=False, aug_preset='none')
    loader = semseg_common.make_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model, forward_logits = build_model_and_forward(run, smp_mod, seg_mod, eff_mod)
    model.to(device)
    model.eval()

    ckpt = torch.load(run.run_dir / 'best.pt', map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    conf = torch.zeros((semseg_common.NUM_FOREGROUND_CLASSES, semseg_common.NUM_FOREGROUND_CLASSES), dtype=torch.int64, device=device)
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = forward_logits(model, x)
            semseg_common.update_confusion(conf, logits, y)

    conf_np = conf.detach().cpu().numpy()
    (
        miou,
        miou_present,
        macro_f1,
        macro_f1_present,
        oa_fg,
        per_class_iou,
        per_class_f1,
        gt_pixels,
    ) = semseg_common.compute_metrics_from_conf(conf.detach().cpu())
    meta = {
        'run': run.name,
        'family': run.family,
        'miou': miou,
        'miou_present': miou_present,
        'macro_f1_present': macro_f1_present,
        'oa_fg': oa_fg,
        'gt_pixels_per_class': gt_pixels,
    }
    np.savez_compressed(cache_path, conf=conf_np, meta=json.dumps(meta))
    return conf_np, meta


def row_normalize(conf: np.ndarray) -> np.ndarray:
    conf = conf.astype(np.float64)
    row_sum = conf.sum(axis=1, keepdims=True)
    out = np.divide(conf, row_sum, out=np.full_like(conf, np.nan, dtype=np.float64), where=row_sum > 0)
    return out


def top_confusions(conf: np.ndarray, topk: int = 8) -> list[tuple[int,int,float,int,int]]:
    # return (gt_idx,pred_idx,row_frac,count,row_total) for off-diagonal confusions, gt rows with support
    rn = row_normalize(conf)
    rows = []
    gt_tot = conf.sum(axis=1)
    for i in range(conf.shape[0]):
        if gt_tot[i] <= 0:
            continue
        for j in range(conf.shape[1]):
            if i == j:
                continue
            if conf[i, j] <= 0:
                continue
            rows.append((i, j, float(rn[i, j]), int(conf[i, j]), int(gt_tot[i])))
    rows.sort(key=lambda t: (t[2], t[3]), reverse=True)
    return rows[:topk]


def plot_one(conf: np.ndarray, title: str, out_path: Path | None = None, ax=None):
    rn = row_normalize(conf)
    gt_pixels = conf.sum(axis=1)
    present_rows = np.where(gt_pixels > 0)[0]
    show = rn[present_rows, :]
    row_names = [CLASS_NAMES[i] for i in present_rows]
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    else:
        fig = ax.figure
    im = ax.imshow(np.ma.masked_invalid(show), cmap='magma', vmin=0, vmax=1, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('GT class (present in test)')
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=8)
    for r in range(show.shape[0]):
        for c in range(show.shape[1]):
            v = show[r, c]
            if np.isnan(v) or v < 0.05:
                continue
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7, color='white' if v < 0.6 else 'black')
    if out_path is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.01)
        cbar.set_label('Row-normalized fraction')
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
    return im


def main() -> None:
    args = parse_args()
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    semseg_common = load_module(ROOT_MISC / 'semseg_common.py', 'goldmdd_semseg_common_cm')
    smp_mod = load_module(SMP_SCRIPT, 'goldmdd_train_smp_cm')
    seg_mod = load_module(SEGFORMER_SCRIPT, 'goldmdd_train_segformer_cm')
    eff_mod = load_module(EFFICIENTVIT_SCRIPT, 'goldmdd_train_efficientvit_cm')

    runs = discover_runs(args.experiments_root)
    if args.runs:
        keep = set(args.runs)
        runs = [r for r in runs if r.name in keep]
    elif args.use_family_bests:
        runs = pick_family_bests(runs, args.diagnostics_dir / 'experiment_suite_summary.csv')

    if not runs:
        raise SystemExit('No runs selected')

    results = []
    top_rows = []
    for run in runs:
        print(f'Computing confusion: {run.name} ({run.family})')
        conf, meta = compute_confusion_for_run(run, semseg_common, smp_mod, seg_mod, eff_mod, device, args.batch_size, args.num_workers)
        results.append((run, conf, meta))
        for gt_i, pr_i, frac, cnt, gt_tot in top_confusions(conf, topk=10):
            top_rows.append({
                'run': run.name,
                'family': run.family,
                'gt_class_id': gt_i + 1,
                'gt_class': CLASS_NAMES[gt_i],
                'pred_class_id': pr_i + 1,
                'pred_class': CLASS_NAMES[pr_i],
                'row_fraction': frac,
                'pixels': cnt,
                'gt_pixels': gt_tot,
            })

    # overall best among selected by miou_present
    best = max(results, key=lambda x: x[2].get('miou_present', float('-inf')))
    best_run, best_conf, best_meta = best
    plot_one(
        best_conf,
        title=f"Test confusion (best selected): {best_run.name}\nrow-normalized, GT-present rows only",
        out_path=args.diagnostics_dir / 'test_confusion_matrix_best_selected.png',
    )

    # grid plot for selected runs
    n = len(results)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 5.8 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    ims = []
    for ax, (run, conf, meta) in zip(axes, results):
        im = plot_one(conf, f"{run.name}\nmiou_present={meta['miou_present']:.4f}, oa_fg={meta['oa_fg']:.4f}", ax=ax)
        ims.append(im)
    for ax in axes[len(results):]:
        ax.axis('off')
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes[:len(results)].tolist(), fraction=0.016, pad=0.01)
        cbar.set_label('Row-normalized fraction')
    fig.savefig(args.diagnostics_dir / 'test_confusion_matrices_family_bests.png', dpi=180)
    plt.close(fig)

    top_df = pd.DataFrame(top_rows).sort_values(['run', 'row_fraction', 'pixels'], ascending=[True, False, False])
    top_df.to_csv(args.diagnostics_dir / 'test_confusion_top_offdiagonal_family_bests.csv', index=False)

    # short markdown summary
    md = ['# Test Confusion Matrix Summary (Family Bests)', '']
    md.append('Selected runs:')
    md.append('')
    md.append(pd.DataFrame([
        {'run': r.name, 'family': r.family, **m} for (r, _, m) in results
    ])[['run','family','miou','miou_present','macro_f1_present','oa_fg']].to_markdown(index=False))
    md.append('')
    md.append('Top off-diagonal confusions per run (row-normalized, GT->Pred):')
    md.append('')
    md.append(top_df.groupby('run').head(5).to_markdown(index=False))
    (args.diagnostics_dir / 'test_confusion_summary_family_bests.md').write_text('\n'.join(md), encoding='utf-8')

    print('Saved:')
    print(args.diagnostics_dir / 'test_confusion_matrix_best_selected.png')
    print(args.diagnostics_dir / 'test_confusion_matrices_family_bests.png')
    print(args.diagnostics_dir / 'test_confusion_top_offdiagonal_family_bests.csv')
    print(args.diagnostics_dir / 'test_confusion_summary_family_bests.md')


if __name__ == '__main__':
    main()
