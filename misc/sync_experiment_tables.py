#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP_ROOT = Path('/deac/csc/yangGrp/cuij/GoldMDD/experiments')
DIAG = EXP_ROOT / 'diagnostics'
SUMMARY_CSV = DIAG / 'experiment_suite_summary.csv'
SUMMARY_MD = DIAG / 'summary.md'
VAL_CSV = DIAG / 'val_per_class_iou_completed_models_present_only.csv'
TEST_CSV = DIAG / 'test_per_class_iou_completed_models_present_only.csv'

RUN_RENAME = {
    'segnext_tiny_goldmdd_l40': 'segnext_tiny_goldmdd',
    'cgrseg_b_goldmdd_l40': 'cgrseg_b_goldmdd',
}

CANON_CLASSES = [
    'Building', 'Mining raft', 'Primary Forest', 'Heavy machinery',
    'Water bodies', 'Agricultural crop', 'Compact mounds', 'Gravel mounds',
    'Grass', 'Type1 regen', 'Type2 regen', 'Bare ground', 'Sluice', 'Vehicles'
]
ALT_CLASS_MAP = {
    'Type 1 natural regeneration': 'Type1 regen',
    'Type 2 natural regeneration': 'Type2 regen',
}

RS_MODELS = {
    'FarSeg', 'FarSeg++', 'LoGCAN', 'LoGCAN++', 'SACANet', 'DOCNet',
    'BANet', 'ABCNet', 'MANet', 'UNetFormer', 'DC-Swin', 'A2FPN', 'PyramidMamba', 'MF-Mamba'
}
VFM_MODELS = {'SAM_RS', 'RSAM-Seg', 'SESSRS', 'SAM2.1', 'SAM3', 'HQ-SAM'}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _canon_from_dict(d: dict[str, Any]) -> list[float]:
    out = [float('nan')] * len(CANON_CLASSES)
    key_map = {k: v for k, v in ALT_CLASS_MAP.items()}
    for idx, cname in enumerate(CANON_CLASSES):
        if cname in d:
            v = d[cname]
        else:
            # reverse alt map lookup
            alts = [k for k, v in key_map.items() if v == cname]
            v = d.get(alts[0], float('nan')) if alts else float('nan')
        out[idx] = float(v) if v is not None else float('nan')
    return out


def _extract_metrics(run_dir: Path, split: str) -> dict[str, Any] | None:
    def _from_sessrs_payload(raw: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        pred = raw.get('sessrs_pred_metrics')
        if not isinstance(pred, dict):
            return None
        per = pred.get('per_class_iou', [])
        if isinstance(per, dict):
            per = _canon_from_dict(per)
        return {
            'miou': pred.get('miou', float('nan')),
            'miou_present': pred.get('miou_present', float('nan')),
            'macro_f1_present': pred.get('macro_f1_present', float('nan')),
            'oa_fg': pred.get('oa_fg', float('nan')),
            'per_class_iou': [float(x) for x in per] if per is not None else [],
            'postprocess_ms_per_image': raw.get('postprocess_ms_per_image', float('nan')),
        }

    if split == 'val':
        p = run_dir / 'val_metrics_best.json'
        if p.exists():
            return _load_json(p)
        p = run_dir / 'val_metrics.json'
        if p.exists():
            raw = _load_json(p)
            sessrs = _from_sessrs_payload(raw)
            if sessrs is not None:
                return sessrs
        # Some runners only dump best per-class IoU without aggregate scalar metrics.
        p = run_dir / 'best_val_per_class_iou.json'
        if p.exists():
            raw = _load_json(p)
            if isinstance(raw, dict):
                per = _canon_from_dict(raw)
            else:
                per = [float(x) for x in raw]
            return {'per_class_iou': per}
        return None

    # test
    p = run_dir / 'test_metrics.json'
    if p.exists():
        raw = _load_json(p)
        sessrs = _from_sessrs_payload(raw)
        if sessrs is not None:
            return sessrs
        # Normalize "test_*" payloads (e.g., RSAM-Seg) to a common schema.
        if any(k in raw for k in ['test_miou', 'test_miou_present', 'test_per_class_iou']):
            per = raw.get('test_per_class_iou', raw.get('per_class_iou', []))
            if isinstance(per, dict):
                per = _canon_from_dict(per)
            return {
                'miou': raw.get('test_miou', raw.get('miou', float('nan'))),
                'miou_present': raw.get('test_miou_present', raw.get('miou_present', float('nan'))),
                'macro_f1_present': raw.get('test_macro_f1_present', raw.get('macro_f1_present', float('nan'))),
                'oa_fg': raw.get('test_oa_fg', raw.get('oa_fg', float('nan'))),
                'per_class_iou': [float(x) for x in per] if per is not None else [],
            }
        return raw

    p = run_dir / 'eval_best_test' / 'test_metrics_unified.json'
    if p.exists():
        raw = _load_json(p)
        per = raw.get('per_class_iou', {})
        per_arr = _canon_from_dict(per) if isinstance(per, dict) else [float(x) for x in per]
        return {
            'miou': raw.get('test_miou', raw.get('miou', float('nan'))),
            'miou_present': raw.get('test_miou_present', raw.get('miou_present', float('nan'))),
            'macro_f1_present': raw.get('test_macro_f1_present', raw.get('macro_f1_present', float('nan'))),
            'oa_fg': raw.get('test_oa_fg', raw.get('oa_fg', float('nan'))),
            'per_class_iou': per_arr,
        }

    p = run_dir / 'summary.json'
    if p.exists():
        raw = _load_json(p)
        if 'test_per_class_iou' in raw:
            return {
                'miou': raw.get('test_miou', float('nan')),
                'miou_present': raw.get('test_miou_present', float('nan')),
                'macro_f1_present': raw.get('test_macro_f1_present', float('nan')),
                'oa_fg': raw.get('test_oa_fg', float('nan')),
                'per_class_iou': [float(x) for x in raw.get('test_per_class_iou', [])],
            }
    return None


def _fmt(v: Any, prec: int = 4) -> str:
    if v is None:
        return '-'
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(fv):
        return '-'
    return f'{fv:.{prec}f}'


def _fmt_epoch(v: Any) -> str:
    if v is None:
        return '-'
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(fv):
        return '-'
    return str(int(round(fv)))


def _md_table_with_placeholders(df: pd.DataFrame, floatfmt: str = '.4f') -> str:
    if df.empty:
        return '| status | model | backbone | loss |\n|:--|:--|:--|:--|'
    show = df.where(pd.notna(df), '-')
    return show.to_markdown(index=False, floatfmt=floatfmt)


def _parse_table(lines: list[str], start_idx: int) -> tuple[int, int, pd.DataFrame]:
    i = start_idx
    while i < len(lines) and not lines[i].startswith('|'):
        i += 1
    header = [c.strip() for c in lines[i].strip().strip('|').split('|')]
    j = i + 2
    rows: list[list[str]] = []
    while j < len(lines) and lines[j].startswith('|'):
        rows.append([c.strip() for c in lines[j].strip().strip('|').split('|')])
        j += 1
    df = pd.DataFrame(rows, columns=header)
    return i, j, df


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    # Keep run names aligned to actual folders.
    df['run'] = df['run'].replace(RUN_RENAME)

    val_existing = pd.read_csv(VAL_CSV) if VAL_CSV.exists() else pd.DataFrame()
    test_existing = pd.read_csv(TEST_CSV) if TEST_CSV.exists() else pd.DataFrame()

    # Update missing metrics for completed runs.
    for idx, row in df[df['status'] == 'completed'].iterrows():
        run_dir = EXP_ROOT / row['run']
        if not run_dir.exists():
            continue

        val_m = _extract_metrics(run_dir, 'val')
        test_m = _extract_metrics(run_dir, 'test')

        if val_m is not None:
            force_refresh = str(row.get('model', '')) == 'SESSRS'
            for src, dst in [
                ('miou', 'val_bestckpt_miou'),
                ('miou_present', 'val_bestckpt_miou_present'),
                ('macro_f1_present', 'val_bestckpt_macro_f1_present'),
                ('oa_fg', 'val_bestckpt_oa_fg'),
            ]:
                if src in val_m and (force_refresh or pd.isna(df.at[idx, dst])):
                    df.at[idx, dst] = float(val_m[src])
            # Keep best-val columns aligned for runners that only provide val_metrics_best.json.
            if 'miou' in val_m and (force_refresh or pd.isna(df.at[idx, 'best_val_miou'])):
                df.at[idx, 'best_val_miou'] = float(val_m['miou'])
            if 'miou_present' in val_m and (force_refresh or pd.isna(df.at[idx, 'best_val_miou_present'])):
                df.at[idx, 'best_val_miou_present'] = float(val_m['miou_present'])
            if ('source_epoch' in val_m) and (force_refresh or pd.isna(df.at[idx, 'best_val_miou_epoch'])):
                try:
                    df.at[idx, 'best_val_miou_epoch'] = float(val_m['source_epoch'])
                except Exception:
                    pass
            if ('source_epoch' in val_m) and (force_refresh or pd.isna(df.at[idx, 'best_val_miou_present_epoch'])):
                try:
                    df.at[idx, 'best_val_miou_present_epoch'] = float(val_m['source_epoch'])
                except Exception:
                    pass
            ckpt = str(val_m.get('checkpoint') or val_m.get('source_checkpoint') or '')
            m_epoch = re.search(r'epoch[=:_-]([0-9]+)', ckpt)
            if m_epoch and pd.isna(df.at[idx, 'best_val_miou_epoch']):
                df.at[idx, 'best_val_miou_epoch'] = float(m_epoch.group(1))
            if m_epoch and pd.isna(df.at[idx, 'best_val_miou_present_epoch']):
                df.at[idx, 'best_val_miou_present_epoch'] = float(m_epoch.group(1))
            # mmseg-style naming: best_mIoU_iter_<N>.pth. Convert iter -> epoch by protocol equivalence.
            m_iter = re.search(r'iter[_=:-]?([0-9]+)', ckpt)
            if m_iter:
                try:
                    iter_id = int(m_iter.group(1))
                    ep = float(iter_id // 8225)  # 658000 iters / 80 epochs
                    if pd.isna(df.at[idx, 'best_val_miou_epoch']):
                        df.at[idx, 'best_val_miou_epoch'] = ep
                    if pd.isna(df.at[idx, 'best_val_miou_present_epoch']):
                        df.at[idx, 'best_val_miou_present_epoch'] = ep
                except Exception:
                    pass

        # Backfill from train_log for runs that don't store val_metrics_best.json.
        if (pd.isna(df.at[idx, 'val_bestckpt_macro_f1_present']) or pd.isna(df.at[idx, 'val_bestckpt_oa_fg'])):
            tl = run_dir / 'train_log.csv'
            if tl.exists():
                tdf = pd.read_csv(tl)
                ep = int(df.at[idx, 'best_val_miou_present_epoch']) if np.isfinite(df.at[idx, 'best_val_miou_present_epoch']) else None
                if ep is None and 'val_miou_present' in tdf.columns and len(tdf) > 0:
                    best_idx = tdf['val_miou_present'].idxmax()
                    ep = int(tdf.loc[best_idx, 'epoch']) if 'epoch' in tdf.columns else int(best_idx + 1)
                    if pd.isna(df.at[idx, 'best_val_miou_present']):
                        df.at[idx, 'best_val_miou_present'] = float(tdf.loc[best_idx, 'val_miou_present'])
                    if pd.isna(df.at[idx, 'best_val_miou']):
                        if 'val_miou' in tdf.columns:
                            df.at[idx, 'best_val_miou'] = float(tdf.loc[best_idx, 'val_miou'])
                        else:
                            df.at[idx, 'best_val_miou'] = float(tdf.loc[best_idx, 'val_miou_present'])
                    if pd.isna(df.at[idx, 'best_val_miou_present_epoch']):
                        df.at[idx, 'best_val_miou_present_epoch'] = float(ep)
                    if pd.isna(df.at[idx, 'best_val_miou_epoch']):
                        df.at[idx, 'best_val_miou_epoch'] = float(ep)
                if ep is not None and 'epoch' in tdf.columns and (tdf['epoch'] == ep).any():
                    r = tdf[tdf['epoch'] == ep].iloc[0]
                    if pd.isna(df.at[idx, 'val_bestckpt_macro_f1_present']) and 'val_macro_f1_present' in r:
                        df.at[idx, 'val_bestckpt_macro_f1_present'] = float(r['val_macro_f1_present'])
                    if pd.isna(df.at[idx, 'val_bestckpt_oa_fg']) and 'val_oa_fg' in r:
                        df.at[idx, 'val_bestckpt_oa_fg'] = float(r['val_oa_fg'])

        if test_m is not None:
            force_refresh = str(row.get('model', '')) == 'SESSRS'
            for src, dst in [
                ('miou', 'test_miou'),
                ('miou_present', 'test_miou_present'),
                ('macro_f1_present', 'test_macro_f1_present'),
                ('oa_fg', 'test_oa_fg'),
            ]:
                if src in test_m and (force_refresh or pd.isna(df.at[idx, dst])):
                    df.at[idx, dst] = float(test_m[src])

    # Fill FarSeg++ efficiency metrics.
    farsegpp_stats = DIAG / 'model_stats' / 'farsegpp_mitb2.json'
    if farsegpp_stats.exists():
        st = _load_json(farsegpp_stats)
        mask = df['run'] == 'farsegpp_mitb2_native'
        if mask.any():
            i = df[mask].index[0]
            df.at[i, 'params_m'] = float(st['params_m'])
            df.at[i, 'gmacs'] = float(st['gmacs'])
            df.at[i, 'gflops'] = float(st['gflops'])
            df.at[i, 'latency_ms_1x3x512x512'] = float(st['latency_ms_1x3x512x512'])
            df.at[i, 'peak_vram_gb'] = float(st['peak_vram_gb'])

    # Fill SAM_RS efficiency metrics.
    samrs_stats = DIAG / 'model_stats' / 'sam_rs_unetformer.json'
    if samrs_stats.exists():
        st = _load_json(samrs_stats)
        mask = (
            (df['model'] == 'SAM_RS')
            & (df['backbone'] == 'UNetFormer + SAM priors')
            & (df['loss_display'] == 'seg+bdy+obj (native)')
        )
        if mask.any():
            i = df[mask].index[0]
            df.at[i, 'params_m'] = float(st['params_m'])
            df.at[i, 'gmacs'] = float(st['gmacs'])
            df.at[i, 'gflops'] = float(st['gflops'])
            df.at[i, 'latency_ms_1x3x512x512'] = float(st['latency_ms_1x3x512x512'])
            df.at[i, 'peak_vram_gb'] = float(st['peak_vram_gb'])

    # Fill additional efficiency metrics from cached model_stats JSON files.
    # SESSRS rows reuse the same backbone stats as their underlying GeoSeg model;
    # SESSRS itself is a post-processing step and does not define a new network.
    fill_specs: list[tuple[pd.Series, str]] = [
        (
            (df['model'] == 'MCPNet')
            & (df['backbone'] == 'ResNet-50'),
            'mcpnet_r50.json',
        ),
        (
            (df['model'] == 'MANet')
            & (df['backbone'] == 'ResNet-50'),
            'geoseg_manet.json',
        ),
        (
            (df['model'] == 'PyramidMamba')
            & (df['backbone'] == 'Swin-Base'),
            'geoseg_pyramidmamba.json',
        ),
        (
            (df['model'] == 'DC-Swin')
            & (df['backbone'] == 'Swin-Small'),
            'geoseg_dcswin.json',
        ),
        (
            (df['model'] == 'SAM2.1')
            & (df['backbone'].str.contains('Hiera-B\\+', na=False)),
            'sam2_1_hierabplus_msfpn.json',
        ),
        (
            (df['model'] == 'HQ-SAM')
            & (df['backbone'].str.contains('ViT-B \\+ HQ decoder', na=False)),
            'hq_sam_vit_b_msfpn.json',
        ),
        (
            (df['model'] == 'RSAM-Seg')
            & (df['backbone'] == 'ViT-B'),
            'rsamseg_vit_b.json',
        ),
        (
            (df['model'] == 'CGRSeg')
            & (df['backbone'] == 'EfficientFormerV2-B'),
            'ssaseg_cgrseg_b.json',
        ),
        (
            (df['model'] == 'DOCNet')
            & (df['backbone'] == 'HRNet-W32')
            & (df['loss_display'] == 'ce+aux_ce (native)'),
            'rsseg_docnet_hrnetw32.json',
        ),
        (
            (df['model'] == 'SACANet')
            & (df['backbone'] == 'HRNet-W32')
            & (df['loss_display'] == 'ce+aux_ce (native)'),
            'rsseg_sacanet_hrnetw32.json',
        ),
        (
            (df['model'] == 'SAM_RS')
            & (df['backbone'] == 'FTUNetFormer + SAM priors')
            & (df['loss_display'] == 'seg+bdy+obj (native)'),
            'sam_rs_ftunetformer.json',
        ),
        (
            (df['model'] == 'SAM_RS')
            & (df['backbone'] == 'ABCNet + SAM priors')
            & (df['loss_display'] == 'seg+bdy+obj (native)'),
            'sam_rs_abcnet.json',
        ),
        (
            (df['model'] == 'SAM_RS')
            & (df['backbone'] == 'CMTFNet + SAM priors')
            & (df['loss_display'] == 'seg+bdy+obj (native)'),
            'sam_rs_cmtfnet.json',
        ),
        (
            (df['model'] == 'SESSRS') & (df['backbone'].str.startswith('UNetFormer', na=False)),
            'geoseg_unetformer.json',
        ),
        (
            (df['model'] == 'SESSRS') & (df['backbone'].str.startswith('ABCNet', na=False)),
            'geoseg_abcnet_r18.json',
        ),
        (
            (df['model'] == 'SESSRS') & (df['backbone'].str.startswith('BANet', na=False)),
            'geoseg_banet.json',
        ),
        (
            (df['model'] == 'SESSRS') & (df['backbone'].str.startswith('MANet', na=False)),
            'geoseg_manet.json',
        ),
        (
            (df['model'] == 'SESSRS') & (df['backbone'].str.startswith('A2FPN', na=False)),
            'geoseg_a2fpn.json',
        ),
        (
            (df['model'] == 'PPMambaSeg')
            & (df['backbone'] == 'swsl-ResNet-18'),
            'ppmambaseg_ppmamba.json',
        ),
    ]
    for mask, fname in fill_specs:
        stats_path = DIAG / 'model_stats' / fname
        if not stats_path.exists() or not mask.any():
            continue
        st = _load_json(stats_path)
        for i in df[mask].index.tolist():
            # Keep deterministic values across repeated sync calls.
            df.at[i, 'params_m'] = float(st['params_m'])
            df.at[i, 'gmacs'] = float(st['gmacs'])
            df.at[i, 'gflops'] = float(st['gflops'])
            df.at[i, 'latency_ms_1x3x512x512'] = float(st['latency_ms_1x3x512x512'])
            df.at[i, 'peak_vram_gb'] = float(st['peak_vram_gb'])

    # Add SESSRS post-process overhead on top of backbone inference latency.
    sessrs_mask = (df['status'] == 'completed') & (df['model'] == 'SESSRS')
    for i in df[sessrs_mask].index.tolist():
        run_dir = EXP_ROOT / str(df.at[i, 'run'])
        tm = _extract_metrics(run_dir, 'test')
        if tm is None:
            continue
        post_ms = float(tm.get('postprocess_ms_per_image', float('nan')))
        if not np.isfinite(post_ms):
            continue
        base_lat = (
            float(df.at[i, 'latency_ms_1x3x512x512'])
            if np.isfinite(df.at[i, 'latency_ms_1x3x512x512'])
            else 0.0
        )
        df.at[i, 'latency_ms_1x3x512x512'] = base_lat + post_ms

    # Persist updated suite summary.
    df.to_csv(SUMMARY_CSV, index=False)

    completed = df[df['status'] == 'completed'].copy()

    val_present_mask = np.array(_load_json(EXP_ROOT / 'baseline1_augv2_ce_dice' / 'val_metrics_best.json')['gt_pixels_per_class']) > 0
    test_present_mask = np.array(_load_json(EXP_ROOT / 'baseline1_augv2_ce_dice' / 'test_metrics.json')['gt_pixels_per_class']) > 0

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for _, row in completed.iterrows():
        run = row['run']
        run_dir = EXP_ROOT / run

        val_m = _extract_metrics(run_dir, 'val')
        if val_m is None and not val_existing.empty:
            m = (
                (val_existing['model'] == row['model'])
                & (val_existing['backbone'] == row['backbone'])
                & (val_existing['loss'] == row['loss_display'])
            )
            if m.any():
                r0 = val_existing[m].iloc[0]
                per = [float(r0.get(c, np.nan)) for c in CANON_CLASSES]
                val_m = {'per_class_iou': per}

        if val_m is not None:
            per = val_m.get('per_class_iou', [])
            if isinstance(per, dict):
                per = _canon_from_dict(per)
            per = [float(x) for x in per]
            if len(per) == len(CANON_CLASSES):
                out = {
                    'model': row['model'],
                    'backbone': row['backbone'],
                    'loss': row['loss_display'],
                    'val_miou_present': float(row['best_val_miou_present']) if np.isfinite(row['best_val_miou_present']) else np.nan,
                }
                for i_cls, cname in enumerate(CANON_CLASSES):
                    if val_present_mask[i_cls]:
                        out[cname] = per[i_cls]
                val_rows.append(out)

        test_m = _extract_metrics(run_dir, 'test')
        if test_m is None and not test_existing.empty:
            m = (
                (test_existing['model'] == row['model'])
                & (test_existing['backbone'] == row['backbone'])
                & (test_existing['loss'] == row['loss_display'])
            )
            if m.any():
                r0 = test_existing[m].iloc[0]
                per = [float(r0.get(c, np.nan)) for c in CANON_CLASSES]
                test_m = {'per_class_iou': per}

        if test_m is not None:
            per = test_m.get('per_class_iou', [])
            if isinstance(per, dict):
                per = _canon_from_dict(per)
            per = [float(x) for x in per]
            if len(per) == len(CANON_CLASSES):
                out = {
                    'model': row['model'],
                    'backbone': row['backbone'],
                    'loss': row['loss_display'],
                    'test_miou_present': float(row['test_miou_present']) if np.isfinite(row['test_miou_present']) else np.nan,
                }
                for i_cls, cname in enumerate(CANON_CLASSES):
                    if test_present_mask[i_cls]:
                        out[cname] = per[i_cls]
                test_rows.append(out)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    # Keep per-class artifacts compact and human-readable.
    for _df in (val_df, test_df):
        for c in _df.columns:
            if c in {'model', 'backbone', 'loss'}:
                continue
            if pd.api.types.is_numeric_dtype(_df[c]):
                _df[c] = _df[c].round(4)

    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    # Update summary.md tables.
    lines = SUMMARY_MD.read_text(encoding='utf-8').splitlines()

    # Main table
    main_start = lines.index('## Ranking (chronological order)') + 1
    t0, t1, tdf = _parse_table(lines, main_start)
    for ridx in range(len(tdf)):
        if str(tdf.at[ridx, 'model']).startswith('---'):
            continue
        m = (
            (df['model'] == tdf.at[ridx, 'model'])
            & (df['backbone'] == tdf.at[ridx, 'backbone'])
            & (df['loss_display'] == tdf.at[ridx, 'loss'])
        )
        if not m.any():
            continue
        r = df[m].iloc[0]
        tdf.at[ridx, 'status'] = str(r['status'])
        if tdf.at[ridx, 'status'] != 'completed':
            continue
        tdf.at[ridx, 'params_m'] = _fmt(r['params_m'])
        tdf.at[ridx, 'gflops'] = _fmt(r['gflops'])
        tdf.at[ridx, 'latency_ms_1x3x512x512'] = _fmt(r['latency_ms_1x3x512x512'])
        tdf.at[ridx, 'peak_vram_gb'] = _fmt(r['peak_vram_gb'])
        tdf.at[ridx, 'test_miou_present'] = _fmt(r['test_miou_present'])
        tdf.at[ridx, 'test_macro_f1_present'] = _fmt(r['test_macro_f1_present'])
        tdf.at[ridx, 'test_oa_fg'] = _fmt(r['test_oa_fg'])
        tdf.at[ridx, 'test_miou'] = _fmt(r['test_miou'])
        tdf.at[ridx, 'best_val_miou'] = _fmt(r['best_val_miou'])
        tdf.at[ridx, 'best_val_miou_present'] = _fmt(r['best_val_miou_present'])

    main_md = tdf.to_markdown(index=False)
    lines[t0:t1] = main_md.splitlines()

    # Training/Validation table
    train_start = lines.index('## Training / Validation Summary') + 1
    u0, u1, udf = _parse_table(lines, train_start)
    for ridx in range(len(udf)):
        if str(udf.at[ridx, 'model']).startswith('---'):
            continue
        m = (
            (df['model'] == udf.at[ridx, 'model'])
            & (df['backbone'] == udf.at[ridx, 'backbone'])
            & (df['loss_display'] == udf.at[ridx, 'loss'])
        )
        if not m.any():
            continue
        r = df[m].iloc[0]
        udf.at[ridx, 'status'] = str(r['status'])
        if udf.at[ridx, 'status'] != 'completed':
            continue
        udf.at[ridx, 'best_val_miou_epoch'] = _fmt_epoch(r['best_val_miou_epoch'])
        udf.at[ridx, 'best_val_miou'] = _fmt(r['best_val_miou'])
        udf.at[ridx, 'best_val_miou_present_epoch'] = _fmt_epoch(r['best_val_miou_present_epoch'])
        udf.at[ridx, 'best_val_miou_present'] = _fmt(r['best_val_miou_present'])
        udf.at[ridx, 'val_bestckpt_macro_f1_present'] = _fmt(r['val_bestckpt_macro_f1_present'])
        udf.at[ridx, 'val_bestckpt_oa_fg'] = _fmt(r['val_bestckpt_oa_fg'])

    train_md = udf.to_markdown(index=False)
    lines[u0:u1] = train_md.splitlines()

    # Rebuild per-class block.
    # Keep per-class table row order consistent with the main ranking table.
    ranking_order: list[tuple[str, str, str]] = []
    for _, r in tdf.iterrows():
        if str(r['model']).startswith('---'):
            continue
        ranking_order.append((str(r['model']), str(r['backbone']), str(r['loss'])))
    order_idx = {k: i for i, k in enumerate(ranking_order)}

    # Add running placeholders to per-class tables so users can track all runs in one place.
    class_cols_val = [c for c in val_df.columns if c not in {'model', 'backbone', 'loss', 'val_miou_present'}]
    class_cols_test = [c for c in test_df.columns if c not in {'model', 'backbone', 'loss', 'test_miou_present'}]
    running_df = df[df['status'] == 'running'][['model', 'backbone', 'loss_display']].drop_duplicates().copy()
    running_df = running_df.rename(columns={'loss_display': 'loss'})
    running_val_rows: list[dict[str, Any]] = []
    running_test_rows: list[dict[str, Any]] = []
    for _, rr in running_df.iterrows():
        vrow = {
            'status': 'running',
            'model': rr['model'],
            'backbone': rr['backbone'],
            'loss': rr['loss'],
            'val_miou_present': float('nan'),
        }
        for cname in class_cols_val:
            vrow[cname] = float('nan')
        running_val_rows.append(vrow)

        trow = {
            'status': 'running',
            'model': rr['model'],
            'backbone': rr['backbone'],
            'loss': rr['loss'],
            'test_miou_present': float('nan'),
        }
        for cname in class_cols_test:
            trow[cname] = float('nan')
        running_test_rows.append(trow)

    if not val_df.empty:
        val_df.insert(0, 'status', 'completed')
    if not test_df.empty:
        test_df.insert(0, 'status', 'completed')

    v_show = pd.concat([val_df, pd.DataFrame(running_val_rows)], ignore_index=True, sort=False)
    t_show = pd.concat([test_df, pd.DataFrame(running_test_rows)], ignore_index=True, sort=False)
    if not v_show.empty:
        v_show['_ord'] = v_show.apply(
            lambda r: order_idx.get((str(r['model']), str(r['backbone']), str(r['loss'])), 10**9),
            axis=1,
        )
        v_show = v_show.sort_values('_ord').drop(columns=['_ord']).reset_index(drop=True)
    if not t_show.empty:
        t_show['_ord'] = t_show.apply(
            lambda r: order_idx.get((str(r['model']), str(r['backbone']), str(r['loss'])), 10**9),
            axis=1,
        )
        t_show = t_show.sort_values('_ord').drop(columns=['_ord']).reset_index(drop=True)
    rs_mask_val = v_show['model'].isin(RS_MODELS)
    rs_mask_test = t_show['model'].isin(RS_MODELS)
    vfm_mask_val = v_show['model'].isin(VFM_MODELS)
    vfm_mask_test = t_show['model'].isin(VFM_MODELS)

    block: list[str] = []
    block.append('<!-- PER_CLASS_TEST_IOU_START -->')
    block.append('## Per-Class IoU Tables (Completed + Running)')
    block.append('')
    block.append('### Validation (best checkpoint)')
    block.append('')
    block.append(f'- Source: `{VAL_CSV}`')
    block.append(f'- Heatmap: `/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics/val_per_class_iou_completed_models_heatmap.png`')
    block.append('- Classes shown: 13 (GT-present in val)')
    block.append('- Running rows are placeholders (`-`) until eval artifacts are generated.')
    block.append('')
    block.append('#### General segmentation methods')
    block.append('')
    block.append(_md_table_with_placeholders(v_show[~rs_mask_val & ~vfm_mask_val], floatfmt='.4f'))
    block.append('')
    block.append('#### Remote sensing segmentation methods')
    block.append('')
    block.append(_md_table_with_placeholders(v_show[rs_mask_val], floatfmt='.4f'))
    block.append('')
    block.append('#### Remote sensing segmentation with VFM methods')
    block.append('')
    block.append(_md_table_with_placeholders(v_show[vfm_mask_val], floatfmt='.4f'))
    block.append('')
    block.append('### Test')
    block.append('')
    block.append(f'- Source: `{TEST_CSV}`')
    block.append(f'- Heatmap: `/deac/csc/yangGrp/cuij/GoldMDD/experiments/diagnostics/test_per_class_iou_completed_models_heatmap.png`')
    block.append('- Classes shown: 10 (GT-present in test; dropped absent classes: Heavy machinery, Compact mounds, Grass, Vehicles)')
    block.append('- Running rows are placeholders (`-`) until eval artifacts are generated.')
    block.append('')
    block.append('#### General segmentation methods')
    block.append('')
    block.append(_md_table_with_placeholders(t_show[~rs_mask_test & ~vfm_mask_test], floatfmt='.4f'))
    block.append('')
    block.append('#### Remote sensing segmentation methods')
    block.append('')
    block.append(_md_table_with_placeholders(t_show[rs_mask_test], floatfmt='.4f'))
    block.append('')
    block.append('#### Remote sensing segmentation with VFM methods')
    block.append('')
    block.append(_md_table_with_placeholders(t_show[vfm_mask_test], floatfmt='.4f'))
    block.append('')
    block.append('<!-- PER_CLASS_TEST_IOU_END -->')

    s_idx = lines.index('<!-- PER_CLASS_TEST_IOU_START -->')
    e_idx = lines.index('<!-- PER_CLASS_TEST_IOU_END -->')
    lines[s_idx:e_idx + 1] = block

    SUMMARY_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
