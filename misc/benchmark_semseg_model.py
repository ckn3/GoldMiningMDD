#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from socket import gethostname
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter


INPUT_SIZE = 512


def count_params(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        if isinstance(p, UninitializedParameter):
            continue
        try:
            total += int(p.numel())
        except (RuntimeError, ValueError):
            continue
    return total / 1e6


def count_trainable_params(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, UninitializedParameter):
            continue
        try:
            total += int(p.numel())
        except (RuntimeError, ValueError):
            continue
    return total / 1e6


def materialize_lazy_params(model: nn.Module, runner, device: torch.device) -> None:
    model.eval()
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        _ = runner(model, x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def benchmark_latency(model: nn.Module, runner, device: torch.device, warmup: int = 10, iters: int = 30) -> tuple[float, float]:
    model.eval()
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = runner(model, x)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = runner(model, x)
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / iters
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return latency_ms, peak_gb


def flop_gflops(model: nn.Module, flop_runner, device: torch.device) -> tuple[float, float, dict[str, int]]:
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(flop_runner(model), (x,))
        total = flops.total()
        unsupported = {k: int(v) for k, v in flops.unsupported_ops().items()}
    # fvcore totals are commonly interpreted as MAC-style counts for conv layers.
    # Report both conventions: GMACs and GFLOPs (= 2 * GMACs).
    gmacs = total / 1e9
    gflops = 2.0 * gmacs
    return gflops, gmacs, unsupported


def flop_gflops_mmcv_fallback(model: nn.Module) -> tuple[float, float]:
    from mmcv.cnn import get_model_complexity_info

    original_forward = None
    if hasattr(model, "forward_dummy"):
        original_forward = model.forward
        model.forward = model.forward_dummy
    try:
        flops, _params = get_model_complexity_info(
            model,
            (3, INPUT_SIZE, INPUT_SIZE),
            as_strings=False,
            print_per_layer_stat=False,
        )
    finally:
        if original_forward is not None:
            model.forward = original_forward
    gflops = float(flops) / 1e9
    gmacs = gflops / 2.0
    return gflops, gmacs


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so decorators/dataclasses can resolve __module__.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_smp(arch: str, encoder: str, encoder_weights: str = "imagenet"):
    mod = load_module_from_path("train_semseg_smp_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_smp.py"))
    model = mod.build_model(arch, encoder, encoder_weights)

    def runner(m, x):
        out = m(x)
        if isinstance(out, dict):
            out = out["out"]
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, dict):
                out = out["out"]
            return out

    family_name = f"SMP-{arch}/tu-{encoder}"
    return model, runner, lambda m: FlopWrap(m), family_name


def build_segformer():
    mod = load_module_from_path("train_semseg_segformer_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_segformer.py"))
    model = mod.build_model("nvidia/segformer-b2-finetuned-ade-512-512", True)

    def runner(m, x):
        return mod.forward_logits(m, x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(pixel_values=x).logits
            if out.shape[-2:] != x.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return out

    return model, runner, lambda m: FlopWrap(m), "SegFormer-B2"


def build_efficientvit():
    mod = load_module_from_path("train_semseg_evit_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_efficientvit.py"))
    repo = Path("/deac/csc/yangGrp/cuij/third_party/efficientvit")
    model = mod.build_model("efficientvit-seg-b2-ade20k", True, repo)

    def runner(m, x):
        return mod.forward_logits(m, x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if out.shape[-2:] != x.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return out

    return model, runner, lambda m: FlopWrap(m), "EfficientViT-Seg-B2"


def build_farseg():
    mod = load_module_from_path("train_semseg_farseg_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_farseg.py"))
    ns = argparse.Namespace(
        model="farseg",
        farseg_repo=Path("/deac/csc/yangGrp/cuij/third_party/FarSeg"),
        pretrained=True,
        farsegpp_backbone="mit_b2",
    )
    model = mod.build_model(ns)

    def runner(m, x):
        return m(x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    return model, runner, lambda m: FlopWrap(m), "FarSeg-R50"


def build_farsegpp_mitb2():
    mod = load_module_from_path("train_semseg_farseg_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_farseg.py"))
    ns = argparse.Namespace(
        model="farsegpp",
        farseg_repo=Path("/deac/csc/yangGrp/cuij/third_party/FarSeg"),
        pretrained=True,
        farsegpp_backbone="mit_b2",
    )
    model = mod.build_model(ns)

    def runner(m, x):
        return m(x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    return model, runner, lambda m: FlopWrap(m), "FarSegPP-MiT-B2"


def build_rsseg_logcan():
    repo = Path("/deac/csc/yangGrp/cuij/third_party/rssegmentation")
    sys.path.insert(0, str(repo))
    try:
        from utils.config import Config
        from rsseg.models.build_model import build_model
    finally:
        sys.path.pop(0)

    cfg = Config.fromfile(str(repo / "configs/goldmdd/logcan_r50_goldmdd.py"))
    model = build_model(cfg.model_config)

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            # Keep a deterministic tensor output if model returns a dict.
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), "LoGCAN-R50"


def build_rsseg_logcanplus():
    repo = Path("/deac/csc/yangGrp/cuij/third_party/rssegmentation")
    sys.path.insert(0, str(repo))
    try:
        from utils.config import Config
        from rsseg.models.build_model import build_model
    finally:
        sys.path.pop(0)

    cfg = Config.fromfile(str(repo / "configs/goldmdd/logcanplus_repvitm23_goldmdd.py"))
    model = build_model(cfg.model_config)

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), "LoGCAN++-RepViT-M2.3"


def build_rsseg_docnet():
    repo = Path("/deac/csc/yangGrp/cuij/third_party/rssegmentation")
    sys.path.insert(0, str(repo))
    try:
        from utils.config import Config
        from rsseg.models.build_model import build_model
    finally:
        sys.path.pop(0)

    cfg = Config.fromfile(str(repo / "configs/goldmdd/docnet_hrnetw32_goldmdd.py"))
    model = build_model(cfg.model_config)

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), "DOCNet-HRNet-W32"


def build_rsseg_sacanet():
    repo = Path("/deac/csc/yangGrp/cuij/third_party/rssegmentation")
    sys.path.insert(0, str(repo))
    try:
        from utils.config import Config
        from rsseg.models.build_model import build_model
    finally:
        sys.path.pop(0)

    cfg = Config.fromfile(str(repo / "configs/goldmdd/sacanet_hrnetw32_goldmdd.py"))
    model = build_model(cfg.model_config)

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), "SACANet-HRNet-W32"


def build_detectron2_family(repo_root: Path, config_file: Path, module_name: str, family_name: str):
    sys.path.insert(0, str(repo_root))
    try:
        mod = load_module_from_path(module_name, repo_root / "train_net.py")
        args = argparse.Namespace(
            config_file=str(config_file),
            opts=["OUTPUT_DIR", "/tmp/ignore_eval_output"],
            eval_only=False,
            resume=False,
            num_gpus=1,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
        )
        cfg = mod.setup(args)
        model = mod.Trainer.build_model(cfg)
    finally:
        sys.path.pop(0)

    def runner(m, x):
        inp = [{"image": x[0], "height": INPUT_SIZE, "width": INPUT_SIZE}]
        return m(inp)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            inp = [{"image": x[0], "height": INPUT_SIZE, "width": INPUT_SIZE}]
            out = self.inner(inp)
            if isinstance(out, list):
                if out and isinstance(out[0], dict) and "sem_seg" in out[0]:
                    return out[0]["sem_seg"].unsqueeze(0)
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            if isinstance(out, dict) and "sem_seg" in out:
                return out["sem_seg"].unsqueeze(0)
            return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)

    return model, runner, lambda m: FlopWrap(m), family_name


def build_pem():
    os.environ.setdefault("GOLDMDD_PEM_ROOT", "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-pem")
    os.environ.setdefault("GOLDMDD_DATA_CROPPED_ROOT", "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped")
    return build_detectron2_family(
        Path("/deac/csc/yangGrp/cuij/third_party/PEM"),
        Path("/deac/csc/yangGrp/cuij/third_party/PEM/configs/goldmdd/semantic-segmentation/pem_R50_bs8_658k.yaml"),
        "pem_train_mod",
        "PEM",
    )


def build_mask2former():
    os.environ.setdefault("GOLDMDD_PEM_ROOT", "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped-pem")
    os.environ.setdefault("GOLDMDD_DATA_CROPPED_ROOT", "/deac/csc/yangGrp/cuij/GoldMDD/data-cropped")
    return build_detectron2_family(
        Path("/deac/csc/yangGrp/cuij/third_party/Mask2Former"),
        Path("/deac/csc/yangGrp/cuij/third_party/Mask2Former/configs/goldmdd/semantic-segmentation/mask2former_R50_bs8_658k.yaml"),
        "m2f_train_mod",
        "Mask2Former-R50",
    )


def build_ssaseg(config_path: Path, family_name: str):
    # SSA-Seg custom modules are registered through repo-local `models`.
    repo = Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg")
    sys.path.insert(0, str(repo))
    try:
        import models  # noqa: F401
        from mmcv import Config
        from mmseg.models import build_segmentor
    finally:
        sys.path.pop(0)

    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))

    def runner(m, x):
        meta = [{
            "img_shape": (INPUT_SIZE, INPUT_SIZE, 3),
            "ori_shape": (INPUT_SIZE, INPUT_SIZE, 3),
            "pad_shape": (INPUT_SIZE, INPUT_SIZE, 3),
            "scale_factor": 1.0,
            "flip": False,
        }]
        out = m.encode_decode(x, meta)
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            meta = [{
                "img_shape": (INPUT_SIZE, INPUT_SIZE, 3),
                "ori_shape": (INPUT_SIZE, INPUT_SIZE, 3),
                "pad_shape": (INPUT_SIZE, INPUT_SIZE, 3),
                "scale_factor": 1.0,
                "flip": False,
            }]
            return self.inner.encode_decode(x, meta)

    return model, runner, lambda m: FlopWrap(m), family_name


def build_geoseg(config_path: Path, family_name: str):
    repo = Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg")
    sys.path.insert(0, str(repo))
    try:
        from tools.cfg import py2cfg  # type: ignore
        cfg = py2cfg(config_path)
        model = cfg.net
    finally:
        sys.path.pop(0)

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), family_name


def build_ppmambaseg():
    repo = Path("/deac/csc/yangGrp/cuij/third_party/PPMambaSeg/GeoSeg")
    sys.path.insert(0, str(repo))
    try:
        from tools.cfg import py2cfg  # type: ignore
        cfg = py2cfg(str(repo / "config/goldmdd/ppmamba.py"))
        model = cfg.net
    finally:
        sys.path.pop(0)

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), "PPMambaSeg-ResNet18"


def build_rs3mamba():
    mod = load_module_from_path("train_semseg_rs3mamba_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_rs3mamba.py"))
    model = mod.build_model(
        Path("/deac/csc/yangGrp/cuij/third_party/SSRS/RS3Mamba"),
        Path("/deac/csc/yangGrp/cuij/third_party/SSRS/RS3Mamba/pretrain/vmamba_tiny_e292.pth"),
    )

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), "RS3Mamba-VMambaTiny"


def build_mfmamba():
    mod = load_module_from_path("train_semseg_mfmamba_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_mfmamba.py"))
    model = mod.build_model(
        Path("/deac/csc/yangGrp/cuij/third_party/MF-Mamba"),
        None,
    )

    def runner(m, x):
        out = m(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return out

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        return v
                return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=x.device)
            return out

    return model, runner, lambda m: FlopWrap(m), "MF-Mamba-HRNetW18"


def build_mcpnet():
    repo = Path("/deac/csc/yangGrp/cuij/third_party/MCPNet")
    sys.path.insert(0, str(repo))
    try:
        from mmengine.config import Config
        from mmseg.utils import register_all_modules
        from mmseg.models import build_segmentor

        register_all_modules(init_default_scope=False)
        cfg = Config.fromfile(
            str(
                Path(
                    "/deac/csc/yangGrp/cuij/third_party/MCPNet/configs/goldmdd_mcpnet_full_80ep_bs8_ce.py"
                )
            )
        )
        model_cfg = cfg.model.copy()
        # Keep benchmarking independent from mmengine data preprocessor registry.
        model_cfg.pop("data_preprocessor", None)
        model = build_segmentor(model_cfg)
    finally:
        sys.path.pop(0)

    def runner(m, x):
        meta = [{
            "img_shape": (INPUT_SIZE, INPUT_SIZE, 3),
            "ori_shape": (INPUT_SIZE, INPUT_SIZE, 3),
            "pad_shape": (INPUT_SIZE, INPUT_SIZE, 3),
            "scale_factor": 1.0,
            "flip": False,
        }]
        return m.encode_decode(x, meta)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            meta = [{
                "img_shape": (INPUT_SIZE, INPUT_SIZE, 3),
                "ori_shape": (INPUT_SIZE, INPUT_SIZE, 3),
                "pad_shape": (INPUT_SIZE, INPUT_SIZE, 3),
                "scale_factor": 1.0,
                "flip": False,
            }]
            return self.inner.encode_decode(x, meta)

    return model, runner, lambda m: FlopWrap(m), "MCPNet-R50"


def build_sam_rs(model_name: str):
    mod = load_module_from_path("train_semseg_samrs_mod", Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_sam_rs.py"))
    model = mod.build_model(Path("/deac/csc/yangGrp/cuij/third_party/SSRS/SAM_RS"), model_name)

    def runner(m, x):
        return mod.forward_logits(m, x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return mod.forward_logits(self.inner, x)

    return model, runner, lambda m: FlopWrap(m), f"SAM_RS-{model_name}"


def build_sam2_1_hierabplus():
    mod = load_module_from_path(
        "train_semseg_samfam_mod",
        Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_sam_family.py"),
    )
    ns = argparse.Namespace(
        family="sam2_1",
        decoder_type="fpn_multiscale",
        decoder_channels=256,
        sam2_root=Path("/deac/csc/yangGrp/cuij/third_party/sam2"),
        sam2_config="configs/sam2.1/sam2.1_hiera_b+.yaml",
        sam2_checkpoint=None,
        sam2_image_size=512,
        sam3_root=Path("/deac/csc/yangGrp/cuij/third_party/sam3"),
        sam3_checkpoint=None,
        sam3_load_from_hf=False,
        sam3_image_size=1008,
        hqsam_root=Path("/deac/csc/yangGrp/cuij/third_party/sam-hq"),
        hqsam_model_type="vit_b",
        hqsam_checkpoint=None,
    )
    model = mod.SAMFamilySemantic(ns)

    def runner(m, x):
        return m(x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    return model, runner, lambda m: FlopWrap(m), "SAM2.1-Hiera-B+-FPN"


def build_hq_sam_vit_b():
    mod = load_module_from_path(
        "train_semseg_sam_family_mod",
        Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_sam_family.py"),
    )
    args = argparse.Namespace(
        family="hq_sam",
        hqsam_root=Path("/deac/csc/yangGrp/cuij/third_party/sam-hq"),
        hqsam_model_type="vit_b",
        hqsam_checkpoint=None,
        decoder_type="fpn_multiscale",
        decoder_channels=256,
    )
    model = mod.SAMFamilySemantic(args)

    def runner(m, x):
        return m(x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    return model, runner, lambda m: FlopWrap(m), "HQ-SAM-ViT-B+FPN"


def build_rsamseg_vit_b():
    mod = load_module_from_path(
        "train_semseg_rsamseg_mod",
        Path("/deac/csc/yangGrp/cuij/GoldMDD/misc/train_semseg_rsamseg.py"),
    )
    encoder_cfg = mod._build_encoder_cfg("vit_b", img_size=512)
    model = mod.RSAMSegSemantic(
        Path("/deac/csc/alqahtaniGrp/cuij/third_party/RSAM-Seg"),
        encoder_cfg,
        num_classes=14,
    )

    def runner(m, x):
        return m(x)

    class FlopWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    return model, runner, lambda m: FlopWrap(m), "RSAM-Seg-ViT-B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family",
        required=True,
        choices=[
            "smp",
            "segformer",
            "efficientvit",
            "farseg_r50",
            "farsegpp_mitb2",
            "rsseg_logcan_r50",
            "rsseg_logcanplus_repvitm23",
            "rsseg_docnet_hrnetw32",
            "rsseg_sacanet_hrnetw32",
            "pem",
            "mask2former",
            "ssaseg_afformer_base",
            "ssaseg_seaformer_base",
            "ssaseg_segnext_tiny",
            "ssaseg_cgrseg_b",
            "ssaseg_upernet_swin_tiny",
            "ssaseg_ocrnet_hr48",
            "geoseg_unetformer",
            "geoseg_a2fpn",
            "geoseg_abcnet",
            "geoseg_banet",
            "geoseg_manet",
            "geoseg_dcswin",
            "geoseg_pyramidmamba",
            "ppmambaseg_ppmamba",
            "rs3mamba",
            "mfmamba",
            "mcpnet",
            "sam_rs",
            "sam2_1_hierabplus",
            "hq_sam_vit_b",
            "rsamseg_vit_b",
        ],
    )
    p.add_argument("--smp-arch", default="deeplabv3plus", choices=["deeplabv3plus", "fpn", "unet"])
    p.add_argument("--smp-encoder", default="convnext_tiny")
    p.add_argument("--smp-encoder-weights", default="imagenet")
    p.add_argument("--samrs-model", default="unetformer", choices=["unetformer", "ftunetformer", "abcnet", "cmtfnet"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-flops", action="store_true", help="Skip FLOPs/GMACs computation; keep/update latency only.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    builders = {
        "smp": lambda: build_smp(args.smp_arch, args.smp_encoder, args.smp_encoder_weights),
        "segformer": build_segformer,
        "efficientvit": build_efficientvit,
        "farseg_r50": build_farseg,
        "farsegpp_mitb2": build_farsegpp_mitb2,
        "rsseg_logcan_r50": build_rsseg_logcan,
        "rsseg_logcanplus_repvitm23": build_rsseg_logcanplus,
        "rsseg_docnet_hrnetw32": build_rsseg_docnet,
        "rsseg_sacanet_hrnetw32": build_rsseg_sacanet,
        "pem": build_pem,
        "mask2former": build_mask2former,
        "ssaseg_afformer_base": lambda: build_ssaseg(
            Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/afformer_base_goldmdd.py"),
            "Afformer-Base",
        ),
        "ssaseg_seaformer_base": lambda: build_ssaseg(
            Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/seaformer_base_goldmdd.py"),
            "SeaFormer-Base",
        ),
        "ssaseg_segnext_tiny": lambda: build_ssaseg(
            Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/segnext_tiny_goldmdd.py"),
            "SegNeXt-Tiny",
        ),
        "ssaseg_cgrseg_b": lambda: build_ssaseg(
            Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/cgrseg_b_goldmdd.py"),
            "CGRSeg-B",
        ),
        "ssaseg_upernet_swin_tiny": lambda: build_ssaseg(
            Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/upernet_swin_tiny_goldmdd.py"),
            "UPerNet-Swin-Tiny",
        ),
        "ssaseg_ocrnet_hr48": lambda: build_ssaseg(
            Path("/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/ocrnet_hr48_goldmdd.py"),
            "OCRNet-HR48",
        ),
        "geoseg_unetformer": lambda: build_geoseg(
            Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg/config/goldmdd/unetformer.py"),
            "UNetFormer-R18",
        ),
        "geoseg_a2fpn": lambda: build_geoseg(
            Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg/config/goldmdd/a2fpn.py"),
            "A2FPN-R18",
        ),
        "geoseg_abcnet": lambda: build_geoseg(
            Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg/config/goldmdd/abcnet.py"),
            "ABCNet-R18",
        ),
        "geoseg_banet": lambda: build_geoseg(
            Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg/config/goldmdd/banet.py"),
            "BANet-ResT-Lite",
        ),
        "geoseg_manet": lambda: build_geoseg(
            Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg/config/goldmdd/manet.py"),
            "MANet-R50",
        ),
        "geoseg_dcswin": lambda: build_geoseg(
            Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg/config/goldmdd/dcswin.py"),
            "DC-Swin-Small",
        ),
        "geoseg_pyramidmamba": lambda: build_geoseg(
            Path("/deac/csc/yangGrp/cuij/third_party/GeoSeg/config/goldmdd/pyramidmamba.py"),
            "PyramidMamba-SwinBase",
        ),
        "ppmambaseg_ppmamba": build_ppmambaseg,
        "rs3mamba": build_rs3mamba,
        "mfmamba": build_mfmamba,
        "mcpnet": build_mcpnet,
        "sam_rs": lambda: build_sam_rs(args.samrs_model),
        "sam2_1_hierabplus": build_sam2_1_hierabplus,
        "hq_sam_vit_b": build_hq_sam_vit_b,
        "rsamseg_vit_b": build_rsamseg_vit_b,
    }
    model, runner, flop_wrap, family_name = builders[args.family]()
    device = torch.device(args.device)
    model = model.to(device)
    materialize_lazy_params(model, runner, device)

    params_m = count_params(model)
    latency_ms, peak_gb = benchmark_latency(model, runner, device)
    gflops = float("nan")
    gmacs = float("nan")
    unsupported = {}
    if args.skip_flops:
        unsupported = {"flop_error": "skipped_by_flag"}
    else:
        try:
            gflops, gmacs, unsupported = flop_gflops(model, flop_wrap, device)
        except Exception as e:
            try:
                gflops, gmacs = flop_gflops_mmcv_fallback(model)
                unsupported = {"flop_error": f"fvcore_failed_fallback_mmcv: {str(e)[:240]}"}
            except Exception as e2:
                unsupported = {"flop_error": f"{str(e2)[:240]}"}

    payload = {
        "family": family_name,
        "params_m": params_m,
        "trainable_params_m": count_trainable_params(model),
        "gflops": gflops,
        "gmacs": gmacs,
        "latency_ms_1x3x512x512": latency_ms,
        "peak_vram_gb": peak_gb,
        "flop_unsupported_ops": unsupported,
        "device": str(device),
        "hostname": gethostname(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_flops and out.exists():
        try:
            prev = json.loads(out.read_text())
            for k in ("gflops", "gmacs"):
                if k in prev:
                    payload[k] = prev[k]
        except Exception:
            pass
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
