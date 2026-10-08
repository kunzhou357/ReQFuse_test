"""Main backbone: light source pyramids and a single Restormer fusion U-Net.

AMG-Fuse motivates separate modality/fusion roles; DSPFusion motivates fused
multi-scale skips. The two method modules live in restoration.py and quality.py.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (Downsample, LightConvResidualBlock, TransformerStage,
                     Upsample, pad_to_multiple, validate_pyramid)
from .restoration import SelectiveRestormer
from .quality import PostRestorationErrorHead, QualityMerge, WindowChannelCrossAttention, quality_from_error


MODEL_VERSION = "restormer_v2"

DEFAULT_MODEL_CONFIG = {
    "restorer": {"widths": [24, 48, 96], "heads": [1, 2, 4], "encoder_blocks": [1, 2, 2],
                 "decoder_blocks": [2, 1], "refinement_blocks": 1, "expansion": 2.66, "residual_bound": 1.0},
    "fusion": {"widths": [32, 64, 128, 256], "heads": [1, 2, 4, 8], "encoder_blocks": [1, 2, 2, 4],
               "decoder_blocks": [2, 2, 1], "refinement_blocks": 1, "expansion": 2.66},
    "state_hidden": 16, "quality_temperature": 0.1, "error_scale": 0.1, "window_size": 8,
    "exchange_scales": [2, 3], "checkpoint_blocks": False,
    "use_demand": True, "use_quality": True, "gate_private": True,
    "gate_messages": True, "local_exchange": True,
}


def model_config(overrides=None):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    for key, value in (overrides or {}).items():
        if key not in config:
            raise ValueError(f"Unknown model option: {key}")
        if isinstance(config[key], dict):
            unknown = set(value) - set(config[key])
            if unknown:
                raise ValueError(f"Unknown {key} options: {sorted(unknown)}")
            config[key].update(value)
        else:
            config[key] = value
    return config


class SourcePyramid(nn.Module):
    def __init__(self, image_channels, widths):
        super().__init__()
        self.stem = nn.Conv2d(image_channels, widths[0], 3, padding=1, bias=False)
        self.stages = nn.ModuleList(LightConvResidualBlock(c) for c in widths)
        self.down = nn.ModuleList(Downsample(c) for c in widths[:-1])

    def forward(self, image):
        x, features = self.stem(image), []
        for s, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if s < len(self.down):
                x = self.down[s](x)
        return features


class FusionBackbone(nn.Module):
    def __init__(self, widths, heads, encoder_blocks, decoder_blocks, refinement_blocks,
                 expansion, checkpoint_blocks=False):
        super().__init__()
        validate_pyramid(widths, heads, encoder_blocks, 4)
        if len(decoder_blocks) != 3 or any(n < 1 for n in decoder_blocks):
            raise ValueError("Fusion decoder_blocks must be three coarse-to-fine counts.")
        self.encoder = nn.ModuleList(TransformerStage(c, h, n, expansion, checkpoint_blocks)
                                     for c, h, n in zip(widths, heads, encoder_blocks))
        self.down = nn.ModuleList(Downsample(c) for c in widths[:-1])
        self.up = nn.ModuleList(Upsample(c) for c in reversed(widths[1:]))
        self.reduce = nn.ModuleList(nn.Conv2d(2*c, c, 1, bias=False) for c in reversed(widths[:-1]))
        self.decoder = nn.ModuleList(TransformerStage(widths[s], heads[s], n, expansion, checkpoint_blocks)
                                    for s, n in zip((2, 1, 0), decoder_blocks))
        self.refinement = TransformerStage(widths[0], heads[0], refinement_blocks, expansion, checkpoint_blocks)
        self.output = nn.Conv2d(widths[0], 3, 3, padding=1)

    def forward(self, merges):
        skips, x = [], merges[0]
        for s, stage in enumerate(self.encoder):
            if s:
                x = self.down[s-1](x) + merges[s]
            x = stage(x)
            skips.append(x)
        for j, s in enumerate((2, 1, 0)):
            x = self.decoder[j](self.reduce[j](torch.cat((self.up[j](x), skips[s]), 1)))
        return self.output(self.refinement(x)).sigmoid()


class RobustFusionNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = cfg = model_config(config)
        r, f = cfg["restorer"], cfg["fusion"]
        validate_pyramid(f["widths"], f["heads"], f["encoder_blocks"], 4)
        if len(set(cfg["exchange_scales"])) != len(cfg["exchange_scales"]) or any(s not in (0,1,2,3) for s in cfg["exchange_scales"]):
            raise ValueError("exchange_scales must be unique pyramid indices 0..3.")
        args = dict(r, state_hidden=cfg["state_hidden"], use_demand=cfg["use_demand"], checkpoint_blocks=cfg["checkpoint_blocks"])
        self.restorer_visible, self.restorer_infrared = SelectiveRestormer(3, **args), SelectiveRestormer(1, **args)
        self.error_visible = PostRestorationErrorHead(3, r["widths"][0], cfg["state_hidden"], error_scale=cfg["error_scale"])
        self.error_infrared = PostRestorationErrorHead(1, r["widths"][0], cfg["state_hidden"], error_scale=cfg["error_scale"])
        self.pyramid_visible, self.pyramid_infrared = SourcePyramid(3, f["widths"]), SourcePyramid(1, f["widths"])
        self.cross_i_to_v, self.cross_v_to_i = nn.ModuleDict(), nn.ModuleDict()
        for s in cfg["exchange_scales"]:
            args = (f["widths"][s], f["heads"][s], cfg["window_size"], cfg["local_exchange"])
            self.cross_i_to_v[str(s)], self.cross_v_to_i[str(s)] = WindowChannelCrossAttention(*args), WindowChannelCrossAttention(*args)
        self.merges = nn.ModuleList(QualityMerge(c, s in cfg["exchange_scales"]) for s,c in enumerate(f["widths"]))
        self.backbone = FusionBackbone(**f, checkpoint_blocks=cfg["checkpoint_blocks"])
        # Validate the temperature even before the first forward.
        quality_from_error(torch.zeros(1), cfg["quality_temperature"])

    def forward(self, visible, infrared, return_aux=True, quality_strength=1.0,
                restoration_only=False, demand_strength=1.0):
        if (visible.ndim != 4 or infrared.ndim != 4 or visible.shape[1] != 3 or infrared.shape[1] != 1
                or visible.shape[0] != infrared.shape[0] or visible.shape[-2:] != infrared.shape[-2:]
                or any(n < 1 for n in visible.shape) or visible.device != infrared.device
                or not visible.is_floating_point() or not infrared.is_floating_point()):
            raise ValueError("Expected matching floating RGB visible and gray infrared NCHW tensors.")
        if not 0 <= quality_strength <= 1:
            raise ValueError("quality_strength must be within [0,1].")
        if not 0 <= demand_strength <= 1:
            raise ValueError("demand_strength must be within [0,1].")
        if restoration_only and not return_aux:
            raise ValueError("restoration_only requires return_aux=True.")
        h, w = visible.shape[-2:]
        visible, infrared = pad_to_multiple(visible), pad_to_multiple(infrared)
        rv, ri = self.restorer_visible(visible, demand_strength), self.restorer_infrared(infrared, demand_strength)
        sv = self.error_visible(rv["restored"], rv["restored"]-visible, rv["demand"], rv["feature"])
        si = self.error_infrared(ri["restored"], ri["restored"]-infrared, ri["demand"], ri["feature"])
        ev, ei = sv["error"], si["error"]
        qv, qi = quality_from_error(ev, self.config["quality_temperature"]), quality_from_error(ei, self.config["quality_temperature"])
        outputs = {key: {"visible": pair[0][..., :h, :w], "infrared": pair[1][..., :h, :w]} for key,pair in
                   {"restored": (rv["restored"],ri["restored"]), "demand": (rv["demand"],ri["demand"]),
                    "demand_logits": (rv["demand_logits"],ri["demand_logits"]),
                    "applied_demand": (rv["applied_demand"],ri["applied_demand"]),
                    "residual_logits": (rv["residual_logits"],ri["residual_logits"]),
                    "error_raw": (sv["raw"],si["raw"]),
                    "error": (ev,ei), "quality": (qv,qi)}.items()}
        if return_aux:
            with torch.no_grad():
                outputs["head_feature_std"] = {m: r["feature"][..., :h, :w].float().std(unbiased=False)
                                               for m,r in (("visible",rv),("infrared",ri))}
        if restoration_only:
            return outputs
        hv, hi = self.pyramid_visible(rv["restored"]), self.pyramid_infrared(ri["restored"])
        if not self.config["use_quality"]:
            quality_strength = 0.0
        qv, qi = ((1-quality_strength)+quality_strength*q.detach() for q in (qv,qi))
        merges = []
        for s in range(4):
            qvs, qis = (F.interpolate(q, size=hv[s].shape[-2:], mode="area") for q in (qv,qi))
            messages = ()
            if s in self.config["exchange_scales"]:
                qs_v = qvs if self.config["gate_messages"] else torch.ones_like(qvs)
                qs_i = qis if self.config["gate_messages"] else torch.ones_like(qis)
                messages = (self.cross_i_to_v[str(s)](hv[s], hi[s], qs_i), self.cross_v_to_i[str(s)](hi[s], hv[s], qs_v))
            if not self.config["gate_private"]:
                qvs, qis = torch.ones_like(qvs), torch.ones_like(qis)
            merges.append(self.merges[s](hv[s], hi[s], qvs, qis, messages))
        fused = self.backbone(merges)[..., :h, :w]
        if not return_aux:
            return fused
        outputs["fused"] = fused
        return outputs
