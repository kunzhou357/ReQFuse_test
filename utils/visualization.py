"""Training curves and labeled image comparisons; no GUI or torchvision needed.

CalibFuse motivates the progress/preview workflow. Comparisons add the two
method modules' states so recovery and quality control can be inspected.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .loss import to_luminance


def _pil(image, heatmap=False):
    image = image.detach().float().cpu()
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("Comparison tiles require CHW RGB or gray tensors.")
    array = image.clamp(0, 1).permute(1, 2, 0).numpy()
    if heatmap:
        # Fixed [0,1] color scale for all samples/steps; no per-image stretching.
        colors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                           [94, 201, 98], [253, 231, 37]], dtype=np.float32)
        values = array[..., 0] * (len(colors)-1)
        lower = np.minimum(values.astype(np.int32), len(colors)-2)
        fraction = (values-lower)[..., None]
        array = colors[lower]*(1-fraction) + colors[lower+1]*fraction
    else:
        array = array * 255
        if array.shape[-1] == 1:
            array = array[..., 0]
    return Image.fromarray(array.round().astype(np.uint8)).convert("RGB")


def save_comparison(batch, outputs, path, max_samples=2, title="",
                    tile_width=192, tile_height=144, error_max=0.2, reference_errors=None):
    """Save labeled LQ/restored/fused and state-map rows at fixed display scales.

    The overview preserves aspect ratios with letterboxing. Native-resolution
    test outputs are saved separately. Warmup has no invented fusion output.
    """
    if max_samples < 1 or tile_width < 64 or tile_height < 32 or error_max <= 0:
        raise ValueError("Invalid comparison size or error display range.")
    count = min(max_samples, batch["visible"].shape[0])
    font = ImageFont.load_default()
    rows = []
    for index in range(count):
        ids = batch.get("sample_id", [])
        sample_id = ids[index] if isinstance(ids, (list, tuple)) else ids
        rows.append((f"{title} | {sample_id}", None))
        fused = outputs.get("fused")
        tiles = [("Input VI", batch["visible"][index], False),
                 ("Input IR", batch["infrared"][index], False),
                 ("Restored VI", outputs["restored"]["visible"][index], False),
                 ("Restored IR", outputs["restored"]["infrared"][index], False),
                 ("Fused RGB" if fused is not None else "Fusion starts after warmup",
                  fused[index] if fused is not None else None, False),
                 ("Fused gray", to_luminance(fused[index:index+1])[0] if fused is not None else None, False)]
        rows.append((None, tiles))
        states = []
        for kind in ("demand", "error", "quality"):
            for modality, label in (("visible", "VI"), ("infrared", "IR")):
                value = outputs[kind][modality][index]
                scale = error_max if kind == "error" else 1.0
                states.append((f"{kind.capitalize()} {label} [0,{scale:g}]", value/scale, True))
        rows.append((None, states))
        applied = [(f"Applied demand {label} [0,1]", outputs["applied_demand"][m][index], True)
                   for m,label in (("visible", "VI"), ("infrared", "IR"))]
        if reference_errors is not None:
            applied.extend((f"HQ error before {label} [0,{error_max:g}]",
                            reference_errors["input"][m][index]/error_max, True)
                           for m,label in (("visible", "VI"), ("infrared", "IR")))
        applied.extend((f"Residual |raw| > 3 {label}",
                        (outputs["residual_logits"][m][index].abs() > 3).float().mean(0, keepdim=True), True)
                       for m,label in (("visible", "VI"), ("infrared", "IR")))
        rows.append((None, applied))
        if "hq_visible" in batch:
            references = [("Reference VI", batch["hq_visible"][index], False),
                          ("Reference IR", batch["hq_infrared"][index], False)]
            for modality, label in (("visible", "VI"), ("infrared", "IR")):
                if f"mask_{modality}" in batch:
                    references.append((f"Degradation mask {label}", batch[f"mask_{modality}"][index], False))
            if reference_errors is not None:
                references.extend((f"HQ error after {label} [0,{error_max:g}]",
                                   reference_errors["restored"][m][index]/error_max, True)
                                  for m,label in (("visible", "VI"), ("infrared", "IR")))
            rows.append((None, references))
    label_height, gap, columns = 25, 6, 6
    width = columns*(tile_width+gap)+gap
    height = sum(28 if header is not None else tile_height+label_height+gap for header, _ in rows)+gap
    canvas = Image.new("RGB", (width, height), (24, 27, 32))
    draw = ImageDraw.Draw(canvas)
    top = gap
    for header, tiles in rows:
        if header is not None:
            draw.text((gap, top+4), header, font=font, fill=(240, 240, 240))
            top += 28
            continue
        for column, (label, tensor, heatmap) in enumerate(tiles):
            left = gap+column*(tile_width+gap)
            draw.text((left+3, top+4), label, font=font, fill=(225, 230, 236))
            draw.rectangle((left, top+label_height, left+tile_width-1,
                            top+label_height+tile_height-1), fill=(44, 48, 55))
            if tensor is not None:
                tile = _pil(tensor, heatmap)
                scale = min(tile_width/tile.width, tile_height/tile.height)
                tile = tile.resize((max(1,round(tile.width*scale)), max(1,round(tile.height*scale))),
                                   Image.Resampling.LANCZOS)
                canvas.paste(tile, (left+(tile_width-tile.width)//2,
                                   top+label_height+(tile_height-tile.height)//2))
        top += tile_height+label_height+gap
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_loss_curves(log_path, path):
    """Plot logged optimizer updates, including records written before resume."""
    log_path, path = Path(log_path), Path(path)
    if not log_path.is_file():
        return
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep plotting caches in the experiment output (tests output for smoke).
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent/".matplotlib"))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    steps = [row["step"] for row in records]
    groups = [(axes[0, 0], "Weighted losses", ("loss", "fusion_loss", "auxiliary_loss")),
              (axes[0, 1], "Fusion components (before weights)",
               ("fusion_intensity", "fusion_gradient", "fusion_ssim", "fusion_color")),
              (axes[1, 0], "Auxiliary components (before weights)",
               ("aux_reconstruction", "aux_keep", "aux_demand", "aux_error"))]
    for axis, title, names in groups:
        for name in names:
            if name in records[0]:
                axis.plot(steps, [row[name] for row in records], label=name, linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Optimizer step (logged samples)")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    schedule = axes[1, 1]
    schedule.plot(steps, [row["lr"] for row in records], color="tab:blue", label="Learning rate")
    schedule.set_ylabel("Learning rate", color="tab:blue")
    strength = schedule.twinx()
    strength.plot(steps, [row["quality_strength"] for row in records], color="tab:orange", label="Joint strength")
    strength.set_ylabel("Fusion/quality/demand strength", color="tab:orange")
    strength.set_ylim(-.05, 1.05)
    schedule.set_title("Training schedule")
    schedule.set_xlabel("Optimizer step (logged samples)")
    schedule.grid(alpha=.25)
    figure.savefig(path, dpi=140)
    figure.clear()


def save_state_curves(log_path, path):
    """Expose collapsed heads and residual saturation before a full run ends."""
    log_path, path = Path(log_path), Path(path)
    if not log_path.is_file():
        return
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent/".matplotlib"))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(16, 9), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    steps = [row["step"] for row in records]
    groups = [
        ("Restoration MAE (fixed training samples)", ("input_mae", "restored_mae")),
        ("Predicted / target / applied demand", ("demand_mean", "demand_target_mean", "applied_demand_mean")),
        ("Predicted / actual remaining error", ("error_mean", "error_target_mean")),
        ("Quality / residual saturation fraction", ("quality_mean", "residual_saturated_fraction")),
        ("Normalized head feature standard deviation", ("head_feature_std",)),
        ("Head gradient norms (unscaled, before clipping)", ("demand_grad_norm", "residual_grad_norm", "error_grad_norm")),
    ]
    for index, (axis, (title, suffixes)) in enumerate(zip(axes.flat, groups)):
        for label in ("vi", "ir"):
            for suffix in suffixes:
                key = f"{label}_{suffix}"
                if key not in records[0]:
                    continue
                values = [row[key] for row in records]
                if index == 5:
                    axis.semilogy(steps, [max(value, 1e-12) for value in values], label=key, linewidth=1.3)
                else:
                    axis.plot(steps, values, label=key, linewidth=1.3)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    figure.savefig(path, dpi=140)
    figure.clear()
