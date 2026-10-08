"""Image IO and explicitly named evaluation metrics; RGB inside the project."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .loss import ssim, to_luminance


def read_image(path, channels=3):
    if channels not in (1,3):
        raise ValueError("read_image supports RGB (3) or grayscale (1).")
    with Image.open(path) as image:
        array = np.array(image.convert("RGB" if channels == 3 else "L"), dtype=np.float32, copy=True) / 255.0
    if channels == 1:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(image, path):
    image = image.detach().float().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("save_image expects one image, not a batch.")
        image = image[0]
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("save_image expects CHW RGB or gray.")
    array = image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    if image.shape[0] == 1:
        array = array[..., 0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def reconstruction_metrics(restored, reference):
    """Same-modality reference metrics, never fusion PSNR against a single source."""
    restored, reference = restored.float(), reference.float()
    mse = (restored-reference).square().mean()
    return {"psnr": float((-10*torch.log10(mse.clamp_min(1e-12))).item()),
            "ssim": float(ssim(restored, reference).item()),
            "mae": float((restored-reference).abs().mean().item())}


def fusion_statistics(fused):
    """Descriptive fused-image statistics, not substitutes for fidelity metrics."""
    y = to_luminance(fused).detach().float().clamp(0, 1)
    hist = torch.histc(y.cpu(), bins=256, min=0, max=1)
    probability = hist[hist > 0] / hist.sum()
    dx, dy = y[..., 1:] - y[..., :-1], y[..., 1:, :] - y[..., :-1, :]
    sx = dx.square().mean() if dx.numel() else y.new_zeros(())
    sy = dy.square().mean() if dy.numel() else y.new_zeros(())
    return {"entropy": float(-(probability*probability.log2()).sum()),
            "std": float(y.std(unbiased=False)), "spatial_frequency": float((sx+sy).sqrt())}
