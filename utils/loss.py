"""可独立复制的 PyTorch 图像融合损失，仅依赖 torch 和 Python 标准库。

输入约定
--------
* 浮点 NCHW 张量，RGB 通道顺序为 R/G/B；默认数值范围为 [0, 1]。
* 单通道直接作为亮度；RGB 使用 BT.601 亮度 Y（可改为通道均值）。
* 不自动 clamp、归一化或 detach 输入，避免改变网络梯度和训练目标。
* 退化融合训练若有清晰参考，应传入清晰的同模态源图作为监督。

默认定义
--------
L_intensity = L1(Y_fused, max(Y_visible, Y_infrared))
L_gradient  = L1(G(Y_fused), max(G(Y_visible), G(Y_infrared)))
L_ssim      = 0.5 * (1 - SSIM(Y_fused, Y_visible))
              + 0.5 * (1 - SSIM(Y_fused, Y_infrared))
L_color     = mean(abs(CbCr_fused - CbCr_visible))
其中 G(x) = abs(Sobel_x(x)) + abs(Sobel_y(x))，Sobel 不额外缩放。

用法
----
    criterion = FusionLoss(intensity_weight=10, gradient_weight=10,
                           ssim_weight=1, color_weight=1)
    total, terms = criterion(fused, visible, infrared, return_components=True)
    total.backward()
    # terms 是未加权分项，保持计算图；日志记录时再调用 .item()。

四项权重默认均为 1，只作为中性默认值，不代表某篇论文的最优配置。
灰度输出无法训练色度，默认颜色项为零；require_color=True 可强制要求 RGB。
所有损失支持 reduction='mean'/'sum'/'none'。none 返回逐像素损失；
组合损失及颜色损失为 [N, 1, H, W]，其余独立损失保留输入通道数。
半精度/bfloat16 输入使用 float32 计算，float64 输入保留双精度。
CPU/CUDA autocast 在损失内部关闭；不绑定 CUDA，也不依赖项目其他文件。
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "to_luminance", "rgb_to_ycbcr", "ssim", "SobelGradient",
    "IntensityLoss", "GradientLoss", "SSIMLoss", "ColorLoss", "FusionLoss",
]


def _check_image(image: Tensor, name: str) -> None:
    if not isinstance(image, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if image.ndim != 4 or any(size == 0 for size in image.shape):
        raise ValueError(f"{name} must have non-empty NCHW shape, got {tuple(image.shape)}.")
    if not image.is_floating_point():
        raise TypeError(f"{name} must be floating point; convert uint8 images before use.")


def _check_images(*images: Tensor, same_channels: bool = True) -> None:
    for index, image in enumerate(images):
        _check_image(image, f"image[{index}]")
    first = images[0]
    for image in images[1:]:
        if image.device != first.device:
            raise ValueError("All images must be on the same device.")
        if (image.shape[0], image.shape[2:]) != (first.shape[0], first.shape[2:]):
            raise ValueError("All images must have the same batch size, height and width.")
        if same_channels and image.shape[1] != first.shape[1]:
            raise ValueError("All images must have the same number of channels.")


def _working_images(*images: Tensor) -> Tuple[Tensor, ...]:
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in images) else torch.float32
    return tuple(x.to(dtype=dtype) for x in images)


def _no_autocast(image: Tensor):
    # Older PyTorch releases do not support an MPS autocast context.
    if image.device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=image.device.type, enabled=False)
    return nullcontext()


def _check_reduction(reduction: str) -> None:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum' or 'none'.")


def _reduce(loss_map: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss_map.mean()
    if reduction == "sum":
        return loss_map.sum()
    return loss_map


def _check_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return value


def _check_mode(mode: str) -> None:
    if mode not in ("max", "mean"):
        raise ValueError("target_mode must be 'max' or 'mean'.")


def _joint_target(a: Tensor, b: Optional[Tensor], mode: str) -> Tensor:
    if b is None:
        return a
    return torch.maximum(a, b) if mode == "max" else (a + b) * 0.5


def to_luminance(image: Tensor, mode: str = "luma") -> Tensor:
    """NCHW 的灰度/RGB 转单通道亮度；mode 可选 luma 或 mean。"""
    _check_image(image, "image")
    if mode not in ("luma", "mean"):
        raise ValueError("luminance_mode must be 'luma' or 'mean'.")
    image, = _working_images(image)
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError("Luminance conversion accepts only 1-channel or 3-channel images.")
    if mode == "mean":
        return image.mean(dim=1, keepdim=True)
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def rgb_to_ycbcr(image: Tensor, data_range: float = 1.0) -> Tensor:
    """BT.601 full-range RGB -> Y/Cb/Cr；Cb/Cr 中心为 data_range / 2。

    不 clamp 结果。颜色损失只比较 Cb/Cr，公共偏置不影响色度差。
    """
    _check_image(image, "image")
    data_range = _check_positive(data_range, "data_range")
    if image.shape[1] != 3:
        raise ValueError("RGB to YCbCr conversion requires 3 channels in RGB order.")
    image, = _working_images(image)
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5 * data_range
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5 * data_range
    return torch.cat((y, cb, cr), dim=1)


class SobelGradient(nn.Module):
    """逐通道 Sobel 梯度幅值；replicate padding 避免零填充制造边界。"""

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.reshape(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_x.t().reshape(1, 1, 3, 3))

    def forward(self, image: Tensor) -> Tensor:
        _check_image(image, "image")
        image, = _working_images(image)
        with _no_autocast(image):
            channels = image.shape[1]
            padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
            kernel_x = self.kernel_x.to(image).expand(channels, 1, 3, 3).contiguous()
            kernel_y = self.kernel_y.to(image).expand(channels, 1, 3, 3).contiguous()
            gx = F.conv2d(padded, kernel_x, groups=channels)
            gy = F.conv2d(padded, kernel_y, groups=channels)
            return gx.abs() + gy.abs()


class IntensityLoss(nn.Module):
    """L1 强度损失；第二源图可省略，此时直接对齐第一个参考图。

    独立使用时不转换颜色，输入须同形状；FusionLoss 会先转换为亮度。
    """

    def __init__(self, target_mode: str = "max", reduction: str = "mean") -> None:
        super().__init__()
        _check_mode(target_mode)
        _check_reduction(reduction)
        self.target_mode, self.reduction = target_mode, reduction

    def forward(self, fused: Tensor, source_a: Tensor, source_b: Optional[Tensor] = None) -> Tensor:
        images = (fused, source_a) if source_b is None else (fused, source_a, source_b)
        _check_images(*images)
        images = _working_images(*images)
        target = _joint_target(images[1], images[2] if source_b is not None else None, self.target_mode)
        return _reduce((images[0] - target).abs(), self.reduction)


class GradientLoss(nn.Module):
    """对齐源图 Sobel 幅值的 max/mean；第二源图可省略。"""

    def __init__(self, target_mode: str = "max", reduction: str = "mean") -> None:
        super().__init__()
        _check_mode(target_mode)
        _check_reduction(reduction)
        self.target_mode, self.reduction = target_mode, reduction
        self.sobel = SobelGradient()

    def forward(self, fused: Tensor, source_a: Tensor, source_b: Optional[Tensor] = None) -> Tensor:
        images = (fused, source_a) if source_b is None else (fused, source_a, source_b)
        _check_images(*images)
        images = _working_images(*images)
        gradients = tuple(self.sobel(image) for image in images)
        target = _joint_target(gradients[1], gradients[2] if source_b is not None else None, self.target_mode)
        return _reduce((gradients[0] - target).abs(), self.reduction)


class SSIMLoss(nn.Module):
    """两幅图的 1-SSIM，逐通道 Gaussian 窗口，使用总体协方差。

    使用 replicate padding，支持小于窗口的图像，包括 1x1 图像。
    data_range 是输入约定的数值跨度，不按每张图动态估计。
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")
        sigma = _check_positive(sigma, "sigma")
        self.data_range = _check_positive(data_range, "data_range")
        _check_reduction(reduction)
        self.window_size, self.reduction = window_size, reduction
        coordinates = torch.arange(window_size, dtype=torch.float64, device="cpu") - window_size // 2
        gaussian = torch.exp(-coordinates.square() / (2 * sigma * sigma))
        gaussian = gaussian / gaussian.sum()
        # 默认 buffer 使用 float32，允许 criterion.to('mps')；MPS 不支持 float64。
        self.register_buffer("window", torch.outer(gaussian, gaussian).float().reshape(1, 1, window_size, window_size))

    def _similarity_map(self, image_a: Tensor, image_b: Tensor) -> Tensor:
        _check_images(image_a, image_b)
        image_a, image_b = _working_images(image_a, image_b)
        with _no_autocast(image_a):
            # 先按已知范围缩放，避免 [0,255] 或半精度时的二阶矩溢出。
            a, b = image_a / self.data_range, image_b / self.data_range
            channels = a.shape[1]
            window = self.window.to(a)
            window = (window / window.sum()).expand(channels, 1, self.window_size, self.window_size).contiguous()
            pad = self.window_size // 2

            def local_mean(image: Tensor) -> Tensor:
                padded = F.pad(image, (pad, pad, pad, pad), mode="replicate")
                return F.conv2d(padded, window, groups=channels)

            mu_a, mu_b = local_mean(a), local_mean(b)
            var_a = local_mean(a.square()) - mu_a.square()
            var_b = local_mean(b.square()) - mu_b.square()
            covariance = local_mean(a * b) - mu_a * mu_b
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            numerator = (2 * mu_a * mu_b + c1) * (2 * covariance + c2)
            denominator = (mu_a.square() + mu_b.square() + c1) * (var_a + var_b + c2)
            # tiny 不会像 eps 那样压过黑色常量图的正常分母 c1*c2。
            similarity = numerator / denominator.clamp_min(torch.finfo(a.dtype).tiny)
            return similarity.clamp(-1.0, 1.0)

    def similarity(self, image_a: Tensor, image_b: Tensor) -> Tensor:
        """返回 SSIM 相似度；forward 返回用于优化的 1-SSIM。"""
        return _reduce(self._similarity_map(image_a, image_b), self.reduction)

    def forward(self, image_a: Tensor, image_b: Tensor) -> Tensor:
        return _reduce(1.0 - self._similarity_map(image_a, image_b), self.reduction)


def ssim(image_a: Tensor, image_b: Tensor, window_size: int = 11,
         sigma: float = 1.5, data_range: float = 1.0, reduction: str = "mean") -> Tensor:
    """便捷 SSIM 指标接口；训练循环建议复用 SSIMLoss 实例。"""
    return SSIMLoss(window_size, sigma, data_range, reduction).similarity(image_a, image_b)


class ColorLoss(nn.Module):
    """RGB 预测与 RGB 参考的 Cb/Cr L1；先对两个色度通道取平均。"""

    def __init__(self, data_range: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.data_range = _check_positive(data_range, "data_range")
        _check_reduction(reduction)
        self.reduction = reduction

    def forward(self, fused: Tensor, reference: Tensor) -> Tensor:
        _check_images(fused, reference)
        if fused.shape[1] != 3:
            raise ValueError("ColorLoss requires two 3-channel RGB images.")
        fused, reference = _working_images(fused, reference)
        fused_chroma = rgb_to_ycbcr(fused, self.data_range)[:, 1:3]
        reference_chroma = rgb_to_ycbcr(reference, self.data_range)[:, 1:3]
        loss_map = (fused_chroma - reference_chroma).abs().mean(dim=1, keepdim=True)
        return _reduce(loss_map, self.reduction)


class FusionLoss(nn.Module):
    """四项加权融合损失，参数顺序统一为 (fused, visible, infrared)。

    权重为零的分项跳过计算，并在分项字典中返回零。
    SSIM 的 visible 权重默认 0.5，infrared 权重为 1-ssim_visible_weight。
    color_reference 可另传 RGB 参考，默认使用 visible。
    灰度场景颜色项默认为零；require_color=True 时会明确报错。
    """

    def __init__(self, intensity_weight: float = 1.0, gradient_weight: float = 1.0,
                 ssim_weight: float = 1.0, color_weight: float = 1.0,
                 intensity_mode: str = "max", gradient_mode: str = "max",
                 ssim_visible_weight: float = 0.5, window_size: int = 11,
                 sigma: float = 1.5, data_range: float = 1.0,
                 luminance_mode: str = "luma", reduction: str = "mean",
                 require_color: bool = False) -> None:
        super().__init__()
        _check_reduction(reduction)
        if luminance_mode not in ("luma", "mean"):
            raise ValueError("luminance_mode must be 'luma' or 'mean'.")
        self.weights = {"intensity": float(intensity_weight), "gradient": float(gradient_weight),
                        "ssim": float(ssim_weight), "color": float(color_weight)}
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights.values()):
            raise ValueError("Loss weights must be finite and non-negative.")
        self.ssim_visible_weight = float(ssim_visible_weight)
        if not math.isfinite(self.ssim_visible_weight) or not 0 <= self.ssim_visible_weight <= 1:
            raise ValueError("ssim_visible_weight must be between 0 and 1.")
        self.luminance_mode, self.reduction = luminance_mode, reduction
        self.require_color = require_color
        self.intensity = IntensityLoss(intensity_mode, reduction="none")
        self.gradient = GradientLoss(gradient_mode, reduction="none")
        self.structure = SSIMLoss(window_size, sigma, data_range, reduction="none")
        self.color = ColorLoss(data_range, reduction="none")

    def forward(self, fused: Tensor, visible: Tensor, infrared: Tensor, *,
                color_reference: Optional[Tensor] = None,
                return_components: bool = False) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        _check_images(fused, visible, infrared, same_channels=False)
        fused, visible, infrared = _working_images(fused, visible, infrared)
        with _no_autocast(fused):
            fused_y = to_luminance(fused, self.luminance_mode)
            visible_y = to_luminance(visible, self.luminance_mode)
            infrared_y = to_luminance(infrared, self.luminance_mode)
            zero = fused_y * 0.0  # 关闭全部损失时仍允许 backward，梯度为零。
            maps = {name: zero for name in self.weights}
            if self.weights["intensity"] > 0:
                maps["intensity"] = self.intensity(fused_y, visible_y, infrared_y)
            if self.weights["gradient"] > 0:
                maps["gradient"] = self.gradient(fused_y, visible_y, infrared_y)
            if self.weights["ssim"] > 0:
                weight = self.ssim_visible_weight
                maps["ssim"] = (weight * self.structure(fused_y, visible_y)
                                + (1 - weight) * self.structure(fused_y, infrared_y))
            if self.weights["color"] > 0:
                reference = visible if color_reference is None else color_reference
                _check_images(fused, reference, same_channels=False)
                if reference.shape[1] not in (1, 3):
                    raise ValueError("color_reference must have 1 or 3 channels.")
                if fused.shape[1] == 3 and reference.shape[1] == 3:
                    maps["color"] = self.color(fused, reference)
                elif self.require_color:
                    raise ValueError("Active color loss requires RGB fused and reference images.")
            terms = {name: _reduce(loss_map, self.reduction) for name, loss_map in maps.items()}
            total = sum(self.weights[name] * term for name, term in terms.items())
        return (total, terms) if return_components else total
