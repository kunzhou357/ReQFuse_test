"""Reproducible, modality-independent local synthetic degradations on CPU.

Masks returned here describe the actual spatial support of modification.
They supervise training only and are never inputs to the network.
"""
import math

import torch
from torch.nn import functional as F


DEFAULT_DEGRADATION = {
    "pair_probabilities": [0.2, 0.3, 0.3, 0.2],  # clean, VI only, IR only, both
    "local_probability": 0.7,
    "visible_types": ["noise", "blur", "contrast", "exposure"],
    "infrared_types": ["noise", "blur", "contrast"],
    "noise_std": [0.015, 0.12], "blur_sigma": [0.6, 2.5],
    "contrast_factor": [0.3, 0.8], "exposure_gain": [0.35, 1.8],
}


def uniform(generator, low=0.0, high=1.0):
    return low + (high-low)*torch.rand((), generator=generator).item()


def randint(generator, low, high):
    return int(torch.randint(low, high, (), generator=generator))


class LocalDegrader:
    def __init__(self, config=None):
        self.config = dict(DEFAULT_DEGRADATION)
        if config:
            unknown = set(config)-set(self.config)
            if unknown:
                raise ValueError(f"Unknown degradation options: {sorted(unknown)}")
            self.config.update(config)
        c = self.config
        probs = c["pair_probabilities"]
        if len(probs) != 4 or any(not math.isfinite(p) or p < 0 for p in probs) or abs(sum(probs)-1) > 1e-6:
            raise ValueError("pair_probabilities must contain four probabilities summing to one.")
        if not 0 <= c["local_probability"] <= 1:
            raise ValueError("local_probability must be in [0,1].")
        for name in ("visible_types", "infrared_types"):
            if not c[name] or set(c[name])-{"noise","blur","contrast","exposure"}:
                raise ValueError(f"Invalid {name}.")
        for name in ("noise_std", "blur_sigma", "contrast_factor", "exposure_gain"):
            bounds = c[name]
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]:
                raise ValueError(f"Invalid {name} bounds.")
        if c["blur_sigma"][0] <= 0:
            raise ValueError("blur_sigma must be positive.")

    def _mask(self, image, generator):
        _, h, w = image.shape
        if uniform(generator) >= self.config["local_probability"]:
            return image.new_ones(1,h,w)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        mask = image.new_zeros(h,w)
        for _ in range(randint(generator, 1, 4)):
            rh, rw = max(1, round(h*uniform(generator,.12,.65))), max(1,round(w*uniform(generator,.12,.65)))
            top, left = randint(generator,0,h-rh+1), randint(generator,0,w-rw+1)
            if uniform(generator) < .5:
                region = (yy >= top) & (yy < top+rh) & (xx >= left) & (xx < left+rw)
            else:
                region = ((yy-(top+(rh-1)/2))/max(rh/2,.5)).square() + ((xx-(left+(rw-1)/2))/max(rw/2,.5)).square() <= 1
            mask = torch.maximum(mask, region.to(image.dtype))
        # Soft boundary for the actual blend; keep supervision uses its support.
        return F.avg_pool2d(F.pad(mask[None,None],(3,3,3,3),mode="replicate"),7,1)[0]

    def _blur(self, image, sigma):
        radius = min(6, max(1, math.ceil(3*sigma)))
        axis = torch.arange(-radius,radius+1,dtype=image.dtype)
        kernel = torch.exp(-axis.square()/(2*sigma*sigma))
        kernel = kernel/kernel.sum()
        kernel = torch.outer(kernel,kernel).expand(image.shape[0],1,-1,-1).contiguous()
        return F.conv2d(F.pad(image[None],(radius,)*4,mode="replicate"),kernel,groups=image.shape[0])[0]

    def _one(self, image, modality, enabled, generator):
        if not enabled:
            return image.clone(), image.new_zeros(1,*image.shape[-2:]), "clean"
        types = self.config[f"{modality}_types"]
        kind = types[randint(generator,0,len(types))]
        mask = self._mask(image,generator)
        if kind == "noise":
            std = uniform(generator,*self.config["noise_std"])
            degraded = image + std*torch.randn(image.shape,generator=generator,dtype=image.dtype)
        elif kind == "blur":
            degraded = self._blur(image,uniform(generator,*self.config["blur_sigma"]))
        elif kind == "contrast":
            factor = uniform(generator,*self.config["contrast_factor"])
            mean = image.mean((-2,-1),keepdim=True)
            degraded = mean + factor*(image-mean)
        else:
            degraded = image*uniform(generator,*self.config["exposure_gain"])
        result = (image*(1-mask)+degraded.clamp(0,1)*mask).clamp(0,1)
        return result, (mask > 0).to(image.dtype), kind

    def __call__(self, visible, infrared, generator):
        draw, cumulative, case = uniform(generator), 0.0, 3
        for i,p in enumerate(self.config["pair_probabilities"]):
            cumulative += p
            if draw < cumulative:
                case = i
                break
        vi,mv,tv = self._one(visible,"visible",case in (1,3),generator)
        ir,mi,ti = self._one(infrared,"infrared",case in (2,3),generator)
        return vi,ir,mv,mi,f"visible:{tv};infrared:{ti}"
