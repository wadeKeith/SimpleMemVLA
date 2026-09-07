
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class ClipAugParams:

    scale: float
    top_frac: float
    left_frac: float
    brightness: float
    contrast: float
    saturation: float
    hue: float


_LUMA = (0.299, 0.587, 0.114)


def _sat_hue_matrix(saturation: float, hue: float) -> torch.Tensor:
    lr, lg, lb = _LUMA
    s = float(saturation)
    sat_m = torch.tensor([
        [s + (1 - s) * lr, (1 - s) * lg, (1 - s) * lb],
        [(1 - s) * lr, s + (1 - s) * lg, (1 - s) * lb],
        [(1 - s) * lr, (1 - s) * lg, s + (1 - s) * lb],
    ], dtype=torch.float32)
    theta = 2.0 * np.pi * float(hue)
    c, si = float(np.cos(theta)), float(np.sin(theta))
    hue_m = torch.tensor([
        [0.299 + 0.701 * c + 0.168 * si, 0.587 - 0.587 * c + 0.330 * si, 0.114 - 0.114 * c - 0.497 * si],
        [0.299 - 0.299 * c - 0.328 * si, 0.587 + 0.413 * c + 0.035 * si, 0.114 - 0.114 * c + 0.292 * si],
        [0.299 - 0.300 * c + 1.250 * si, 0.587 - 0.588 * c - 1.050 * si, 0.114 + 0.886 * c - 0.203 * si],
    ], dtype=torch.float32)
    return hue_m @ sat_m


class ClipAugmenter:

    def __init__(
        self,
        scale_min: float = 0.9,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ):
        if not 0.0 < scale_min <= 1.0:
            raise ValueError(f"scale_min must be in (0, 1], got {scale_min}")
        self.scale_min = float(scale_min)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def draw(self) -> ClipAugParams:
        return ClipAugParams(
            scale=float(np.random.uniform(self.scale_min, 1.0)),
            top_frac=float(np.random.uniform(0.0, 1.0)),
            left_frac=float(np.random.uniform(0.0, 1.0)),
            brightness=float(np.random.uniform(1 - self.brightness, 1 + self.brightness)),
            contrast=float(np.random.uniform(1 - self.contrast, 1 + self.contrast)),
            saturation=float(np.random.uniform(1 - self.saturation, 1 + self.saturation)),
            hue=float(np.random.uniform(-self.hue, self.hue)),
        )

    def apply(self, frames: np.ndarray, params: ClipAugParams) -> np.ndarray:
        arr = np.asarray(frames)
        single = arr.ndim == 3
        if single:
            arr = arr[None]
        if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
            raise ValueError(
                f"Expected (T, H, W, 3) or (H, W, 3) uint8, got shape "
                f"{arr.shape} dtype {arr.dtype}"
            )
        t, h, w, _ = arr.shape
        side = float(np.sqrt(params.scale))
        ch = max(1, round(h * side))
        cw = max(1, round(w * side))
        top = round(params.top_frac * (h - ch))
        left = round(params.left_frac * (w - cw))

        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
        if (ch, cw) != (h, w):
            x = TF.resized_crop(
                x, top, left, ch, cw, [h, w],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
        if params.brightness != 1.0:
            x = TF.adjust_brightness(x, params.brightness)
        if params.contrast != 1.0:
            x = TF.adjust_contrast(x, params.contrast)
        if params.saturation != 1.0 or params.hue != 0.0:
            m = _sat_hue_matrix(params.saturation, params.hue).to(x.dtype)
            x = torch.einsum("ij,tjhw->tihw", m, x)
        out = (
            x.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
            .permute(0, 2, 3, 1).contiguous().numpy()
        )
        return out[0] if single else out
