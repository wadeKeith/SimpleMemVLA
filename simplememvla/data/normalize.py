import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def _to_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def load_stats_file(stats_path: str | Path, key: str) -> dict[str, torch.Tensor]:
    path = Path(stats_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Expected a LeRobot v3.0 stats.json carrying the "
            f"{key!r} mean/std (the trainer copies <root>/meta/stats.json into each "
            "checkpoint as stats.json)."
        )
    with path.open("r") as f:
        payload = json.load(f)
    if key in payload:
        stats = payload[key]
    elif "mean" in payload and "std" in payload:
        stats = payload
    else:
        raise KeyError(
            f"{path} has no stats for {key!r} (available keys: {sorted(payload)})."
        )
    missing = [k for k in ("mean", "std") if k not in stats]
    if missing:
        raise KeyError(f"{path} is missing {missing} for {key!r} (needed by MEAN_STD).")
    return {k: _to_tensor(v) for k, v in stats.items() if isinstance(v, (list, tuple))}


def _load_stats(root: str | Path, key: str) -> dict[str, torch.Tensor]:
    return load_stats_file(Path(root) / "meta" / "stats.json", key)


def load_action_stats(root: str | Path) -> dict[str, torch.Tensor]:
    return _load_stats(root, "action")


def load_state_stats(root: str | Path) -> dict[str, torch.Tensor]:
    return _load_stats(root, "observation.state")


class MeanStdNormalizer:

    def __init__(self, stats: Mapping[str, torch.Tensor], std_floor: float = 1e-2):
        self.stats = {k: _to_tensor(v) for k, v in stats.items()}
        self.std_floor = float(std_floor)
        missing = {"mean", "std"} - set(self.stats)
        if missing:
            raise KeyError(f"Normalization stats missing keys: {sorted(missing)}")
        if self.stats["mean"].shape != self.stats["std"].shape:
            raise ValueError("mean and std normalization stats must have identical shape")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        mean = self.stats["mean"].to(x.device)
        std = self.stats["std"].to(x.device).clamp_min(self.std_floor)
        if x.shape[-1] != mean.numel():
            raise ValueError(
                f"Input dim {x.shape[-1]} does not match mean/std dim {mean.numel()}"
            )
        return (x - mean) / std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.stats["mean"].to(x.device)
        std = self.stats["std"].to(x.device).clamp_min(self.std_floor)
        return x * std + mean
