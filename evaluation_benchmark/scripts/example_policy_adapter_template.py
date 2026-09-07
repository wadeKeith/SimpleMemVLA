from __future__ import annotations

from typing import Any

import numpy as np

from policy_adapter import BasePolicyAdapter


class YourPolicyAdapter(BasePolicyAdapter):
    """Template adapter. Replace this with your own model loading and inference."""

    def __init__(self, checkpoint_dir: str | None = None, device: str = "cuda") -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        # Load your model here.

    def reset(self) -> None:
        # Reset recurrent state or caches here if needed.
        return None

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        raise NotImplementedError(
            "Implement model-specific inference here. "
            "The benchmark already provides eval26-style processed keys such as "
            "'observation/image', 'observation/wrist_image', and 'observation/state'. "
            "Raw env obs is still available in obs and obs['_raw_obs']. "
            "Return a float32 numpy array with shape [horizon, action_dim]."
        )


def build_adapter(**kwargs: Any) -> BasePolicyAdapter:
    return YourPolicyAdapter(**kwargs)
