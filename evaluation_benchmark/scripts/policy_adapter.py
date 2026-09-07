from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


class BasePolicyAdapter(ABC):
    """Model-agnostic adapter interface for benchmark evaluation."""

    def reset(self) -> None:
        """Reset any per-episode internal state if needed."""

    @abstractmethod
    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        """Return an action chunk with shape [horizon, action_dim]."""


class AdapterLoadError(RuntimeError):
    pass


def _load_module_from_path(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise AdapterLoadError(f"Cannot load adapter module from path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_factory_spec(factory_spec: str) -> tuple[str, str]:
    if ":" in factory_spec:
        module_spec, factory_name = factory_spec.split(":", 1)
        return module_spec, factory_name
    return factory_spec, "build_adapter"


def load_policy_adapter(factory_spec: str, **factory_kwargs: Any) -> BasePolicyAdapter:
    if not factory_spec:
        raise AdapterLoadError("Missing adapter spec. Expected 'module.path:build_adapter' or '/abs/path.py:build_adapter'.")

    module_spec, factory_name = _split_factory_spec(factory_spec)
    module_path = Path(module_spec)
    if module_path.suffix == ".py" and module_path.exists():
        module = _load_module_from_path(module_path)
    else:
        module = importlib.import_module(module_spec)

    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise AdapterLoadError(f"Factory '{factory_name}' not found in adapter module '{module_spec}'.")

    adapter = factory(**factory_kwargs)
    if not isinstance(adapter, BasePolicyAdapter):
        raise AdapterLoadError(
            f"Adapter factory '{factory_name}' returned {type(adapter)!r}, expected BasePolicyAdapter subclass."
        )
    return adapter


def ensure_action_chunk(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Policy adapter must return shape [horizon, action_dim], got {arr.shape}.")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"Policy adapter returned invalid empty action chunk: {arr.shape}.")
    return arr


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64).copy()
    norm = np.linalg.norm(quat)
    if norm > 1e-12:
        quat /= norm
    if quat[3] < 0:
        quat = -quat

    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arccos(quat[3])
    return ((quat[:3] * angle) / den).astype(np.float32)


def _process_image_match_eval26(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    image = np.flipud(image)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def _extract_state(obs: dict[str, Any]) -> np.ndarray:
    eef_pos = obs.get("robot0_eef_pos")
    eef_quat = obs.get("robot0_eef_quat")
    gripper = obs.get("robot0_gripper_qpos")

    if eef_pos is None:
        eef_pos = obs.get("eef_pos")
    if eef_quat is None:
        eef_quat = obs.get("eef_quat")
    if gripper is None:
        gripper = obs.get("gripper_qpos")

    if eef_pos is None:
        eef_pos = np.zeros(3, dtype=np.float32)
    if eef_quat is None:
        eef_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if gripper is None:
        gripper = np.zeros(1, dtype=np.float32)

    return np.concatenate(
        [
            np.asarray(eef_pos, dtype=np.float32),
            _quat2axisangle(np.asarray(eef_quat, dtype=np.float32)),
            np.asarray(gripper, dtype=np.float32),
        ]
    )


def _extract_images(obs: dict[str, Any], resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    main = obs.get("agentview_image")
    if main is None:
        main = obs.get("agentview_rgb")
    if main is None:
        raise KeyError("Neither 'agentview_image' nor 'agentview_rgb' found in obs.")

    wrist = obs.get("robot0_eye_in_hand_image")
    if wrist is None:
        wrist = obs.get("wrist_image")
    if wrist is None:
        raise KeyError("Neither 'robot0_eye_in_hand_image' nor 'wrist_image' found in obs.")

    return (
        _process_image_match_eval26(np.asarray(main), resize_size),
        _process_image_match_eval26(np.asarray(wrist), resize_size),
    )


def build_eval26_policy_input(
    raw_obs: dict[str, Any],
    prompt: str,
    resize_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build eval26-aligned VLA input and keep raw obs available for adapters."""
    processed_main, processed_wrist = _extract_images(raw_obs, resize_size)
    state = _extract_state(raw_obs)

    adapter_obs = dict(raw_obs)
    adapter_obs.update(
        {
            "observation/image": processed_main,
            "observation/wrist_image": processed_wrist,
            "base_0_rgb": processed_main,
            "left_wrist_0_rgb": np.zeros_like(processed_wrist),
            "right_wrist_0_rgb": processed_wrist,
            "observation/state": state,
            "prompt": str(prompt),
            "_raw_obs": raw_obs,
        }
    )
    return adapter_obs, processed_main, processed_wrist
