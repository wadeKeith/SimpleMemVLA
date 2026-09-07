
from __future__ import annotations

import numpy as np
import torch

from mikasa_sim.policy import MikasaPolicy

SENSOR_TO_TAG = {
    "base_camera": "top",
    "hand_camera": "wrist",
}


class SimpleMemVLAChunkPolicy:

    chunk_size = 1

    def __init__(
        self,
        policy: MikasaPolicy,
        execute_horizon: int,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        if execute_horizon < 1:
            raise ValueError(f"execute_horizon must be >= 1, got {execute_horizon}")
        if execute_horizon > policy.action_horizon:
            raise ValueError(
                f"execute_horizon {execute_horizon} > model action_horizon "
                f"{policy.action_horizon}"
            )
        runner = getattr(policy, "runner", None)
        if runner is not None and runner.execute_horizon != int(execute_horizon):
            raise ValueError(
                f"execute_horizon {execute_horizon} != the pipelined runner's "
                f"{runner.execute_horizon}; the decision cadence must have one value"
            )
        self.policy = policy
        self.execute_horizon = int(execute_horizon)
        self.action_low = None if action_low is None else np.asarray(action_low, np.float32)
        self.action_high = None if action_high is None else np.asarray(action_high, np.float32)
        self._instruction: str | None = None
        self._queue: list[np.ndarray] = []
        self._channel_map: list[tuple[str, int]] | None = None
        self.last_subtask: str = ""
        self.n_predictions: int = 0

    def bind_sensor_names(self, sensor_names: list[str]) -> None:
        tag_to_channel: dict[str, int] = {}
        for i, name in enumerate(sensor_names):
            tag = SENSOR_TO_TAG.get(name, name)
            tag_to_channel[tag] = 3 * i
        channel_map: list[tuple[str, int]] = []
        for key in self.policy.image_keys:
            tag = key.split(".")[-1]
            if tag not in tag_to_channel:
                raise RuntimeError(
                    f"Checkpoint camera {key!r} (tag {tag!r}) has no matching env "
                    f"sensor. Env sensors: {sensor_names} -> tags "
                    f"{sorted(tag_to_channel)}."
                )
            channel_map.append((key, tag_to_channel[tag]))
        self._channel_map = channel_map

    def start_episode(self, instruction: str) -> None:
        if not str(instruction).strip():
            raise ValueError("episode instruction must be a non-empty string")
        self._instruction = str(instruction)
        self._queue = []
        self.policy.reset()
        self.last_subtask = ""
        self.n_predictions = 0

    def _split_frames(self, obs) -> dict[str, np.ndarray]:
        if self._channel_map is None:
            raise RuntimeError("bind_sensor_names() must run before forward()")
        rgb = obs["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            if rgb.shape[0] != 1:
                raise ValueError(f"expected num_envs=1 obs, got rgb {rgb.shape}")
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[-1] < 3 * len(self._channel_map):
            raise ValueError(f"unexpected rgb obs shape {rgb.shape}")
        return {
            key: np.ascontiguousarray(rgb[:, :, c0 : c0 + 3]).astype(np.uint8, copy=False)
            for key, c0 in self._channel_map
        }

    @staticmethod
    def _state_vec(obs) -> np.ndarray:
        proprio = obs.get("proprio")
        if proprio is None:
            raise RuntimeError(
                "obs has no 'proprio'; the canonical apply_mikasa_vla_wrappers "
                "stack must be active."
            )
        if torch.is_tensor(proprio):
            proprio = proprio.detach().cpu().numpy()
        return np.asarray(proprio, dtype=np.float32).reshape(-1)

    def forward(self, obs) -> np.ndarray:
        if self._instruction is None:
            raise RuntimeError("start_episode() must run before forward()")
        self.policy.observe(self._split_frames(obs))
        if not self._queue:
            state = self._state_vec(obs) if self.policy.use_proprio else None
            actions, subtask = self.policy.predict(self._instruction, state=state)
            self.last_subtask = subtask
            self.n_predictions += 1
            self._queue = [np.asarray(a, dtype=np.float32) for a in actions[: self.execute_horizon]]
        action = self._queue.pop(0)
        if self.action_low is not None and self.action_high is not None:
            action = np.clip(action, self.action_low, self.action_high)
        return action
