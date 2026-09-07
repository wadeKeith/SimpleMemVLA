
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_EVAL_SCRIPTS = REPO_ROOT / "evaluation_benchmark" / "scripts"
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from robomemarena_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

from policy_adapter import BasePolicyAdapter

from simplememvla.benchmarks.robomemarena import CAM_FRONT_KEY, CAM_WRIST_KEY


def quat2axisangle_dataset(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * np.arccos(w)) / den).astype(np.float32)


def encode_state(raw_obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
            quat2axisangle_dataset(raw_obs["robot0_eef_quat"]).reshape(3),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        ]
    ).astype(np.float32)


class SimpleMemVLARoboMemArenaAdapter(BasePolicyAdapter):

    def __init__(
        self,
        checkpoint_dir: str,
        compute_dtype: str = "bfloat16",
        attn_implementation: str | None = "flash_attention_2",
        num_denoising_steps: int = 10,
        temperature: float = 1.0,
        max_reasoning_tokens: int = 256,
        log_every: int = 0,
        episode_seed_base: int | None = None,
    ):
        from transformers import AutoProcessor

        from robomemarena_sim.policy import RoboMemArenaPolicy, get_vla

        vla, unnormalize_action, normalize_state = get_vla(
            str(checkpoint_dir),
            compute_dtype=compute_dtype,
            attn_implementation=attn_implementation,
        )
        processor = AutoProcessor.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        self.policy = RoboMemArenaPolicy(
            vla,
            processor,
            unnormalize_action,
            normalize_state,
            num_denoising_steps=num_denoising_steps,
            temperature=temperature,
            max_reasoning_tokens=max_reasoning_tokens,
        )
        self.log_every = int(log_every)
        self._steps = 0
        self._decisions = 0
        self.last_reasoning = ""
        self._episode_seed_base = None if episode_seed_base is None else int(episode_seed_base)
        self._episodes_seen = 0
        self._decision_seconds: deque[float] = deque(maxlen=4096)
        self._total_decisions = 0

    def set_episode_seed_base(self, base: int) -> None:
        self._episode_seed_base = int(base)
        self._episodes_seen = 0

    @property
    def runaway_decodes(self) -> int:
        return int(self.policy.runaway_decodes)

    @property
    def episodes_seen(self) -> int:
        return self._episodes_seen

    def begin_shard(self) -> None:
        self._decision_seconds.clear()

    def decision_seconds_median(self) -> float | None:
        if not self._decision_seconds:
            return None
        ordered = sorted(self._decision_seconds)
        return ordered[len(ordered) // 2]

    @staticmethod
    def _processed_frames(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            CAM_FRONT_KEY: np.ascontiguousarray(obs["observation/image"]),
            CAM_WRIST_KEY: np.ascontiguousarray(obs["observation/wrist_image"]),
        }

    def reset(self) -> None:
        self.policy.reset()
        self._steps = 0
        self._decisions = 0
        self.last_reasoning = ""
        if self._episode_seed_base is not None:
            seed = self._episode_seed_base + self._episodes_seen
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self._episodes_seen += 1

    def observe(self, obs: dict[str, Any], prompt: str, resize_size: int) -> None:
        self.policy.observe(self._processed_frames(obs))
        self._steps += 1

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        t0 = time.perf_counter()
        actions, reasoning = self.policy.predict(prompt, state=encode_state(obs["_raw_obs"]))
        self._decision_seconds.append(time.perf_counter() - t0)
        self.last_reasoning = reasoning
        self._decisions += 1
        self._total_decisions += 1
        if self.log_every and self._total_decisions % self.log_every == 0:
            logging.info(
                "  [decision %d (ep %d) @ step %d] clip=%d frames | %.2fs/decision "
                "(median %.2fs) | truncated=%s | reasoning=%r",
                self._total_decisions,
                self._decisions,
                self._steps,
                self.policy._clip_len(),
                self._decision_seconds[-1],
                self.decision_seconds_median(),
                self.policy.last_decode_truncated,
                reasoning,
            )
        return np.stack(actions).astype(np.float32)


def build_adapter(**kwargs: Any) -> BasePolicyAdapter:
    return SimpleMemVLARoboMemArenaAdapter(**kwargs)
