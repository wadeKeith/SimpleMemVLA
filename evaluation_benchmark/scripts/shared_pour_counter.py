"""Live MuJoCo-state counter for strict counting-pour stages.

The counter deliberately does not read policy actions, end-effector pose, or
demonstration labels. A completed pour is a lifted source body that tilts over
the intended target for several frames and then returns upright long enough to
re-arm the detector for a later pour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PourCounterConfig:
    lift_delta: float = 0.03
    target_radius: float = 0.20
    tilt_enter_rad: float = 0.70
    tilt_return_rad: float = 0.20
    enter_dwell: int = 3
    return_dwell: int = 5


def _name_variants(name: str) -> tuple[str, ...]:
    variants = [name]
    if not name.endswith("_main"):
        variants.append(f"{name}_main")
    if name.endswith("_main"):
        variants.append(name[:-5])
    return tuple(variants)


def _body_id(env: Any, name: str) -> int | None:
    for candidate in _name_variants(name):
        try:
            return int(env.sim.model.body_name2id(candidate))
        except Exception:
            continue
    return None


def _site_id(env: Any, name: str) -> int | None:
    for candidate in _name_variants(name):
        try:
            return int(env.sim.model.site_name2id(candidate))
        except Exception:
            continue
    return None


def _body_pos(env: Any, name: str) -> np.ndarray | None:
    identifier = _body_id(env, name)
    if identifier is None:
        return None
    return np.asarray(env.sim.data.body_xpos[identifier], dtype=np.float64).copy()


def _body_rotation(env: Any, name: str) -> np.ndarray | None:
    identifier = _body_id(env, name)
    if identifier is None:
        return None
    return np.asarray(env.sim.data.body_xmat[identifier], dtype=np.float64).reshape(3, 3).copy()


def _target_pos(env: Any, kind: str, name: str) -> np.ndarray | None:
    if kind == "body":
        return _body_pos(env, name)
    if kind == "site":
        identifier = _site_id(env, name)
        if identifier is None:
            return None
        return np.asarray(env.sim.data.site_xpos[identifier], dtype=np.float64).copy()
    raise ValueError(f"Unsupported target kind: {kind}")


def _rotation_angle(reference: np.ndarray, current: np.ndarray) -> float:
    relative = reference.T @ current
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


@dataclass
class SharedPourCounter:
    source_name: str
    target_kind: str
    target_name: str
    initial_source_pos: np.ndarray
    config: PourCounterConfig = field(default_factory=PourCounterConfig)
    reference_rotation: np.ndarray | None = None
    event_count: int = 0
    available: bool = True
    enter_run: int = 0
    return_run: int = 0
    events: list[dict[str, float | int]] = field(default_factory=list)

    def update(self, env: Any, step: int) -> int:
        source_pos = _body_pos(env, self.source_name)
        source_rotation = _body_rotation(env, self.source_name)
        target_pos = _target_pos(env, self.target_kind, self.target_name)
        if source_pos is None or source_rotation is None or target_pos is None:
            return self.event_count

        lifted = float(source_pos[2] - self.initial_source_pos[2]) > self.config.lift_delta
        if self.reference_rotation is None and lifted:
            # Reset yaw varies by episode. The first physically lifted pose is
            # the upright reference for this source object in this episode.
            self.reference_rotation = source_rotation.copy()

        tilt = (
            _rotation_angle(self.reference_rotation, source_rotation)
            if self.reference_rotation is not None
            else float("nan")
        )
        target_distance = float(np.linalg.norm(source_pos[:2] - target_pos[:2]))
        at_target_and_tilted = bool(
            lifted
            and np.isfinite(tilt)
            and target_distance <= self.config.target_radius
            and tilt >= self.config.tilt_enter_rad
        )

        if self.available:
            self.enter_run = self.enter_run + 1 if at_target_and_tilted else 0
            if self.enter_run >= self.config.enter_dwell:
                event_step = step - self.config.enter_dwell + 1
                self.event_count += 1
                self.events.append(
                    {
                        "event_index": self.event_count,
                        "start_step": event_step,
                        "tilt_rad": float(tilt),
                        "target_xy_distance": target_distance,
                    }
                )
                self.available = False
                self.return_run = 0
        else:
            # A long, moving pour stays one event until the physical source
            # returns close to its acquired upright orientation.
            returned = bool(lifted and np.isfinite(tilt) and tilt <= self.config.tilt_return_rad)
            self.return_run = self.return_run + 1 if returned else 0
            if self.return_run >= self.config.return_dwell:
                self.available = True
                self.enter_run = 0

        return self.event_count
