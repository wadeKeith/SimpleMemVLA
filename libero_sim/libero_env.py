
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from libero_sim._mujoco_env import prepare_mujoco_runtime

prepare_mujoco_runtime()

CAM_FRONT = "image"
CAM_WRIST = "wrist_image"

RAW_FRONT = "agentview_image"
RAW_WRIST = "robot0_eye_in_hand_image"

TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

ACTION_SPACES = {"osc": 7, "joint": 8}

DUMMY_ACTIONS = {
    "osc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    "joint": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
}
DEFAULT_NUM_STEPS_WAIT = 10

JOINT_DELTA_MAX = 0.5

JOINT_KP = float(os.environ.get("SIMPLEMEMVLA_JOINT_KP", 300.0))

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]

ENV_BUILD_LOCK = threading.Lock()

_CONTROLLER_PATCH_LOCK = threading.Lock()


def pin_worker_gpu(gpu_id: int) -> str:
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    if visible:
        if gpu_id >= len(visible):
            raise RuntimeError(
                f"num_gpus asks for worker {gpu_id} but CUDA_VISIBLE_DEVICES "
                f"grants only {len(visible)} GPU(s) ({','.join(visible)}); set "
                "GPUS to the full device list (scripts/eval_libero.sh: GPUS=0,1,...)."
            )
        dev = visible[gpu_id]
    else:
        dev = str(gpu_id)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    if os.environ.get("MUJOCO_GL", "egl") == "egl":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = dev
    return dev


def _flip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def encode_frame(raw_obs: dict) -> dict[str, np.ndarray]:
    return {
        CAM_FRONT: _flip(raw_obs[RAW_FRONT]).astype(np.uint8, copy=False),
        CAM_WRIST: _flip(raw_obs[RAW_WRIST]).astype(np.uint8, copy=False),
    }


def encode_state(raw_obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(raw_obs["robot0_joint_pos"], dtype=np.float32).reshape(7),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        ]
    ).astype(np.float32)


class LiberoSimEnv:

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        max_steps: int | None = None,
        image_size: int = 256,
        num_steps_wait: int = DEFAULT_NUM_STEPS_WAIT,
        action_space: str = "osc",
    ):
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if suite_name not in TASK_SUITE_MAX_STEPS:
            raise ValueError(f"Unknown LIBERO suite {suite_name!r}")
        if action_space not in ACTION_SPACES:
            raise ValueError(
                f"Unknown action_space {action_space!r}; expected one of "
                f"{sorted(ACTION_SPACES)}"
            )
        self.action_space = action_space
        self.action_dim = ACTION_SPACES[action_space]
        self.dummy_action = list(DUMMY_ACTIONS[action_space])
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(task_id)
        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.task_name = task.name
        self.instruction = task.language
        self.num_steps_wait = int(num_steps_wait)
        self.max_steps = int(max_steps) if max_steps else TASK_SUITE_MAX_STEPS[suite_name]

        init_states_path = os.path.join(
            get_libero_path("init_states"), task.problem_folder, task.init_states_file
        )
        self.init_states = torch.load(init_states_path, weights_only=False)
        self.num_episodes = len(self.init_states)

        bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        extra_env_kwargs = {}
        if self.max_steps + self.num_steps_wait >= 990:
            extra_env_kwargs["horizon"] = self.max_steps + self.num_steps_wait + 10
        if action_space == "osc":
            self.env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                camera_heights=image_size,
                camera_widths=image_size,
                controller="OSC_POSE",
                **extra_env_kwargs,
            )
        else:
            import robosuite

            with _CONTROLLER_PATCH_LOCK:
                orig_loader = robosuite.load_controller_config

                def _patched(custom_fpath=None, default_controller=None):
                    cfg = orig_loader(
                        custom_fpath=custom_fpath, default_controller=default_controller
                    )
                    if default_controller == "JOINT_POSITION":
                        cfg["output_max"] = JOINT_DELTA_MAX
                        cfg["output_min"] = -JOINT_DELTA_MAX
                        cfg["kp"] = JOINT_KP
                    return cfg

                robosuite.load_controller_config = _patched
                try:
                    self.env = OffScreenRenderEnv(
                        bddl_file_name=bddl_file,
                        camera_heights=image_size,
                        camera_widths=image_size,
                        controller="JOINT_POSITION",
                        **extra_env_kwargs,
                    )
                finally:
                    robosuite.load_controller_config = orig_loader

        inner = self.env.env
        low, _high = inner.action_spec
        if low.shape[0] != self.action_dim:
            raise RuntimeError(
                f"{action_space} env action dim {low.shape[0]} != {self.action_dim} "
                "— controller selection did not take effect"
            )
        ctrl = inner.robots[0].controller
        expected_ctrl = ("OperationalSpaceController" if action_space == "osc"
                         else "JointPositionController")
        if type(ctrl).__name__ != expected_ctrl:
            raise RuntimeError(
                f"Unexpected controller {type(ctrl).__name__} (want {expected_ctrl})"
            )
        self._q_low = self._q_high = None
        if action_space == "joint":
            if float(np.max(ctrl.output_max)) != JOINT_DELTA_MAX:
                raise RuntimeError(
                    f"controller output_max {ctrl.output_max} != {JOINT_DELTA_MAX}"
                )
            if float(np.max(ctrl.kp)) != JOINT_KP:
                raise RuntimeError(f"controller kp {ctrl.kp} != {JOINT_KP}")
            rng = inner.sim.model.jnt_range[inner.robots[0]._ref_joint_indexes]
            limited = rng[:, 1] > rng[:, 0]
            self._q_low = np.where(limited, rng[:, 0], -np.inf)
            self._q_high = np.where(limited, rng[:, 1], np.inf)
        self._steps = 0
        self._success = False
        self._done = False

    def _bounded_inner_reset(self, max_attempts: int = 20) -> None:
        from robosuite.utils.errors import RandomizationError

        for _ in range(max_attempts):
            try:
                self.env.env.reset()
                return
            except RandomizationError:
                continue
        raise RuntimeError(
            f"env.reset: RandomizationError retries exhausted ({max_attempts})"
        )

    def reset(self, episode_idx: int, init_state=None) -> dict:
        if init_state is None:
            if not (0 <= episode_idx < self.num_episodes):
                raise IndexError(
                    f"init state {episode_idx} out of range [0, {self.num_episodes})"
                )
            init_state = self.init_states[episode_idx]
        self._bounded_inner_reset()
        raw_obs = self.env.set_init_state(init_state)
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self.env.step(self.dummy_action)
        self._steps = 0
        self._success = False
        self._done = False
        return raw_obs

    def step_one(self, action: np.ndarray) -> dict:
        if self._done:
            raise RuntimeError("episode already done; call reset()")
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] != self.action_dim:
            raise ValueError(
                f"Expected a {self.action_dim}-dim {self.action_space} action, got "
                f"shape {act.shape}"
            )
        if self.action_space == "osc":
            cmd = np.clip(act, -1.0, 1.0)
        else:
            q_now = np.asarray(
                self.env.env.robots[0]._joint_positions, dtype=np.float64
            )
            q_target = np.clip(act[:7], self._q_low, self._q_high)
            arm_cmd = np.clip((q_target - q_now) / JOINT_DELTA_MAX, -1.0, 1.0)
            cmd = np.concatenate([arm_cmd, [np.clip(act[7], -1.0, 1.0)]])
        raw_obs, _reward, env_done, _info = self.env.step(cmd)
        self._steps += 1
        success = bool(self.env.check_success())
        self._success = success
        self._done = success or bool(env_done) or self._steps >= self.max_steps
        return raw_obs

    @property
    def success(self) -> bool:
        return self._success

    @property
    def done(self) -> bool:
        return self._done

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


class SimEnvService:

    def __init__(self, max_steps: int | None = None,
                 num_steps_wait: int = DEFAULT_NUM_STEPS_WAIT,
                 image_size: int = 256,
                 reset_retries: int = 2,
                 action_space: str = "osc"):
        self.max_steps = max_steps
        self.num_steps_wait = int(num_steps_wait)
        self.image_size = int(image_size)
        self.reset_retries = int(reset_retries)
        self.action_space = action_space
        self.env: LiberoSimEnv | None = None
        self._env_key: tuple[str, int] | None = None

    def _ensure_env(self, suite_name: str, task_id: int):
        key = (suite_name, int(task_id))
        if self.env is None or self._env_key != key:
            if self.env is not None:
                self.env.close()
                self.env = None
            with ENV_BUILD_LOCK:
                self.env = LiberoSimEnv(
                    suite_name,
                    task_id,
                    max_steps=self.max_steps,
                    image_size=self.image_size,
                    num_steps_wait=self.num_steps_wait,
                    action_space=self.action_space,
                )
            self._env_key = key

    def reset(self, payload: dict) -> dict:
        suite_name = payload["suite"]
        task_id = int(payload["task_id"])
        episode = int(payload["episode"])
        last_err = "unknown"
        for _attempt in range(self.reset_retries + 1):
            try:
                self._ensure_env(suite_name, task_id)
                raw_obs = self.env.reset(episode)
                return {
                    "ok": True,
                    "episode": episode,
                    "instruction": self.env.instruction,
                    "frames": [encode_frame(raw_obs)],
                    "states": [encode_state(raw_obs)],
                    "max_steps": self.env.max_steps,
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                try:
                    if self.env is not None:
                        self.env.close()
                except Exception:
                    pass
                self.env = None
                self._env_key = None
        return {"ok": False, "reason": f"reset_failed: {last_err}"}

    def step(self, payload: dict) -> dict:
        action_chunk = np.asarray(payload["action_chunk"], dtype=np.float64)
        frames, states, consumed = [], [], 0
        error = None
        for action in action_chunk:
            try:
                raw_obs = self.env.step_one(action)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                break
            consumed += 1
            frames.append(encode_frame(raw_obs))
            states.append(encode_state(raw_obs))
            if self.env.done:
                break
        return {
            "frames": frames,
            "states": states,
            "consumed": consumed,
            "done": bool((self.env is not None and self.env.done) or error is not None),
            "success": bool(self.env.success) if self.env is not None else False,
            "status": "error" if error else (
                "success" if (self.env is not None and self.env.success) else
                ("done" if (self.env is not None and self.env.done) else "ongoing")
            ),
            "error_message": error,
        }

    def close(self, _payload=None) -> dict:
        if self.env is not None:
            self.env.close()
            self.env = None
            self._env_key = None
        return {"ok": True}
