
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import yaml

from rmbench_sim._sapien_env import prepare_sapien_runtime

prepare_sapien_runtime()

SIM_CAMERAS = ("head_camera", "left_camera", "right_camera")

VENDORED_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rmbench")


def _materialize_curobo_configs(repo: str) -> None:
    embodiments = os.path.join(repo, "assets", "embodiments")
    if not os.path.isdir(embodiments):
        return
    for dirpath, _dirnames, filenames in os.walk(embodiments):
        for name in filenames:
            if not (name.startswith("curobo") and name.endswith("_tmp.yml")):
                continue
            template = os.path.join(dirpath, name)
            target = os.path.join(dirpath, name[: -len("_tmp.yml")] + ".yml")
            try:
                with open(template, "r", encoding="utf-8") as f:
                    rendered = f.read().replace("${ASSETS_PATH}", repo)
                current = None
                if os.path.isfile(target):
                    with open(target, "r", encoding="utf-8") as f:
                        current = f.read()
                if current != rendered:
                    with open(target, "w", encoding="utf-8") as f:
                        f.write(rendered)
                    print(f"[sim] rendered CuRobo config for this checkout: {target}", flush=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not render CuRobo config '{target}' from '{template}': {exc}"
                ) from exc


def _resolve_sim_dir(repo_root: str | None) -> str:
    repo = repo_root or VENDORED_SIM_DIR
    if not os.path.isdir(repo) or not os.path.isdir(os.path.join(repo, "envs")):
        raise RuntimeError(
            f"RMBench sim not found at '{repo}' (no envs/). The sim is vendored at "
            "rmbench_sim/rmbench/; if you just cloned SimpleMemVLA, fetch the gitignored sim data "
            "(assets/ and envs/curobo/) from ModelScope into that dir (see README)."
        )
    if not os.path.isdir(os.path.join(repo, "assets")):
        raise RuntimeError(
            f"RMBench sim at '{repo}' is missing assets/ (gitignored heavy data). Fetch assets/ "
            "and envs/curobo/ from ModelScope into rmbench_sim/rmbench/ (see README)."
        )
    repo = os.path.abspath(repo)
    _materialize_curobo_configs(repo)
    return repo


def pin_worker_gpu(gpu_id: int) -> str:
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    dev = visible[gpu_id] if gpu_id < len(visible) else str(gpu_id)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    return dev


def _setup_rmbench_path(repo_root: str) -> None:
    for p in (
        repo_root,
        os.path.join(repo_root, "envs", "curobo", "src"),
        os.path.join(repo_root, "policy"),
        os.path.join(repo_root, "description", "utils"),
        os.path.join(repo_root, "script"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


class RMBenchSimEnv:
    def __init__(
        self,
        task_name: str,
        task_config: str,
        repo_root: str,
        eval_video_log: bool = False,
    ):
        self.task_name = task_name
        self.task_config = task_config
        self.repo_root = _resolve_sim_dir(repo_root)
        _setup_rmbench_path(self.repo_root)
        os.chdir(self.repo_root)

        import importlib

        from envs import CONFIGS_PATH
        from envs.utils.create_actor import UnStableError

        self._UnStableError = UnStableError
        self._CONFIGS_PATH = CONFIGS_PATH

        with open(os.path.join(self.repo_root, "task_config", f"{task_config}.yml"), "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        args["task_name"] = task_name
        args["task_config"] = task_config
        args["ckpt_setting"] = None
        if not eval_video_log:
            args["eval_video_log"] = False

        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        def embodiment_file(name):
            robot_file = embodiment_types[name]["file_path"]
            if robot_file is None:
                raise ValueError("No embodiment files")
            return robot_file

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        def embodiment_cfg(robot_file):
            with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        args["left_embodiment_config"] = embodiment_cfg(args["left_robot_file"])
        args["right_embodiment_config"] = embodiment_cfg(args["right_robot_file"])
        args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
        args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
        args["eval_mode"] = True
        args["render_freq"] = 0

        self.args = args
        envs_module = importlib.import_module(f"envs.{task_name}")
        self.TASK_ENV = getattr(envs_module, task_name)()

        from generate_episode_instructions import generate_episode_descriptions

        self._gen_instructions = generate_episode_descriptions
        self._closed = True

    def find_solvable_seed(self, start_seed: int, max_tries: int = 200) -> tuple[int, dict] | None:
        seed = start_seed
        for _ in range(max_tries):
            try:
                self.TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
                episode_info = self.TASK_ENV.play_once()
                solved = bool(self.TASK_ENV.plan_success and self.TASK_ENV.check_success())
                self.TASK_ENV.close_env()
            except self._UnStableError:
                self._safe_close()
                seed += 1
                continue
            except Exception:
                self._safe_close()
                seed += 1
                continue
            if solved:
                return seed, episode_info
            seed += 1
        return None

    def make_instruction(self, episode_info: dict, instruction_type: str = "seen", test_num: int = 100) -> str:
        info_list = [episode_info["info"]]
        results = self._gen_instructions(self.task_name, info_list, test_num)
        return str(np.random.choice(results[0][instruction_type]))

    def reset(self, seed: int, instruction: str | None = None) -> dict[str, Any]:
        self.TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
        if instruction is not None:
            self.TASK_ENV.set_instruction(instruction=instruction)
        self._closed = False
        return self.TASK_ENV.get_obs()

    def get_obs(self) -> dict[str, Any]:
        return self.TASK_ENV.get_obs()

    @property
    def instruction(self) -> str:
        return self.TASK_ENV.get_instruction()

    @property
    def step_lim(self) -> int:
        return int(self.TASK_ENV.step_lim)

    @property
    def take_action_cnt(self) -> int:
        return int(self.TASK_ENV.take_action_cnt)

    @property
    def success(self) -> bool:
        return bool(self.TASK_ENV.eval_success)

    @property
    def max_reward(self) -> float:
        return float(self.TASK_ENV.max_reward)

    @property
    def done(self) -> bool:
        return self.success or (self.take_action_cnt >= self.step_lim)

    def take_action_single(self, action: np.ndarray) -> None:
        self.TASK_ENV.take_action(np.asarray(action, dtype=np.float64), action_type="qpos")

    def take_action_dense(self, action: np.ndarray, substeps: int = 15) -> bool:
        TE = self.TASK_ENV
        if TE.take_action_cnt >= TE.step_lim or TE.eval_success:
            return bool(TE.eval_success)

        robot = TE.robot
        la = int(self.args["left_arm_dim"])
        ra = int(self.args["right_arm_dim"])
        q = np.asarray(action, dtype=np.float64)
        left_arm = q[:la]
        left_grip = float(q[la])
        right_arm = q[la + 1: la + 1 + ra]
        right_grip = float(q[la + 1 + ra])

        cur = np.asarray(
            robot.get_left_arm_jointState() + robot.get_right_arm_jointState(),
            dtype=np.float64,
        )
        dt = float(TE.scene.get_timestep())
        denom = max(substeps, 1) * dt
        left_vel = (left_arm - cur[:la]) / denom
        right_vel = (right_arm - cur[la + 1: la + 1 + ra]) / denom

        n_sub = max(int(substeps), 1)
        left_grip_start = float(cur[la])
        right_grip_start = float(cur[la + 1 + ra])
        left_grip_seq = np.linspace(left_grip_start, left_grip, n_sub + 1)[1:]
        right_grip_seq = np.linspace(right_grip_start, right_grip, n_sub + 1)[1:]

        TE.take_action_cnt += 1
        latched = False
        for s in range(n_sub):
            robot.set_arm_joints(left_arm, left_vel, "left")
            robot.set_gripper(float(left_grip_seq[s]), "left")
            robot.set_arm_joints(right_arm, right_vel, "right")
            robot.set_gripper(float(right_grip_seq[s]), "right")
            TE.scene.step()
            if TE.check_success():
                TE.eval_success = True
                latched = True
                break
        TE._update_render()
        return bool(latched or TE.eval_success)

    def step(self, action_chunk: np.ndarray, substeps: int = 15) -> tuple[dict, float, bool, dict]:
        consumed = 0
        for action in np.asarray(action_chunk, dtype=np.float64):
            self.take_action_dense(action, substeps=substeps)
            consumed += 1
            if self.success:
                break
            if self.take_action_cnt >= self.step_lim:
                break
        obs = self.TASK_ENV.get_obs()
        info = {"consumed": consumed, "take_action_cnt": self.take_action_cnt}
        return obs, self.max_reward, self.done, info

    def _safe_close(self):
        try:
            self.TASK_ENV.close_env()
        except Exception:
            pass

    def close(self, clear_cache: bool = False):
        if not self._closed:
            try:
                self.TASK_ENV.close_env(clear_cache=clear_cache)
            except Exception:
                self._safe_close()
            self._closed = True


def encode_obs(observation: dict) -> dict[str, np.ndarray]:
    cam_map = {"head_camera": "cam_high", "left_camera": "cam_left_wrist", "right_camera": "cam_right_wrist"}
    cams = observation["observation"]
    return {dst: np.asarray(cams[src]["rgb"], dtype=np.uint8) for src, dst in cam_map.items()}


def obs_state(observation: dict) -> np.ndarray:
    return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)


class SimEnvService:

    def __init__(self, task_config: str, repo_root: str, instruction_type: str, dense_substeps: int = 15):
        self.task_config = task_config
        self.repo_root = repo_root
        self.instruction_type = instruction_type
        self.dense_substeps = int(dense_substeps)
        self.env: RMBenchSimEnv | None = None
        self.task_name: str | None = None

    def _ensure_env(self, task_name: str):
        if self.env is None or self.task_name != task_name:
            if self.env is not None:
                self.env.close()
            self.env = RMBenchSimEnv(task_name, self.task_config, self.repo_root)
            self.task_name = task_name

    def reset(self, payload: dict) -> dict:
        task_name = payload["task"]
        seed = int(payload["seed"])
        instruction = payload.get("instruction")
        expert_check = bool(payload.get("expert_check", False))
        max_seed_search = int(payload.get("max_seed_search", 50))
        self._ensure_env(task_name)

        used_seed = seed
        if expert_check:
            found = self.env.find_solvable_seed(seed, max_tries=max_seed_search)
            if found is None:
                return {"ok": False, "reason": "no_solvable_seed"}
            used_seed, episode_info = found
            if not instruction:
                instruction = self.env.make_instruction(episode_info, self.instruction_type)
        if not instruction:
            instruction = f"Complete the {task_name} task."
        intended_seed = used_seed

        for _ in range(max_seed_search):
            try:
                obs = self.env.reset(used_seed, instruction=instruction)
                if used_seed != intended_seed:
                    print(f"[sim] WARNING: {task_name} reset seed drifted "
                          f"{intended_seed} -> {used_seed} (setup retries); "
                          "instruction kept from the intended seed, "
                          "expert-solvability NOT re-checked.", flush=True)
                return {
                    "ok": True,
                    "seed": used_seed,
                    "instruction": instruction,
                    "frame": encode_obs(obs),
                    "state": obs_state(obs),
                    "step_lim": self.env.step_lim,
                }
            except Exception:
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env = None
                self.task_name = None
                used_seed += 1
                try:
                    self._ensure_env(task_name)
                except Exception:
                    return {"ok": False, "reason": "reset_rebuild_failed"}
        return {"ok": False, "reason": "reset_failed"}

    def step(self, payload: dict) -> dict:
        action_chunk = np.asarray(payload["action_chunk"], dtype=np.float64)
        substeps = int(payload.get("dense_substeps", self.dense_substeps))
        stride = max(1, int(payload.get("capture_stride", 1)))
        base = int(payload.get("frame_index", 0))
        n = len(action_chunk)
        frames, states, offsets, consumed = [], [], [], 0
        for j, action in enumerate(action_chunk, start=1):
            self.env.take_action_dense(action, substeps=substeps)
            consumed += 1
            last = (j == n) or self.env.success or self.env.take_action_cnt >= self.env.step_lim
            if (base + j) % stride == 0 or last:
                obs = self.env.get_obs()
                frames.append(encode_obs(obs))
                states.append(obs_state(obs))
                offsets.append(j)
            elif stride > 1:
                self.env.TASK_ENV._update_render()
            if last:
                break
        return {
            "frames": frames,
            "frame_offsets": offsets,
            "states": states,
            "consumed": consumed,
            "done": self.env.done,
            "success": self.env.success,
            "finish_step": self.env.take_action_cnt,
            "step_lim": self.env.step_lim,
        }

    def obs(self, _payload=None) -> dict:
        obs = self.env.get_obs()
        return {"frame": encode_obs(obs), "state": obs_state(obs)}

    def close(self, _payload=None) -> dict:
        if self.env is not None:
            self.env.close()
            self.env = None
        return {"ok": True}
