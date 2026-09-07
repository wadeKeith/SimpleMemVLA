#!/usr/bin/env python3
"""Replay recorded expert actions through the public counting-pour scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASK_IDS = (6, 7, 8, 9, 10, 15, 16, 22)
SEED_RE = re.compile(r"_seed(?P<seed>\d+)_task(?P<task>\d+)\.hdf5$")
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "evaluation_benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_common as ec  # noqa: E402
import task2_26_reference_stage as stages  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(TASK_IDS))
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument(
        "--stop-after-pour-events",
        type=int,
        default=None,
        help="Stop after this many detected pour events to validate an incomplete prefix.",
    )
    return parser.parse_args()


def _seed_for(path: Path, task_id: int) -> int:
    match = SEED_RE.search(path.name)
    if match is None or int(match.group("task")) != task_id:
        raise ValueError(f"Cannot parse matching task/seed from {path}")
    return int(match.group("seed"))


def _task_paths(hdf_root: Path, task_id: int, count: int) -> list[Path]:
    task_dirs = sorted(path for path in hdf_root.glob(f"{task_id}_*") if path.is_dir())
    if len(task_dirs) != 1:
        raise RuntimeError(f"Expected one task directory for {task_id}, found {task_dirs}")
    paths = sorted(task_dirs[0].glob("*.hdf5"))
    if len(paths) < count:
        raise RuntimeError(f"Task {task_id} has {len(paths)} demonstrations, need {count}")
    return paths[:count]


def _load_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        first_eef = np.asarray(demo["obs/ee_pos"][0], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"Expected [T, 7] actions in {path}, got {actions.shape}")
    return actions, first_eef


def _make_env(task_id: int) -> Any:
    return ec._get_env_class()(
        bddl_file_name=str(ec._resolve_bddl_path(task_id)),
        camera_heights=480,
        camera_widths=640,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
        initialization_noise=None,
    )


def _replay_one(task_id: int, path: Path, stop_after_pour_events: int | None) -> dict[str, Any]:
    seed = _seed_for(path, task_id)
    actions, hdf_first_eef = _load_actions(path)
    np.random.seed(seed)
    env = _make_env(task_id)
    try:
        try:
            env.seed(seed)
        except AttributeError:
            pass
        obs = env.reset()
        env_first_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        state = stages._build_initial_state(env)
        specs = stages._task_specs(task_id)
        stage_done = {spec.name: False for spec in specs}
        stage_idx = 0
        stage_start = int(state["step_idx"])
        trigger_steps: dict[str, int] = {}
        extra_check = stages._extra_pour_check(task_id)
        monitor_start: int | None = None
        monitor_deadline: int | None = None
        extra_pour = False
        executed_steps = 0
        truncated_after_pour_event = False

        for step, action in enumerate(actions):
            obs, _, _, _ = env.step(action.tolist())
            executed_steps = step + 1
            stages._update_state(obs, state)
            if stage_idx < len(specs):
                spec = specs[stage_idx]
                if spec.check_fn(env, state, stage_start):
                    stage_done[spec.name] = True
                    trigger_steps[spec.name] = step
                    stage_idx += 1
                    stage_start = int(state["step_idx"])
                    if spec.name.endswith("_Pour_Two"):
                        monitor_start = int(state["step_idx"])
                        monitor_deadline = step + 30
            if (
                extra_check is not None
                and monitor_start is not None
                and monitor_deadline is not None
                and step <= monitor_deadline
                and int(state["step_idx"]) > monitor_start
                and extra_check(env, state, monitor_start)
            ):
                extra_pour = True
            if stop_after_pour_events is not None:
                detected_events = max(
                    (counter.event_count for counter in state.get("shared_pour_counters", {}).values()),
                    default=0,
                )
                if detected_events >= stop_after_pour_events:
                    truncated_after_pour_event = True
                    break

        all_stages_done = bool(stage_done) and all(stage_done.values())
        monitor_complete = monitor_deadline is None or executed_steps - 1 >= monitor_deadline
        counter_rows = []
        for key, counter in state.get("shared_pour_counters", {}).items():
            counter_rows.append({"key": list(key), "event_count": counter.event_count, "events": counter.events})
        stage_success = all_stages_done and monitor_complete and not extra_pour
        expected_incomplete = stop_after_pour_events is not None
        validation_passed = (
            not stage_success
            and truncated_after_pour_event
            and any(counter["event_count"] == stop_after_pour_events for counter in counter_rows)
            and any(name.endswith("_Pour_One") and done for name, done in stage_done.items())
            and any(name.endswith("_Pour_Two") and not done for name, done in stage_done.items())
            if expected_incomplete
            else stage_success
        )
        return {
            "task_id": task_id,
            "seed": seed,
            "source_hdf": str(path),
            "source_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "executed_action_sha256": hashlib.sha256(actions[:executed_steps].tobytes()).hexdigest(),
            "source_num_actions": int(len(actions)),
            "executed_num_actions": int(executed_steps),
            "stop_after_pour_events": stop_after_pour_events,
            "truncated_after_pour_event": truncated_after_pour_event,
            "reset_eef_l2": float(np.linalg.norm(env_first_eef - hdf_first_eef)),
            "stage_done": stage_done,
            "stage_trigger_steps": trigger_steps,
            "stage_score_pct": stages._stage_score_pct(task_id, stage_done),
            "stage_success": stage_success,
            "validation_passed": validation_passed,
            "extra_pour_detected": extra_pour,
            "counter_rows": counter_rows,
        }
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    unknown = sorted(set(args.task_ids) - set(TASK_IDS))
    if unknown:
        raise ValueError(f"Unsupported counting-pour tasks: {unknown}")
    if args.samples_per_task < 1:
        raise ValueError("samples-per-task must be positive")
    if args.stop_after_pour_events is not None and args.stop_after_pour_events < 1:
        raise ValueError("stop-after-pour-events must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ec.patch_env_resolution(480, 640)

    rows = []
    for task_id in args.task_ids:
        for path in _task_paths(args.hdf_root, task_id, args.samples_per_task):
            row = _replay_one(task_id, path, args.stop_after_pour_events)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "task_ids": args.task_ids,
        "samples_per_task": args.samples_per_task,
        "checked": len(rows),
        "mode": "incomplete_prefix" if args.stop_after_pour_events is not None else "full_trajectory",
        "passed": sum(bool(row["validation_passed"]) for row in rows),
        "all_checked_passed": all(bool(row["validation_passed"]) for row in rows),
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({"complete": summary["all_checked_passed"], "out_dir": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
