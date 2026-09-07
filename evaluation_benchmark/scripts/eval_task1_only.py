from __future__ import annotations

import argparse
import dataclasses
import logging

import numpy as np

import eval_common as ec
import task2_26_reference_stage as stage_eval
from eval_common import parse_adapter_kwargs

TASK_ID = 1

STAGE_CHECKS = [
    (spec.name, lambda env, spec=spec: spec.check_fn(env, {}, 0))
    for spec in stage_eval._task_specs(TASK_ID)
]


@dataclasses.dataclass
class Args:
    adapter_spec: str
    adapter_kwargs: str = ""
    resize_size: int = 256
    env_camera_height: int = 480
    env_camera_width: int = 640
    replan_steps: int = 10
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    max_steps: int = 2500
    post_goal_steps: int = 200
    video_out_path: str = "outputs/task1_eval"
    seed: int = 100


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate task1 with a custom policy adapter.")
    parser.add_argument("--adapter-spec", required=True)
    parser.add_argument("--adapter-kwargs", default="")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--env-camera-height", type=int, default=480)
    parser.add_argument("--env-camera-width", type=int, default=640)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-trials-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--post-goal-steps", type=int, default=200)
    parser.add_argument("--video-out-path", default="outputs/task1_eval")
    parser.add_argument("--seed", type=int, default=100)
    return parser


def main(args: Args) -> dict:
    ec.patch_env_resolution(args.env_camera_height, args.env_camera_width)
    return ec.run_eval(
        task_id=TASK_ID,
        num_trials_per_task=args.num_trials_per_task,
        adapter_spec=args.adapter_spec,
        adapter_kwargs=parse_adapter_kwargs(args.adapter_kwargs),
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        post_goal_steps=args.post_goal_steps,
        video_out_path=args.video_out_path,
        seed=args.seed,
        stage_checks=STAGE_CHECKS,
        seed_everywhere_fn=lambda s: np.random.seed(s),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ns = build_argparser().parse_args()
    main(Args(**vars(ns)))
