from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import eval_common as ec
import eval_task1_only as task1_eval
import eval_tasks2_26 as tasks26
from policy_adapter import load_policy_adapter


EPISODES_HEADER = [
    "task_id",
    "ep",
    "seed",
    "TSR",
    "CSR",
    "extra_pour_detected",
    "pour_1_step",
    "pour_2_step",
    "extra_monitor_end_step",
    "failure_reason",
    "prompt",
    "video_dir",
]

TASK_SUMMARY_HEADER = [
    "task_id",
    "num_trials",
    "seed_start",
    "TSR",
    "CSR",
    "prompt",
    "video_dir",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-seed RoboMemArena Task 1-26 sweep with a policy adapter. "
            "This records every episode; it does not retry seeds or filter by stage score."
        )
    )
    parser.add_argument("--adapter-spec", required=True)
    parser.add_argument("--adapter-kwargs", default="")
    parser.add_argument("--task-start", type=int, default=1)
    parser.add_argument("--task-end", type=int, default=26)
    parser.add_argument("--num-trials-per-task", type=int, default=51)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--post-goal-steps", type=int, default=200)
    parser.add_argument(
        "--fail-on-extra-pour",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--extra-pour-monitor-steps",
        "--post-stage-steps",
        dest="extra_pour_monitor_steps",
        type=int,
        default=30,
    )
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--out-root", required=True)
    return parser


def _ensure_tsv(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)


def _run_task(
    *,
    task_id: int,
    adapter: Any,
    num_trials_per_task: int,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    post_goal_steps: int,
    fail_on_extra_pour: bool,
    extra_pour_monitor_steps: int,
    video_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if task_id == 1:
        return ec.run_eval(
            task_id=1,
            num_trials_per_task=num_trials_per_task,
            adapter=adapter,
            resize_size=resize_size,
            replan_steps=replan_steps,
            num_steps_wait=num_steps_wait,
            max_steps=max_steps,
            post_goal_steps=post_goal_steps,
            video_out_path=str(video_dir),
            seed=seed,
            stage_checks=task1_eval.STAGE_CHECKS,
            seed_everywhere_fn=lambda s: np.random.seed(s),
        )

    return tasks26.run_eval_task(
        task_id=task_id,
        num_trials_per_task=num_trials_per_task,
        adapter=adapter,
        resize_size=resize_size,
        replan_steps=replan_steps,
        num_steps_wait=num_steps_wait,
        max_steps=max_steps,
        post_goal_steps=post_goal_steps,
        fail_on_extra_pour=fail_on_extra_pour,
        extra_pour_monitor_steps=extra_pour_monitor_steps,
        video_out_path=str(video_dir),
        seed=seed,
    )


def _write_outputs(out_root: Path, results: list[dict[str, Any]], seed: int) -> None:
    episodes_tsv = out_root / "episodes.tsv"
    task_summary_tsv = out_root / "task_summary.tsv"
    summary_json = out_root / "summary.json"
    aggregate_json = out_root / "aggregate.json"

    _ensure_tsv(episodes_tsv, EPISODES_HEADER)
    _ensure_tsv(task_summary_tsv, TASK_SUMMARY_HEADER)

    with episodes_tsv.open("w", newline="", encoding="utf-8") as episodes_f:
        writer = csv.writer(episodes_f, delimiter="\t")
        writer.writerow(EPISODES_HEADER)
        for result in results:
            for episode in result["episodes"]:
                writer.writerow(
                    [
                        result["task_id"],
                        episode["ep"],
                        episode["seed"],
                        f"{float(episode.get('TSR', 0.0)):.1f}",
                        f"{float(episode.get('CSR', 0.0)):.1f}",
                        "Y" if episode.get("extra_pour_detected", False) else "N",
                        episode.get("pour_1_step"),
                        episode.get("pour_2_step"),
                        episode.get("extra_monitor_end_step"),
                        episode.get("failure_reason"),
                        result["prompt"],
                        result["video_dir"],
                    ]
                )

    with task_summary_tsv.open("w", newline="", encoding="utf-8") as summary_f:
        writer = csv.writer(summary_f, delimiter="\t")
        writer.writerow(TASK_SUMMARY_HEADER)
        for result in results:
            writer.writerow(
                [
                    result["task_id"],
                    len(result["episodes"]),
                    seed,
                    f"{float(result.get('TSR', 0.0)):.1f}",
                    f"{float(result.get('CSR', 0.0)):.1f}",
                    result["prompt"],
                    result["video_dir"],
                ]
            )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    num_tasks = len(results)
    aggregate = {
        "num_tasks": num_tasks,
        "num_episodes": sum(len(result["episodes"]) for result in results),
        "seed_start": seed,
        "TSR": sum(float(result.get("TSR", 0.0)) for result in results) / max(1, num_tasks),
        "CSR": sum(float(result.get("CSR", 0.0)) for result in results) / max(1, num_tasks),
    }
    aggregate_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()

    if args.task_start < 1 or args.task_end > 26 or args.task_start > args.task_end:
        raise ValueError(f"Invalid task range: {args.task_start}..{args.task_end}; expected within 1..26.")
    if args.num_trials_per_task < 1:
        raise ValueError("--num-trials-per-task must be >= 1.")

    adapter_kwargs = ec.parse_adapter_kwargs(args.adapter_kwargs)
    adapter = load_policy_adapter(args.adapter_spec, **adapter_kwargs)
    out_root = Path(args.out_root)
    video_root = out_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)

    tasks26._patch_env_resolution()
    results: list[dict[str, Any]] = []
    try:
        for task_id in range(args.task_start, args.task_end + 1):
            _, task_key = ec._resolve_task_id(task_id)
            bddl_path = ec._resolve_bddl_path(task_id)
            prompt = ec.get_prompt(task_key, bddl_path.stem)
            video_dir = video_root / f"task{task_id}"
            logging.info("task=%s seed_start=%s trials=%s prompt=%s", task_id, args.seed, args.num_trials_per_task, prompt)
            results.append(
                _run_task(
                    task_id=task_id,
                    adapter=adapter,
                    num_trials_per_task=args.num_trials_per_task,
                    resize_size=args.resize_size,
                    replan_steps=args.replan_steps,
                    num_steps_wait=args.num_steps_wait,
                    max_steps=args.max_steps,
                    post_goal_steps=args.post_goal_steps,
                    fail_on_extra_pour=args.fail_on_extra_pour,
                    extra_pour_monitor_steps=args.extra_pour_monitor_steps,
                    video_dir=video_dir,
                    seed=args.seed,
                )
            )
    finally:
        close_fn = getattr(adapter, "close", None)
        if callable(close_fn):
            close_fn()

    _write_outputs(out_root, results, args.seed)
    logging.info("Wrote Task 1-26 sweep outputs to %s", out_root)


if __name__ == "__main__":
    main()
