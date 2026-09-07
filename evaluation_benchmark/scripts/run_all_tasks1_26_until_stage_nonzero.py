from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

import eval_common as ec
import eval_task1_only as task1_eval
import eval_tasks2_26 as tasks26
from policy_adapter import load_policy_adapter


ATTEMPTS_HEADER = [
    "task_id",
    "attempt",
    "seed",
    "score_pct",
    "goal",
    "stage_nonzero",
    "prompt",
    "video_dir",
    "status",
]

TASK_SUMMARY_HEADER = [
    "task_id",
    "final_attempt",
    "final_seed",
    "score_pct",
    "goal",
    "prompt",
    "video_dir",
    "status",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run task1..26 until first non-zero stage score per task.")
    parser.add_argument("--adapter-spec", required=True)
    parser.add_argument("--adapter-kwargs", default="")
    parser.add_argument("--task-start", type=int, default=1)
    parser.add_argument("--task-end", type=int, default=26)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--out-root", required=True)
    return parser


def _ensure_tsv(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)


def _run_single_task(
    *,
    task_id: int,
    adapter,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    video_dir: Path,
    seed: int,
) -> dict:
    if task_id == 1:
        return ec.run_eval(
            task_id=1,
            num_trials_per_task=1,
            adapter=adapter,
            resize_size=resize_size,
            replan_steps=replan_steps,
            num_steps_wait=num_steps_wait,
            max_steps=max_steps,
            video_out_path=str(video_dir),
            seed=seed,
            stage_checks=task1_eval.STAGE_CHECKS,
            seed_everywhere_fn=lambda s: np.random.seed(s),
        )

    return tasks26.run_eval_task(
        task_id=task_id,
        num_trials_per_task=1,
        adapter=adapter,
        resize_size=resize_size,
        replan_steps=replan_steps,
        num_steps_wait=num_steps_wait,
        max_steps=max_steps,
        video_out_path=str(video_dir),
        seed=seed,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()
    adapter_kwargs = ec.parse_adapter_kwargs(args.adapter_kwargs)
    adapter = load_policy_adapter(args.adapter_spec, **adapter_kwargs)

    out_root = Path(args.out_root)
    video_root = out_root / "videos"
    attempts_tsv = out_root / "attempts.tsv"
    task_summary_tsv = out_root / "task_summary.tsv"
    attempts_jsonl = out_root / "attempts.jsonl"
    progress_txt = out_root / "progress.txt"

    _ensure_tsv(attempts_tsv, ATTEMPTS_HEADER)
    _ensure_tsv(task_summary_tsv, TASK_SUMMARY_HEADER)
    video_root.mkdir(parents=True, exist_ok=True)

    tasks26._patch_env_resolution()

    with attempts_tsv.open("a", newline="", encoding="utf-8") as attempts_f, \
        task_summary_tsv.open("a", newline="", encoding="utf-8") as summary_f, \
        attempts_jsonl.open("a", encoding="utf-8") as attempts_jsonl_f, \
        progress_txt.open("a", encoding="utf-8") as progress_f:

        attempts_writer = csv.writer(attempts_f, delimiter="\t")
        summary_writer = csv.writer(summary_f, delimiter="\t")

        for task_id in range(args.task_start, args.task_end + 1):
            attempt = 0
            seed = args.seed_start
            _, task_key = ec._resolve_task_id(task_id)
            bddl_path = ec._resolve_bddl_path(task_id)
            prompt = ec.get_prompt(task_key, bddl_path.stem)
            progress_f.write(f"task={task_id} prompt={prompt}\n")
            progress_f.flush()
            logging.info("task=%s prompt=%s", task_id, prompt)

            while True:
                attempt += 1
                video_dir = video_root / f"task{task_id}" / f"attempt_{attempt}_seed_{seed}"
                video_dir.mkdir(parents=True, exist_ok=True)

                result = _run_single_task(
                    task_id=task_id,
                    adapter=adapter,
                    resize_size=args.resize_size,
                    replan_steps=args.replan_steps,
                    num_steps_wait=args.num_steps_wait,
                    max_steps=args.max_steps,
                    video_out_path=str(video_dir),
                    seed=seed,
                )

                episode = result["episodes"][0]
                score_pct = float(episode["score_pct"])
                goal = "Y" if episode["goal_success"] else "N"
                stage_nonzero = "Y" if score_pct > 0 else "N"
                status = "stage_nonzero" if score_pct > 0 else "retry"

                attempts_writer.writerow(
                    [
                        task_id,
                        attempt,
                        seed,
                        f"{score_pct:.0f}",
                        goal,
                        stage_nonzero,
                        prompt,
                        str(video_dir),
                        status,
                    ]
                )
                attempts_f.flush()
                attempts_jsonl_f.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "attempt": attempt,
                            "seed": seed,
                            "score_pct": score_pct,
                            "goal": goal,
                            "stage_nonzero": stage_nonzero,
                            "prompt": prompt,
                            "video_dir": str(video_dir),
                            "status": status,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                attempts_jsonl_f.flush()

                if stage_nonzero == "Y":
                    summary_writer.writerow(
                        [
                            task_id,
                            attempt,
                            seed,
                            f"{score_pct:.0f}",
                            goal,
                            prompt,
                            str(video_dir),
                            status,
                        ]
                    )
                    summary_f.flush()
                    progress_f.write(
                        f"task={task_id} final_attempt={attempt} final_seed={seed} score={score_pct:.0f} goal={goal}\n"
                    )
                    progress_f.flush()
                    logging.info(
                        "[SUCCESS] task=%s attempt=%s seed=%s score=%.0f%% goal=%s",
                        task_id,
                        attempt,
                        seed,
                        score_pct,
                        goal,
                    )
                    break

                seed += 1

    close_fn = getattr(adapter, "close", None)
    if callable(close_fn):
        close_fn()


if __name__ == "__main__":
    main()
