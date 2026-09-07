
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simplememvla.data.messages import derive_video_sampling
from simplememvla.benchmarks.robomemarena import IMAGE_SIZE

EPISODES_HEADER = [
    "task_id",
    "ep",
    "seed",
    "TSR",
    "CSR",
    "stages_done",
    "extra_pour_detected",
    "pour_1_step",
    "pour_2_step",
    "extra_monitor_end_step",
    "failure_reason",
    "video_basename",
    "prompt",
    "video_dir",
]
TASK_SUMMARY_HEADER = [
    "task_id",
    "num_trials",
    "seed_start",
    "TSR",
    "TSR_ci_lo",
    "TSR_ci_hi",
    "CSR",
    "extra_pour_rate",
    "scene_mismatch",
    "prompt",
    "video_dir",
]

KNOWN_SCENE_MISMATCH_TASKS = (1, 2, 3)

RESUME_COMPATIBLE_KEYS = (
    "checkpoint",
    "checkpoint_fingerprint",
    "task_start",
    "task_end",
    "num_trials_per_task",
    "seed",
    "seed_policy",
    "max_steps",
    "post_goal_steps",
    "replan_steps",
    "num_steps_wait",
    "resize_size",
    "extra_pour_monitor_steps",
    "fail_on_extra_pour",
    "num_denoising_steps",
    "temperature",
    "max_reasoning_tokens",
    "compute_dtype",
    "attn_implementation",
    "episodes_per_shard",
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="SimpleMemVLA checkpoint dir (contains stats.json)")
    p.add_argument("--out-root", required=True)
    p.add_argument("--task-start", type=int, default=1)
    p.add_argument("--task-end", type=int, default=26)
    p.add_argument("--num-trials-per-task", type=int, default=51)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--post-goal-steps", type=int, default=200)
    p.add_argument("--replan-steps", type=int, default=10)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--resize-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--extra-pour-monitor-steps", "--post-stage-steps", dest="extra_pour_monitor_steps", type=int, default=30)
    p.add_argument("--fail-on-extra-pour", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-denoising-steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-reasoning-tokens", type=int, default=256)
    p.add_argument("--compute-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--log-every", type=int, default=0, help="log the generated sub-task every N decisions")
    p.add_argument(
        "--seed-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="seed the DiT's sampling noise per episode from seed+global_ep, so results "
        "do not depend on how the work was sharded",
    )
    p.add_argument("--gpus", default="0", help="comma-separated physical GPU ids")
    p.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="processes per GPU. Each holds its own ~13 GB bf16 model plus a MuJoCo/EGL "
        "context and peaks around 26 GB, so 2 fits an 80 GB card comfortably.",
    )
    p.add_argument("--episodes-per-shard", type=int, default=0, help="0 = one shard per task")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip shards whose result JSON already exists under --out-root",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="re-aggregate an existing --out-root and exit; touches no GPU",
    )
    p.add_argument(
        "--worker-poll-seconds",
        type=float,
        default=60.0,
        help="how often the parent wakes to reap workers that died without reporting",
    )
    p.add_argument(
        "--gpu-memory-cap-gib",
        type=float,
        default=None,
        help="hard cap on each worker's TORCH allocations, in GiB. On a SHARED GPU this "
        "turns 'do not starve the other tenant' from a hope into a mechanism: past the "
        "cap this process raises a clean CUDA OOM (its shard is reported and a resume "
        "re-runs it) instead of taking memory the co-tenant was going to need. "
        "Note the cap covers torch only -- MuJoCo/EGL's "
        "render context lives outside it, so leave it a few GiB below the free space.",
    )
    return p


class _SwallowedEpisodeCounter:

    PREFIX = "Episode failed:"

    def __init__(self) -> None:
        self.count = 0
        self.first_message: str | None = None

    def begin_shard(self) -> int:
        self.first_message = None
        return self.count

    def install(self) -> None:
        import logging

        counter = self

        class _Handler(logging.Handler):
            def emit(self, record: "logging.LogRecord") -> None:
                if record.levelno >= logging.ERROR:
                    try:
                        message = record.getMessage()
                    except Exception:
                        return
                    if message.startswith(counter.PREFIX):
                        counter.count += 1
                        if counter.first_message is None:
                            counter.first_message = message[:300]

        logging.getLogger().addHandler(_Handler())


def checkpoint_fingerprint(ckpt: Path) -> str:
    h = hashlib.sha256()
    for name in ("config.json", "stats.json"):
        h.update((ckpt / name).read_bytes())
    index = ckpt / "model.safetensors.index.json"
    shards = (
        sorted({ckpt / s for s in json.loads(index.read_text())["weight_map"].values()})
        if index.is_file()
        else [ckpt / "model.safetensors"]
    )
    for shard in shards:
        size = shard.stat().st_size
        h.update(str(size).encode())
        with shard.open("rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            h.update(f.read(header_len))
            for frac in (0.0, 0.25, 0.5, 0.75, 0.999):
                f.seek(max(0, min(int(size * frac), size - 16 * 1024 * 1024)))
                h.update(f.read(16 * 1024 * 1024))
    return h.hexdigest()


def shard_path(out_root: Path, task_id: int, ep_start: int) -> Path:
    return out_root / "shards" / f"task{task_id:02d}_ep{ep_start:04d}.json"


def plan_units(task_start: int, task_end: int, num_trials: int, per_shard: int) -> list[tuple[int, int, int]]:
    step = per_shard or num_trials
    units: list[tuple[int, int, int]] = []
    for task_id in range(task_start, task_end + 1):
        for start in range(0, num_trials, step):
            units.append((task_id, start, min(step, num_trials - start)))
    return units


ENV_BUILD_SEEDS = (None, *range(8))


def _worker(slot: int, gpu: str, units: list[tuple[int, int, int]], args: argparse.Namespace, out_dir: str, q) -> None:
    adapter = None
    try:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
        sys.path.insert(0, str(REPO_ROOT))

        from robomemarena_sim import assert_libero_fork, prepare_mujoco_runtime

        prepare_mujoco_runtime()

        import logging

        logging.basicConfig(level=logging.INFO, format=f"[w{slot}-gpu{gpu}] %(levelname)s %(message)s")
        swallowed = _SwallowedEpisodeCounter()
        swallowed.install()

        import eval_common as ec
        import eval_task1_only as task1_eval
        import eval_tasks2_26 as tasks26
        import numpy as np
        from robosuite.utils.errors import RandomizationError

        from robomemarena_sim.policy_adapter import build_adapter

        assert_libero_fork()
        tasks26._patch_env_resolution()

        if args.gpu_memory_cap_gib is not None:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("--gpu-memory-cap-gib was given but CUDA is unavailable")
            total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            fraction = float(args.gpu_memory_cap_gib) / total_gib
            if not 0.0 < fraction <= 1.0:
                raise ValueError(
                    f"--gpu-memory-cap-gib {args.gpu_memory_cap_gib} is not in (0, {total_gib:.1f}]"
                )
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            logging.info(
                "torch allocations capped at %.1f GiB of %.1f GiB (fraction %.3f)",
                args.gpu_memory_cap_gib,
                total_gib,
                fraction,
            )

        adapter = build_adapter(
            checkpoint_dir=args.checkpoint,
            compute_dtype=args.compute_dtype,
            attn_implementation=args.attn_implementation,
            num_denoising_steps=args.num_denoising_steps,
            temperature=args.temperature,
            max_reasoning_tokens=args.max_reasoning_tokens,
            log_every=args.log_every,
        )

        for task_id, ep_start, ep_count in units:
            try:
                video_dir = Path(out_dir) / "videos" / f"task{task_id}"
                common = dict(
                    task_id=task_id,
                    num_trials_per_task=ep_count,
                    adapter=adapter,
                    resize_size=args.resize_size,
                    replan_steps=args.replan_steps,
                    num_steps_wait=args.num_steps_wait,
                    max_steps=args.max_steps,
                    post_goal_steps=args.post_goal_steps,
                    video_out_path=str(video_dir),
                    seed=args.seed + ep_start,
                )
                last_build_err = ""
                for build_seed in ENV_BUILD_SEEDS:
                    t0 = time.time()
                    runaway_before = adapter.runaway_decodes
                    swallowed_before = swallowed.begin_shard()
                    adapter.begin_shard()
                    if args.seed_policy:
                        adapter.set_episode_seed_base(args.seed + ep_start)
                    episodes_before = adapter.episodes_seen
                    if build_seed is not None:
                        np.random.seed(build_seed)
                    try:
                        if task_id == 1:
                            result = ec.run_eval(
                                **common,
                                stage_checks=task1_eval.STAGE_CHECKS,
                                seed_everywhere_fn=lambda s: np.random.seed(s),
                            )
                        else:
                            result = tasks26.run_eval_task(
                                **common,
                                fail_on_extra_pour=args.fail_on_extra_pour,
                                extra_pour_monitor_steps=args.extra_pour_monitor_steps,
                            )
                        break
                    except RandomizationError as build_err:
                        if adapter.episodes_seen != episodes_before:
                            raise
                        last_build_err = f"{type(build_err).__name__}: {build_err}"
                        logging.warning(
                            "task%s ep%s: RandomizationError at env construction under "
                            "numpy state %r; retrying under the next one",
                            task_id,
                            ep_start,
                            build_seed,
                        )
                        build_err.__traceback__ = None
                        del build_err
                        gc.collect()
                else:
                    raise RuntimeError(
                        f"task{task_id} ep{ep_start}: robosuite could not seat every "
                        f"object under any of the {len(ENV_BUILD_SEEDS)} construction-"
                        f"time numpy states tried; investigate the task's placement "
                        f"regions. Last error: {last_build_err}"
                    )
                hit = swallowed.count - swallowed_before
                if hit:
                    raise RuntimeError(
                        f"{hit} episode(s) in this shard died inside the vendored runner and "
                        f"were scored on a truncated rollout; discarding the shard so a resume "
                        f"re-runs it. First: {swallowed.first_message!r}"
                    )
                for i, episode in enumerate(result.get("episodes", [])):
                    episode["ep"] = ep_start + i
                dt = time.time() - t0
                result["_shard"] = {
                    "task_id": task_id,
                    "ep_start": ep_start,
                    "ep_count": ep_count,
                    "seconds": dt,
                    "gpu": gpu,
                    "slot": slot,
                    "env_build_seed": build_seed,
                    "runaway_decodes": adapter.runaway_decodes - runaway_before,
                    "decision_seconds_median": adapter.decision_seconds_median(),
                }
                out = shard_path(Path(out_dir), task_id, ep_start)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                os.replace(tmp, out)
                q.put(("done", slot, gpu, task_id, ep_start, ep_count, dt))
            except Exception as exc:
                import traceback

                q.put((
                    "shard_error",
                    slot,
                    gpu,
                    task_id,
                    ep_start,
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                ))
                continue
    except BaseException as exc:
        import traceback

        try:
            q.put(("worker_error", slot, gpu, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if adapter is not None:
            close_fn = getattr(adapter, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        try:
            q.put(("exit", slot, gpu))
            q.close()
            q.join_thread()
        except Exception:
            pass


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0)


def _stage_rollup(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    order: list[str] = []
    excluded: set[str] = set()
    for e in episodes:
        stages = e.get("stage_done") or {}
        success = float(e.get("TSR", 0.0)) >= 100.0
        for name, ok in stages.items():
            if name not in counts:
                counts[name] = 0
                order.append(name)
            counts[name] += int(bool(ok))
            if success and not ok:
                excluded.add(name)
    n = len(episodes)
    return {
        "n": n,
        "excluded_from_scoring": sorted(excluded),
        "stages": [
            {
                "name": name,
                "completed": counts[name],
                "rate": 100.0 * counts[name] / max(1, n),
                "scored": name not in excluded,
            }
            for name in order
        ],
    }


def _aggregate(out_root: Path) -> dict[str, Any]:
    shard_dir = out_root / "shards"
    shards = sorted(shard_dir.glob("*.json"))
    if not shards:
        raise FileNotFoundError(f"No shard results under {shard_dir}")

    expected_units: list[tuple[int, int, int]] = []
    run_args: dict[str, Any] = {}
    run_args_path = out_root / "run_args.json"
    if run_args_path.is_file():
        run_args = json.loads(run_args_path.read_text())
        expected_units = plan_units(
            int(run_args["task_start"]),
            int(run_args["task_end"]),
            int(run_args["num_trials_per_task"]),
            int(run_args.get("episodes_per_shard") or 0),
        )

    by_task: dict[int, dict[str, Any]] = {}
    unreadable: list[str] = []
    duplicates: list[str] = []
    shard_seconds: dict[int, float] = {}
    runaway_total = 0
    decision_medians: list[float] = []
    for path in shards:
        try:
            payload = json.loads(path.read_text())
            tid = int(payload["task_id"])
        except Exception as exc:
            unreadable.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        entry = by_task.setdefault(
            tid,
            {
                "task_id": tid,
                "prompt": payload.get("prompt", ""),
                "video_dir": payload.get("video_dir", ""),
                "episodes": [],
                "_seen_eps": set(),
            },
        )
        for episode in payload.get("episodes", []):
            ep = int(episode["ep"])
            if ep in entry["_seen_eps"]:
                duplicates.append(f"task{tid} ep{ep} ({path.name})")
                continue
            entry["_seen_eps"].add(ep)
            entry["episodes"].append(episode)
        meta = payload.get("_shard") or {}
        if "seconds" in meta:
            shard_seconds[tid] = shard_seconds.get(tid, 0.0) + float(meta["seconds"])
        runaway_total += int(meta.get("runaway_decodes") or 0)
        if meta.get("decision_seconds_median"):
            decision_medians.append(float(meta["decision_seconds_median"]))

    expected_trials = int(run_args.get("num_trials_per_task", 0)) or None
    results = []
    for tid in sorted(by_task):
        entry = by_task[tid]
        entry.pop("_seen_eps", None)
        eps = sorted(entry["episodes"], key=lambda e: int(e["ep"]))
        entry["episodes"] = eps
        n = max(1, len(eps))
        tsr_successes = sum(1 for e in eps if float(e.get("TSR", 0.0)) >= 100.0)
        entry["TSR"] = sum(float(e.get("TSR", 0.0)) for e in eps) / n
        entry["CSR"] = sum(float(e.get("CSR", 0.0)) for e in eps) / n
        entry["TSR_ci"] = _wilson(tsr_successes, len(eps))
        entry["is_counting_pour"] = any(
            "Pour_One" in name for e in eps for name in (e.get("stage_done") or {})
        )
        entry["extra_pour_rate"] = (
            100.0 * sum(1 for e in eps if e.get("extra_pour_detected")) / len(eps) if eps else 0.0
        )
        entry["stage_rollup"] = _stage_rollup(eps)
        entry["scene_mismatch"] = tid in KNOWN_SCENE_MISMATCH_TASKS
        entry["complete"] = expected_trials is None or len(eps) == expected_trials
        entry["seconds"] = shard_seconds.get(tid, 0.0)
        results.append(entry)

    def _video_basename(task_id: int, episode: dict[str, Any]) -> str:
        succ = float(episode.get("TSR", 0.0)) >= 100.0
        return f"task{task_id}_{'success' if succ else 'failure'}_ep?_seed{episode.get('seed')}"

    with (out_root / "episodes.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(EPISODES_HEADER)
        for r in results:
            for e in r["episodes"]:
                stages = e.get("stage_done") or {}
                w.writerow(
                    [
                        r["task_id"],
                        e.get("ep"),
                        e.get("seed"),
                        f"{float(e.get('TSR', 0.0)):.1f}",
                        f"{float(e.get('CSR', 0.0)):.1f}",
                        ";".join(f"{k}={'Y' if v else 'N'}" for k, v in stages.items()),
                        "Y" if e.get("extra_pour_detected", False) else "N",
                        e.get("pour_1_step"),
                        e.get("pour_2_step"),
                        e.get("extra_monitor_end_step"),
                        e.get("failure_reason"),
                        _video_basename(r["task_id"], e),
                        r["prompt"],
                        r["video_dir"],
                    ]
                )
    with (out_root / "task_summary.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(TASK_SUMMARY_HEADER)
        for r in results:
            lo, hi = r["TSR_ci"]
            w.writerow(
                [
                    r["task_id"],
                    len(r["episodes"]),
                    min((int(e["seed"]) for e in r["episodes"]), default=0),
                    f"{r['TSR']:.1f}",
                    f"{lo:.1f}",
                    f"{hi:.1f}",
                    f"{r['CSR']:.1f}",
                    f"{r['extra_pour_rate']:.1f}" if r["is_counting_pour"] else "",
                    "Y" if r["scene_mismatch"] else "",
                    r["prompt"],
                    r["video_dir"],
                ]
            )
    (out_root / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

    def _macro(rows: list[dict[str, Any]], key: str) -> float | None:
        return sum(r[key] for r in rows) / len(rows) if rows else None

    def _macro_se(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        var = 0.0
        for r in rows:
            n = len(r["episodes"])
            if n <= 1:
                continue
            p = r["TSR"] / 100.0
            var += p * (1 - p) / n
        return 100.0 * math.sqrt(var) / len(rows)

    found_units = {(t, s) for t, s, _ in expected_units if shard_path(out_root, t, s).is_file()}
    missing_units = [
        {"task_id": t, "ep_start": s, "ep_count": c}
        for t, s, c in expected_units
        if (t, s) not in found_units
    ]
    without_mismatch = [r for r in results if not r["scene_mismatch"]]
    complete = (
        bool(expected_units)
        and not missing_units
        and not unreadable
        and not duplicates
        and all(r["complete"] for r in results)
    )
    aggregate = {
        "complete": complete,
        "num_tasks": len(results),
        "num_episodes": sum(len(r["episodes"]) for r in results),
        "expected_tasks": len({t for t, _, _ in expected_units}) or None,
        "expected_episodes": (
            len({t for t, _, _ in expected_units}) * int(run_args["num_trials_per_task"])
            if expected_units
            else None
        ),
        "expected_shards": len(expected_units) or None,
        "found_shards": len(found_units) or len(shards),
        "missing_shards": missing_units,
        "unreadable_shards": unreadable,
        "duplicate_episodes": duplicates,
        "TSR": _macro(results, "TSR"),
        "CSR": _macro(results, "CSR"),
        "TSR_macro_se": _macro_se(results),
        "TSR_excl_scene_mismatch": _macro(without_mismatch, "TSR"),
        "CSR_excl_scene_mismatch": _macro(without_mismatch, "CSR"),
        "scene_mismatch_tasks": list(KNOWN_SCENE_MISMATCH_TASKS),
        "extra_pour_rate_counting_tasks": _macro(
            [r for r in results if r["is_counting_pour"]], "extra_pour_rate"
        ),
        "runaway_subtask_decodes": runaway_total,
        "decision_seconds_median": (
            sorted(decision_medians)[len(decision_medians) // 2] if decision_medians else None
        ),
        "gpu_seconds": sum(r["seconds"] for r in results),
        "per_task": {
            r["task_id"]: {
                "TSR": r["TSR"],
                "TSR_ci": r["TSR_ci"],
                "CSR": r["CSR"],
                "n": len(r["episodes"]),
                "complete": r["complete"],
                "scene_mismatch": r["scene_mismatch"],
                "extra_pour_rate": r["extra_pour_rate"] if r["is_counting_pour"] else None,
                "stages": r["stage_rollup"]["stages"],
                "seconds": r["seconds"],
            }
            for r in results
        },
    }
    (out_root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    return aggregate


def _print_report(aggregate: dict[str, Any], out_root: Path) -> None:
    print("\n" + "=" * 78, flush=True)
    status = "COMPLETE" if aggregate["complete"] else "PARTIAL -- do not report as a full run"
    print(f"{status}   tasks={aggregate['num_tasks']} episodes={aggregate['num_episodes']}", flush=True)
    for tid, v in sorted(aggregate["per_task"].items(), key=lambda kv: int(kv[0])):
        lo, hi = v["TSR_ci"]
        flags = []
        if v["scene_mismatch"]:
            flags.append("scene-mismatch")
        if not v["complete"]:
            flags.append("SHORT")
        pour = f"  extra_pour={v['extra_pour_rate']:5.1f}%" if v["extra_pour_rate"] is not None else ""
        print(
            f"  task{int(tid):2d}: TSR={v['TSR']:5.1f}% [{lo:5.1f},{hi:5.1f}]  "
            f"CSR={v['CSR']:5.1f}%  (n={v['n']:2d}){pour}"
            + (f"  {' '.join(flags)}" if flags else ""),
            flush=True,
        )
        scored = [s for s in v["stages"] if s["scored"]]
        worst = sorted(scored, key=lambda s: s["rate"])[:1]
        if worst and worst[0]["rate"] < 100.0:
            print(f"           weakest scored stage: {worst[0]['name']} {worst[0]['rate']:.1f}%", flush=True)
        skipped = [s for s in v["stages"] if not s["scored"]]
        if skipped:
            print(
                "           not scored by the benchmark: "
                + ", ".join(f"{s['name']} {s['rate']:.1f}%" for s in skipped),
                flush=True,
            )
    def _pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}%"

    print(
        f"MEAN  TSR={_pct(aggregate['TSR'])} (+-{aggregate['TSR_macro_se']:.2f} SE)   "
        f"CSR={_pct(aggregate['CSR'])}",
        flush=True,
    )
    print(
        f"      excl. tasks {aggregate['scene_mismatch_tasks']} (known scene mismatch): "
        f"TSR={_pct(aggregate['TSR_excl_scene_mismatch'])}  "
        f"CSR={_pct(aggregate['CSR_excl_scene_mismatch'])}",
        flush=True,
    )
    if aggregate["extra_pour_rate_counting_tasks"] is not None:
        print(
            f"      counting-pour tasks, extra-pour rate: "
            f"{aggregate['extra_pour_rate_counting_tasks']:.2f}%",
            flush=True,
        )
    if aggregate["runaway_subtask_decodes"]:
        print(
            f"      WARNING: {aggregate['runaway_subtask_decodes']} sub-task decode(s) hit the token "
            "budget without emitting <|im_end|> -- the DiT condition span was un-terminated there",
            flush=True,
        )
    if aggregate["missing_shards"]:
        print(f"      missing shards: {len(aggregate['missing_shards'])}", flush=True)
    if aggregate["unreadable_shards"]:
        print(f"      unreadable shards: {aggregate['unreadable_shards']}", flush=True)
    if aggregate["duplicate_episodes"]:
        print(f"      duplicate episodes skipped: {len(aggregate['duplicate_episodes'])}", flush=True)
    print("=" * 78, flush=True)
    print(f"outputs: {out_root}", flush=True)


def _preflight(args: argparse.Namespace, ckpt: Path) -> None:
    config = json.loads((ckpt / "config.json").read_text())
    action_horizon = int(config["action_horizon"])
    if args.replan_steps > action_horizon:
        raise ValueError(
            f"--replan-steps {args.replan_steps} exceeds the checkpoint's action_horizon "
            f"{action_horizon}; the runner would silently execute only {action_horizon} "
            "steps per chunk."
        )
    if args.resize_size != IMAGE_SIZE:
        raise ValueError(
            f"--resize-size {args.resize_size} != the trained frame size {IMAGE_SIZE}. The "
            "vendored `_process_image_match_eval26` really does resize to this, so the "
            "policy would receive frames the checkpoint was never trained on."
        )
    n_frames, stride = derive_video_sampling(
        float(config["history_video_sec"]),
        float(config["history_video_fps"]),
        float(config["native_video_fps"]),
    )
    span = (n_frames - 1) * stride + 1
    if args.max_steps > span:
        raise ValueError(
            f"--max-steps {args.max_steps} exceeds the history window span {span} native "
            f"frames ({n_frames} frames at stride {stride}). Past that the window SLIDES and "
            "episode frame 0 leaves the clip -- on a memory benchmark that silently removes "
            "the very thing every task has to remember."
        )


def main() -> int:
    args = build_argparser().parse_args()
    out_root = Path(args.out_root)

    if args.aggregate_only:
        if not out_root.is_dir():
            raise FileNotFoundError(f"--aggregate-only: {out_root} does not exist")
        aggregate = _aggregate(out_root)
        _print_report(aggregate, out_root)
        return 0 if aggregate["complete"] else 1

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --aggregate-only is given")
    if not (1 <= args.task_start <= args.task_end <= 26):
        raise ValueError(f"Invalid task range {args.task_start}..{args.task_end}; expected within 1..26.")
    if args.num_trials_per_task < 1:
        raise ValueError("--num-trials-per-task must be >= 1.")
    if args.episodes_per_shard < 0:
        raise ValueError("--episodes-per-shard must be >= 0 (0 means one shard per task).")
    ckpt = Path(args.checkpoint)
    if not (ckpt / "stats.json").is_file():
        raise FileNotFoundError(
            f"{ckpt}/stats.json is missing. The trainer copies <data_root>/meta/stats.json "
            "into every checkpoint; without it the action unnormalization reference is unknown."
        )
    _preflight(args, ckpt)

    (out_root / "shards").mkdir(parents=True, exist_ok=True)
    (out_root / "videos").mkdir(parents=True, exist_ok=True)

    record = dict(vars(args))
    record["checkpoint_fingerprint"] = checkpoint_fingerprint(ckpt)

    run_args_path = out_root / "run_args.json"
    if run_args_path.is_file():
        previous = json.loads(run_args_path.read_text())
        absent = [key for key in RESUME_COMPATIBLE_KEYS if key not in previous]
        if absent:
            raise ValueError(
                f"{run_args_path} predates the {absent} flag(s), so its protocol cannot be "
                "compared with this run's; refusing to guess. Use a fresh --out-root (or "
                "aggregate the old one read-only with --aggregate-only)."
            )
        incompatible = {
            key: (previous[key], record[key])
            for key in RESUME_COMPATIBLE_KEYS
            if previous[key] != record[key]
        }
        if incompatible:
            hint = ""
            if "checkpoint_fingerprint" in incompatible:
                hint = (
                    " The checkpoint at this path has DIFFERENT WEIGHTS than the one whose "
                    "results are already in this out-root -- resuming would mix two models "
                    "into one aggregate."
                )
            raise ValueError(
                f"{run_args_path} records a DIFFERENT experiment; refusing to mix results "
                f"in one out-root. Mismatched: {sorted(incompatible)}.{hint} "
                "Use a fresh --out-root."
            )
    else:
        run_args_path.write_text(json.dumps(record, indent=2) + "\n")

    all_units = plan_units(args.task_start, args.task_end, args.num_trials_per_task, args.episodes_per_shard)
    if args.resume:
        units = [u for u in all_units if not shard_path(out_root, u[0], u[1]).is_file()]
        skipped = len(all_units) - len(units)
        if skipped:
            print(f"resume: {skipped} of {len(all_units)} shard(s) already on disk, skipping them", flush=True)
    else:
        units = list(all_units)
    if not units:
        print("nothing left to run; aggregating what is on disk", flush=True)
        aggregate = _aggregate(out_root)
        _print_report(aggregate, out_root)
        return 0 if aggregate["complete"] else 1

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError(f"--gpus {args.gpus!r} parsed to an empty GPU list")
    slots = [g for g in gpus for _ in range(max(1, args.workers_per_gpu))]
    buckets: list[list[tuple[int, int, int]]] = [[] for _ in slots]
    for i, unit in enumerate(units):
        buckets[i % len(slots)].append(unit)

    print(f"{len(units)} shard(s) over {len(slots)} worker(s) on GPUs {gpus}", flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs: dict[int, Any] = {}
    for slot, (gpu, bucket) in enumerate(zip(slots, buckets)):
        if not bucket:
            continue
        p = ctx.Process(target=_worker, args=(slot, gpu, bucket, args, str(out_root), q), name=f"w{slot}-gpu{gpu}")
        p.start()
        procs[slot] = p

    pending = set(procs)
    done, shard_errors, worker_errors, vanished = 0, [], [], []
    t0 = time.time()
    while pending:
        try:
            msg = q.get(timeout=args.worker_poll_seconds)
        except queue_mod.Empty:
            for slot in sorted(pending):
                if not procs[slot].is_alive():
                    vanished.append(slot)
                    pending.discard(slot)
                    print(
                        f"worker w{slot} died without reporting (exitcode={procs[slot].exitcode}); "
                        "its remaining shards are simply absent and a resume will re-run them",
                        file=sys.stderr,
                        flush=True,
                    )
            continue
        kind = msg[0]
        if kind == "done":
            _, slot, gpu, task_id, ep_start, ep_count, dt = msg
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (len(units) - done)
            print(
                f"[{done}/{len(units)}] task{task_id} ep{ep_start}..{ep_start + ep_count - 1} "
                f"on w{slot}-gpu{gpu} in {dt / 60:.1f} min "
                f"(elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min)",
                flush=True,
            )
        elif kind == "shard_error":
            _, slot, gpu, task_id, ep_start, err, tb = msg
            shard_errors.append(msg)
            print(
                f"SHARD FAILED task{task_id} ep{ep_start} on w{slot}-gpu{gpu}: {err}\n{tb}",
                file=sys.stderr,
                flush=True,
            )
        elif kind == "worker_error":
            _, slot, gpu, err, tb = msg
            worker_errors.append(msg)
            print(f"WORKER FAILED w{slot}-gpu{gpu}: {err}\n{tb}", file=sys.stderr, flush=True)
        elif kind == "exit":
            pending.discard(msg[1])
    for p in procs.values():
        p.join()

    failures = len(shard_errors) + len(worker_errors) + len(vanished)
    if failures:
        print(
            f"\n{len(worker_errors)} worker failure(s), {len(shard_errors)} shard failure(s), "
            f"{len(vanished)} worker(s) vanished. Re-run the same command to retry the missing "
            "shards.",
            file=sys.stderr,
            flush=True,
        )
    if not any((out_root / "shards").glob("*.json")):
        print(
            f"No shard completed, so there is nothing to aggregate under {out_root}/shards. "
            "See the failures above.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    aggregate = _aggregate(out_root)
    _print_report(aggregate, out_root)
    return 0 if aggregate["complete"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
