
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rmbench_sim._sapien_env import prepare_sapien_runtime

prepare_sapien_runtime()

import argparse
import contextlib
import fcntl
import gc
import json
import os
import tempfile
import traceback
import urllib.request
from datetime import datetime, timezone

_FATAL_MARKERS = (
    "out of memory", "cuda error", "illegal", "no kernel image",
    "device-side assert", "left_planner", "right_planner", "invalid argument",
)
_RENDER_DEVICE_MARKER = "rendering device"
_UNSUPPORTED_SETUP_FAIL_LIMIT = 10
_RENDER_INIT_LOCKFILE = os.path.join(tempfile.gettempdir(), "simplememvla_sapien_render_init.lock")
_VULKAN_INITED = False

DEFAULT_TASKS = [
    "observe_and_pickup", "put_back_block", "swap_T", "press_button", "place_block_mat",
    "battery_try", "rearrange_blocks", "swap_blocks", "cover_blocks", "blocks_ranking_try",
]

_DEMO_SEED_HF_URL = ("https://huggingface.co/datasets/TianxingChen/RMBench/resolve/main/"
                     "data/{task}/{task_config}/seed.txt")


def _read_demo_seeds(task: str, task_config: str, seed_txt_dir: str | None) -> list[int]:
    if seed_txt_dir:
        path = os.path.join(seed_txt_dir, f"{task}.txt")
        if not os.path.isfile(path):
            path = os.path.join(seed_txt_dir, task, task_config, "seed.txt")
        with open(path, encoding="utf-8") as f:
            text = f.read()
    else:
        url = _DEMO_SEED_HF_URL.format(task=task, task_config=task_config)
        with urllib.request.urlopen(url, timeout=30) as r:
            text = r.read().decode("utf-8")
    return [int(tok) for tok in text.split() if tok.strip().lstrip("-").isdigit()]


def _demo_seed_guard(base_seed: int, tasks: list[str], task_config: str,
                     seed_txt_dir: str | None) -> None:
    overall_max = -1
    checked = 0
    for task in tasks:
        try:
            seeds = _read_demo_seeds(task, task_config, seed_txt_dir)
        except Exception as e:
            print(f"[precollect] demo-seed guard: could not read {task} seed.txt ({e}); skipping it.", flush=True)
            continue
        if seeds:
            overall_max = max(overall_max, max(seeds))
            checked += 1
    if checked == 0:
        print("[precollect] demo-seed guard: no demo seeds readable — skipping held-out check "
              "(pass --seed_txt_dir for an offline copy, or --no_check_demo_seeds).", flush=True)
        return
    if base_seed <= overall_max:
        raise SystemExit(
            f"[precollect] base_seed={base_seed} is NOT held-out: it is <= the max demo-collection "
            f"seed ({overall_max}) across {checked} task(s), so eval would reuse training layouts. "
            f"Raise --base_seed above {overall_max} (official RMBench uses 100000).")
    print(f"[precollect] demo-seed guard OK: base_seed={base_seed} > max demo seed {overall_max} "
          f"(checked {checked} task(s)) — eval seeds are held-out from the demonstrations.", flush=True)


def _is_fatal(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _FATAL_MARKERS)


def _is_render_race(text: str) -> bool:
    return bool(text) and _RENDER_DEVICE_MARKER in (text or "").lower()


def _cleanup() -> None:
    with contextlib.suppress(Exception):
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@contextlib.contextmanager
def _render_init_guard():
    global _VULKAN_INITED
    if _VULKAN_INITED:
        yield
        return
    fd = os.open(_RENDER_INIT_LOCKFILE, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield
        _VULKAN_INITED = True
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _scan_task(task_name, task_config, instruction_type, sim_dir, base_seed, num_seeds, max_scan):
    from rmbench_sim.rmbench_env import RMBenchSimEnv

    success_seeds: list[int] = []
    instructions: dict[str, str] = {}

    print(f"[{task_name}] building env (task_config={task_config}, sim={sim_dir})", flush=True)
    try:
        env = RMBenchSimEnv(task_name, task_config, sim_dir)
    except Exception:
        tb = traceback.format_exc()
        print(f"[{task_name}] BUILD-FAIL:\n{tb}", flush=True)
        return {"success_seeds": [], "instructions": {}, "scanned": 0,
                "build_failed": True, "error": tb.strip().splitlines()[-1]}

    TE = env.TASK_ENV
    scanned = 0
    seed = base_seed
    consecutive_setup_fail = 0
    try:
        while len(success_seeds) < num_seeds and scanned < max_scan:
            try:
                with _render_init_guard():
                    TE.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env.args)
            except env._UnStableError:
                env._safe_close(); _cleanup(); seed += 1; scanned += 1
                continue
            except Exception:
                tb = traceback.format_exc()
                print(f"[{task_name}] seed {seed} setup fail: {tb.strip().splitlines()[-1]}", flush=True)
                env._safe_close(); _cleanup(); seed += 1; scanned += 1
                if _is_fatal(tb):
                    print(f"[{task_name}] FATAL during setup; stopping task.", flush=True)
                    break
                if not _is_render_race(tb):
                    consecutive_setup_fail += 1
                if not success_seeds and consecutive_setup_fail >= _UNSUPPORTED_SETUP_FAIL_LIMIT:
                    print(f"[{task_name}] aborting: {consecutive_setup_fail} setup fails, 0 solvable "
                          f"— task likely UNSUPPORTED (missing asset?).", flush=True)
                    break
                continue

            consecutive_setup_fail = 0
            try:
                episode_info = TE.play_once()
                solved = bool(TE.plan_success and TE.check_success())
            except Exception:
                tb = traceback.format_exc()
                print(f"[{task_name}] seed {seed} play_once fail: {tb.strip().splitlines()[-1]}", flush=True)
                env._safe_close(); _cleanup(); seed += 1; scanned += 1
                if _is_fatal(tb):
                    print(f"[{task_name}] FATAL during play_once; stopping task.", flush=True)
                    break
                continue

            if solved:
                try:
                    instr = env.make_instruction(episode_info, instruction_type)
                except Exception:
                    instr = f"Complete the {task_name} task."
                success_seeds.append(seed)
                instructions[str(seed)] = instr
                print(f"[{task_name}] seed {seed} SUCCESS ({len(success_seeds)}/{num_seeds})", flush=True)
            env._safe_close(); _cleanup(); seed += 1; scanned += 1
    finally:
        with contextlib.suppress(Exception):
            env.close()
        _cleanup()

    print(f"[{task_name}] done: {len(success_seeds)} success / {scanned} scanned", flush=True)
    return {"success_seeds": success_seeds, "instructions": instructions, "scanned": scanned,
            "build_failed": False, "error": ""}


def _worker(gpu_id, tasks, task_config, instruction_type, sim_dir, base_seed, num_seeds, max_scan, result_queue):
    from rmbench_sim.rmbench_env import pin_worker_gpu

    pin_worker_gpu(gpu_id)
    try:
        import torch

        torch.cuda.set_device(0)
    except Exception:
        pass
    out = {}
    for task_name in tasks:
        try:
            out[task_name] = _scan_task(task_name, task_config, instruction_type, sim_dir,
                                        base_seed, num_seeds, max_scan)
        except Exception:
            tb = traceback.format_exc()
            print(f"[gpu {gpu_id}] task {task_name} crashed:\n{tb}", flush=True)
            out[task_name] = {"success_seeds": [], "instructions": {}, "scanned": 0,
                              "build_failed": True, "error": tb.strip().splitlines()[-1]}
    result_queue.put(out)


def _partition(items, n):
    buckets: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def default_out_path(task_config: str, instruction_type: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds",
                        f"{task_config}__{instruction_type}.json")


def main():
    ap = argparse.ArgumentParser(description="Offline expert-solvable seed pre-collection for the RMBench eval.")
    ap.add_argument("--tasks", default=None, help="comma-separated task names (default: the 10 implemented tasks)")
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--instruction_type", default="seen")
    ap.add_argument("--num_seeds", type=int, default=100,
                    help="target expert-solvable seeds per task (default 100 == official test_num).")
    ap.add_argument("--base_seed", type=int, default=100000,
                    help="first seed to scan from. Default 100000 == official eval st_seed "
                         "(100000 * (1 + seed)), which is held-out from the demo-collection seeds.")
    ap.add_argument("--max_scan", type=int, default=1000, help="max seeds to scan per task")
    ap.add_argument("--seed_txt_dir", default=None,
                    help="local dir of demo <task>.txt (or <task>/<task_config>/seed.txt) for the "
                         "held-out base-seed guard; default downloads seed.txt from the HF dataset.")
    ap.add_argument("--check_demo_seeds", action="store_true", default=True,
                    help="verify base_seed is held-out from the demo-collection seeds (default on).")
    ap.add_argument("--no_check_demo_seeds", dest="check_demo_seeds", action="store_false",
                    help="skip the demo-seed held-out guard (e.g. fully offline, no seed_txt_dir).")
    ap.add_argument("--num_gpus", type=int, default=1, help="GPU workers (one process per GPU)")
    ap.add_argument("--worker_timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None, help="output JSON (default: seeds/<task_config>__<instruction_type>.json)")
    args = ap.parse_args()

    from rmbench_sim.rmbench_env import VENDORED_SIM_DIR

    sim_dir = VENDORED_SIM_DIR
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else list(DEFAULT_TASKS)
    out_path = args.out or default_out_path(args.task_config, args.instruction_type)
    num_gpus = max(1, int(args.num_gpus))

    if args.check_demo_seeds:
        _demo_seed_guard(args.base_seed, tasks, args.task_config, args.seed_txt_dir)
    print(f"[precollect] tasks={tasks} task_config={args.task_config} instruction_type={args.instruction_type}\n"
          f"[precollect] num_seeds={args.num_seeds} base_seed={args.base_seed} max_scan={args.max_scan} "
          f"num_gpus={num_gpus} out={out_path}", flush=True)

    tasks_out: dict[str, dict] = {}
    if num_gpus == 1:
        for task_name in tasks:
            tasks_out[task_name] = _scan_task(task_name, args.task_config, args.instruction_type,
                                              sim_dir, args.base_seed, args.num_seeds, args.max_scan)
    else:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        result_queue: "mp.Queue" = mp.Queue()
        _vis = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        procs = []
        for gpu_id, assigned in enumerate(_partition(tasks, num_gpus)):
            if not assigned:
                continue
            p = mp.Process(target=_worker, args=(gpu_id, assigned, args.task_config, args.instruction_type,
                                                 sim_dir, args.base_seed, args.num_seeds, args.max_scan,
                                                 result_queue))
            p.start()
            procs.append(p)
            phys = _vis[gpu_id] if gpu_id < len(_vis) else gpu_id
            print(f"[precollect] started worker {gpu_id} (physical GPU {phys}) for tasks {assigned}", flush=True)
        remaining = len(procs)
        while remaining > 0:
            try:
                out = result_queue.get(timeout=args.worker_timeout)
                tasks_out.update(out)
                remaining -= 1
            except Exception:
                alive = [p for p in procs if p.is_alive()]
                if not alive:
                    print("[precollect] all workers exited; stopping wait.", flush=True)
                    break
                print(f"[precollect] still waiting on {len(alive)} worker(s)...", flush=True)
        for p in procs:
            p.join(timeout=10)

    data = {
        "meta": {"task_config": args.task_config, "instruction_type": args.instruction_type,
                 "base_seed": args.base_seed, "num_seeds_per_task": args.num_seeds,
                 "generated_utc": datetime.now(timezone.utc).isoformat()},
        "tasks": tasks_out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[precollect] wrote {out_path}", flush=True)
    print(f"{'task':40s}  success/scanned", flush=True)
    n_unsupported = 0
    for task_name in tasks:
        b = tasks_out.get(task_name, {})
        n_succ = len(b.get("success_seeds", []))
        flag = "  [BUILD-FAIL: re-run]" if b.get("build_failed") else ("  [UNSUPPORTED/no solvable seed]" if n_succ == 0 else "")
        if n_succ == 0 and not b.get("build_failed"):
            n_unsupported += 1
        print(f"{task_name:40s}  {n_succ}/{b.get('scanned', 0)}{flag}", flush=True)
    if n_unsupported:
        print(f"[precollect] WARNING: {n_unsupported} task(s) produced 0 seeds — drop them from --tasks "
              f"(e.g. battery_try may be missing an asset) or they make empty eval groups.", flush=True)


if __name__ == "__main__":
    main()
