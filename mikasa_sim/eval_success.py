
from __future__ import annotations

import sys

sys.path.append("./")
import mikasa_sim

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import struct
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASKS = [
    "ShellGameTouch-VLA-v0",
    "InterceptMedium-VLA-v0",
    "RememberColor3-VLA-v0",
    "RememberColor5-VLA-v0",
    "RememberColor9-VLA-v0",
]

CLAIM_BLOCK = 4
EXIT_OK = 0
EXIT_CRASHED = 17


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", default="./checkpoints/sft/simplememvla/mikasa_baseline")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Env IDs to evaluate (default: the 5 MemoryVLA-paper MIKASA tasks).")
    ap.add_argument("--n_episodes", type=int, default=100,
                    help="Seeded episodes per task (the MemoryVLA paper evaluates 100; the "
                         "benchmark's own default stream is the same, just longer).")
    ap.add_argument("--start_seed", type=int, default=None,
                    help="First episode seed (default: the benchmark's canonical START_SEED).")
    ap.add_argument("--execute_horizon", type=int, default=8,
                    help="Receding horizon: execute the first N of the predicted chunk "
                         "(model chunk = config.action_horizon), then re-decide.")
    ap.add_argument("--decision_path", default="pipelined",
                    choices=["pipelined", "recompute"],
                    help="How each decision is computed. 'pipelined' "
                         "(mikasa_sim/pipelined_policy.py) reuses what provably did "
                         "not change since the last decision — per-temporal-patch ViT "
                         "features, the video-prefix KV/recurrent state, and the "
                         "sub-task decode's own hidden states (dropping the recompute "
                         "path's second full backbone pass). Same computation, ~2x "
                         "faster; 'recompute' is the plain "
                         "MikasaPolicy.predict reference path.")
    ap.add_argument("--num_denoising_steps", type=int, default=10)
    ap.add_argument("--eval_temperature", type=float, default=1.0,
                    help="Deterministic-ODE init-noise scale. 1.0 matches training (the flow "
                         "field is trained to denoise from unit-variance noise at t=1).")
    ap.add_argument("--max_subtask_tokens", type=int, default=64)
    ap.add_argument("--compute_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help="VLM backbone full-attention backend: sdpa (portable) | flash_attention_2 (fast).")
    ap.add_argument("--sim_backend", default="gpu", choices=["gpu", "cpu"],
                    help="ManiSkill simulation backend (the canonical benchmark uses gpu).")
    ap.add_argument("--control_mode", default=None,
                    help="Env controller. Default: the official benchmark's pd_ee_delta_pose. "
                         "Pass pd_joint_pos for checkpoints trained on the joint-space dataset "
                         "(the joint-space mikasa_lerobot dataset); the wrapper stack then serves the "
                         "matching [qpos(7), gripper_width] proprio.")
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="GPU workers (one process per GPU), each loading its own model replica "
                         "and claiming episode blocks from a shared counter. Set "
                         "CUDA_VISIBLE_DEVICES to the GPU list (scripts/eval_mikasa.sh passes GPUS). "
                         "Results are invariant to this value.")
    ap.add_argument("--retry_rounds", type=int, default=1,
                    help="After the workers exit, if any canonical episode is still missing "
                         "(crashed/killed worker), respawn workers for just those episodes this "
                         "many more times before failing.")
    ap.add_argument("--worker_timeout", type=float, default=28800.0,
                    help="seconds to wait for a worker process to exit before giving up on it.")
    ap.add_argument("--save_videos", action="store_true",
                    help="Record every episode as a composed benchmark video "
                         "(general render | top / wrist) under <output_dir>/videos/<env_id>/.")
    ap.add_argument("--output_dir", default=None,
                    help="Where the episode store, per-task JSON and summary.json land (default: "
                         "next to --log_file, or ./logs/mikasa_sim/adhoc). Re-running with the "
                         "same directory RESUMES: episodes already recorded for this checkpoint "
                         "are skipped.")
    ap.add_argument("--log_file", default=None,
                    help="If set, results land next to this log path (logs/ convention).")
    return ap.parse_args()


def _check_attn_available(attn_implementation: str) -> None:
    if attn_implementation in (None, "", "sdpa", "eager"):
        return
    if "flash" in attn_implementation and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError(
            f"attn_implementation='{attn_implementation}' requested but flash_attn is not "
            "installed. Run scripts/install/install_fast_path.sh, or use --attn_implementation sdpa."
        )


def pin_worker_gpu(gpu_id: int) -> str:
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    dev = visible[gpu_id] if gpu_id < len(visible) else str(gpu_id)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    return dev


def _sampled_sha256(path: str, chunks: int = 16, size: int = 4 << 20) -> str:
    total = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(total).encode())
    with open(path, "rb") as f:
        if total <= chunks * size:
            h.update(f.read())
        else:
            for i in range(chunks):
                f.seek(int(i * (total - size) / (chunks - 1)))
                h.update(f.read(size))
    return h.hexdigest()


def checkpoint_fingerprint(checkpoint: str) -> str:
    files = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No *.safetensors under {checkpoint}")
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        h.update(_sampled_sha256(p).encode())
    return h.hexdigest()[:32]


_IDENTITY_FILES = ("stats.json", "config.json", "chat_template.jinja")
_IDENTITY_ARGS = (
    "execute_horizon", "num_denoising_steps", "eval_temperature",
    "max_subtask_tokens", "compute_dtype", "attn_implementation",
    "control_mode", "sim_backend", "decision_path",
)


def run_identity(args) -> tuple[str, str]:
    ckpt_fp = checkpoint_fingerprint(args.pretrained_checkpoint)
    h = hashlib.sha256()
    h.update(ckpt_fp.encode())
    for name in _IDENTITY_FILES:
        p = os.path.join(args.pretrained_checkpoint, name)
        h.update(name.encode())
        h.update(_sampled_sha256(p).encode() if os.path.isfile(p) else b"<absent>")
    for name in _IDENTITY_ARGS:
        h.update(f"{name}={getattr(args, name, None)!r}".encode())
    return h.hexdigest()[:32], ckpt_fp


def checkpoint_has_proprio(checkpoint: str) -> bool:
    for p in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        if any("action_head.state_proj" in k for k in header):
            return True
    return False


def _episode_dir(output_dir: str, env_id: str) -> Path:
    return Path(output_dir) / "episodes" / env_id


def _write_episode(output_dir: str, env_id: str, seed: int, payload: dict) -> None:
    d = _episode_dir(output_dir, env_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{seed}.{os.getpid()}.json.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, d / f"{seed}.json")


def read_episodes(output_dir: str, env_id: str, identity: str) -> tuple[dict[int, dict], int]:
    recs: dict[int, dict] = {}
    stale = 0
    for p in sorted(_episode_dir(output_dir, env_id).glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            seed = int(rec["seed"])
            success = bool(rec["success_once"])
        except Exception:
            continue
        if rec.get("run_identity") != identity:
            stale += 1
            continue
        recs[seed] = rec
    return recs, stale


def _benchmark_config(args):
    from mikasa_robo_suite.vla import benchmarking as bench

    return bench.BenchmarkConfig(
        start_seed=args.start_seed if args.start_seed is not None else bench.START_SEED,
        n_episodes=args.n_episodes,
        sim_backend=args.sim_backend,
        save_videos=bool(args.save_videos),
        **({"control_mode": args.control_mode} if args.control_mode else {}),
    )


def build_adapter(args):
    from transformers import AutoProcessor

    from mikasa_sim.policy import MikasaPolicy, get_vla
    from mikasa_sim.policy_adapter import SimpleMemVLAChunkPolicy

    _check_attn_available(args.attn_implementation)
    vla, unnormalize_action, normalize_state = get_vla(
        args.pretrained_checkpoint, args.compute_dtype,
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(
        args.pretrained_checkpoint,
        trust_remote_code=True,
        padding_side="right",
        model_max_length=8192,
    )
    kwargs = dict(
        vla=vla,
        processor=processor,
        unnormalize_action=unnormalize_action,
        normalize_state=normalize_state,
        num_denoising_steps=args.num_denoising_steps,
        temperature=args.eval_temperature,
        max_subtask_tokens=args.max_subtask_tokens,
    )
    if args.decision_path == "pipelined":
        from mikasa_sim.pipelined_policy import PipelinedMikasaPolicy

        policy = PipelinedMikasaPolicy(**kwargs, execute_horizon=args.execute_horizon)
    else:
        policy = MikasaPolicy(**kwargs)
    return SimpleMemVLAChunkPolicy(policy, execute_horizon=args.execute_horizon)


def _bind_env(adapter, env, task) -> None:
    adapter.bind_sensor_names(list(env.unwrapped.scene.sensors.keys()))
    space = env.action_space
    low = np.asarray(space.low, np.float32).reshape(-1)
    high = np.asarray(space.high, np.float32).reshape(-1)
    if low.shape[0] != adapter.policy.action_dim:
        raise RuntimeError(
            f"{task.env_id}: env action dim {low.shape[0]} != model action dim "
            f"{adapter.policy.action_dim}"
        )
    adapter.action_low, adapter.action_high = low, high


def _claim_block(counter, lock, total: int) -> int | None:
    with lock:
        i = counter.value
        if i >= total:
            return None
        counter.value = i + 1
    return i


def _run_blocks(gpu_id, args, blocks, counter, lock, output_dir, identity, ckpt_fp) -> None:
    from mikasa_robo_suite.vla import benchmarking as bench

    adapter = build_adapter(args)
    config = _benchmark_config(args)
    tasks = {
        t.env_id: t
        for t in bench.select_benchmark_tasks(env_ids=sorted({b[0] for b in blocks}))
    }
    env = None
    cur_env_id = None
    n_done = 0
    try:
        while True:
            idx = _claim_block(counter, lock, len(blocks))
            if idx is None:
                break
            env_id, seeds = blocks[idx]
            task = tasks[env_id]
            if task.max_episode_steps <= 0:
                raise ValueError(
                    f"{env_id}: unknown benchmark task (not in mikasa_robo_vla_envs.csv)"
                )
            instruction = task.language_instruction.strip()
            if not instruction:
                raise ValueError(f"{env_id}: benchmark CSV has no language_instruction")

            if env_id != cur_env_id:
                if env is not None:
                    env.close()
                    env = None
                env = bench.make_benchmark_env(env_id, config)
                cur_env_id = env_id
                _bind_env(adapter, env, task)

            video_dir = None
            if config.save_videos and output_dir is not None:
                video_dir = Path(output_dir) / "videos" / env_id

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed % (2 ** 32))
                adapter.start_episode(instruction)
                ep, frames = bench.run_episode(
                    env, adapter, seed, collect_video=(video_dir is not None)
                )
                if frames and video_dir is not None:
                    idx_in_stream = seed - config.start_seed
                    bench._write_video(
                        video_dir / f"episode_{idx_in_stream:04d}.mp4", frames, fps=30.0
                    )
                _write_episode(output_dir, env_id, seed, {
                    "env_id": env_id,
                    "seed": int(seed),
                    "success_once": bool(ep.success_once),
                    "episode_return": float(ep.episode_return),
                    "n_steps": int(ep.n_steps),
                    "n_predictions": int(adapter.n_predictions),
                    "run_identity": identity,
                    "checkpoint_fingerprint": ckpt_fp,
                })
                n_done += 1
            print(f"[eval] gpu{gpu_id} {env_id} seeds {seeds[0]}..{seeds[-1]} done "
                  f"({n_done} episodes on this worker)", flush=True)
    finally:
        if env is not None:
            env.close()


def _worker(gpu_id, args, blocks, counter, lock, output_dir, identity, ckpt_fp):
    pin_worker_gpu(gpu_id)
    code = EXIT_OK
    try:
        _run_blocks(gpu_id, args, blocks, counter, lock, output_dir, identity, ckpt_fp)
    except BaseException:
        print(f"[eval] gpu {gpu_id} crashed:\n{traceback.format_exc()}", flush=True)
        code = EXIT_CRASHED
    sys.stdout.flush()
    os._exit(code)


def _work_blocks(task_ids, missing: dict[str, list[int]]) -> list[tuple[str, list[int]]]:
    blocks: list[tuple[str, list[int]]] = []
    for env_id in task_ids:
        seeds = missing.get(env_id, [])
        for i in range(0, len(seeds), CLAIM_BLOCK):
            blocks.append((env_id, seeds[i:i + CLAIM_BLOCK]))
    return blocks


def _missing(output_dir, task_ids, expected_seeds, identity) -> dict[str, list[int]]:
    out = {}
    for env_id in task_ids:
        recs, _ = read_episodes(output_dir, env_id, identity)
        out[env_id] = [s for s in expected_seeds if s not in recs]
    return out


def _progress_printer(output_dir, task_ids, expected_seeds, identity, stop_event, period=120.0):
    total = len(task_ids) * len(expected_seeds)
    while not stop_event.wait(period):
        done = 0
        parts = []
        for env_id in task_ids:
            recs, _ = read_episodes(output_dir, env_id, identity)
            done += len(recs)
            if recs:
                sr = float(np.mean([r["success_once"] for r in recs.values()]))
                parts.append(f"{env_id.replace('-VLA-v0','')} {len(recs)}/{len(expected_seeds)}"
                             f"@{sr * 100:.0f}%")
        print(f"[eval] progress {done}/{total} episodes | " + "  ".join(parts), flush=True)


def run_round(args, task_ids, blocks, output_dir, identity, ckpt_fp, num_workers) -> list[int]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    num_workers = max(1, min(num_workers, len(blocks)))
    counter = ctx.Value("i", 0)
    lock = ctx.Lock()
    procs = []
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    for gpu_id in range(num_workers):
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, args, blocks, counter, lock, output_dir, identity, ckpt_fp),
        )
        p.start()
        procs.append(p)
        phys = visible[gpu_id] if gpu_id < len(visible) else gpu_id
        print(f"[eval] started worker {gpu_id} (physical GPU {phys})", flush=True)

    deadline = time.time() + args.worker_timeout
    for p in procs:
        p.join(timeout=max(1.0, deadline - time.time()))
        if p.is_alive():
            print(f"[eval] worker pid {p.pid} exceeded --worker_timeout; terminating", flush=True)
            p.terminate()
            p.join(timeout=30)
    codes = [p.exitcode for p in procs]
    bad = [(i, c) for i, c in enumerate(codes) if c != EXIT_OK]
    if bad:
        print(f"[eval] WARNING: {len(bad)} worker(s) did not exit cleanly: "
              f"{[f'worker{i}=exit{c}' for i, c in bad]}", flush=True)
    return codes


def reduce_run(args, task_ids, expected_seeds, identity, ckpt_fp, output_dir):
    from mikasa_robo_suite.vla import benchmarking as bench
    from simplememvla.model.configuration_simplememvla import SimpleMemVLAConfig

    cfg = SimpleMemVLAConfig.from_pretrained(args.pretrained_checkpoint)
    config = _benchmark_config(args)
    tasks = {t.env_id: t for t in bench.select_benchmark_tasks(env_ids=task_ids)}
    model_meta = {
        "checkpoint": os.path.abspath(args.pretrained_checkpoint),
        "checkpoint_fingerprint": ckpt_fp,
        "run_identity": identity,
        "model_action_horizon": int(cfg.action_horizon),
        "execute_horizon": int(args.execute_horizon),
        "decision_path": str(args.decision_path),
        "num_denoising_steps": int(args.num_denoising_steps),
        "eval_temperature": float(args.eval_temperature),
        "attn_implementation": str(args.attn_implementation),
        "compute_dtype": str(args.compute_dtype),
        "use_proprio": bool(cfg.use_proprio and checkpoint_has_proprio(args.pretrained_checkpoint)),
    }
    commit = bench.benchmark_commit()

    results, missing = [], {}
    for env_id in task_ids:
        recs, stale = read_episodes(output_dir, env_id, identity)
        if stale:
            print(f"[eval] {env_id}: ignored {stale} episode record(s) recorded under a "
                  "different run identity (weights / stats / rollout knobs)", flush=True)
        gaps = [s for s in expected_seeds if s not in recs]
        if gaps:
            missing[env_id] = gaps
            continue
        ordered = [recs[s] for s in expected_seeds]
        successes = [bool(r["success_once"]) for r in ordered]
        returns = [float(r["episode_return"]) for r in ordered]
        task = tasks[env_id]
        results.append({
            "env_id": env_id,
            "split": task.split.title(),
            "memory_type": task.memory_type,
            "start_seed": config.start_seed,
            "n_episodes": len(expected_seeds),
            "successes": successes,
            "returns": returns,
            "sr": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "benchmark_commit": commit,
            "control_mode": config.control_mode,
            "obs_mode": config.obs_mode,
            "wrapper_chain": bench.WRAPPER_CHAIN,
            "action_chunk_size": 1,
            "model": dict(model_meta),
            "episode_lengths": [int(r["n_steps"]) for r in ordered],
            "episode_seeds": list(expected_seeds),
        })
    return results, missing


def publish(args, results, task_ids, expected_seeds, identity, ckpt_fp, output_dir) -> None:
    from mikasa_robo_suite.vla import benchmarking as bench

    for result in results:
        bench._write_json(Path(output_dir) / f"{result['env_id']}.json", result)

    print("\n==== MIKASA-Robo-VLA closed-loop success rates ====")
    for r in results:
        print(f"  {r['env_id']:32s} {r['sr'] * 100:5.1f}% (n={r['n_episodes']})")
    macro = float(np.mean([r["sr"] for r in results]))
    print(f"  {'OVERALL (macro avg)':32s} {macro * 100:5.1f}%", flush=True)

    summary = dict(bench.summarize_task_results(results))
    summary.update({
        "complete": True,
        "tasks_evaluated": list(task_ids),
        "n_episodes_per_task": len(expected_seeds),
        "start_seed": int(expected_seeds[0]),
        "checkpoint": os.path.abspath(args.pretrained_checkpoint),
        "checkpoint_fingerprint": ckpt_fp,
        "run_identity": identity,
        "execute_horizon": int(args.execute_horizon),
        "decision_path": str(args.decision_path),
        "control_mode": results[0]["control_mode"],
        "episode_seeded_policy": True,
    })
    bench._write_json(Path(output_dir) / "summary.json", summary)
    print(f"[eval] wrote results -> {output_dir}", flush=True)


def fail_incomplete(missing, task_ids, expected_seeds, identity, ckpt_fp, output_dir) -> None:
    from mikasa_robo_suite.vla import benchmarking as bench

    total = sum(len(v) for v in missing.values())
    print(f"\n[eval] RUN INCOMPLETE: {total} canonical episode(s) missing across "
          f"{len(missing)} task(s). No summary.json written.", flush=True)
    for env_id, seeds in missing.items():
        head = ", ".join(str(s) for s in seeds[:8])
        more = "" if len(seeds) <= 8 else f", ... (+{len(seeds) - 8} more)"
        print(f"  {env_id:32s} missing {len(seeds)}/{len(expected_seeds)}: {head}{more}",
              flush=True)
    bench._write_json(Path(output_dir) / "INCOMPLETE.json", {
        "complete": False,
        "checkpoint_fingerprint": ckpt_fp,
        "run_identity": identity,
        "n_episodes_per_task": len(expected_seeds),
        "tasks": list(task_ids),
        "missing_counts": {k: len(v) for k, v in missing.items()},
        "missing_seeds": {k: list(v) for k, v in missing.items()},
    })
    print("[eval] re-run the SAME command with the same --output_dir to resume "
          "(recorded episodes are skipped).", flush=True)


def main():
    args = parse_args()
    task_ids = args.tasks if args.tasks else list(DEFAULT_TASKS)

    if args.output_dir:
        output_dir = args.output_dir
    elif args.log_file:
        output_dir = os.path.dirname(os.path.abspath(args.log_file))
    else:
        output_dir = "./logs/mikasa_sim/adhoc"
    os.makedirs(output_dir, exist_ok=True)

    from mikasa_robo_suite.vla import benchmarking as bench

    start_seed = args.start_seed if args.start_seed is not None else bench.START_SEED
    expected_seeds = [start_seed + i for i in range(args.n_episodes)]
    identity, ckpt_fp = run_identity(args)

    print(f"[eval] tasks={task_ids}", flush=True)
    print(f"[eval] n_episodes/task={args.n_episodes}, seeds {expected_seeds[0]}.."
          f"{expected_seeds[-1]}, execute_horizon={args.execute_horizon}, "
          f"decision_path={args.decision_path}", flush=True)
    print(f"[eval] checkpoint={os.path.abspath(args.pretrained_checkpoint)}", flush=True)
    print(f"[eval] checkpoint fingerprint={ckpt_fp}", flush=True)
    print(f"[eval] run identity={identity} (weights + stats/config/template + rollout knobs)", flush=True)
    for stale_name in ("summary.json", "INCOMPLETE.json"):
        stale_path = Path(output_dir) / stale_name
        if stale_path.exists():
            stale_path.unlink()
            print(f"[eval] cleared stale {stale_name} from a previous run", flush=True)
    print(f"[eval] results -> {output_dir}", flush=True)

    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    if visible and int(args.num_gpus) > len(visible):
        raise ValueError(
            f"--num_gpus={args.num_gpus} exceeds the {len(visible)} visible GPU(s) "
            f"{visible}; workers beyond the list would land on unrequested devices."
        )
    num_workers = max(1, int(args.num_gpus))
    stop_event = threading.Event()
    progress = threading.Thread(
        target=_progress_printer,
        args=(output_dir, task_ids, expected_seeds, identity, stop_event),
        daemon=True,
    )
    progress.start()

    try:
        prev_todo = None
        for round_idx in range(max(1, int(args.retry_rounds) + 1)):
            missing = _missing(output_dir, task_ids, expected_seeds, identity)
            todo = sum(len(v) for v in missing.values())
            if todo == 0:
                if round_idx == 0:
                    print("[eval] every canonical episode is already recorded for this "
                          "checkpoint; reducing without running.", flush=True)
                break
            if prev_todo is not None and todo >= prev_todo:
                print(f"[eval] retry made no progress ({todo} still missing); "
                      "stopping instead of retrying again.", flush=True)
                break
            if round_idx > 0:
                print(f"[eval] retry round {round_idx}: {todo} episode(s) still missing",
                      flush=True)
            blocks = _work_blocks(task_ids, missing)
            print(f"[eval] round {round_idx}: {todo} episodes in {len(blocks)} blocks "
                  f"across {min(num_workers, len(blocks))} worker(s)", flush=True)
            run_round(args, task_ids, blocks, output_dir, identity, ckpt_fp, num_workers)
            prev_todo = todo
    finally:
        stop_event.set()

    results, missing = reduce_run(args, task_ids, expected_seeds, identity, ckpt_fp, output_dir)
    if missing:
        fail_incomplete(missing, task_ids, expected_seeds, identity, ckpt_fp, output_dir)
        sys.stdout.flush()
        os._exit(1)

    publish(args, results, task_ids, expected_seeds, identity, ckpt_fp, output_dir)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
