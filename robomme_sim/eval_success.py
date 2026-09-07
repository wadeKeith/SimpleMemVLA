
from __future__ import annotations

import sys
sys.path.append("./")
import robomme_sim

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robomme_sim.batched_policy import BatchedEvalPolicy
from robomme_sim.inproc_pool import InProcSimPool
from robomme_sim.robomme_env import pin_worker_gpu

DEFAULT_TASKS = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", default="./checkpoints/sft/simplememvla/robomme_baseline")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of tasks to evaluate (default: all 16 RoboMME tasks).")
    ap.add_argument("--dataset_split", default="test", choices=["test", "val", "train"],
                    help="Benchmark episode split (official frozen lists; test=50/task).")
    ap.add_argument("--episodes_per_task", type=int, default=50,
                    help="Episodes per task (capped at the split's episode count; test has 50).")
    ap.add_argument("--group_size", type=int, default=2,
                    help="Envs evaluated in parallel per group (= envs batched into one forward).")
    ap.add_argument("--execute_horizon", type=int, default=16,
                    help="Receding horizon: execute the first N of the predicted chunk, then re-decide.")
    ap.add_argument("--max_steps", type=int, default=1300,
                    help="Max policy control steps per episode (the official MME-VLA setting).")
    ap.add_argument("--num_denoising_steps", type=int, default=10)
    ap.add_argument("--eval_temperature", type=float, default=1.0,
                    help="Deterministic-ODE init-noise scale. 1.0 matches training (the flow "
                         "field is trained to denoise from unit-variance noise at t=1) and the "
                         "open-loop eval default, so closed-loop uses the same sampling "
                         "distribution. (<1.0 sharpens but starts the ODE off the trained t=1 "
                         "manifold; not used by default.)")
    ap.add_argument("--max_subtask_tokens", type=int, default=64)
    ap.add_argument("--pipelined", action="store_true", default=False,
                    help="PIPELINED decisions (robomme_sim/pipelined_policy.py): the same exact "
                         "recompute, split so the 59 already-determined temporal patches are "
                         "prefilled while the arm executes the previous chunk and only [last "
                         "patch + wrist + instruction] stays on the decision-time critical path. "
                         "Semantically identical to the default path (chunk splitting is an "
                         "identity for causal attention and the gated-delta scan). Requires "
                         "--execute_horizon == 2*stride (20 here) so the window slides exactly "
                         "one temporal patch per decision.")
    ap.add_argument("--compute_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help="VLM backbone full-attention backend: sdpa (portable) | flash_attention_2 (fast).")
    ap.add_argument("--step_timeout", type=float, default=300.0)
    ap.add_argument("--reset_timeout", type=float, default=3600.0,
                    help="Per-episode reset timeout. RoboMME resets PLAY the conditioning demo "
                         "(motion-planner driven), so they are much longer than a plain scene reset.")
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="GPU workers (one process per GPU); the task list is sharded across them, "
                         "each worker loading its own model replica + SAPIEN pool. Set "
                         "CUDA_VISIBLE_DEVICES to the GPU list (scripts/eval_robomme.sh passes GPUS).")
    ap.add_argument("--worker_timeout", type=float, default=14400.0,
                    help="seconds to wait for each multi-GPU worker's results.")
    ap.add_argument("--video_dir", default=None,
                    help="If set, record rollout MP4s under <video_dir>/<task>/<task>__ep<idx>__<succ|fail>.mp4 "
                         "(front + wrist tiled side-by-side; demo frames included). For each task ONE "
                         "success + ONE failure clip is kept (frames buffer in RAM). Default None disables.")
    ap.add_argument("--video_max_per_task", type=int, default=1,
                    help="Clips to keep per outcome bucket (success / failure) per task when --video_dir is set. "
                         "1 -> one success + one failure example per task; 0 disables.")
    ap.add_argument("--log_file", default=None,
                    help="If set, results.json is placed next to this log path (logs/ convention).")
    return ap.parse_args()


def _check_attn_available(attn_implementation: str) -> None:
    if attn_implementation in (None, "", "sdpa", "eager"):
        return
    if "flash" in attn_implementation and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError(
            f"attn_implementation='{attn_implementation}' requested but flash_attn is not "
            "installed. Run scripts/install/install_fast_path.sh, or use --attn_implementation sdpa."
        )


def build_policy(args):
    from transformers import AutoProcessor

    from robomme_sim.policy import RoboMMEPolicy, get_vla

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
    device = next(vla.parameters()).device
    if getattr(args, "pipelined", False):
        from robomme_sim.pipelined_policy import PipelinedBufferPolicy, PipelinedEvalGroup

        def buffer_factory():
            return PipelinedBufferPolicy(
                vla=vla, processor=processor,
                unnormalize_action=unnormalize_action, normalize_state=normalize_state,
                num_denoising_steps=args.num_denoising_steps, temperature=args.eval_temperature,
                max_subtask_tokens=args.max_subtask_tokens,
            )

        stride = int(buffer_factory().stride)
        if int(args.execute_horizon) != 2 * stride:
            raise ValueError(
                f"--pipelined requires --execute_horizon == 2*stride == {2 * stride} "
                f"(got {args.execute_horizon}): the window must slide exactly one "
                "temporal patch per decision for the prefetched prefix to be the next "
                "decision's prefix."
            )
        print("[eval] PIPELINED decisions: exact recompute, split so the first "
              "n-1 temporal patches prefill during action execution; critical path = "
              "last patch + wrist + instruction + sub-task decode + DiT.", flush=True)
        batched = PipelinedEvalGroup(
            model=vla, processor=processor,
            unnormalize_action=unnormalize_action, normalize_state=normalize_state,
            num_denoising_steps=args.num_denoising_steps, eval_temperature=args.eval_temperature,
            max_subtask_tokens=args.max_subtask_tokens, device=device,
        )
        return batched, buffer_factory, normalize_state

    batched = BatchedEvalPolicy(
        model=vla, processor=processor,
        unnormalize_action=unnormalize_action, normalize_state=normalize_state,
        num_denoising_steps=args.num_denoising_steps, eval_temperature=args.eval_temperature,
        max_subtask_tokens=args.max_subtask_tokens, device=device,
    )

    def buffer_factory():
        return RoboMMEPolicy(
            vla=vla, processor=processor,
            unnormalize_action=unnormalize_action, normalize_state=normalize_state,
            num_denoising_steps=args.num_denoising_steps, temperature=args.eval_temperature,
            max_subtask_tokens=args.max_subtask_tokens,
        )

    return batched, buffer_factory, normalize_state


def run_group(args, pool, batched, buffer_factory, normalize_state, task, specs,
              video_dir=None, video_quota=None):
    G = len(specs)
    image_keys = list(buffer_factory().image_keys)
    cam_map = {k.split(".")[-1]: k for k in image_keys}

    def to_full(frame):
        return {cam_map.get(k, k): np.asarray(v, dtype=np.uint8) for k, v in frame.items()}

    def state_norm(state):
        if normalize_state is None:
            return None
        s = normalize_state.normalize(torch.from_numpy(np.asarray(state, dtype=np.float32)))
        return s.unsqueeze(0).to(batched.device)

    record = (video_dir is not None and video_quota is not None
              and (video_quota.get("succ", 0) > 0 or video_quota.get("fail", 0) > 0))
    recorders = [None] * G
    if record:
        from robomme_sim.video_writer import RolloutVideoRecorder
        native_fps = float(getattr(buffer_factory(), "native_fps", 0) or 0) or 20.0
        recorders = [RolloutVideoRecorder(fps=native_fps) for _ in range(G)]

    def save_video(g, ok):
        rec = recorders[g]
        if rec is None or len(rec) == 0:
            return
        bucket = "succ" if ok else "fail"
        if not video_quota or video_quota.get(bucket, 0) <= 0:
            return
        episode = specs[g].get("episode", g)
        out_path = os.path.join(video_dir, f"{task}__ep{episode}__{bucket}.mp4")
        nframes = len(rec)
        try:
            if rec.save(out_path):
                video_quota[bucket] -= 1
                print(f"[eval] saved {bucket} video: {out_path} ({nframes} frames)", flush=True)
        except Exception as e:
            print(f"[eval] WARNING: video save failed for {task} ep{episode}: {e}", flush=True)

    resets = pool.reset(specs)

    buffers = [buffer_factory() for _ in range(G)]
    instructions = [None] * G
    cur_state = [None] * G
    active = []
    reset_ok = [False] * G
    for g in range(G):
        r = resets[g]
        if not (isinstance(r, dict) and r.get("ok")):
            continue
        reset_ok[g] = True
        buffers[g].reset()
        for fr in r["frames"]:
            buffers[g].observe(to_full(fr))
            if recorders[g] is not None:
                recorders[g].add(fr)
        instructions[g] = r["instruction"]
        cur_state[g] = np.asarray(r["states"][-1], dtype=np.float32)
        active.append(g)
    if len(active) < G:
        print(f"[eval] {task}: {G - len(active)}/{G} env reset(s) failed -> those episodes "
              "are EXCLUDED from the success rate (None), not counted as failures.", flush=True)

    success = [False] * G

    def _results():
        return [success[g] if reset_ok[g] else None for g in range(G)]

    if not active:
        return _results()

    hard_bound = max(1, -(-int(args.max_steps) // max(1, args.execute_horizon))) + 2

    decisions_taken = 0
    while active and decisions_taken < hard_bound:
        decisions_taken += 1
        act = [g for g in active if not success[g]]
        if not act:
            break
        processed_list = [buffers[g]._prepare_inputs(instructions[g]) for g in act]
        state_list = [state_norm(cur_state[g]) for g in act]
        decisions = batched.generate_batch(processed_list, state_list)

        action_chunks = [None] * G
        for k, g in enumerate(act):
            actions_unnorm, _subtask = decisions[k]
            action_chunks[g] = actions_unnorm[: args.execute_horizon]

        step_results = pool.step(action_chunks, act)
        for g in act:
            res = step_results[g]
            if not isinstance(res, dict) or "error" in res:
                success[g] = False
                if g in active:
                    active.remove(g)
                    save_video(g, ok=False)
                continue
            for fr in res.get("frames", []):
                buffers[g].observe(to_full(fr))
                if recorders[g] is not None:
                    recorders[g].add(fr)
            if res.get("states"):
                cur_state[g] = np.asarray(res["states"][-1], dtype=np.float32)
            success[g] = bool(res.get("success", False))
            if res.get("done", False) and g in active:
                active.remove(g)
                save_video(g, ok=success[g])

    for g in active:
        save_video(g, ok=success[g])

    return _results()


def evaluate_tasks(args, tasks) -> dict:
    import gc

    torch.manual_seed(0)
    np.random.seed(0)

    n_req = int(args.episodes_per_task)
    group_size = int(min(args.group_size, n_req))

    batched, buffer_factory, normalize_state = build_policy(args)
    pool = InProcSimPool(
        num_envs=group_size, dataset_split=args.dataset_split, max_steps=args.max_steps,
        step_timeout=args.step_timeout, reset_timeout=args.reset_timeout,
    )
    pool.start()

    def episode_count(task) -> int:
        from robomme_sim.robomme_env import _setup_robomme_path

        _setup_robomme_path()
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        builder = BenchmarkEnvBuilder(env_id=task, dataset=args.dataset_split,
                                      action_space="joint_angle", max_steps=args.max_steps)
        return builder.get_episode_num()

    def build_specs(task, done, g):
        return [{"task": task, "episode": done + e} for e in range(g)]

    video_root = getattr(args, "video_dir", None)
    video_per_bucket = max(0, int(getattr(args, "video_max_per_task", 1) or 0))

    def run_group_oom_safe(task, done, g, video_quota=None):
        specs = build_specs(task, done, g)
        vdir = os.path.join(video_root, task) if video_root else None
        try:
            return list(run_group(args, pool, batched, buffer_factory, normalize_state, task,
                                  specs, video_dir=vdir, video_quota=video_quota)[:g]), False
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        if g <= 1:
            print(f"[eval] group {task}:{done} OOM even at group=1 -- skipped (not counted as failure).",
                  flush=True)
            return [], True
        half = g // 2
        print(f"[eval] group {task}:{done} CUDA OOM at group={g} -- retrying as {half}+{g - half}.",
              flush=True)
        left, _ = run_group_oom_safe(task, done, half, video_quota=video_quota)
        right, _ = run_group_oom_safe(task, done + half, g - half, video_quota=video_quota)
        return left + right, True

    per_task: dict[str, float | None] = {}
    per_task_n: dict[str, int] = {}
    per_task_oom: dict[str, bool] = {}
    for task in tasks:
        try:
            n_avail = episode_count(task)
        except Exception as e:
            print(f"[eval] {task}: could not resolve episode count ({e}) -> N/A", flush=True)
            per_task[task] = None
            per_task_n[task] = 0
            per_task_oom[task] = False
            continue
        n = min(n_req, n_avail)
        if n < n_req:
            print(f"[eval] {task}: split has only {n_avail} episodes (requested {n_req}) "
                  f"-> evaluating {n}.", flush=True)
        successes = []
        any_oom = False
        done = 0
        video_quota = ({"succ": video_per_bucket, "fail": video_per_bucket}
                       if (video_root and video_per_bucket > 0) else None)
        while done < n:
            g = min(group_size, n - done)
            try:
                grp_outcomes, oom = run_group_oom_safe(task, done, g, video_quota=video_quota)
                successes.extend(s for s in grp_outcomes if s is not None)
                any_oom = any_oom or oom
            except Exception as e:
                import traceback
                print(f"[eval] group {task}:{done} failed: {e}\n{traceback.format_exc()[:1200]}", flush=True)
            done += g
        torch.cuda.empty_cache()
        gc.collect()
        per_task_n[task] = len(successes)
        per_task_oom[task] = any_oom
        if successes:
            per_task[task] = float(np.mean(successes))
            print(f"[eval] {task}: success_rate={per_task[task]:.3f} (n={len(successes)}"
                  f"{', some groups OOM' if any_oom else ''})", flush=True)
        else:
            per_task[task] = None
            reason = "all groups OOM -> N/A" if any_oom else "no valid episodes (all resets failed?) -> N/A"
            print(f"[eval] {task}: {reason} (n=0)", flush=True)
    return {"per_task": per_task, "n": per_task_n, "oom": per_task_oom}


def _partition(items, n):
    buckets: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def _gpu_worker(gpu_id, args, tasks, result_queue):
    pin_worker_gpu(gpu_id)
    try:
        res = evaluate_tasks(args, tasks)
    except Exception:
        import traceback
        print(f"[eval] gpu {gpu_id} crashed:\n{traceback.format_exc()}", flush=True)
        res = {}
    result_queue.put(res)
    result_queue.close()
    result_queue.join_thread()
    sys.stdout.flush()
    os._exit(0)


def _print_summary(results, tasks, out_json: str | None = None) -> None:
    per_task = results.get("per_task", {})
    n_map = results.get("n", {})
    oom_map = results.get("oom", {})
    valid = [per_task.get(t) for t in tasks if per_task.get(t) is not None]
    overall = float(np.mean(valid)) if valid else 0.0
    print("\n==== RoboMME closed-loop success rates ====")
    for t in tasks:
        s = per_task.get(t)
        n = n_map.get(t, 0)
        flag = "  [OOM]" if oom_map.get(t) else ""
        if s is None:
            print(f"  {t:24s} {'N/A':>6s} (n={n}){flag}")
        else:
            print(f"  {t:24s} {s * 100:5.1f}% (n={n}){flag}")
    n_excluded = sum(1 for t in tasks if per_task.get(t) is None)
    note = f"  [excluded {n_excluded} N/A task(s)]" if n_excluded else ""
    print(f"  {'OVERALL (macro avg)':24s} {overall * 100:5.1f}%{note}", flush=True)

    if out_json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
            payload = {
                "tasks": {
                    t: {"success_rate": per_task.get(t), "n": int(n_map.get(t, 0)),
                        "oom": bool(oom_map.get(t))}
                    for t in tasks
                },
                "macro_avg": overall,
                "n_excluded_na": n_excluded,
            }
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[eval] wrote results JSON -> {out_json}", flush=True)
        except Exception as e:
            print(f"[eval] WARNING: could not write results JSON '{out_json}': {e}", flush=True)


def _validate_pipelined_horizon(args) -> None:
    if not getattr(args, "pipelined", False):
        return
    from simplememvla.data.messages import derive_video_sampling
    from simplememvla.model.configuration_simplememvla import SimpleMemVLAConfig

    cfg = SimpleMemVLAConfig.from_pretrained(args.pretrained_checkpoint)
    if bool(getattr(cfg, "variable_history", False)):
        raise SystemExit(
            "[eval] --pipelined supports fixed-window checkpoints only: its prefetch "
            "assumes the window SLIDES by one temporal patch per decision, but a "
            "variable_history=True checkpoint GROWS the window early in the episode. "
            "Drop --pipelined (the default batched path handles variable history)."
        )
    native = (
        cfg.native_video_fps
        if getattr(cfg, "native_video_fps", None) is not None
        else cfg.control_frequency_hz
    )
    _, stride = derive_video_sampling(
        cfg.history_video_sec, cfg.history_video_fps, float(native)
    )
    if int(args.execute_horizon) != 2 * stride:
        raise SystemExit(
            f"[eval] --pipelined requires --execute_horizon == 2*stride == {2 * stride} "
            f"(got {args.execute_horizon}). The pipelined split prefetches the NEXT "
            "window's first n-1 temporal patches, which is only what the next decision "
            "sees when the window slides by exactly ONE patch per decision."
        )


def main():
    args = parse_args()
    tasks = args.tasks if args.tasks else DEFAULT_TASKS
    _validate_pipelined_horizon(args)
    print(f"[eval] split={args.dataset_split}, episodes_per_task<={args.episodes_per_task} "
          f"(official frozen benchmark episodes; no seed search).", flush=True)

    out_json = None
    if args.log_file:
        out_json = os.path.join(os.path.dirname(os.path.abspath(args.log_file)), "results.json")
    elif args.video_dir:
        out_json = os.path.join(args.video_dir, "results.json")
    if args.video_dir:
        print(f"[eval] recording up to {args.video_max_per_task} success + {args.video_max_per_task} "
              f"failure video(s)/task under {args.video_dir}/<task>/ ; results JSON -> {out_json}", flush=True)

    num_gpus = max(1, int(args.num_gpus))
    if num_gpus == 1:
        results = evaluate_tasks(args, tasks)
        _print_summary(results, tasks, out_json=out_json)
        sys.stdout.flush()
        os._exit(0)
    else:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        result_queue: "mp.Queue" = mp.Queue()
        _vis = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        procs = []
        for gpu_id, assigned in enumerate(_partition(tasks, num_gpus)):
            if not assigned:
                continue
            p = mp.Process(target=_gpu_worker, args=(gpu_id, args, assigned, result_queue))
            p.start()
            procs.append(p)
            phys = _vis[gpu_id] if gpu_id < len(_vis) else gpu_id
            print(f"[eval] started worker {gpu_id} (physical GPU {phys}) for tasks {assigned}", flush=True)
        merged = {"per_task": {}, "n": {}, "oom": {}}
        remaining = len(procs)
        while remaining > 0:
            try:
                res = result_queue.get(timeout=args.worker_timeout)
                for key in ("per_task", "n", "oom"):
                    merged[key].update(res.get(key, {}))
                remaining -= 1
            except Exception:
                alive = [p for p in procs if p.is_alive()]
                if not alive:
                    print("[eval] all workers exited; stopping wait.", flush=True)
                    break
                print(f"[eval] still waiting on {len(alive)} worker(s)...", flush=True)
        for p in procs:
            p.join(timeout=10)
        _print_summary(merged, tasks, out_json=out_json)


if __name__ == "__main__":
    main()
