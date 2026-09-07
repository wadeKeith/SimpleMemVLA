
from __future__ import annotations

import sys
sys.path.append("./")
import rmbench_sim

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmbench_sim.batched_policy import BatchedEvalPolicy
from rmbench_sim.inproc_pool import InProcSimPool
from rmbench_sim.rmbench_env import VENDORED_SIM_DIR, pin_worker_gpu

_LOG_SUBTASKS = os.environ.get("SIMPLEMEMVLA_LOG_SUBTASKS", "") not in ("", "0")

DEFAULT_TASKS = [
    "observe_and_pickup",
    "put_back_block",
    "swap_T",
    "press_button",
    "place_block_mat",
    "battery_try",
    "rearrange_blocks",
    "swap_blocks",
    "cover_blocks",
    "blocks_ranking_try",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", default="checkpoints/simplememvla_rmbench")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of tasks to evaluate (default: the 10 implemented tasks).")
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--instruction_type", default="seen")
    ap.add_argument("--seeds_per_task", type=int, default=100)
    ap.add_argument("--group_size", type=int, default=2,
                    help="Envs evaluated in parallel per group (= envs batched into one forward).")
    ap.add_argument("--execute_horizon", type=int, default=16,
                    help="Receding horizon: execute the first N of the predicted chunk, then re-decide.")
    ap.add_argument("--dense_substeps", type=int, default=15,
                    help="Physics sub-steps per native waypoint for the continuous dense executor. "
                         "RMBench scene is 250 Hz and collection saves every save_freq=15 steps, so "
                         "one dataset frame = 15 physics steps (the converter's 50 Hz label is wrong). "
                         "15 matches the trained history cadence and the step_lim budget.")
    ap.add_argument("--capture_stride", type=int, default=1,
                    help="Native frames per RENDERED observation. 1 (default) = the official "
                         "every-frame cadence. 0 = auto (the policy's history stride, currently "
                         "8): RMBench renders with path tracing (rt shader, 32 spp, OIDN — the "
                         "most expensive op of a decision cycle) but the policy only READS "
                         "frames on its stride grid, so capturing just those is input-identical "
                         "and ~8x cheaper. Skipped slots are placeholders the policy raises on "
                         "if ever read; check_success is polled per physics sub-step and is "
                         "unaffected.")
    ap.add_argument("--max_seed_search", type=int, default=50)
    ap.add_argument("--num_denoising_steps", type=int, default=10)
    ap.add_argument("--eval_temperature", type=float, default=1.0,
                    help="Deterministic-ODE init-noise scale. 1.0 matches training (the flow "
                         "field is trained to denoise from unit-variance noise at t=1) and the "
                         "open-loop eval default, so closed-loop uses the same sampling "
                         "distribution. (<1.0 sharpens but starts the ODE off the trained t=1 "
                         "manifold; not used by default.)")
    ap.add_argument("--max_subtask_tokens", type=int, default=64)
    ap.add_argument("--pipelined", action="store_true", default=False,
                    help="PIPELINED decisions (rmbench_sim/pipelined_policy.py): the same exact "
                         "recompute, split so the 59 already-determined slots are prefilled while "
                         "the arm executes the previous chunk and only [last patch + wrists + "
                         "instruction] (~490 tokens, one single-frame VLA's worth) stays on the "
                         "decision-time critical path. Semantically identical to the default path "
                         "(chunk splitting is an identity for causal attention and the gated-delta "
                         "scan); measured 94.3%% macro vs 93.1%% over 9 tasks x 100 seeds, and "
                         "0.60 s critical path vs ~0.86 s fully serial.")
    ap.add_argument("--expert_check", action="store_true", default=True,
                    help="Validate each seed is expert-solvable + generate proper task language (official eval).")
    ap.add_argument("--no_expert_check", dest="expert_check", action="store_false",
                    help="Skip the (slow) CuRobo expert solve / seed filtering.")
    ap.add_argument("--base_seed", type=int, default=100000)
    ap.add_argument("--seed_cache_path", default=None,
                    help="Precollected expert-solvable seed cache (from precollect_seeds.py). If omitted, "
                         "auto-loads the default path seeds/<task_config>__<instruction_type>.json when it "
                         "exists. When a cache is used, seeds + frozen instructions are drawn from it and the "
                         "per-seed CuRobo expert solve is bypassed. Use --no_seed_cache to force live expert_check.")
    ap.add_argument("--no_seed_cache", dest="seed_cache_path", action="store_const", const="",
                    help="Ignore any seed cache and run the live expert_check seed search.")
    ap.add_argument("--compute_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help="VLM backbone full-attention backend: sdpa (portable) | flash_attention_2 (fast).")
    ap.add_argument("--step_timeout", type=float, default=300.0)
    ap.add_argument("--reset_timeout", type=float, default=1800.0)
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="GPU workers (one process per GPU); the task list is sharded across them, "
                         "each worker loading its own model replica + SAPIEN pool. Set "
                         "CUDA_VISIBLE_DEVICES to the GPU list (scripts/eval_rmbench.sh passes GPUS).")
    ap.add_argument("--worker_timeout", type=float, default=7200.0,
                    help="seconds to wait for each multi-GPU worker's results.")
    ap.add_argument("--video_dir", default=None,
                    help="If set, record rollout MP4s under <video_dir>/<task>/<task>__seed<seed>__<succ|fail>.mp4 "
                         "(three cameras tiled side-by-side). For each task ONE success + ONE failure clip is "
                         "kept (frames buffer in RAM). Default None disables recording.")
    ap.add_argument("--video_max_per_task", type=int, default=1,
                    help="Clips to keep per outcome bucket (success / failure) per task when --video_dir is set. "
                         "1 -> one success + one failure example per task; 0 disables.")
    ap.add_argument("--log_file", default=None,
                    help="If set, tee all eval stdout to this file (logs/ convention). The launcher passes a "
                         "timestamped path so each run keeps its own full log next to the videos.")
    return ap.parse_args()


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


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

    from rmbench_sim.policy import RMBenchPolicy, get_vla

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
        from rmbench_sim.pipelined_policy import PipelinedBufferPolicy, PipelinedEvalGroup

        print("[eval] PIPELINED decisions: exact recompute, split so 59/60 slots prefill "
              "during action execution; critical path = last patch + wrists + instruction",
              flush=True)
        batched = PipelinedEvalGroup(
            model=vla, processor=processor,
            unnormalize_action=unnormalize_action, normalize_state=normalize_state,
            num_denoising_steps=args.num_denoising_steps, eval_temperature=args.eval_temperature,
            max_subtask_tokens=args.max_subtask_tokens, device=device,
        )

        def buffer_factory():
            return PipelinedBufferPolicy(
                vla=vla, processor=processor,
                unnormalize_action=unnormalize_action, normalize_state=normalize_state,
                num_denoising_steps=args.num_denoising_steps, temperature=args.eval_temperature,
                max_subtask_tokens=args.max_subtask_tokens,
            )

        return batched, buffer_factory, normalize_state
    batched = BatchedEvalPolicy(
        model=vla, processor=processor,
        unnormalize_action=unnormalize_action, normalize_state=normalize_state,
        num_denoising_steps=args.num_denoising_steps, eval_temperature=args.eval_temperature,
        max_subtask_tokens=args.max_subtask_tokens, device=device,
    )

    def buffer_factory():
        return RMBenchPolicy(
            vla=vla, processor=processor,
            unnormalize_action=unnormalize_action, normalize_state=normalize_state,
            num_denoising_steps=args.num_denoising_steps, temperature=args.eval_temperature,
            max_subtask_tokens=args.max_subtask_tokens,
        )

    return batched, buffer_factory, normalize_state


def load_seed_cache(args):
    from rmbench_sim.precollect_seeds import default_out_path

    path = args.seed_cache_path
    if path == "":
        return {}, None
    if path is None:
        path = default_out_path(args.task_config, args.instruction_type)
        if not os.path.isfile(path):
            return {}, None
    if not os.path.isfile(path):
        print(f"[eval] WARNING: seed_cache_path '{path}' not found — using live expert_check.", flush=True)
        return {}, None
    try:
        with open(path) as f:
            data = json.load(f)
        pool = {}
        for t, b in data.get("tasks", {}).items():
            seeds = b.get("success_seeds") or []
            if seeds:
                pool[t] = {"seeds": [int(s) for s in seeds], "instr": dict(b.get("instructions") or {})}
        return pool, path
    except Exception as e:
        print(f"[eval] WARNING: could not load seed cache '{path}': {e} — using live expert_check.", flush=True)
        return {}, None


def resolve_capture_stride(args, policy) -> int:
    cached = getattr(args, "_capture_stride_resolved", None)
    if cached is not None:
        return cached
    requested = int(getattr(args, "capture_stride", 1) or 0)
    stride = int(getattr(policy, "stride", 1) or 1)
    horizon = int(args.execute_horizon)

    def _remember(value: int) -> int:
        args._capture_stride_resolved = value
        return value

    if requested == 1:
        return _remember(1)
    if requested == 0:
        if stride > 1 and horizon % stride == 0:
            print(f"[eval] capture_stride=auto -> {stride}: rendering only the native "
                  f"frames the policy reads (history stride {stride}, execute_horizon "
                  f"{horizon}); un-rendered slots are placeholders the policy asserts it "
                  "never reads.", flush=True)
            return _remember(stride)
        print(f"[eval] capture_stride=auto -> 1 (history stride {stride} does not divide "
              f"execute_horizon {horizon}): rendering every native frame.", flush=True)
        return _remember(1)
    if requested < 1 or stride % requested or horizon % requested:
        raise ValueError(
            f"--capture_stride {requested} must divide BOTH the policy history stride "
            f"({stride}) and --execute_horizon ({horizon}); otherwise the policy would "
            "read un-rendered frames.")
    print(f"[eval] capture_stride={requested}: rendering 1 of every {requested} native "
          "frames (the policy's read grid).", flush=True)
    return _remember(requested)


def run_group(args, pool, batched, buffer_factory, normalize_state, task, specs,
              video_dir=None, video_quota=None):
    G = len(specs)
    probe = buffer_factory()
    image_keys = list(probe.image_keys)
    cam_map = {k.split(".")[-1]: k for k in image_keys}
    capture_stride = resolve_capture_stride(args, probe)
    if getattr(probe, "variable_history", False) and args.execute_horizon != 2 * probe.stride:
        print(f"[eval] WARNING: execute_horizon {args.execute_horizon} != 2*stride "
              f"({2 * probe.stride}); with variable_history the clip then advances by "
              "something other than one temporal patch per decision -> prompt-cache and "
              "ViT-patch-cache misses every step and the pipelined prefetch never hits "
              "(still exact, just slow).", flush=True)
    frame_index = [0] * G
    last_real = [None] * G

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
        from rmbench_sim.video_writer import RolloutVideoRecorder
        native_fps = (float(getattr(probe, "native_fps", 0) or 0) or 16.6667) / capture_stride
        recorders = [RolloutVideoRecorder(fps=native_fps) for _ in range(G)]

    def save_video(g, ok):
        rec = recorders[g]
        if rec is None or len(rec) == 0:
            return
        bucket = "succ" if ok else "fail"
        if not video_quota or video_quota.get(bucket, 0) <= 0:
            return
        seed = specs[g].get("seed", g)
        out_path = os.path.join(video_dir, f"{task}__seed{seed}__{bucket}.mp4")
        nframes = len(rec)
        try:
            if rec.save(out_path):
                video_quota[bucket] -= 1
                print(f"[eval] saved {bucket} video: {out_path} ({nframes} frames)", flush=True)
        except Exception as e:
            print(f"[eval] WARNING: video save failed for {task} seed{seed}: {e}", flush=True)

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
        buffers[g].observe(to_full(r["frame"]))
        last_real[g] = r["frame"]
        if recorders[g] is not None:
            recorders[g].add(r["frame"])
        instructions[g] = r["instruction"]
        cur_state[g] = np.asarray(r["state"], dtype=np.float32)
        active.append(g)
    if len(active) < G:
        print(f"[eval] {task}: {G - len(active)}/{G} env reset(s) failed -> those seeds "
              "are EXCLUDED from the success rate (None), not counted as failures.", flush=True)

    success = [False] * G
    errored = [False] * G

    def _results():
        return [
            success[g] if (reset_ok[g] and not errored[g]) else None for g in range(G)
        ]

    if not active:
        return _results()

    step_lims = [int(resets[g]["step_lim"]) for g in active if resets[g].get("step_lim")]
    max_step_lim = max(step_lims) if step_lims else 10**9
    hard_bound = max(1, -(-max_step_lim // max(1, args.execute_horizon))) + 2

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
            if _LOG_SUBTASKS:
                print(f"[subtask] {task} seed={specs[g].get('seed', g)} "
                      f"dec={decisions_taken} env={g}: {_subtask}", flush=True)

        step_results = pool.step(action_chunks, act, capture_stride=capture_stride,
                                 frame_index=frame_index)
        for g in act:
            res = step_results[g]
            if not isinstance(res, dict) or "error" in res:
                err = res.get("error") if isinstance(res, dict) else res
                print(f"[eval] {task} seed {specs[g].get('seed')}: step FAILED ({err}) -> "
                      "EXCLUDED from the success rate (not counted as a failure).",
                      flush=True)
                errored[g] = True
                success[g] = False
                if g in active:
                    active.remove(g)
                continue
            frames = res.get("frames", [])
            offsets = res.get("frame_offsets") or list(range(1, len(frames) + 1))
            by_offset = dict(zip(offsets, frames))
            for j in range(1, int(res.get("consumed", len(frames))) + 1):
                fr = by_offset.get(j)
                if fr is None:
                    buffers[g].observe(to_full(last_real[g]), placeholder=True)
                    continue
                last_real[g] = fr
                buffers[g].observe(to_full(fr))
                if recorders[g] is not None:
                    recorders[g].add(fr)
            frame_index[g] += int(res.get("consumed", len(frames)))
            if res.get("states"):
                cur_state[g] = np.asarray(res["states"][-1], dtype=np.float32)
            success[g] = bool(res.get("success", False))
            if res.get("done", False) and g in active:
                active.remove(g)
                save_video(g, ok=success[g])

    for g in active:
        save_video(g, ok=success[g])

    return _results()


def evaluate_tasks(args, tasks, seed_pool) -> dict:
    import gc

    torch.manual_seed(0)
    np.random.seed(0)

    n = int(args.seeds_per_task)
    group_size = int(min(args.group_size, n))

    batched, buffer_factory, normalize_state = build_policy(args)
    pool = InProcSimPool(
        num_envs=group_size, repo_root=VENDORED_SIM_DIR, task_config=args.task_config,
        instruction_type=args.instruction_type, step_timeout=args.step_timeout, reset_timeout=args.reset_timeout,
        dense_substeps=args.dense_substeps,
    )
    pool.start()

    def build_specs(task, done, g):
        tp = seed_pool.get(task)
        specs = []
        for e in range(g):
            idx = done + e
            if tp and tp["seeds"]:
                seed = tp["seeds"][idx % len(tp["seeds"])]
                specs.append({"task": task, "seed": int(seed),
                              "instruction": tp["instr"].get(str(seed)),
                              "expert_check": False, "max_seed_search": args.max_seed_search})
            else:
                specs.append({"task": task, "seed": args.base_seed + idx * 1000,
                              "expert_check": args.expert_check, "max_seed_search": args.max_seed_search})
        return specs

    video_root = getattr(args, "video_dir", None)
    video_per_bucket = max(0, int(getattr(args, "video_max_per_task", 1) or 0))

    def run_group_oom_safe(task, done, g, video_quota=None):
        specs = build_specs(task, done, g)
        vdir = os.path.join(video_root, task) if video_root else None
        try:
            return list(run_group(args, pool, batched, buffer_factory, normalize_state, task,
                                  specs, video_dir=vdir, video_quota=video_quota)[:g]), False

        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            pass
        torch.cuda.empty_cache()
        gc.collect()
        if g <= 1:
            print(f"[eval] group {task}:{done} OOM even at group=1 -- skipped (not counted as failure).",
                  flush=True)
            return [None], True
        half = g // 2
        print(f"[eval] group {task}:{done} CUDA OOM at group={g} -- retrying as {half}+{g - half}.",
              flush=True)
        left, _ = run_group_oom_safe(task, done, half, video_quota=video_quota)
        right, _ = run_group_oom_safe(task, done + half, g - half, video_quota=video_quota)
        return left + right, True

    per_task: dict[str, float | None] = {}
    per_task_n: dict[str, int] = {}
    per_task_oom: dict[str, bool] = {}
    per_task_seed: dict[str, dict] = {}
    for task in tasks:
        successes = []
        seed_outcomes: dict[int, list] = {}
        any_oom = False
        done = 0
        n = int(args.seeds_per_task)
        pooled = len((seed_pool.get(task) or {}).get("seeds") or [])
        if pooled and pooled < n:
            print(f"[eval] {task}: seed cache holds only {pooled} expert-solvable seed(s) "
                  f"(< {n} requested) — evaluating those {pooled}, not re-running any "
                  "seed twice.", flush=True)
            n = pooled
        video_quota = ({"succ": video_per_bucket, "fail": video_per_bucket}
                       if (video_root and video_per_bucket > 0) else None)
        while done < n:
            g = min(group_size, n - done)
            specs_g = build_specs(task, done, g)
            try:
                grp_outcomes, oom = run_group_oom_safe(task, done, g, video_quota=video_quota)
                for idx_g, outcome in enumerate(grp_outcomes):
                    if outcome is not None:
                        successes.append(outcome)
                        seed = int(specs_g[idx_g].get("seed", -1))
                        seed_outcomes.setdefault(seed, []).append(bool(outcome))
                any_oom = any_oom or oom
            except Exception as e:
                import traceback
                print(f"[eval] group {task}:{done} failed: {e}\n{traceback.format_exc()[:1200]}", flush=True)
            done += g
        torch.cuda.empty_cache()
        gc.collect()
        per_task_n[task] = len(successes)
        per_task_oom[task] = any_oom
        per_task_seed[task] = seed_outcomes
        if successes:
            per_task[task] = float(np.mean(successes))
            print(f"[eval] {task}: success_rate={per_task[task]:.3f} (n={len(successes)}"
                  f"{', some groups OOM' if any_oom else ''})", flush=True)
        else:
            per_task[task] = None
            reason = "all groups OOM -> N/A" if any_oom else "no valid seeds (all resets failed?) -> N/A"
            print(f"[eval] {task}: {reason} (n=0)", flush=True)
    return {"per_task": per_task, "n": per_task_n, "oom": per_task_oom,
            "per_seed": per_task_seed}


def _partition(items, n):
    buckets: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def _gpu_worker(gpu_id, args, tasks, seed_pool, result_queue):
    pin_worker_gpu(gpu_id)
    try:
        res = evaluate_tasks(args, tasks, seed_pool)
    except Exception:
        import traceback
        print(f"[eval] gpu {gpu_id} crashed:\n{traceback.format_exc()}", flush=True)
        res = {}
    result_queue.put(res)
    result_queue.close()
    result_queue.join_thread()
    sys.stdout.flush()
    os._exit(0)


def _print_seed_cache_info(args, tasks, seed_pool, cache_path) -> None:
    n = int(args.seeds_per_task)
    if seed_pool:
        n_seeds = sum(len(v["seeds"]) for v in seed_pool.values())
        print(f"[eval] seed cache: {len(seed_pool)} tasks, {n_seeds} expert-solvable seeds from "
              f"'{cache_path}' — drawing seeds + frozen instructions, per-seed expert_check bypassed.", flush=True)
        thin = sorted(t for t in tasks if seed_pool.get(t) and len(seed_pool[t]["seeds"]) < n)
        if thin:
            print(f"[eval] NOTE: {len(thin)} task(s) have fewer than seeds_per_task={n} cached seeds "
                  f"(seeds will be reused cyclically): {thin}. Re-run precollect with a larger "
                  f"--num_seeds for more distinct seeds.", flush=True)
        missing = sorted(t for t in tasks if not seed_pool.get(t))
        if missing:
            print(f"[eval] NOTE: {len(missing)} task(s) not in the cache — falling back to live "
                  f"expert_check for them: {missing}.", flush=True)
    else:
        print("[eval] seed cache: none — using live expert_check (slow CuRobo solve per seed).", flush=True)


def _print_summary(results, tasks, out_json: str | None = None) -> float | None:
    per_task = results.get("per_task", {})
    n_map = results.get("n", {})
    oom_map = results.get("oom", {})
    valid = [per_task.get(t) for t in tasks if per_task.get(t) is not None]
    overall = float(np.mean(valid)) if valid else None
    print("\n==== RMBench closed-loop success rates ====")
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
    shown_overall = "   N/A" if overall is None else f"{overall * 100:5.1f}%"
    print(f"  {'OVERALL (macro avg)':24s} {shown_overall}{note}", flush=True)

    if out_json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
            per_seed = results.get("per_seed", {})
            payload = {
                "tasks": {
                    t: {"success_rate": per_task.get(t), "n": int(n_map.get(t, 0)),
                        "oom": bool(oom_map.get(t)),
                        "per_seed": {str(k): v for k, v in (per_seed.get(t) or {}).items()}}
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
    return overall


def main():
    args = parse_args()
    tasks = args.tasks if args.tasks else DEFAULT_TASKS
    seed_pool, cache_path = load_seed_cache(args)
    _print_seed_cache_info(args, tasks, seed_pool, cache_path)

    out_json = None
    if args.video_dir:
        out_json = os.path.join(args.video_dir, "results.json")
        print(f"[eval] recording up to {args.video_max_per_task} success + {args.video_max_per_task} "
              f"failure video(s)/task under {args.video_dir}/<task>/ ; results JSON -> {out_json}", flush=True)
    elif args.log_file:
        out_json = os.path.join(os.path.dirname(os.path.abspath(args.log_file)), "results.json")

    num_gpus = max(1, int(args.num_gpus))
    if num_gpus == 1:
        results = evaluate_tasks(args, tasks, seed_pool)
        overall = _print_summary(results, tasks, out_json=out_json)
        sys.stdout.flush()
        os._exit(0 if overall is not None else 3)
    else:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        result_queue: "mp.Queue" = mp.Queue()
        _vis = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        procs = []
        for gpu_id, assigned in enumerate(_partition(tasks, num_gpus)):
            if not assigned:
                continue
            p = mp.Process(target=_gpu_worker, args=(gpu_id, args, assigned, seed_pool, result_queue))
            p.start()
            procs.append(p)
            phys = _vis[gpu_id] if gpu_id < len(_vis) else gpu_id
            print(f"[eval] started worker {gpu_id} (physical GPU {phys}) for tasks {assigned}", flush=True)
        merged = {"per_task": {}, "n": {}, "oom": {}, "per_seed": {}}
        remaining = len(procs)
        while remaining > 0:
            try:
                res = result_queue.get(timeout=args.worker_timeout)
                for key in ("per_task", "n", "oom", "per_seed"):
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
        overall = _print_summary(merged, tasks, out_json=out_json)
        sys.exit(0 if overall is not None else 3)


if __name__ == "__main__":
    main()
