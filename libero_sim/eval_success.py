
from __future__ import annotations

import sys
sys.path.append("./")
import libero_sim

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero_sim.batched_policy import BatchedEvalPolicy
from libero_sim.inproc_pool import InProcSimPool
from libero_sim.libero_env import (
    ACTION_SPACES,
    JOINT_DELTA_MAX,
    JOINT_KP,
    SUITES,
    TASK_SUITE_MAX_STEPS,
    pin_worker_gpu,
)

DEFAULT_TASK_KEYS = [f"{suite}/{tid}" for suite in SUITES for tid in range(10)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", default="./checkpoints/sft/simplememvla/libero_baseline")
    ap.add_argument("--task_suites", nargs="*", default=None, choices=SUITES,
                    help="Suites to evaluate (default: all four).")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Explicit task keys 'suite/task_id' (overrides --task_suites).")
    ap.add_argument("--episodes_per_task", type=int, default=50,
                    help="Episodes per task (capped at the 50 frozen init states).")
    ap.add_argument("--group_size", type=int, default=10,
                    help="Envs evaluated in parallel per group (= envs batched into one forward).")
    ap.add_argument("--execute_horizon", type=int, default=8,
                    help="Receding horizon: execute the first N of the predicted 16-step chunk, "
                         "then re-decide (0.4 s of control per decision at 20 Hz).")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="Max policy control steps per episode; 0 = the per-suite community "
                         "budget (spatial/object 280, goal 300, libero_10 520).")
    ap.add_argument("--num_steps_wait", type=int, default=10,
                    help="No-op settle steps after set_init_state (not shown to the policy).")
    ap.add_argument("--num_denoising_steps", type=int, default=10)
    ap.add_argument("--eval_temperature", type=float, default=1.0,
                    help="Deterministic-ODE init-noise scale. 1.0 matches training (the flow "
                         "field is trained to denoise from unit-variance noise at t=1) and the "
                         "open-loop eval default, so closed-loop uses the same sampling "
                         "distribution.")
    ap.add_argument("--max_reasoning_tokens", type=int, default=256,
                    help="Budget for the generated sub-task answer. The dataset's longest "
                         "sub-task is ~13 tokens; 256 leaves ample headroom without letting "
                         "a degenerate decode run away.")
    ap.add_argument("--pipelined", action="store_true", default=False,
                    help="PIPELINED decisions (libero_sim/pipelined_policy.py): the same exact "
                         "recompute, split so every already-determined temporal patch of the "
                         "NEXT window is encoded and prefilled while the arm executes the "
                         "current chunk, leaving only [last patch + wrist + instruction] on the "
                         "decision-time critical path. Semantically identical to the default "
                         "path (chunk splitting is an identity for causal attention and the "
                         "gated-delta scan); decisions then run one env at a time instead of as "
                         "one padded batch, so this trades eval wall-clock for the deployment "
                         "latency it measures.")
    ap.add_argument("--compute_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help="VLM backbone full-attention backend: sdpa (portable) | flash_attention_2 (fast).")
    ap.add_argument("--step_timeout", type=float, default=300.0)
    ap.add_argument("--reset_timeout", type=float, default=600.0,
                    help="Per-episode reset timeout (scene build + set_init_state + settle).")
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="GPU workers (one process per GPU); the task list is sharded across them, "
                         "each worker loading its own model replica + env pool. Set "
                         "CUDA_VISIBLE_DEVICES to the GPU list (scripts/eval_libero.sh passes GPUS).")
    ap.add_argument("--worker_timeout", type=float, default=14400.0,
                    help="IDLE timeout (s): how long to wait with NO worker "
                         "reporting a finished task before declaring the sweep "
                         "truncated (killing the stragglers and exiting 2). Every "
                         "task result re-arms it, so a slow-but-healthy sweep of "
                         "any total length is fine.")
    ap.add_argument("--video_dir", default=None,
                    help="If set, record rollout MP4s under <video_dir>/<suite>/<task>__ep<idx>__<succ|fail>.mp4 "
                         "(agentview + wrist tiled side-by-side). For each task ONE success + ONE "
                         "failure clip is kept (frames buffer in RAM). Default None disables.")
    ap.add_argument("--video_max_per_task", type=int, default=1,
                    help="Clips to keep per outcome bucket (success / failure) per task when "
                         "--video_dir is set. 0 disables.")
    ap.add_argument("--log_file", default=None,
                    help="If set, results.json is placed next to this log path (logs/ convention).")
    return ap.parse_args()


def resolve_task_keys(args) -> list[str]:
    if args.tasks:
        keys = []
        for key in args.tasks:
            suite, _, tid = key.partition("/")
            if suite not in SUITES or not tid.isdigit() or not (0 <= int(tid) < 10):
                raise ValueError(f"Bad task key {key!r}; expected e.g. libero_10/3")
            keys.append(f"{suite}/{int(tid)}")
        return keys
    suites = args.task_suites if args.task_suites else SUITES
    return [f"{suite}/{tid}" for suite in suites for tid in range(10)]


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

    from libero_sim.policy import LiberoPolicy, get_vla

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
        from libero_sim.pipelined_policy import PipelinedBufferPolicy, PipelinedEvalGroup

        print("[eval] PIPELINED decisions: exact recompute, split so the next window's "
              "already-determined patches are encoded + prefilled during action "
              "execution; critical path = last patch + wrist + instruction.", flush=True)
        batched = PipelinedEvalGroup(
            model=vla, processor=processor,
            unnormalize_action=unnormalize_action, normalize_state=normalize_state,
            num_denoising_steps=args.num_denoising_steps,
            eval_temperature=args.eval_temperature,
            max_subtask_tokens=args.max_reasoning_tokens,
            execute_horizon=args.execute_horizon, device=device,
        )

        def buffer_factory():
            return PipelinedBufferPolicy(
                vla=vla, processor=processor,
                unnormalize_action=unnormalize_action, normalize_state=normalize_state,
                num_denoising_steps=args.num_denoising_steps,
                temperature=args.eval_temperature,
                max_reasoning_tokens=args.max_reasoning_tokens,
            )

        return batched, buffer_factory, normalize_state
    batched = BatchedEvalPolicy(
        model=vla, processor=processor,
        unnormalize_action=unnormalize_action, normalize_state=normalize_state,
        num_denoising_steps=args.num_denoising_steps, eval_temperature=args.eval_temperature,
        max_subtask_tokens=args.max_reasoning_tokens, device=device,
    )

    def buffer_factory():
        return LiberoPolicy(
            vla=vla, processor=processor,
            unnormalize_action=unnormalize_action, normalize_state=normalize_state,
            num_denoising_steps=args.num_denoising_steps, temperature=args.eval_temperature,
            max_reasoning_tokens=args.max_reasoning_tokens,
        )

    return batched, buffer_factory, normalize_state


def run_group(args, pool, batched, buffer_factory, normalize_state, task_key, specs,
              max_steps, video_dir=None, video_quota=None):
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
        from libero_sim.video_writer import RolloutVideoRecorder
        native_fps = float(getattr(buffer_factory(), "native_fps", 0) or 0) or 20.0
        recorders = [RolloutVideoRecorder(fps=native_fps) for _ in range(G)]

    task_tag = task_key.replace("/", "_t")

    def save_video(g, ok):
        rec = recorders[g]
        if rec is None or len(rec) == 0:
            return
        bucket = "succ" if ok else "fail"
        if not video_quota or video_quota.get(bucket, 0) <= 0:
            return
        episode = specs[g].get("episode", g)
        out_path = os.path.join(video_dir, f"{task_tag}__ep{episode}__{bucket}.mp4")
        nframes = len(rec)
        try:
            if rec.save(out_path):
                video_quota[bucket] -= 1
                print(f"[eval] saved {bucket} video: {out_path} ({nframes} frames)", flush=True)
        except Exception as e:
            print(f"[eval] WARNING: video save failed for {task_key} ep{episode}: {e}", flush=True)

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
        print(f"[eval] {task_key}: {G - len(active)}/{G} env reset(s) failed -> those episodes "
              "are EXCLUDED from the success rate (None), not counted as failures.", flush=True)

    success = [False] * G

    def _results():
        return [success[g] if reset_ok[g] else None for g in range(G)]

    if not active:
        return _results()

    hard_bound = max(1, -(-int(max_steps) // max(1, args.execute_horizon))) + 2

    crashed = 0
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
            actions_unnorm, _reasoning = decisions[k]
            action_chunks[g] = actions_unnorm[: args.execute_horizon]

        step_results = pool.step(action_chunks, act)
        for g in act:
            res = step_results[g]
            if not isinstance(res, dict) or "error" in res or res.get("error_message"):
                success[g] = False
                crashed += 1
                reason = res.get("error") or res.get("error_message") if isinstance(res, dict) else res
                print(f"[eval] {task_key}: episode slot {g} CRASHED mid-rollout "
                      f"({reason}) — counted as a FAILURE.", flush=True)
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

    if crashed:
        print(f"[eval] {task_key}: {crashed}/{G} episode(s) in this group ended in an "
              "infra crash and are inside the reported failure count.", flush=True)

    return _results()


def evaluate_tasks(args, task_keys, on_task_done=None) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)

    n_req = int(args.episodes_per_task)
    group_size = int(min(args.group_size, n_req))

    batched, buffer_factory, normalize_state = build_policy(args)
    if int(args.execute_horizon) > batched.action_horizon:
        raise ValueError(
            f"--execute_horizon {args.execute_horizon} exceeds the checkpoint's "
            f"action_horizon {batched.action_horizon}; a chunk has no more actions."
        )
    by_dim = {dim: name for name, dim in ACTION_SPACES.items()}
    if batched.action_dim not in by_dim:
        raise ValueError(
            f"Checkpoint action_dim {batched.action_dim} matches no known action "
            f"space {ACTION_SPACES}"
        )
    action_space = by_dim[batched.action_dim]
    gains = (f" JOINT_KP={JOINT_KP} delta_max={JOINT_DELTA_MAX}"
             if action_space == "joint" else "")
    print(f"[eval] action_space={action_space} (action_dim={batched.action_dim}){gains} "
          f"execute_horizon={args.execute_horizon}/{batched.action_horizon}", flush=True)
    override_steps = int(args.max_steps) if int(args.max_steps) > 0 else None
    pool = InProcSimPool(
        num_envs=group_size, max_steps=override_steps,
        num_steps_wait=args.num_steps_wait,
        step_timeout=args.step_timeout, reset_timeout=args.reset_timeout,
        action_space=action_space,
    )
    pool.start()

    def build_specs(suite, task_id, done, g):
        return [{"suite": suite, "task_id": task_id, "episode": done + e} for e in range(g)]

    video_root = getattr(args, "video_dir", None)
    video_per_bucket = max(0, int(getattr(args, "video_max_per_task", 1) or 0))

    def run_group_oom_safe(task_key, suite, task_id, done, g, max_steps, video_quota=None):
        specs = build_specs(suite, task_id, done, g)
        vdir = os.path.join(video_root, suite) if video_root else None
        if vdir:
            os.makedirs(vdir, exist_ok=True)
        try:
            return list(run_group(args, pool, batched, buffer_factory, normalize_state,
                                  task_key, specs, max_steps,
                                  video_dir=vdir, video_quota=video_quota)[:g]), False
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if g <= 1:
                print(f"[eval] group {task_key}:{done} OOM even at group=1 — skipped "
                      "(not counted as failure).", flush=True)
                return [], True
            half = g // 2
            print(f"[eval] group {task_key}:{done} CUDA OOM at group={g} — retrying as "
                  f"{half}+{g - half}.", flush=True)
            left, oom_l = run_group_oom_safe(task_key, suite, task_id, done, half,
                                            max_steps, video_quota=video_quota)
            right, oom_r = run_group_oom_safe(task_key, suite, task_id, done + half,
                                             g - half, max_steps, video_quota=video_quota)
            return left + right, (oom_l or oom_r)

    per_task: dict[str, float | None] = {}
    per_task_n: dict[str, int] = {}
    per_task_oom: dict[str, bool] = {}
    for task_key in task_keys:
        suite, _, tid = task_key.partition("/")
        task_id = int(tid)
        max_steps = override_steps or TASK_SUITE_MAX_STEPS[suite]
        n = min(n_req, 50)
        successes = []
        any_oom = False
        done = 0
        video_quota = ({"succ": video_per_bucket, "fail": video_per_bucket}
                       if (video_root and video_per_bucket > 0) else None)
        while done < n:
            g = min(group_size, n - done)
            try:
                grp_outcomes, oom = run_group_oom_safe(
                    task_key, suite, task_id, done, g, max_steps, video_quota=video_quota
                )
                successes.extend(s for s in grp_outcomes if s is not None)
                any_oom = any_oom or oom
            except Exception as e:
                import traceback
                print(f"[eval] group {task_key}:{done} failed: {e} — its {g} episode(s) are "
                      f"DROPPED from the success rate (N/A), so the reported n shrinks.\n"
                      f"{traceback.format_exc()[:1200]}", flush=True)
            done += g
        torch.cuda.empty_cache()
        gc.collect()
        per_task_n[task_key] = len(successes)
        per_task_oom[task_key] = any_oom
        if successes:
            per_task[task_key] = float(np.mean(successes))
            print(f"[eval] {task_key}: success_rate={per_task[task_key]:.3f} "
                  f"(n={len(successes)}{', some groups OOM' if any_oom else ''})", flush=True)
        else:
            per_task[task_key] = None
            reason = ("all groups OOM -> N/A" if any_oom
                      else "no valid episodes (all resets failed?) -> N/A")
            print(f"[eval] {task_key}: {reason} (n=0)", flush=True)
        if on_task_done is not None:
            on_task_done(task_key, per_task[task_key], per_task_n[task_key],
                         per_task_oom[task_key])
    pool.close()
    return {"per_task": per_task, "n": per_task_n, "oom": per_task_oom}


def _partition(items, n):
    buckets: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def _gpu_worker(gpu_id, args, task_keys, result_queue):
    pin_worker_gpu(gpu_id)

    def deliver(task_key, rate, n, oom):
        result_queue.put({"type": "task", "per_task": {task_key: rate},
                          "n": {task_key: n}, "oom": {task_key: oom}})

    try:
        evaluate_tasks(args, task_keys, on_task_done=deliver)
    except Exception:
        import traceback
        print(f"[eval] gpu {gpu_id} crashed:\n{traceback.format_exc()}", flush=True)
    result_queue.put({"type": "worker_done", "gpu_id": gpu_id})
    result_queue.close()
    result_queue.join_thread()
    sys.stdout.flush()
    os._exit(0)


def _print_summary(results, task_keys, out_json: str | None = None,
                   truncated: int = 0) -> None:
    per_task = results.get("per_task", {})
    n_map = results.get("n", {})
    oom_map = results.get("oom", {})
    valid = [per_task.get(t) for t in task_keys if per_task.get(t) is not None]
    overall = float(np.mean(valid)) if valid else None
    print("\n==== LIBERO closed-loop success rates ====")
    suite_rates: dict[str, list[float]] = {}
    for t in task_keys:
        s = per_task.get(t)
        n = n_map.get(t, 0)
        flag = "  [OOM]" if oom_map.get(t) else ""
        if s is None:
            print(f"  {t:24s} {'N/A':>6s} (n={n}){flag}")
        else:
            print(f"  {t:24s} {s * 100:5.1f}% (n={n}){flag}")
            suite_rates.setdefault(t.split("/")[0], []).append(s)
    print("  ---- per-suite (macro over valid tasks) ----")
    suite_avgs = {}
    for suite in SUITES:
        if any(t.startswith(suite + "/") for t in task_keys):
            rates = suite_rates.get(suite, [])
            if rates:
                suite_avgs[suite] = float(np.mean(rates))
                print(f"  {suite:24s} {np.mean(rates) * 100:5.1f}% ({len(rates)} tasks)")
            else:
                suite_avgs[suite] = None
                print(f"  {suite:24s} {'N/A':>6s}")
    n_excluded = sum(1 for t in task_keys if per_task.get(t) is None)
    note = f"  [excluded {n_excluded} N/A task(s)]" if n_excluded else ""
    overall_str = f"{overall * 100:5.1f}%" if overall is not None else "  N/A"
    print(f"  {'OVERALL (macro avg)':24s} {overall_str}{note}", flush=True)

    if out_json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
            payload = {
                "tasks": {
                    t: {"success_rate": per_task.get(t), "n": int(n_map.get(t, 0)),
                        "oom": bool(oom_map.get(t))}
                    for t in task_keys
                },
                "suites": suite_avgs,
                "macro_avg": overall,
                "n_excluded_na": n_excluded,
                "truncated": bool(truncated),
                "n_workers_truncated": int(truncated),
            }
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[eval] wrote results JSON -> {out_json}", flush=True)
        except Exception as e:
            print(f"[eval] WARNING: could not write results JSON '{out_json}': {e}", flush=True)


def main():
    args = parse_args()
    task_keys = resolve_task_keys(args)
    print(f"[eval] {len(task_keys)} task(s), episodes_per_task<={args.episodes_per_task} "
          "(frozen benchmark init states; success = env.check_success()).", flush=True)

    out_json = None
    if args.video_dir:
        out_json = os.path.join(args.video_dir, "results.json")
        print(f"[eval] recording up to {args.video_max_per_task} success + "
              f"{args.video_max_per_task} failure video(s)/task under "
              f"{args.video_dir}/<suite>/ ; results JSON -> {out_json}", flush=True)
    elif args.log_file:
        out_json = os.path.join(os.path.dirname(os.path.abspath(args.log_file)), "results.json")

    num_gpus = max(1, int(args.num_gpus))
    if num_gpus == 1:
        results = evaluate_tasks(args, task_keys)
        _print_summary(results, task_keys, out_json=out_json)
        sys.stdout.flush()
        os._exit(0)
    else:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        result_queue: "mp.Queue" = mp.Queue()
        _vis = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        procs = []
        for gpu_id, assigned in enumerate(_partition(task_keys, num_gpus)):
            if not assigned:
                continue
            p = mp.Process(target=_gpu_worker, args=(gpu_id, args, assigned, result_queue))
            p.start()
            procs.append(p)
            phys = _vis[gpu_id] if gpu_id < len(_vis) else gpu_id
            print(f"[eval] started worker {gpu_id} (physical GPU {phys}) for tasks {assigned}",
                  flush=True)
        merged = {"per_task": {}, "n": {}, "oom": {}}
        import time as _time

        remaining = len(procs)
        poll = min(300.0, args.worker_timeout)
        deadline = _time.time() + args.worker_timeout
        while remaining > 0 and _time.time() < deadline:
            try:
                res = result_queue.get(timeout=poll)
            except Exception:
                dead = [p for p in procs if not p.is_alive()]
                crashed = len(dead) - (len(procs) - remaining)
                if crashed > 0:
                    print(f"[eval] {crashed} worker(s) died without finishing; "
                          "keeping their streamed results.", flush=True)
                    remaining -= crashed
                else:
                    print(f"[eval] still waiting on {remaining} worker(s)...", flush=True)
                continue
            deadline = _time.time() + args.worker_timeout
            if isinstance(res, dict) and res.get("type") == "worker_done":
                remaining -= 1
                continue
            for key in ("per_task", "n", "oom"):
                merged[key].update(res.get(key, {}) if isinstance(res, dict) else {})
        stragglers = []
        if remaining > 0:
            stragglers = [p for p in procs if p.is_alive()]
            for p in stragglers:
                p.terminate()
        for p in procs:
            p.join(timeout=10)
        _print_summary(merged, task_keys, out_json=out_json,
                       truncated=len(stragglers))
        if stragglers:
            print(f"[eval] ERROR: no worker produced a result for "
                  f"{args.worker_timeout:.0f}s and {len(stragglers)} worker(s) were "
                  "still running — the sweep was TRUNCATED. The rates above cover "
                  "only the finished tasks; do not report them as a full run.",
                  flush=True)
        sys.stdout.flush()
        os._exit(2 if stragglers else 0)


if __name__ == "__main__":
    main()
