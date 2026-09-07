# RoboMemArena Evaluation Benchmark

- Project: RoboMemArena evaluation benchmark
- Repo: RoboMemArena
- Type: evaluation package

## What this directory contains

This directory contains a model-agnostic evaluation benchmark for RoboMemArena tasks 1-26.
It is meant for users who already trained their own model and now want to evaluate it on the complete benchmark.

The public interface is one adapter contract. The benchmark creates the RoboMemArena/LIBERO environment, loads the BDDL task, resets the environment, collects the current observation and task prompt, passes them to the adapter, executes the returned action chunk, records videos, and computes CSR/TSR.

The adapter can wrap any model structure:

- a single policy that maps `obs + prompt` to actions
- a VLM planner plus a VLA policy
- a remote policy server
- a model that needs custom image, state, prompt, or action conversion

The evaluation loop does not need to know which structure is inside the adapter.

It focuses on:

- evaluation entry scripts
- benchmark task definitions and prompts
- adapter-based model integration
- reference VLM/VLA integration examples
- official CSR/TSR reporting for the 1-26 task benchmark

## Directory layout

```text
evaluation_benchmark/
  README.md
  docs/
    evaluate_your_model.md
    task_evaluation_code_guide.md
  scripts/
    policy_adapter.py
    example_policy_adapter_template.py
    eval_common.py
    run_all_tasks1_26.py
  reference_evaluation/
    README.md
  async_vlm26_reference/
    README.md
    eval_fullvlm26_async_vlm_vla.py
    run_fullvlm26_async_vlm_vla_csr_tsr.sh
    fullvlm_v2_26_memory_tasks.json
  bddl/
  libero_fork/
```

Some source files keep historical names for compatibility, but the public benchmark setting is the full 1-26 task evaluation.

## Quick start

For a focused guide on plugging in your own checkpoint or model, see [Evaluate Your Model on RoboMemArena](docs/evaluate_your_model.md).

1. Make sure your environment can import the local LIBERO fork and its dependencies.
   You typically need a working `mujoco + robosuite + OpenGL/EGL` environment before running actual evaluation.
2. Implement your own adapter by following `scripts/example_policy_adapter_template.py`.
3. Run the 1-26 task sweep:

```bash
cd evaluation_benchmark
python scripts/run_all_tasks1_26.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/ckpt"}' \
  --num-trials-per-task 51 \
  --seed 50 \
  --out-root outputs/tasks1_26_eval
```

The sweep uses the benchmark reference stage/goal checkers, so external model evaluation follows the same 1-26 scoring setting as the reference evaluation path.
This wrapper records every episode. It does not retry seeds and does not filter for non-zero stage scores.

## Adapter contract

Your adapter must return a numpy array with shape `[horizon, action_dim]`.
Each row is one action to send to the environment. The benchmark code will reuse up to `replan_steps` actions before querying the adapter again. The default is 10 steps unless overridden by the evaluation command.

Required methods:

- `reset()`
- `infer_actions(obs, prompt, resize_size)`

See `scripts/policy_adapter.py` and `scripts/example_policy_adapter_template.py`.

## Reference implementation

If you want to inspect how our VLM/VLA stack is connected to the same benchmark setting, see:

```text
evaluation_benchmark/async_vlm26_reference/
evaluation_benchmark/reference_evaluation/
```

These folders are examples of how to connect a planner/policy stack to RoboMemArena. They are not a separate benchmark definition. Internal helper files may keep historical names for compatibility with earlier experiments, but users should treat evaluation as one 1-26 task setting.

Metric names:

- `CSR`: final BDDL goal success rate
- `TSR`: all-stage success rate; an episode counts as successful only when every reference stage is completed
- `average_score_pct`: diagnostic partial stage/process completion score
