#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-./checkpoints/sft/simplememvla/libero_baseline}"
TASK_SUITES="${TASK_SUITES:-}"
TASKS="${TASKS:-}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
GROUP_SIZE="${GROUP_SIZE:-10}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-8}"
PIPELINED="${PIPELINED:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
MAX_REASONING_TOKENS="${MAX_REASONING_TOKENS:-256}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

GPU="${GPU:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
GPUS="${GPUS:-$GPU}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/logs/libero_sim/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval.log}"
VIDEO_DIR="${VIDEO_DIR:-$RUN_DIR/videos}"
VIDEO_MAX_PER_TASK="${VIDEO_MAX_PER_TASK:-1}"

PYBIN="${PYBIN:-python}"
if ! "$PYBIN" -c "import torch, transformers, robosuite, libero" >/dev/null 2>&1; then
  echo "PYBIN=$PYBIN cannot import torch/transformers/robosuite/libero." >&2
  echo "Activate the LIBERO eval env first, or pass PYBIN=/path/to/env/bin/python" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset __EGL_VENDOR_LIBRARY_DIRS
if [[ "$NUM_GPUS" == "1" && "$GPUS" != *,* && "$MUJOCO_GL" == "egl" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPUS}"
fi

mkdir -p "$RUN_DIR"

EXTRA_ARGS=()
if [[ -n "$TASKS" ]]; then
  EXTRA_ARGS+=(--tasks $TASKS)
elif [[ -n "$TASK_SUITES" ]]; then
  EXTRA_ARGS+=(--task_suites $TASK_SUITES)
fi
if [[ -n "$VIDEO_DIR" ]]; then
  EXTRA_ARGS+=(--video_dir "$VIDEO_DIR" --video_max_per_task "$VIDEO_MAX_PER_TASK")
fi
if [[ "$PIPELINED" != "0" ]]; then
  EXTRA_ARGS+=(--pipelined)
fi

cd "$REPO_ROOT"
set +e
"$PYBIN" -m libero_sim.eval_success \
  --pretrained_checkpoint "$CHECKPOINT" \
  --episodes_per_task "$EPISODES_PER_TASK" \
  --group_size "$GROUP_SIZE" \
  --execute_horizon "$EXECUTE_HORIZON" \
  --max_steps "$MAX_STEPS" \
  --num_steps_wait "$NUM_STEPS_WAIT" \
  --num_denoising_steps "$NUM_DENOISING_STEPS" \
  --eval_temperature "$EVAL_TEMPERATURE" \
  --max_reasoning_tokens "$MAX_REASONING_TOKENS" \
  --compute_dtype "$COMPUTE_DTYPE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --num_gpus "$NUM_GPUS" \
  --log_file "$LOG_FILE" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
exit "$status"
