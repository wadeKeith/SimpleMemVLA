#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CHECKPOINT="${CHECKPOINT:-checkpoints/simplememvla_rmbench}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-seen}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-100}"
GROUP_SIZE="${GROUP_SIZE:-4}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-16}"
CAPTURE_STRIDE="${CAPTURE_STRIDE:-1}"
PIPELINED="${PIPELINED:-0}"
DENSE_SUBSTEPS="${DENSE_SUBSTEPS:-15}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
MAX_SUBTASK_TOKENS="${MAX_SUBTASK_TOKENS:-64}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
GPU="${GPU:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
GPUS="${GPUS:-$GPU}"
TASKS="${TASKS:-}"
EXPERT_CHECK="${EXPERT_CHECK:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/logs/rmbench_sim/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval.log}"
VIDEO_DIR="${VIDEO_DIR-$RUN_DIR/videos}"
VIDEO_MAX_PER_TASK="${VIDEO_MAX_PER_TASK:-1}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXPERT_FLAG="--expert_check"
[ "$EXPERT_CHECK" = "0" ] && EXPERT_FLAG="--no_expert_check"

unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
PYBIN="${PYBIN:-python}"

mkdir -p "$RUN_DIR"
[ -n "$VIDEO_DIR" ] && mkdir -p "$VIDEO_DIR"
echo "[eval_rmbench_sim] run dir: $RUN_DIR"
echo "[eval_rmbench_sim] log    : $LOG_FILE"
echo "[eval_rmbench_sim] videos : ${VIDEO_DIR:-<disabled>} (max ${VIDEO_MAX_PER_TASK}/task)"

set +e
"$PYBIN" -m rmbench_sim.eval_success \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_config "$TASK_CONFIG" \
  --instruction_type "$INSTRUCTION_TYPE" \
  --seeds_per_task "$SEEDS_PER_TASK" \
  --group_size "$GROUP_SIZE" \
  --execute_horizon "$EXECUTE_HORIZON" \
  --capture_stride "$CAPTURE_STRIDE" \
  ${PIPELINED:+$([ "$PIPELINED" != 0 ] && echo --pipelined)} \
  --dense_substeps "$DENSE_SUBSTEPS" \
  --num_denoising_steps "$NUM_DENOISING_STEPS" \
  --eval_temperature "$EVAL_TEMPERATURE" \
  --max_subtask_tokens "$MAX_SUBTASK_TOKENS" \
  --compute_dtype "$COMPUTE_DTYPE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --num_gpus "$NUM_GPUS" \
  --log_file "$LOG_FILE" \
  ${VIDEO_DIR:+--video_dir "$VIDEO_DIR"} \
  --video_max_per_task "$VIDEO_MAX_PER_TASK" \
  $EXPERT_FLAG \
  ${TASKS:+--tasks $TASKS} \
  "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
echo "[eval_rmbench_sim] done (exit $status). Log: $LOG_FILE  Results: $RUN_DIR/results.json"
exit "$status"
