#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CHECKPOINT="${CHECKPOINT:-./checkpoints/sft/simplememvla/robomme_baseline}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
GROUP_SIZE="${GROUP_SIZE:-17}"
PIPELINED="${PIPELINED:-0}"
if [ "$PIPELINED" != 0 ]; then
  EXECUTE_HORIZON="${EXECUTE_HORIZON:-20}"
else
  EXECUTE_HORIZON="${EXECUTE_HORIZON:-16}"
fi
MAX_STEPS="${MAX_STEPS:-1300}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
MAX_SUBTASK_TOKENS="${MAX_SUBTASK_TOKENS:-64}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
GPU="${GPU:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
GPUS="${GPUS:-$GPU}"
TASKS="${TASKS:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/logs/robomme_sim/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval.log}"
VIDEO_DIR="${VIDEO_DIR-$RUN_DIR/videos}"
VIDEO_MAX_PER_TASK="${VIDEO_MAX_PER_TASK:-1}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/torchinductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
PYBIN="${PYBIN:-python}"

mkdir -p "$RUN_DIR"
[ -n "$VIDEO_DIR" ] && mkdir -p "$VIDEO_DIR"
echo "[eval_robomme_sim] run dir: $RUN_DIR"
echo "[eval_robomme_sim] log    : $LOG_FILE"
echo "[eval_robomme_sim] videos : ${VIDEO_DIR:-<disabled>} (max ${VIDEO_MAX_PER_TASK}/task)"

set +e
"$PYBIN" -m robomme_sim.eval_success \
  --pretrained_checkpoint "$CHECKPOINT" \
  --dataset_split "$DATASET_SPLIT" \
  --episodes_per_task "$EPISODES_PER_TASK" \
  --group_size "$GROUP_SIZE" \
  --execute_horizon "$EXECUTE_HORIZON" \
  ${PIPELINED:+$([ "$PIPELINED" != 0 ] && echo --pipelined)} \
  --max_steps "$MAX_STEPS" \
  --num_denoising_steps "$NUM_DENOISING_STEPS" \
  --eval_temperature "$EVAL_TEMPERATURE" \
  --max_subtask_tokens "$MAX_SUBTASK_TOKENS" \
  --compute_dtype "$COMPUTE_DTYPE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --num_gpus "$NUM_GPUS" \
  --log_file "$LOG_FILE" \
  ${VIDEO_DIR:+--video_dir "$VIDEO_DIR"} \
  --video_max_per_task "$VIDEO_MAX_PER_TASK" \
  ${TASKS:+--tasks $TASKS} \
  "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
echo "[eval_robomme_sim] done (exit $status). Log: $LOG_FILE  Results: $RUN_DIR/results.json"
exit "$status"
