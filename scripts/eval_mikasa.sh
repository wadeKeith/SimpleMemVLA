#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CHECKPOINT="${CHECKPOINT:-./checkpoints/sft/simplememvla/mikasa_baseline}"
N_EPISODES="${N_EPISODES:-100}"
START_SEED="${START_SEED:-}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-8}"
DECISION_PATH="${DECISION_PATH:-pipelined}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
MAX_SUBTASK_TOKENS="${MAX_SUBTASK_TOKENS:-64}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bfloat16}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
JOINT_DELTA_LIMIT="${JOINT_DELTA_LIMIT:-}"
if [ -n "$JOINT_DELTA_LIMIT" ]; then
  export MIKASA_JOINT_DELTA_LIMIT="$JOINT_DELTA_LIMIT"
fi
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
GPU="${GPU:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
GPUS="${GPUS:-$GPU}"
TASKS="${TASKS:-}"
SAVE_VIDEOS="${SAVE_VIDEOS:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/logs/mikasa_sim/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval.log}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -z "${TRITON_CACHE_DIR:-}" ]; then
  TRITON_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/triton"
fi
if [ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
  TORCHINDUCTOR_CACHE_DIR="$(dirname "$TRITON_CACHE_DIR")/torchinductor"
fi
export TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
PYBIN="${PYBIN:-python}"

mkdir -p "$RUN_DIR"
echo "[eval_mikasa_sim] run dir: $RUN_DIR"
echo "[eval_mikasa_sim] log    : $LOG_FILE"

set +e
"$PYBIN" -m mikasa_sim.eval_success \
  --pretrained_checkpoint "$CHECKPOINT" \
  --n_episodes "$N_EPISODES" \
  --execute_horizon "$EXECUTE_HORIZON" \
  --decision_path "$DECISION_PATH" \
  --num_denoising_steps "$NUM_DENOISING_STEPS" \
  --eval_temperature "$EVAL_TEMPERATURE" \
  --max_subtask_tokens "$MAX_SUBTASK_TOKENS" \
  --compute_dtype "$COMPUTE_DTYPE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --num_gpus "$NUM_GPUS" \
  --output_dir "$RUN_DIR" \
  --log_file "$LOG_FILE" \
  ${START_SEED:+--start_seed "$START_SEED"} \
  ${CONTROL_MODE:+--control_mode "$CONTROL_MODE"} \
  ${SAVE_VIDEOS:+--save_videos} \
  ${TASKS:+--tasks $TASKS} \
  "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
echo "[eval_mikasa_sim] done (exit $status). Log: $LOG_FILE  Results: $RUN_DIR/summary.json"
exit "$status"
