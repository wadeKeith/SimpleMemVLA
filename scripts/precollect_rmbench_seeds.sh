#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-seen}"
NUM_SEEDS="${NUM_SEEDS:-100}"
BASE_SEED="${BASE_SEED:-100000}"
MAX_SCAN="${MAX_SCAN:-1000}"
NUM_GPUS="${NUM_GPUS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TASKS="${TASKS:-}"
OUT="${OUT:-}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUTF8="${PYTHONUTF8:-1}"

unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
PYBIN="${PYBIN:-python}"

"$PYBIN" -m rmbench_sim.precollect_seeds \
  --task_config "$TASK_CONFIG" \
  --instruction_type "$INSTRUCTION_TYPE" \
  --num_seeds "$NUM_SEEDS" \
  --base_seed "$BASE_SEED" \
  --max_scan "$MAX_SCAN" \
  --num_gpus "$NUM_GPUS" \
  ${TASKS:+--tasks "$TASKS"} \
  ${OUT:+--out "$OUT"} \
  "$@"
