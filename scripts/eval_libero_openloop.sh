#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-./checkpoints/sft/simplememvla/libero_baseline}"
ROOT_DIR="${ROOT_DIR:-./data/datasets/yinchenghust/libero_lerobot}"
REPO_ID="${REPO_ID:-yinchenghust/libero_lerobot}"
NUM_EPISODES="${NUM_EPISODES:-5}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
MAX_REASONING_TOKENS="${MAX_REASONING_TOKENS:-256}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYBIN="${PYBIN:-python}"
if ! "$PYBIN" -c "import torch, transformers" >/dev/null 2>&1; then
  echo "PYBIN=$PYBIN cannot import torch/transformers." >&2
  echo "Activate the simplememvla_libero env or pass PYBIN=/path/to/envs/simplememvla_libero/bin/python" >&2
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$PYBIN" -m libero_sim.openloop_eval \
  --pretrained_checkpoint "$CHECKPOINT" \
  --root "$ROOT_DIR" \
  --repo_id "$REPO_ID" \
  --num_episodes "$NUM_EPISODES" \
  --num_denoising_steps "$NUM_DENOISING_STEPS" \
  --temperature "$TEMPERATURE" \
  --num_open_loop_steps "$NUM_OPEN_LOOP_STEPS" \
  --max_reasoning_tokens "$MAX_REASONING_TOKENS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  "$@"
