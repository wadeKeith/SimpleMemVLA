#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-./checkpoints/sft/simplememvla/robomme_baseline}"
ROOT_DIR="${ROOT_DIR:-./data/datasets/yinchenghust/robomme_lerobot}"
REPO_ID="${REPO_ID:-yinchenghust/robomme_lerobot}"
NUM_EPISODES="${NUM_EPISODES:-5}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-16}"
MAX_SUBTASK_TOKENS="${MAX_SUBTASK_TOKENS:-64}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYBIN="${PYBIN:-python}"

"$PYBIN" -m robomme_sim.openloop_eval \
  --pretrained_checkpoint "$CHECKPOINT" \
  --root "$ROOT_DIR" \
  --repo_id "$REPO_ID" \
  --num_episodes "$NUM_EPISODES" \
  --num_denoising_steps "$NUM_DENOISING_STEPS" \
  --temperature "$TEMPERATURE" \
  --num_open_loop_steps "$NUM_OPEN_LOOP_STEPS" \
  --max_subtask_tokens "$MAX_SUBTASK_TOKENS"
