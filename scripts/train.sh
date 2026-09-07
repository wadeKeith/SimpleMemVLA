#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

BENCHMARK="${1:-${BENCHMARK:-}}"
if [[ -z "$BENCHMARK" ]]; then
  echo "usage: bash scripts/train.sh <rmbench|robomme|mikasa|robomemarena|libero>" >&2
  exit 1
fi

case "$BENCHMARK" in
  rmbench)      DEF_BATCH=8;  DEF_ACCUM=1; DEF_STEPS=60000 ;;
  robomme)      DEF_BATCH=8;  DEF_ACCUM=1; DEF_STEPS=60000 ;;
  mikasa)       DEF_BATCH=4;  DEF_ACCUM=4; DEF_STEPS=30000 ;;
  robomemarena) DEF_BATCH=8;  DEF_ACCUM=1; DEF_STEPS=30000 ;;
  libero)       DEF_BATCH=48; DEF_ACCUM=1; DEF_STEPS=30000 ;;
  *) echo "unknown benchmark: $BENCHMARK" >&2; exit 1 ;;
esac

BACKBONE="${BACKBONE:-Qwen/Qwen3.5-4B}"
RUN_NAME="${RUN_NAME:-${BENCHMARK}_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/sft/simplememvla/${RUN_NAME}}"

REPO_ID="${REPO_ID:-}"
ROOT_DIR="${ROOT_DIR:-}"
HISTORY_VIDEO_SEC="${HISTORY_VIDEO_SEC:-}"
HISTORY_VIDEO_FPS="${HISTORY_VIDEO_FPS:-}"
HISTORY_IMAGE_KEYS="${HISTORY_IMAGE_KEYS:-}"
VARIABLE_HISTORY="${VARIABLE_HISTORY:-}"
IMAGE_AUG="${IMAGE_AUG:-}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-}"

USE_PROPRIO="${USE_PROPRIO:-True}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
export WANDB_PROJECT="${WANDB_PROJECT:-simplememvla}"
export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "$OUTPUT_DIR"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6001}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./configs/zero2.json}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-$DEF_BATCH}"
GRAD_ACCUM="${GRAD_ACCUM:-$DEF_ACCUM}"

MAX_STEPS="${MAX_STEPS:-$DEF_STEPS}"
SAVE_STEPS="${SAVE_STEPS:-3000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
BACKBONE_LR="${BACKBONE_LR:-5e-6}"
ACTION_HEAD_LR="${ACTION_HEAD_LR:-5e-5}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-32}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

RESUME="${RESUME:-false}"

EXTRA_ARGS=()
[[ -n "$REPO_ID" ]]           && EXTRA_ARGS+=(--repo_id "$REPO_ID")
[[ -n "$ROOT_DIR" ]]          && EXTRA_ARGS+=(--root "$ROOT_DIR")
[[ -n "$HISTORY_VIDEO_SEC" ]] && EXTRA_ARGS+=(--history_video_sec "$HISTORY_VIDEO_SEC")
[[ -n "$HISTORY_VIDEO_FPS" ]] && EXTRA_ARGS+=(--history_video_fps "$HISTORY_VIDEO_FPS")
[[ -n "$VARIABLE_HISTORY" ]]  && EXTRA_ARGS+=(--variable_history "$VARIABLE_HISTORY")
[[ -n "$IMAGE_AUG" ]]         && EXTRA_ARGS+=(--image_aug "$IMAGE_AUG")
[[ -n "${SUBTASK_INDEX_KEY:-}" ]]       && EXTRA_ARGS+=(--subtask_index_key "$SUBTASK_INDEX_KEY")
[[ -n "${SKIP_VIDEO_DEMO_FRAMES:-}" ]]  && EXTRA_ARGS+=(--skip_video_demo_frames "$SKIP_VIDEO_DEMO_FRAMES")
[[ -n "$CONTROL_FREQUENCY" ]] && EXTRA_ARGS+=(--control_frequency_hz "$CONTROL_FREQUENCY")
if [[ -n "$HISTORY_IMAGE_KEYS" ]]; then
  EXTRA_ARGS+=(--history_image_keys $HISTORY_IMAGE_KEYS)
fi

torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  train.py \
  --benchmark "$BENCHMARK" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --backbone_model_name_or_path "$BACKBONE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --model_max_length 8192 \
  --dit_hidden_size 2048 \
  --dit_depth 16 \
  --dit_num_heads 16 \
  --dit_mlp_ratio 3.5 \
  --timestep_beta_alpha 1.5 \
  --timestep_beta_beta 1.0 \
  --action_loss_weight 1.0 \
  --vl_loss_weight 1.0 \
  --use_proprio "$USE_PROPRIO" \
  --state_dropout_prob "$STATE_DROPOUT_PROB" \
  --run_name "$RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --warmup_steps "$WARMUP_STEPS" \
  --backbone_lr "$BACKBONE_LR" \
  --action_head_lr "$ACTION_HEAD_LR" \
  --weight_decay 1e-8 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --optim adamw_torch \
  --bf16 true \
  --tf32 true \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --report_to wandb \
  --logging_steps "$LOGGING_STEPS" \
  --log_level info \
  --seed 429 \
  --resume "$RESUME" \
  "${EXTRA_ARGS[@]}"
