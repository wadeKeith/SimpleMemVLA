#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${1:?usage: bash eval_robomemarena.sh <checkpoint_dir> [out_dir]}"
OUT="${2:-logs/eval_$(basename "$CKPT")_$(date +%Y%m%d_%H%M%S)}"

TRIALS="${TRIALS:-51}"
TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-26}"
SEED="${SEED:-50}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
MAX_STEPS="${MAX_STEPS:-2500}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
MAX_REASONING_TOKENS="${MAX_REASONING_TOKENS:-256}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-0}"
LOG_EVERY="${LOG_EVERY:-0}"
PYBIN="${PYBIN:-python}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

export ATTN_IMPLEMENTATION
if ! "$PYBIN" - <<'PREFLIGHT'
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
fork = pathlib.Path("evaluation_benchmark/libero_fork/libero").resolve()
if not fork.is_dir():
    sys.exit(f"vendored LIBERO fork missing at {fork}; run bash scripts/install/install_robomemarena_sim.sh")

from robomemarena_sim._compat import apply_fla_torch_compat
apply_fla_torch_compat()

import transformers
if transformers.__version__ != "5.13.1":
    sys.exit(f"transformers {transformers.__version__} != 5.13.1, the version this "
             f"checkpoint was trained and saved under ({sys.executable}). The rendered "
             "prompt is version-dependent. Run: conda activate simplememvla_robomemarena")

if os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2") == "flash_attention_2":
    try:
        import flash_attn
    except Exception as exc:
        sys.exit(f"flash-attn is not importable ({exc}); build it with "
                 "bash scripts/install/install_fast_path.sh, or pass ATTN_IMPLEMENTATION=sdpa to accept a "
                 "different attention backend than training used")

from robomemarena_sim import prepare_mujoco_runtime
prepare_mujoco_runtime()
import libero
where = pathlib.Path(libero.__path__[0]).resolve()
if where != fork:
    sys.exit(f"`import libero` resolves to {where}, not the vendored fork {fork}; upstream "
             "LIBERO lacks this benchmark's objects. An editable install is shadowing it.")
print(f"preflight ok: {sys.executable} (transformers {transformers.__version__})")
PREFLIGHT
then
  echo "preflight FAILED -- not launching any workers" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "checkpoint : $CKPT"
echo "output     : $OUT"
echo "tasks      : $TASK_START..$TASK_END x $TRIALS trials (seed base $SEED)"
echo "gpus       : $GPUS x $WORKERS_PER_GPU worker(s)"

"$PYBIN" -u -m robomemarena_sim.eval_success \
  --checkpoint "$CKPT" \
  --out-root "$OUT" \
  --task-start "$TASK_START" \
  --task-end "$TASK_END" \
  --num-trials-per-task "$TRIALS" \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --replan-steps "$REPLAN_STEPS" \
  --num-steps-wait "$NUM_STEPS_WAIT" \
  --num-denoising-steps "$NUM_DENOISING_STEPS" \
  --max-reasoning-tokens "$MAX_REASONING_TOKENS" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  --gpus "$GPUS" \
  --workers-per-gpu "$WORKERS_PER_GPU" \
  --episodes-per-shard "$EPISODES_PER_SHARD" \
  --log-every "$LOG_EVERY" \
  $( [[ "${SEED_POLICY:-1}" == "0" ]] && echo --no-seed-policy ) \
  $( [[ "${RESUME:-1}" == "0" ]] && echo --no-resume ) \
  ${GPU_MEMORY_CAP_GIB:+--gpu-memory-cap-gib "$GPU_MEMORY_CAP_GIB"} \
  2>&1 | tee -a "$OUT/eval.log"
