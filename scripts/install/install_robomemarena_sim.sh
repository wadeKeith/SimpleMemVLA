#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${1:-${CONDA_DEFAULT_ENV:-simplememvla_robomemarena}}"
UPSTREAM="${UPSTREAM:-https://github.com/OpenHelix-Team/RoboMemArena.git}"
FORK_DIR="$REPO_ROOT/evaluation_benchmark/libero_fork"

echo "==> conda env: $ENV_NAME"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  echo "    (activate it first: conda activate $ENV_NAME)"
fi

echo "==> simulator + benchmark python deps"
pip install --no-input \
  "robosuite==1.4.0" \
  "mujoco==2.3.2" \
  "bddl==1.0.1" \
  "gym==0.25.2" \
  "imageio[ffmpeg]" \
  "opencv-python" \
  "scipy" \
  "pyyaml" \
  "easydict" \
  "termcolor" \
  "thop" \
  "cloudpickle"

echo "==> vendored LIBERO fork"
if [[ -d "$FORK_DIR/libero/assets" ]]; then
  echo "    already present at $FORK_DIR"
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "    cloning $UPSTREAM (shallow) -> $TMP"
  git clone --depth 1 "$UPSTREAM" "$TMP/RoboMemArena"
  mkdir -p "$FORK_DIR"
  rsync -a "$TMP/RoboMemArena/evaluation_benchmark/libero_fork/" "$FORK_DIR/"
  for d in bddl scripts docs tests; do
    if [[ ! -d "$REPO_ROOT/evaluation_benchmark/$d" ]]; then
      rsync -a "$TMP/RoboMemArena/evaluation_benchmark/$d/" "$REPO_ROOT/evaluation_benchmark/$d/"
    fi
  done
  echo "    installed fork into $FORK_DIR"
fi

echo "==> smoke check (imports the fork, builds one env, renders one frame)"
PYTHONPATH="$REPO_ROOT" "${PYBIN:-python}" - <<'PY'
from pathlib import Path

from robomemarena_sim import assert_libero_fork, prepare_mujoco_runtime

prepare_mujoco_runtime()
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

assert_libero_fork()
import libero

print("    libero fork:", libero.__path__[0])
bddl = sorted(Path("evaluation_benchmark/bddl").glob("1_*.bddl"))[0]
env = OffScreenRenderEnv(
    bddl_file_name=str(bddl), camera_heights=480, camera_widths=640,
    ignore_done=True, reward_shaping=True, control_freq=20, initialization_noise=None,
)
np.random.seed(0)
obs = env.reset()
print("    rendered:", {k: np.asarray(v).shape for k, v in obs.items() if "image" in k})
env.close()
print("    OK")
PY

echo
echo "Done. Next:"
echo "  bash scripts/install/install_fast_path.sh          # flash-attn + the Qwen3.5 gated-delta kernels"
echo "  bash scripts/eval_robomemarena.sh <checkpoint>      # closed-loop CSR/TSR"
