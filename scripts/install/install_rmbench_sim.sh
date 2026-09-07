#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${PYBIN:-python}"
DATASET_ID="${RMBENCH_SIM_DATASET:-keithyc/RMBench_sim}"
TARGET="$REPO_ROOT/rmbench_sim/rmbench"

echo "[install_rmbench_sim] installing the SAPIEN sim python deps"
"$PYBIN" -m pip install -r "$REPO_ROOT/requirements/rmbench_sim.txt"

if [ -d "$TARGET/assets" ] && [ -n "$(ls -A "$TARGET/assets" 2>/dev/null)" ]; then
  echo "[install_rmbench_sim] assets already present: $TARGET/assets"
else
  echo "[install_rmbench_sim] downloading $DATASET_ID -> $TARGET (~1.5 GB)"
  "$PYBIN" -m pip show modelscope >/dev/null 2>&1 || "$PYBIN" -m pip install modelscope
  MS_CLI="$(dirname "$(command -v "$PYBIN")")/modelscope"
  [ -x "$MS_CLI" ] || MS_CLI="modelscope"
  "$MS_CLI" download --dataset "$DATASET_ID" --local_dir "$TARGET"
fi

"$PYBIN" - <<EOF
from pathlib import Path
root = Path("$TARGET")
missing = [p for p in ("assets", "envs/curobo/src") if not (root / p).is_dir()]
if missing:
    raise SystemExit(f"[install_rmbench_sim] MISSING after download: {missing}")
print("[install_rmbench_sim] OK: assets + curobo in place")
EOF
