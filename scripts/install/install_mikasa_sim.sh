#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${PYBIN:-python}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple/}"

echo "[install_mikasa_sim] python: $PYBIN"

"$PYBIN" -m pip install -i "$PIP_INDEX" "setuptools<81"
"$PYBIN" -m pip install --no-deps -i "$PIP_INDEX" "sapien==3.0.0.b1" "mani_skill==3.0.0b15"
"$PYBIN" -m pip install --no-deps -i "$PIP_INDEX" colorama tensorboard
"$PYBIN" -m pip install -i "$PIP_INDEX" ipython
"$PYBIN" -m pip install --no-deps -i "$PIP_INDEX" \
  gymnasium dacite transforms3d trimesh tabulate GitPython gitdb smmap mplib toppra
"$PYBIN" -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT/third_party/MIKASA-Robo"

YCB_ROOT="${MS_ASSET_DIR:-$HOME/.maniskill}/data/assets"
if [ -d "$YCB_ROOT/mani_skill2_ycb/models" ] && [ -n "$(ls -A "$YCB_ROOT/mani_skill2_ycb/models" 2>/dev/null)" ]; then
  echo "[install_mikasa_sim] YCB assets already present: $YCB_ROOT/mani_skill2_ycb"
else
  YCB_URL="${HF_ENDPOINT:-https://huggingface.co}/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip"
  YCB_ZIP="$(mktemp -t mani_skill2_ycb.XXXXXX.zip)"
  echo "[install_mikasa_sim] downloading YCB assets from $YCB_URL"
  mkdir -p "$YCB_ROOT"
  curl -L --fail --retry 5 --retry-delay 2 -o "$YCB_ZIP" "$YCB_URL"
  "$PYBIN" - "$YCB_ZIP" "$YCB_ROOT" <<'PY'
import sys, zipfile
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    bad = z.testzip()
    if bad is not None:
        raise SystemExit(f"[install_mikasa_sim] corrupt YCB zip entry: {bad}")
    z.extractall(dest)
print(f"[install_mikasa_sim] extracted YCB -> {dest}/mani_skill2_ycb")
PY
  rm -f "$YCB_ZIP"
fi

"$PYBIN" - <<'EOF'
import mani_skill, sapien, mikasa_robo_suite
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
import inspect
src = inspect.getsource(PandaWristCam._sensor_configs.fget)
assert "width=128" in src and "height=128" in src, (
    "panda_wristcam hand_camera is not 128x128 — wrong mani_skill version for the "
    "MIKASA dataset (need 3.0.0b15)."
)
print(f"[install_mikasa_sim] OK: mani_skill {getattr(mani_skill, '__version__', '?')}, "
      f"sapien {sapien.__version__}, mikasa_robo_suite importable, wrist cam 128x128")
EOF
