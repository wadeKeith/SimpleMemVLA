#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${PYBIN:-python}"

echo "[install] python: $PYBIN ($("$PYBIN" --version 2>&1))"

if [[ ! -d "$REPO_ROOT/third_party/LIBERO" ]]; then
  echo "[install] cloning LIBERO -> third_party/LIBERO"
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    "$REPO_ROOT/third_party/LIBERO"
fi

echo "[install] pinning simulator stack (robosuite 1.4.0 / mujoco 2.3.2 / bddl 1.0.1)"
"$PYBIN" -m pip install \
  "robosuite==1.4.0" \
  "mujoco==2.3.2" \
  "bddl==1.0.1" \
  "easydict>=1.9" \
  "future>=0.18" \
  "gym==0.25.2" \
  "cloudpickle>=2.1" \
  "hydra-core>=1.2" \
  "thop" \
  "imageio[ffmpeg]"

echo "[install] exposing third_party/LIBERO as a namespace package (.pth)"
"$PYBIN" -m pip uninstall -y libero >/dev/null 2>&1 || true
SITE_DIR="$("$PYBIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_ROOT/third_party/LIBERO" > "$SITE_DIR/zz_libero_repo.pth"
echo "[install] wrote $SITE_DIR/zz_libero_repo.pth"

echo "[install] self-check: benchmark + offscreen render"
MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONPATH="$REPO_ROOT" "$PYBIN" - <<'EOF'
import os

from libero_sim._mujoco_env import prepare_mujoco_runtime

prepare_mujoco_runtime()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()
suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
for s in suites:
    suite = bench[s]()
    assert len(suite.tasks) == 10, (s, len(suite.tasks))
print("benchmark OK:", {s: len(bench[s]().tasks) for s in suites})

suite = bench["libero_10"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.reset()
obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
img = obs["agentview_image"]
assert img.shape == (256, 256, 3), img.shape
print("offscreen render OK:", img.shape, img.dtype, "task:", task.language)
env.close()
EOF

echo "[install] DONE. If the self-check failed on EGL, retry with MUJOCO_GL=osmesa."
