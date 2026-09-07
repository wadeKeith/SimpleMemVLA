#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${PYBIN:-python}"

MANISKILL_REPO="${MANISKILL_REPO:-https://github.com/YinpeiDai/ManiSkill.git}"
MANISKILL_COMMIT="${MANISKILL_COMMIT:-07be6fbc}"
MANISKILL_DIR="$REPO_ROOT/third_party/ManiSkill"

if [ ! -d "$MANISKILL_DIR/mani_skill" ]; then
  echo "[install_robomme_sim] cloning $MANISKILL_REPO @ $MANISKILL_COMMIT"
  mkdir -p "$REPO_ROOT/third_party"
  git clone "$MANISKILL_REPO" "$MANISKILL_DIR"
  git -C "$MANISKILL_DIR" checkout "$MANISKILL_COMMIT"
else
  echo "[install_robomme_sim] third_party/ManiSkill already present, skipping clone"
fi

echo "[install_robomme_sim] python: $PYBIN"
"$PYBIN" -m pip install -e "$MANISKILL_DIR"

"$PYBIN" - <<'EOF'
import mani_skill, sapien
print(f"[install_robomme_sim] OK: mani_skill {getattr(mani_skill, '__version__', '?')}, "
      f"sapien {sapien.__version__}")
EOF
