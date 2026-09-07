#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
FLA_VERSION="${FLA_VERSION:-0.2.2}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.5.2}"

echo "[fast-path] python: $(command -v python)"

echo "[fast-path] checking prerequisites ..."
command -v nvcc >/dev/null 2>&1 || {
  echo "ERROR: nvcc not found. Install a CUDA toolkit matching torch's CUDA," >&2
  echo "       e.g. 'conda install -c nvidia cuda-toolkit=12.8'. The nvidia-*-cu12" >&2
  echo "       wheels that torch pulls in are runtime-only and cannot compile." >&2
  exit 1
}
python -c "import torch" >/dev/null 2>&1 || {
  echo "ERROR: torch not importable. Run 'pip install -r requirements.txt' first." >&2
  exit 1
}

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; c=torch.cuda.get_device_capability(0); print(f"{c[0]}.{c[1]}")' 2>/dev/null || true)"
fi
export TORCH_CUDA_ARCH_LIST
echo "[fast-path] TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST:-<auto/all>}'"

echo "[fast-path] (1/3) installing flash-attn==${FLASH_ATTN_VERSION} (prebuilt wheel) ..."
if python -c "
import sys, flash_attn, flash_attn_2_cuda  # noqa: F401
sys.exit(0 if flash_attn.__version__ == '${FLASH_ATTN_VERSION}' else 1)
" >/dev/null 2>&1; then
  echo "[fast-path]   flash-attn ${FLASH_ATTN_VERSION} already installed, skipping"
else
  if [ -z "${FLASH_ATTN_WHEEL_URL:-}" ]; then
    FLASH_ATTN_WHEEL_URL="$(python - "$FLASH_ATTN_VERSION" <<'PY'
import sys, torch
ver = sys.argv[1]
tmaj, tmin = torch.__version__.split("+")[0].split(".")[:2]
cu = "".join((torch.version.cuda or "12.1").split(".")[:2])  # e.g. "12.1" -> "121"; wheel uses "12"
cu_tag = "cu" + cu[:2]  # Dao-AILab tags torch2.4 cu12 wheels as "cu12...", not cu121
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
fn = (f"flash_attn-{ver}+{cu_tag}torch{tmaj}.{tmin}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{ver}/{fn}")
PY
)"
  fi
  echo "[fast-path]   wheel: ${FLASH_ATTN_WHEEL_URL}"
  pip install --no-build-isolation --no-deps --no-cache-dir "${FLASH_ATTN_WHEEL_URL}"
fi

echo "[fast-path] (2/3) installing flash-linear-attention==${FLA_VERSION} ..."
pip install --no-deps "flash-linear-attention==${FLA_VERSION}"

echo "[fast-path] (3/3) compiling causal-conv1d ${CAUSAL_CONV1D_VERSION} (can take several minutes) ..."
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation "causal-conv1d==${CAUSAL_CONV1D_VERSION}"

echo "[fast-path] verifying ..."
python - <<'PY'
from rmbench_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

import flash_attn  # noqa: F401
import flash_attn_2_cuda  # noqa: F401  -- the compiled ext; ImportError => wrong wheel
from transformers.utils import is_flash_attn_2_available
assert is_flash_attn_2_available(), "flash-attn installed but transformers cannot use it"
print(f"OK: flash-attn {flash_attn.__version__} (flash_attention_2 available)")

import transformers.models.qwen3_5.modeling_qwen3_5 as m

missing = [
    name
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    )
    if getattr(m, name) is None
]
assert m.is_fast_path_available and not missing, (
    f"FAST PATH NOT ENABLED (missing: {missing or 'unknown'})"
)
print("OK: Qwen3.5 fast path enabled (flash-linear-attention + causal-conv1d)")
PY

echo "[fast-path] done. Training/eval use flash_attention_2 + the fused gated-delta kernels."
