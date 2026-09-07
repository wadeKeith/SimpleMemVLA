
from __future__ import annotations

import os
import sys
from pathlib import Path

_PREPARED = False

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_BENCH = REPO_ROOT / "evaluation_benchmark"
LIBERO_FORK_ROOT = EVAL_BENCH / "libero_fork"
LIBERO_FORK = LIBERO_FORK_ROOT / "libero"
EVAL_SCRIPTS = EVAL_BENCH / "scripts"
DEFAULT_CONFIG_DIR = REPO_ROOT / ".libero_robomemarena"


def ensure_libero_fork_on_path() -> None:
    for path in (str(LIBERO_FORK_ROOT), str(EVAL_SCRIPTS)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(EVAL_SCRIPTS))
    sys.path.insert(0, str(LIBERO_FORK_ROOT))


def assert_libero_fork() -> None:
    libero = sys.modules.get("libero")
    if libero is None:
        return
    where = Path(libero.__path__[0])
    if LIBERO_FORK.resolve() != where.resolve():
        raise RuntimeError(
            f"The imported `libero` package is {where}, not RoboMemArena's fork at "
            f"{LIBERO_FORK}. Upstream LIBERO lacks this benchmark's objects "
            "(wooden_cabinet, wine_bottle, microwave, ...). Call "
            "robomemarena_sim.prepare_mujoco_runtime() BEFORE importing libero."
        )


def _ensure_libero_config() -> None:
    config_dir = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", str(DEFAULT_CONFIG_DIR)))
    config_file = config_dir / "config.yaml"
    benchmark_root = LIBERO_FORK / "libero"
    want = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    import yaml

    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.dump(want))


def prepare_mujoco_runtime() -> None:
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("ROBOSUITE_MACROS_SILENCE", "1")
    os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)
    ensure_libero_fork_on_path()
    _ensure_libero_config()
    assert_libero_fork()
