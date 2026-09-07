
from __future__ import annotations

import os
from pathlib import Path

_PREPARED = False


def _ensure_libero_config() -> None:
    config_dir = Path(
        os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    )
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_root = repo_root / "third_party" / "LIBERO" / "libero" / "libero"
    if not benchmark_root.exists():
        return
    import yaml

    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        yaml.dump(
            {
                "benchmark_root": str(benchmark_root),
                "bddl_files": str(benchmark_root / "bddl_files"),
                "init_states": str(benchmark_root / "init_files"),
                "datasets": str(benchmark_root.parent / "datasets"),
                "assets": str(benchmark_root / "assets"),
            }
        )
    )


def prepare_mujoco_runtime() -> None:
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("ROBOSUITE_MACROS_SILENCE", "1")
    os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)
    _ensure_libero_config()
