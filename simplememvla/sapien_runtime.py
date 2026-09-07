
from __future__ import annotations

import os

_PREPARED = False

_VULKAN_SONAME = "libvulkan.so.1"

_EXTRA_LIB_DIRS = (
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
)


def _conda_lib_dir() -> str | None:
    prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if not prefix:
        import sys

        prefix = getattr(sys, "prefix", "")
    if prefix:
        lib = os.path.join(prefix, "lib")
        if os.path.isdir(lib):
            return lib
    return None


def _libvulkan_search_dirs() -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        path = path.strip()
        if path and path not in seen:
            seen.add(path)
            dirs.append(path)

    for part in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        _add(part)
    conda_lib = _conda_lib_dir()
    if conda_lib:
        _add(conda_lib)
    for path in _EXTRA_LIB_DIRS:
        _add(path)
    return dirs


def _find_libvulkan_dir() -> str | None:
    for path in _libvulkan_search_dirs():
        if os.path.isfile(os.path.join(path, _VULKAN_SONAME)):
            return path
    return None


def _prepend_ld_library_path(path: str) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in existing.split(":") if p]
    if path in parts:
        return
    os.environ["LD_LIBRARY_PATH"] = f"{path}:{existing}" if existing else path


def prepare_sapien_runtime() -> None:
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True

    os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)

    vulkan_dir = _find_libvulkan_dir()
    if vulkan_dir is not None:
        _prepend_ld_library_path(vulkan_dir)
