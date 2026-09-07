
from __future__ import annotations

_ALIASED_SUBMODULES = ("_utils", "placement_types")


def apply_fla_torch_compat() -> None:
    try:
        import torch.distributed._tensor as _priv
        import torch.distributed.tensor as _pub
    except Exception:
        return

    def _reexport(src) -> None:
        for name in dir(src):
            if not name.startswith("__") and not hasattr(_pub, name):
                setattr(_pub, name, getattr(src, name))

    _reexport(_priv)
    try:
        import torch.distributed._tensor.placement_types as _plt

        _reexport(_plt)
    except Exception:
        pass

    _alias_submodules(_pub)


def _alias_submodules(pub) -> None:
    import importlib
    import sys

    for name in _ALIASED_SUBMODULES:
        public_path = f"torch.distributed.tensor.{name}"
        try:
            importlib.import_module(public_path)
            continue
        except ImportError:
            pass
        try:
            module = importlib.import_module(f"torch.distributed._tensor.{name}")
        except ImportError:
            continue
        sys.modules[public_path] = module
        setattr(pub, name, module)
