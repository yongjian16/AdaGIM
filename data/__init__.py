"""Dataset registry."""
from typing import Any, Dict

_REGISTRY: Dict[str, Any] = {}

def register(name: str):
    def _decorator(fn):
        if name in _REGISTRY:
            raise ValueError(f"Dataset already registered: {name}")
        _REGISTRY[name] = fn
        return fn
    return _decorator

def build(name: str, cfg: Dict[str, Any]):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)

def _lazy_import_all():
    try:
        from . import chain_parity  # noqa: F401
    except (ImportError, AttributeError, FileNotFoundError) as e:
        print(f"[data] chain_parity unavailable: {e}")
    try:
        from . import lrgb  # noqa: F401
    except (ImportError, AttributeError) as e:
        print(f"[data] LRGB unavailable: {e}")
    try:
        from . import zinc  # noqa: F401
    except (ImportError, AttributeError) as e:
        print(f"[data] ZINC unavailable: {e}")
    try:
        from . import ogb  # noqa: F401
    except (ImportError, AttributeError) as e:
        print(f"[data] OGB unavailable: {e}")

_lazy_import_all()

__all__ = ["build", "register"]
