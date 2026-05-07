"""Model registry."""
from typing import Any, Dict

_REGISTRY: Dict[str, Any] = {}

def register(name: str):
    def _decorator(fn):
        if name in _REGISTRY:
            raise ValueError(f"Model already registered: {name}")
        _REGISTRY[name] = fn
        return fn
    return _decorator

def build(name: str, cfg: Dict[str, Any], dataset_meta: Dict[str, Any]):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, dataset_meta)

def _lazy_import_all() -> None:
    from . import ignn               # noqa: F401
    from . import eignn              # noqa: F401
    from . import ignn_finite_loop   # noqa: F401
    from . import ignn_finite_chain  # noqa: F401
    from . import ignn_finite_act    # noqa: F401
    from . import adagim             # noqa: F401
    from . import gcn                # noqa: F401
    from . import gcnii              # noqa: F401
    from . import appnp              # noqa: F401
    for mod in ("gind", "mgnni", "monotone_mignn"):
        try:
            __import__(f"models.{mod}")
        except Exception as e:  # noqa: BLE001
            print(f"[models] {mod} unavailable: {type(e).__name__}: {e}")

_lazy_import_all()

__all__ = ["build", "register"]
