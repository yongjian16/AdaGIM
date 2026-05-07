"""Bridge to the bundled chain-parity / synthetic-mode helper code."""
import importlib.util
import sys
from pathlib import Path

_HELPER = Path(__file__).resolve().parents[1] / "helper"

if str(_HELPER) not in sys.path:
    sys.path.append(str(_HELPER))

_KEY = "_helper_models"
if _KEY not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_KEY, str(_HELPER / "models.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_KEY] = _mod
    _spec.loader.exec_module(_mod)
_existing = sys.modules[_KEY]

ParityIGNN = _existing.ParityIGNN
ParityEIGNN = _existing.ParityEIGNN
ParityGCN = _existing.ParityGCN
ParityDiagUnrollIGNN = _existing.ParityDiagUnrollIGNN

from ignn_utils.layers import ImplicitGraph as _ImplicitGraph  # noqa: E402
from ignn_utils.utils import projection_norm_inf as _projection_norm_inf  # noqa: E402
ImplicitGraph = _ImplicitGraph
projection_norm_inf = _projection_norm_inf
