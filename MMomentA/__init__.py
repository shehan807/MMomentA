"""MMomentA: Fast, transferable Multipole Moment-based charge Assignment.

A machine learning framework for learning multipole moment-based charge
fitting (MPFIT) for polarizable force fields, combining quantum chemistry
calculations with graph neural network architectures.
"""

from . import comparison
from . import utils

# Optional QM imports (require OpenFF/Psi4)
try:
    from . import qm
    _HAS_QM = True
except ImportError:
    _HAS_QM = False

# Optional ML imports (require PyTorch/DGL)
try:
    from . import data
    from . import models
    from . import training
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

# Build __all__ based on what's available
__all__ = ["comparison", "utils"]
if _HAS_QM:
    __all__.append("qm")
if _HAS_ML:
    __all__.extend(["data", "models", "training"])

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
