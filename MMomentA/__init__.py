"""MMomentA: Fast, transferable Multipole Moment-based charge Assignment.

A machine learning framework for learning multipole moment-based charge
fitting (MPFIT) for polarizable force fields, combining quantum chemistry
calculations with graph neural network architectures.
"""

from . import qm
from . import comparison
from . import utils

# Optional imports (require PyTorch/DGL)
try:
    from . import data
    from . import models
    from . import training
    __all__ = ["qm", "data", "comparison", "models", "training", "utils"]
except ImportError:
    # Data environment without PyTorch
    __all__ = ["qm", "comparison", "utils"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
