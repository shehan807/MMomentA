"""MMomentA: Fast, transferable Multipole Moment-based charge Assignment.

A machine learning framework for learning multipole moment-based charge
fitting (MPFIT) for polarizable force fields, combining quantum chemistry
calculations with graph neural network architectures.
"""

from . import qm
from . import data
from . import comparison
from . import models
from . import training
from . import utils

__all__ = ["qm", "data", "comparison", "models", "training", "utils"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
