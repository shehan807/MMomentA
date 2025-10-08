"""Configuration classes for quantum chemistry calculations."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QMSettings:
    """Base quantum chemistry settings.

    Parameters
    ----------
    method : str
        Quantum chemistry method (e.g., 'hf', 'pbe0', 'b3lyp')
    basis : str
        Basis set (e.g., '6-31G*', 'def2-SVP')
    minimize : bool
        Whether to minimize geometry before calculations
    """
    method: str = "hf"
    basis: str = "6-31G*"
    minimize: bool = True


@dataclass
class GDMAConfig(QMSettings):
    """GDMA multipole analysis configuration.

    Parameters
    ----------
    limit : int
        Multipole expansion limit (rank)
    switch : float
        Switching distance for multipole expansion (Angstrom)
    svd_threshold : float
        SVD cutoff threshold for MPFIT solver
    """
    limit: int = 8
    switch: float = 0.0
    svd_threshold: float = 1.0e-4


@dataclass
class ESPConfig(QMSettings):
    """Electrostatic potential calculation configuration.

    Parameters
    ----------
    restraint_weight : float
        RESP restraint weight
    grid_density : str
        Grid density setting ('coarse', 'medium', 'fine')
    """
    restraint_weight: float = 0.0005
    grid_density: str = "medium"


@dataclass
class AM1BCCConfig:
    """AM1-BCC configuration (no QM calculations needed).

    Parameters
    ----------
    fallback_methods : list of str
        Fallback charge methods if AM1-BCC fails
    """
    fallback_methods: list = field(default_factory=lambda: ["am1bcc", "mmff94", "gasteiger"])
