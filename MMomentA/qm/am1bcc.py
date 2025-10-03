"""AM1-BCC charge calculation using OpenFF Toolkit."""

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from openff.toolkit.topology import Molecule
from openff.units import unit

from .settings import AM1BCCConfig


@dataclass
class AM1BCCResult:
    """Container for AM1-BCC calculation results.

    Attributes
    ----------
    charges : np.ndarray
        Atomic partial charges (n_atoms,)
    time_seconds : float
        Computation time in seconds
    smiles : str
        Molecule SMILES string
    metadata : dict
        Additional calculation metadata
    success : bool
        Whether calculation succeeded
    error_message : Optional[str]
        Error message if calculation failed
    """
    charges: np.ndarray
    time_seconds: float
    smiles: str
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class AM1BCCCalculator:
    """Calculator for AM1-BCC charges using OpenFF Toolkit.

    Parameters
    ----------
    config : AM1BCCConfig, optional
        Configuration for AM1-BCC calculations

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> calculator = AM1BCCCalculator()
    >>> molecule = Molecule.from_smiles("CCO")
    >>> result = calculator.compute(molecule)
    >>> print(result.charges)
    """

    def __init__(self, config: Optional[AM1BCCConfig] = None):
        self.config = config or AM1BCCConfig()

    def compute(self, molecule: Molecule) -> AM1BCCResult:
        """Compute AM1-BCC charges for a molecule.

        Parameters
        ----------
        molecule : Molecule
            OpenFF molecule

        Returns
        -------
        AM1BCCResult
            Calculation results including charges
        """
        start_time = time.time()
        smiles = molecule.to_smiles(mapped=False)

        errors = []
        method_used = None

        for method in self.config.fallback_methods:
            molecule.assign_partial_charges(method)
            charges = molecule.partial_charges.m_as(unit.elementary_charge)
            method_used = method
            break

        if method_used is None:
            elapsed = time.time() - start_time
            error_msg = "; ".join(errors)
            return AM1BCCResult(
                charges=np.zeros(molecule.n_atoms),
                time_seconds=elapsed,
                smiles=smiles,
                metadata=self._get_metadata(None),
                success=False,
                error_message=f"All charge methods failed: {error_msg}"
            )

        elapsed = time.time() - start_time

        return AM1BCCResult(
            charges=charges.flatten(),
            time_seconds=elapsed,
            smiles=smiles,
            metadata=self._get_metadata(method_used),
            success=True,
            error_message=None
        )

    def _get_metadata(self, method_used: Optional[str]) -> Dict[str, Any]:
        """Get metadata dictionary for calculation settings."""
        return {
            "charge_method": method_used,
            "fallback_methods": self.config.fallback_methods,
            "calculator": "AM1-BCC"
        }


def compute_am1bcc_charges(
    molecule: Molecule,
    config: Optional[AM1BCCConfig] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute AM1-BCC charges for a molecule (convenience function).

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule
    config : AM1BCCConfig, optional
        Configuration for AM1-BCC calculations

    Returns
    -------
    charges : np.ndarray
        Atomic partial charges (n_atoms,)
    metadata : dict
        Calculation metadata including timing and settings

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> molecule = Molecule.from_smiles("CCO")
    >>> charges, metadata = compute_am1bcc_charges(molecule)
    """
    calculator = AM1BCCCalculator(config)
    result = calculator.compute(molecule)

    if not result.success:
        raise RuntimeError(f"AM1-BCC calculation failed: {result.error_message}")

    return result.charges, result.metadata
