"""MPFIT charge calculation using GDMA multipole moments."""

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from openff.toolkit.topology import Molecule
from openff.recharge.charges.library import LibraryChargeCollection, LibraryChargeGenerator
from openff.recharge.charges.mpfit import generate_mpfit_charge_parameter
from openff.recharge.charges.mpfit.solvers import MPFITSVDSolver
from openff.recharge.gdma import GDMASettings
from openff.recharge.gdma.psi4 import Psi4GDMAGenerator
from openff.recharge.gdma.storage import MoleculeGDMARecord
from openff.recharge.utilities.molecule import extract_conformers

from .settings import GDMAConfig


@dataclass
class MPFITResult:
    """Container for MPFIT calculation results.

    Attributes
    ----------
    charges : np.ndarray
        Atomic partial charges (n_atoms,)
    multipole_moments : np.ndarray
        GDMA multipole moments (n_atoms, n_moments)
    conformer : np.ndarray
        Optimized conformer coordinates (n_atoms, 3)
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
    multipole_moments: Optional[np.ndarray]
    conformer: np.ndarray
    time_seconds: float
    smiles: str
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class MPFITCalculator:
    """Calculator for MPFIT charges using GDMA.

    Parameters
    ----------
    config : GDMAConfig, optional
        Configuration for GDMA calculations

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> calculator = MPFITCalculator()
    >>> molecule = Molecule.from_smiles("CCO")
    >>> molecule.generate_conformers(n_conformers=1)
    >>> result = calculator.compute(molecule)
    >>> print(result.charges)
    """

    def __init__(self, config: Optional[GDMAConfig] = None):
        self.config = config or GDMAConfig()
        self.qc_settings = GDMASettings(
            method=self.config.method,
            basis=self.config.basis,
            switch=self.config.switch,
            limit=self.config.limit
        )
        self.solver = MPFITSVDSolver(self.config.svd_threshold)

    def compute(self, molecule: Molecule) -> MPFITResult:
        """Compute MPFIT charges for a molecule.

        Parameters
        ----------
        molecule : Molecule
            OpenFF molecule with at least one conformer

        Returns
        -------
        MPFITResult
            Calculation results including charges and multipole moments
        """
        start_time = time.time()
        smiles = molecule.to_smiles(mapped=False)

        if not molecule.conformers:
            elapsed = time.time() - start_time
            return MPFITResult(
                charges=np.zeros(molecule.n_atoms),
                multipole_moments=None,
                conformer=np.zeros((molecule.n_atoms, 3)),
                time_seconds=elapsed,
                smiles=smiles,
                metadata=self._get_metadata(),
                success=False,
                error_message="Molecule has no conformers"
            )

        try:
            [input_conformer] = extract_conformers(molecule)

            conformer, multipoles = Psi4GDMAGenerator.generate(
                molecule, input_conformer, self.qc_settings, minimize=self.config.minimize
            )

            qc_record = MoleculeGDMARecord.from_molecule(
                molecule, conformer, multipoles, self.qc_settings
            )

            charge_parameter = generate_mpfit_charge_parameter([qc_record], self.solver)

            charges = np.array(charge_parameter.value)

            elapsed = time.time() - start_time

            return MPFITResult(
                charges=charges,
                multipole_moments=multipoles,
                conformer=conformer,
                time_seconds=elapsed,
                smiles=charge_parameter.smiles,
                metadata=self._get_metadata(),
                success=True,
                error_message=None
            )

        except Exception as e:
            # Catch all exceptions and convert to string to avoid pickling issues
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            return MPFITResult(
                charges=np.zeros(molecule.n_atoms),
                multipole_moments=None,
                conformer=np.zeros((molecule.n_atoms, 3)),
                time_seconds=elapsed,
                smiles=smiles,
                metadata=self._get_metadata(),
                success=False,
                error_message=error_msg
            )

    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary for calculation settings."""
        return {
            "method": self.config.method,
            "basis": self.config.basis,
            "limit": self.config.limit,
            "switch": self.config.switch,
            "svd_threshold": self.config.svd_threshold,
            "minimize": self.config.minimize,
            "calculator": "MPFIT"
        }


def compute_mpfit_charges(
    molecule: Molecule,
    config: Optional[GDMAConfig] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute MPFIT charges for a molecule (convenience function).

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule with at least one conformer
    config : GDMAConfig, optional
        Configuration for GDMA calculations

    Returns
    -------
    charges : np.ndarray
        Atomic partial charges (n_atoms,)
    multipole_moments : np.ndarray
        GDMA multipole moments (n_atoms, n_moments)
    metadata : dict
        Calculation metadata including timing and settings

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> molecule = Molecule.from_smiles("CCO")
    >>> molecule.generate_conformers(n_conformers=1)
    >>> charges, multipoles, metadata = compute_mpfit_charges(molecule)
    """
    calculator = MPFITCalculator(config)
    result = calculator.compute(molecule)

    if not result.success:
        raise RuntimeError(f"MPFIT calculation failed: {result.error_message}")

    return result.charges, result.multipole_moments, result.metadata
