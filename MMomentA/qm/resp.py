"""RESP charge calculation using electrostatic potential fitting."""

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from openff.toolkit.topology import Molecule
from openff.recharge.charges.library import LibraryChargeCollection, LibraryChargeGenerator
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.resp.solvers import IterativeSolver
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings
from openff.recharge.utilities.molecule import extract_conformers

from .settings import ESPConfig


@dataclass
class RESPResult:
    """Container for RESP calculation results.

    Attributes
    ----------
    charges : np.ndarray
        Atomic partial charges (n_atoms,)
    esp_record : MoleculeESPRecord
        ESP data record for validation
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
    esp_record: Optional[MoleculeESPRecord]
    conformer: np.ndarray
    time_seconds: float
    smiles: str
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class RESPCalculator:
    """Calculator for RESP charges using ESP fitting.

    Parameters
    ----------
    config : ESPConfig, optional
        Configuration for ESP calculations

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> calculator = RESPCalculator()
    >>> molecule = Molecule.from_smiles("CCO")
    >>> molecule.generate_conformers(n_conformers=1)
    >>> result = calculator.compute(molecule)
    >>> print(result.charges)
    """

    def __init__(self, config: Optional[ESPConfig] = None):
        self.config = config or ESPConfig()
        self.qc_settings = ESPSettings(
            method=self.config.method,
            basis=self.config.basis,
            grid_settings=MSKGridSettings()
        )
        self.solver = IterativeSolver()

    def compute(self, molecule: Molecule) -> RESPResult:
        """Compute RESP charges for a molecule.

        Parameters
        ----------
        molecule : Molecule
            OpenFF molecule with at least one conformer

        Returns
        -------
        RESPResult
            Calculation results including charges and ESP data
        """
        start_time = time.time()
        smiles = molecule.to_smiles(mapped=False)

        if not molecule.conformers:
            elapsed = time.time() - start_time
            return RESPResult(
                charges=np.zeros(molecule.n_atoms),
                esp_record=None,
                conformer=np.zeros((molecule.n_atoms, 3)),
                time_seconds=elapsed,
                smiles=smiles,
                metadata=self._get_metadata(),
                success=False,
                error_message="Molecule has no conformers"
            )

        try:
            [input_conformer] = extract_conformers(molecule)

            conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
                molecule, input_conformer, self.qc_settings, minimize=self.config.minimize
            )

            qc_record = MoleculeESPRecord.from_molecule(
                molecule, conformer, grid, esp, None, self.qc_settings
            )

            charge_parameter = generate_resp_charge_parameter([qc_record], self.solver)

            charges = LibraryChargeGenerator.generate(
                molecule, LibraryChargeCollection(parameters=[charge_parameter])
            )

            elapsed = time.time() - start_time

            return RESPResult(
                charges=charges.flatten(),
                esp_record=qc_record,
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

            # Get conformer for error result
            conformer_array = (
                molecule.conformers[0].m_as(unit.angstrom)
                if molecule.conformers
                else np.zeros((molecule.n_atoms, 3))
            )

            return RESPResult(
                charges=np.zeros(molecule.n_atoms),
                esp_record=None,
                conformer=conformer_array,
                time_seconds=elapsed,
                smiles=smiles,
                metadata={**self._get_metadata(), "error": error_msg},
                success=False,
                error_message=f"RESP calculation failed: {error_msg}"
            )

    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary for calculation settings."""
        return {
            "method": self.config.method,
            "basis": self.config.basis,
            "restraint_weight": self.config.restraint_weight,
            "grid_density": self.config.grid_density,
            "minimize": self.config.minimize,
            "calculator": "RESP"
        }


def compute_resp_charges(
    molecule: Molecule,
    config: Optional[ESPConfig] = None
) -> Tuple[np.ndarray, MoleculeESPRecord, Dict[str, Any]]:
    """Compute RESP charges for a molecule (convenience function).

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule with at least one conformer
    config : ESPConfig, optional
        Configuration for ESP calculations

    Returns
    -------
    charges : np.ndarray
        Atomic partial charges (n_atoms,)
    esp_record : MoleculeESPRecord
        ESP data record for validation
    metadata : dict
        Calculation metadata including timing and settings

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> molecule = Molecule.from_smiles("CCO")
    >>> molecule.generate_conformers(n_conformers=1)
    >>> charges, esp_record, metadata = compute_resp_charges(molecule)
    """
    calculator = RESPCalculator(config)
    result = calculator.compute(molecule)

    if not result.success:
        raise RuntimeError(f"RESP calculation failed: {result.error_message}")

    return result.charges, result.esp_record, result.metadata
