"""ESP reproduction validation utilities."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from openff.toolkit.topology import Molecule
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.units import unit


@dataclass
class ESPValidation:
    """Container for ESP validation metrics.

    Attributes
    ----------
    rmse : float
        Root mean squared error (kcal/mol/e)
    mae : float
        Mean absolute error (kcal/mol/e)
    max_error : float
        Maximum absolute error (kcal/mol/e)
    correlation : float
        Pearson correlation coefficient
    """
    rmse: float
    mae: float
    max_error: float
    correlation: float


def validate_esp_reproduction(
    molecule: Molecule,
    charges: np.ndarray,
    esp_record: MoleculeESPRecord,
    verbose: bool = False
) -> ESPValidation:
    """Validate how well charges reproduce quantum mechanical ESP.

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule
    charges : np.ndarray
        Atomic partial charges to validate (n_atoms,)
    esp_record : MoleculeESPRecord
        Reference ESP data from QM calculation
    verbose : bool
        Print detailed validation results

    Returns
    -------
    ESPValidation
        ESP reproduction metrics

    Examples
    --------
    >>> from mmomenta.qm import RESPCalculator
    >>> calculator = RESPCalculator()
    >>> result = calculator.compute(molecule)
    >>> validation = validate_esp_reproduction(
    ...     molecule, result.charges, result.esp_record
    ... )
    >>> print(f"ESP RMSE: {validation.rmse:.4f} kcal/mol/e")
    """
    conformer = esp_record.conformer
    grid_coordinates = esp_record.grid_coordinates
    esp_qm = esp_record.esp

    esp_predicted = _compute_esp_from_charges(
        charges, conformer, grid_coordinates, molecule
    )

    esp_qm_array = np.array(esp_qm)
    esp_predicted_array = np.array(esp_predicted)

    diff = esp_predicted_array - esp_qm_array

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_error = float(np.max(np.abs(diff)))

    correlation = float(np.corrcoef(esp_qm_array, esp_predicted_array)[0, 1])

    if verbose:
        print(f"ESP Validation:")
        print(f"  RMSE: {rmse:.4f} kcal/mol/e")
        print(f"  MAE: {mae:.4f} kcal/mol/e")
        print(f"  Max Error: {max_error:.4f} kcal/mol/e")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Grid Points: {len(esp_qm_array)}")

    return ESPValidation(
        rmse=rmse,
        mae=mae,
        max_error=max_error,
        correlation=correlation
    )


def _compute_esp_from_charges(
    charges: np.ndarray,
    conformer: np.ndarray,
    grid_coordinates: np.ndarray,
    molecule: Molecule
) -> np.ndarray:
    """Compute electrostatic potential from point charges.

    Parameters
    ----------
    charges : np.ndarray
        Atomic charges (n_atoms,)
    conformer : np.ndarray
        Atomic coordinates (n_atoms, 3) in Angstrom
    grid_coordinates : np.ndarray
        Grid point coordinates (n_grid, 3) in Angstrom
    molecule : Molecule
        OpenFF molecule (used for unit conversion)

    Returns
    -------
    np.ndarray
        ESP values at grid points (n_grid,) in kcal/mol/e
    """
    COULOMB_CONSTANT = 332.063713  # kcal*Angstrom/(mol*e^2)

    n_atoms = len(charges)
    n_grid = len(grid_coordinates)

    esp = np.zeros(n_grid)

    for i_grid in range(n_grid):
        grid_point = grid_coordinates[i_grid]

        for i_atom in range(n_atoms):
            atom_pos = conformer[i_atom]
            distance = np.linalg.norm(grid_point - atom_pos)

            if distance > 1e-6:
                esp[i_grid] += charges[i_atom] / distance

    esp *= COULOMB_CONSTANT

    return esp
