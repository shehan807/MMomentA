"""
Utility functions for ESP validation of charge fitting methods.
Adapted from openff-PyMPFIT for MMomentA framework.
"""
import numpy as np
import os

# Physical constants
bohr_to_angstrom = 0.529177249


def charges_to_esp(molecule, charges, psi4_grid_file="1_default_grid.dat",
                   psi4_esp_file="1_default_grid_esp.dat", verbose=True):
    """Calculate ESP from charges and compare with reference ESP values.

    Parameters
    ----------
    molecule : psi4.Molecule
        Molecule instance with geometry
    charges : array_like
        Atomic charges to use for ESP calculation
    psi4_grid_file : str, optional
        Path to grid points file (default: "1_default_grid.dat")
    psi4_esp_file : str, optional
        Path to reference ESP values file (default: "1_default_grid_esp.dat")
    verbose : bool, optional
        Print detailed comparison metrics (default: True)

    Returns
    -------
    dict
        Dictionary containing comparison metrics
    """

    # Get coordinates in angstrom
    coordinates = molecule.geometry()
    coordinates = coordinates.np.astype('float') * bohr_to_angstrom

    # Read grid points from grid.dat
    points = np.loadtxt(psi4_grid_file)
    if 'Bohr' in str(molecule.units()):
        points *= bohr_to_angstrom

    # Calculate ESP values
    esp_values = np.zeros(len(points))
    for i in range(len(points)):
        for j in range(len(coordinates)):
            distance = np.linalg.norm(points[i] - coordinates[j])
            # Convert to atomic units: ESP (a.u.) = charge * (1/r_angstrom) * bohr_to_angstrom
            esp_values[i] += charges[j] / distance * bohr_to_angstrom

    # Read ESP values from grid_esp.dat
    esp_psi4 = np.loadtxt(psi4_esp_file)
    metrics = compare_grid_esp(esp_psi4, esp_values, verbose=verbose)

    return metrics


def compare_grid_esp(esp1, esp2, verbose=True):
    """Compare two ESP arrays and calculate error metrics.

    Parameters
    ----------
    esp1 : array_like
        First ESP array (reference values)
    esp2 : array_like
        Second ESP array (calculated values)
    verbose : bool, optional
        Print detailed comparison metrics (default: True)

    Returns
    -------
    dict
        Dictionary containing comparison metrics:
        - l2_norm: L2 norm of the difference
        - rmse: Root mean square error
        - mae: Mean absolute error
        - max_error: Maximum absolute error
        - correlation: Pearson correlation coefficient
        - r_squared: R-squared value
    """

    # Check if arrays have same shape
    if esp1.shape != esp2.shape:
        raise ValueError(f"ESP arrays have different shapes: {esp1.shape} vs {esp2.shape}")

    # Calculate metrics
    metrics = {}

    # L2 norm - this is analogous to ||A @ q - b|| in RESP fitting
    diff = esp1 - esp2
    metrics['l2_norm'] = np.sqrt(np.sum(diff**2))

    # RMSE (L2 norm normalized by number of points)
    metrics['rmse'] = np.sqrt(np.mean(diff**2))

    # MAE (L1 norm normalized by number of points)
    metrics['mae'] = np.mean(np.abs(diff))

    # Max absolute error
    metrics['max_error'] = np.max(np.abs(diff))

    # Pearson correlation coefficient
    metrics['correlation'] = np.corrcoef(esp1, esp2)[0, 1]

    # R-squared
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((esp1 - np.mean(esp1))**2)
    metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if verbose:
        print(f"\nESP Comparison Results:")
        print(f"=======================")
        print(f"Number of grid points: {len(esp1)}")
        print(f"\nError Metrics:")
        print(f"  L2 norm:               {metrics['l2_norm']:.6e}")
        print(f"  RMSE:                  {metrics['rmse']:.6e}")
        print(f"  MAE:                   {metrics['mae']:.6e}")
        print(f"  Max absolute error:    {metrics['max_error']:.6e}")
        print(f"\nCorrelation Metrics:")
        print(f"  Pearson correlation:   {metrics['correlation']:.6f}")
        print(f"  R-squared:             {metrics['r_squared']:.6f}")

    return metrics


def charges_to_esp_openff(molecule, charges, grid_file="grid.dat",
                          esp_file="grid_esp.dat", coordinates=None, verbose=True):
    """Calculate ESP from charges using OpenFF/MMomentA objects and compare with QM ESP.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        Molecule instance
    charges : array_like
        Atomic charges to use for ESP calculation
    grid_file : str, optional
        Path to grid points file (default: "grid.dat")
    esp_file : str, optional
        Path to reference ESP values file (default: "grid_esp.dat")
    coordinates : array_like, optional
        Molecular coordinates in angstrom (if not provided, extracted from molecule)
    verbose : bool, optional
        Print detailed comparison metrics (default: True)

    Returns
    -------
    dict
        Dictionary containing comparison metrics
    """

    # Get coordinates if not provided
    if coordinates is None:
        conformer = molecule.conformers[0]
        # Handle OpenFF units
        if hasattr(conformer, 'm_as'):
            from openff.units import unit
            coordinates = conformer.m_as(unit.angstrom)
        else:
            coordinates = conformer  # Already numpy array in angstrom

    # Check if grid files exist
    if not os.path.exists(grid_file) or not os.path.exists(esp_file):
        if verbose:
            print(f"Warning: Could not find {grid_file} or {esp_file} in current directory")
        return {}

    # Read grid points and ESP values
    grid_points = np.loadtxt(grid_file)
    qm_esp = np.loadtxt(esp_file)

    # Calculate ESP values at grid points
    calculated_esp = np.zeros(len(grid_points))

    for i in range(len(grid_points)):
        for j in range(len(coordinates)):
            distance = np.linalg.norm(grid_points[i] - coordinates[j])
            calculated_esp[i] += charges[j] / distance * bohr_to_angstrom

    # Compare with QM ESP
    metrics = compare_grid_esp(qm_esp, calculated_esp, verbose=verbose)

    return metrics
