"""Charge comparison and analysis utilities."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from openff.toolkit.topology import Molecule
from openff.units.elements import SYMBOLS


@dataclass
class ChargeComparison:
    """Container for charge comparison statistics.

    Attributes
    ----------
    rmsd : float
        Root mean squared deviation
    mae : float
        Mean absolute error
    max_abs_diff : float
        Maximum absolute difference
    correlation : float
        Pearson correlation coefficient
    mean_diff : float
        Mean signed difference
    """
    rmsd: float
    mae: float
    max_abs_diff: float
    correlation: float
    mean_diff: float


def compare_charges(
    charges1: np.ndarray,
    charges2: np.ndarray,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2"
) -> ChargeComparison:
    """Compare two sets of atomic charges.

    Parameters
    ----------
    charges1 : np.ndarray
        First set of charges (n_atoms,)
    charges2 : np.ndarray
        Second set of charges (n_atoms,)
    method1_name : str
        Name of first method (for logging)
    method2_name : str
        Name of second method (for logging)

    Returns
    -------
    ChargeComparison
        Comparison statistics

    Examples
    --------
    >>> charges_mpfit = np.array([0.1, -0.1, 0.0])
    >>> charges_resp = np.array([0.12, -0.11, 0.01])
    >>> comparison = compare_charges(charges_mpfit, charges_resp)
    >>> print(f"RMSD: {comparison.rmsd:.4f}")
    """
    if charges1.shape != charges2.shape:
        raise ValueError(
            f"Charge arrays must have same shape: {charges1.shape} vs {charges2.shape}"
        )

    diff = charges1 - charges2

    rmsd = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_diff = float(np.mean(diff))

    correlation = float(np.corrcoef(charges1, charges2)[0, 1])

    return ChargeComparison(
        rmsd=rmsd,
        mae=mae,
        max_abs_diff=max_abs_diff,
        correlation=correlation,
        mean_diff=mean_diff
    )


def analyze_charge_distribution(
    molecule: Molecule,
    charges_dict: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Analyze charge distribution across multiple methods.

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule
    charges_dict : dict
        Dictionary mapping method names to charge arrays

    Returns
    -------
    dict
        Analysis results including per-atom charges and comparisons

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> molecule = Molecule.from_smiles("CCO")
    >>> charges_dict = {
    ...     "MPFIT": np.array([0.1, 0.2, -0.3]),
    ...     "RESP": np.array([0.11, 0.19, -0.30])
    ... }
    >>> analysis = analyze_charge_distribution(molecule, charges_dict)
    """
    analysis = {
        "molecule_info": {
            "smiles": molecule.to_smiles(mapped=False),
            "n_atoms": molecule.n_atoms,
            "formula": molecule.hill_formula
        },
        "atoms": [],
        "comparisons": {},
        "total_charges": {},
        "charge_statistics": {}
    }

    for i, atom in enumerate(molecule.atoms):
        atom_data = {
            "index": i,
            "element": SYMBOLS[atom.atomic_number],
            "charges": {
                method: float(charges[i])
                for method, charges in charges_dict.items()
            }
        }
        analysis["atoms"].append(atom_data)

    methods = list(charges_dict.keys())
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            comparison = compare_charges(
                charges_dict[method1],
                charges_dict[method2],
                method1,
                method2
            )

            comparison_key = f"{method1}_vs_{method2}"
            analysis["comparisons"][comparison_key] = {
                "rmsd": comparison.rmsd,
                "mae": comparison.mae,
                "max_abs_diff": comparison.max_abs_diff,
                "correlation": comparison.correlation,
                "mean_diff": comparison.mean_diff
            }

    for method, charges in charges_dict.items():
        analysis["total_charges"][method] = float(np.sum(charges))
        analysis["charge_statistics"][method] = {
            "mean": float(np.mean(charges)),
            "std": float(np.std(charges)),
            "min": float(np.min(charges)),
            "max": float(np.max(charges))
        }

    return analysis


def print_charge_comparison(
    molecule: Molecule,
    charges_dict: Dict[str, np.ndarray],
    methods_to_show: Optional[List[str]] = None
):
    """Print formatted charge comparison table.

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule
    charges_dict : dict
        Dictionary mapping method names to charge arrays
    methods_to_show : list of str, optional
        Specific methods to display (default: all)
    """
    analysis = analyze_charge_distribution(molecule, charges_dict)

    print(f"\nMolecule: {analysis['molecule_info']['smiles']}")
    print(f"Formula: {analysis['molecule_info']['formula']}")
    print(f"Atoms: {analysis['molecule_info']['n_atoms']}")
    print("=" * 70)

    if methods_to_show is None:
        methods_to_show = list(charges_dict.keys())

    header = f"{'Atom':<4} {'Element':<8}"
    for method in methods_to_show:
        header += f" {method:<12}"
    print(f"\n{header}")
    print("-" * 70)

    for atom in analysis["atoms"]:
        row = f"{atom['index']:<4} {atom['element']:<8}"
        for method in methods_to_show:
            charge = atom["charges"].get(method, 0.0)
            row += f" {charge:>12.4f}"
        print(row)

    print("-" * 70)
    for method in methods_to_show:
        total = analysis["total_charges"].get(method, 0.0)
        print(f"Total {method}: {total:>10.6f}")

    print("\nMethod Comparisons:")
    print("-" * 50)
    for comparison_key, stats in analysis["comparisons"].items():
        methods_in_comparison = comparison_key.split("_vs_")
        if all(m in methods_to_show for m in methods_in_comparison):
            print(f"\n{comparison_key}:")
            print(f"  RMSD: {stats['rmsd']:.4f}")
            print(f"  MAE: {stats['mae']:.4f}")
            print(f"  Max |Δ|: {stats['max_abs_diff']:.4f}")
            print(f"  Correlation: {stats['correlation']:.4f}")
