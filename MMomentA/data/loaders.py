"""Dataset loading utilities for ZINC, SPICE, and other molecular datasets."""

import pickle
import logging
from pathlib import Path
from typing import List, Optional, Union

from openff.toolkit.topology import Molecule


logger = logging.getLogger(__name__)


def load_molecules_from_file(
    file_path: Union[str, Path],
    max_molecules: Optional[int] = None,
    allow_undefined_stereo: bool = True
) -> List[Molecule]:
    """Load molecules from various file formats.

    Parameters
    ----------
    file_path : str or Path
        Path to molecule file (SDF, OEB, PKL, etc.)
    max_molecules : int, optional
        Maximum number of molecules to load
    allow_undefined_stereo : bool
        Allow molecules with undefined stereochemistry

    Returns
    -------
    list of Molecule
        Loaded OpenFF molecules

    Examples
    --------
    >>> molecules = load_molecules_from_file("molecules.sdf", max_molecules=100)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading molecules from {file_path}")

    if file_path.suffix in [".pkl", ".pickle"]:
        with open(file_path, 'rb') as f:
            molecules = pickle.load(f)
    elif file_path.suffix in [".sdf", ".mol2", ".pdb", ".oeb"]:
        molecules = Molecule.from_file(
            str(file_path),
            allow_undefined_stereo=allow_undefined_stereo
        )
        if not isinstance(molecules, list):
            molecules = [molecules]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    if max_molecules is not None:
        molecules = molecules[:max_molecules]

    logger.info(f"Loaded {len(molecules)} molecules from {file_path}")

    return molecules


def load_zinc_molecules(
    pickle_file: Union[str, Path],
    max_molecules: Optional[int] = None,
    inspect: bool = False
) -> List[Molecule]:
    """Load ZINC molecules from pickle file.

    Parameters
    ----------
    pickle_file : str or Path
        Path to ZINC pickle file
    max_molecules : int, optional
        Maximum number of molecules to load
    inspect : bool
        Print dataset statistics

    Returns
    -------
    list of Molecule
        ZINC molecules

    Examples
    --------
    >>> molecules = load_zinc_molecules("zinc_molecules.pkl", max_molecules=1000)
    """
    molecules = load_molecules_from_file(pickle_file, max_molecules)

    if inspect:
        _inspect_dataset(molecules, "ZINC")

    return molecules


def load_spice_molecules(
    file_path: Union[str, Path],
    max_molecules: Optional[int] = None,
    inspect: bool = False
) -> List[Molecule]:
    """Load SPICE molecules from file.

    Parameters
    ----------
    file_path : str or Path
        Path to SPICE molecule file (OEB, SDF, etc.)
    max_molecules : int, optional
        Maximum number of molecules to load
    inspect : bool
        Print dataset statistics

    Returns
    -------
    list of Molecule
        SPICE molecules

    Examples
    --------
    >>> molecules = load_spice_molecules("spice.oeb", max_molecules=500)
    """
    molecules = load_molecules_from_file(file_path, max_molecules)

    if inspect:
        _inspect_dataset(molecules, "SPICE")

    return molecules


def _inspect_dataset(molecules: List[Molecule], dataset_name: str):
    """Print dataset statistics.

    Parameters
    ----------
    molecules : list of Molecule
        Molecules to inspect
    dataset_name : str
        Name of dataset for logging
    """
    import numpy as np
    from collections import Counter

    logger.info(f"\n=== {dataset_name} Dataset Overview ===")
    logger.info(f"Total molecules: {len(molecules)}")

    atom_counts = [mol.n_atoms for mol in molecules]
    logger.info(f"Atom count range: {min(atom_counts)} - {max(atom_counts)}")
    logger.info(f"Average atoms: {np.mean(atom_counts):.1f}")

    with_conformers = sum(1 for mol in molecules if mol.conformers)
    logger.info(f"Molecules with conformers: {with_conformers}/{len(molecules)}")

    charged = sum(1 for mol in molecules if mol.total_charge != 0)
    logger.info(f"Charged molecules: {charged}")

    logger.info("\nSample molecules:")
    for i, mol in enumerate(molecules[:5]):
        logger.info(f"  {i+1}. {mol.to_smiles(mapped=False)} ({mol.hill_formula})")
