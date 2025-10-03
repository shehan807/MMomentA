"""Dataset splitting utilities for train/validation/test sets.

Implements scaffold-based splitting to test transferability across
chemical space.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from openff.toolkit.topology import Molecule


def scaffold_split(
    molecules: List[Molecule],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 2666
) -> Tuple[List[int], List[int], List[int]]:
    """Split molecules by Bemis-Murcko scaffolds.

    This ensures that structurally similar molecules stay together,
    testing the model's ability to generalize to new scaffolds.

    Parameters
    ----------
    molecules : list of Molecule
        Molecules to split
    train_frac : float
        Fraction for training set
    val_frac : float
        Fraction for validation set
    test_frac : float
        Fraction for test set
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    train_indices : list of int
        Indices for training set
    val_indices : list of int
        Indices for validation set
    test_indices : list of int
        Indices for test set

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> molecules = [Molecule.from_smiles(s) for s in smiles_list]
    >>> train_idx, val_idx, test_idx = scaffold_split(molecules)
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    np.random.seed(random_seed)

    scaffold_to_indices = defaultdict(list)

    for i, mol in enumerate(molecules):
        rdkit_mol = mol.to_rdkit()
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=rdkit_mol, includeChirality=False
        )
        scaffold_to_indices[scaffold].append(i)

    scaffolds = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffolds)

    train_indices = []
    val_indices = []
    test_indices = []

    train_cutoff = int(len(scaffolds) * train_frac)
    val_cutoff = train_cutoff + int(len(scaffolds) * val_frac)

    for i, scaffold in enumerate(scaffolds):
        indices = scaffold_to_indices[scaffold]

        if i < train_cutoff:
            train_indices.extend(indices)
        elif i < val_cutoff:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return train_indices, val_indices, test_indices


def random_split(
    n_molecules: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 2666
) -> Tuple[List[int], List[int], List[int]]:
    """Random split of molecule indices.

    Parameters
    ----------
    n_molecules : int
        Number of molecules
    train_frac : float
        Fraction for training set
    val_frac : float
        Fraction for validation set
    test_frac : float
        Fraction for test set
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    train_indices : list of int
        Indices for training set
    val_indices : list of int
        Indices for validation set
    test_indices : list of int
        Indices for test set
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    np.random.seed(random_seed)

    indices = np.arange(n_molecules)
    np.random.shuffle(indices)

    train_cutoff = int(n_molecules * train_frac)
    val_cutoff = train_cutoff + int(n_molecules * val_frac)

    train_indices = indices[:train_cutoff].tolist()
    val_indices = indices[train_cutoff:val_cutoff].tolist()
    test_indices = indices[val_cutoff:].tolist()

    return train_indices, val_indices, test_indices


def stratified_split(
    molecules: List[Molecule],
    property_name: str,
    n_bins: int = 5,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 2666
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified split based on molecular property.

    Ensures balanced representation of property distribution across splits.

    Parameters
    ----------
    molecules : list of Molecule
        Molecules to split
    property_name : str
        Property to stratify by ('n_atoms', 'molecular_weight', etc.)
    n_bins : int
        Number of bins for stratification
    train_frac : float
        Fraction for training set
    val_frac : float
        Fraction for validation set
    test_frac : float
        Fraction for test set
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    train_indices : list of int
        Indices for training set
    val_indices : list of int
        Indices for validation set
    test_indices : list of int
        Indices for test set
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    np.random.seed(random_seed)

    if property_name == 'n_atoms':
        properties = np.array([mol.n_atoms for mol in molecules])
    elif property_name == 'molecular_weight':
        properties = np.array([mol.to_topology().to_openmm().mass()._value for mol in molecules])
    else:
        raise ValueError(f"Unknown property: {property_name}")

    bins = np.percentile(properties, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(properties, bins[1:-1])

    train_indices = []
    val_indices = []
    test_indices = []

    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        bin_molecule_indices = np.where(bin_mask)[0]

        np.random.shuffle(bin_molecule_indices)

        n_bin = len(bin_molecule_indices)
        train_cutoff = int(n_bin * train_frac)
        val_cutoff = train_cutoff + int(n_bin * val_frac)

        train_indices.extend(bin_molecule_indices[:train_cutoff].tolist())
        val_indices.extend(bin_molecule_indices[train_cutoff:val_cutoff].tolist())
        test_indices.extend(bin_molecule_indices[val_cutoff:].tolist())

    return train_indices, val_indices, test_indices


def split_by_names(
    molecule_names: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 2666
) -> Tuple[List[str], List[str], List[str]]:
    """Split molecules by unique names (for datasets with multiple conformers).

    Ensures all conformers of a molecule stay in the same split.

    Parameters
    ----------
    molecule_names : list of str
        Molecule names (may have duplicates for different conformers)
    train_frac : float
        Fraction for training set
    val_frac : float
        Fraction for validation set
    test_frac : float
        Fraction for test set
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    train_names : list of str
        Names for training set
    val_names : list of str
        Names for validation set
    test_names : list of str
        Names for test set
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    np.random.seed(random_seed)

    unique_names = list(set(molecule_names))
    np.random.shuffle(unique_names)

    n_total = len(unique_names)
    train_cutoff = int(n_total * train_frac)
    val_cutoff = train_cutoff + int(n_total * val_frac)

    train_names = unique_names[:train_cutoff]
    val_names = unique_names[train_cutoff:val_cutoff]
    test_names = unique_names[val_cutoff:]

    return train_names, val_names, test_names
