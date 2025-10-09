"""Data extraction from QM calculations for ML training.

This module extracts multipole moments, atomic features, and graph
structure from MPFIT calculations for ML model training.
"""

import numpy as np
from typing import Dict, Any, Optional
from openff.toolkit.topology import Molecule
from openff.units.elements import SYMBOLS

from .schema import MoleculeData
from ..qm.mpfit import MPFITResult


def extract_multipole_features(multipole_moments: np.ndarray) -> np.ndarray:
    """Extract multipole moment features for ML.

    GDMA returns multipole moments organized by rank:
    - Rank 0 (monopole): Q00 (1 component)
    - Rank 1 (dipole): Q10, Q11c, Q11s (3 components)
    - Rank 2 (quadrupole): Q20, Q21c, Q21s, Q22c, Q22s (5 components)
    - Rank 3 (octupole): 7 components
    - etc.

    Parameters
    ----------
    multipole_moments : np.ndarray
        Raw GDMA multipole moments (n_atoms, n_components)

    Returns
    -------
    np.ndarray
        Processed multipole features (n_atoms, n_features)
    """
    return multipole_moments


def extract_atomic_features(molecule: Molecule) -> Dict[str, np.ndarray]:
    """Extract atomic-level features for graph neural network.

    Features match espaloma-charge conventions:
    - Atomic number (one-hot encoding handled in model)
    - Formal charge
    - Hybridization
    - Aromaticity
    - Ring membership
    - Degree and valence

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule

    Returns
    -------
    dict
        Atomic features dictionary
    """
    rdkit_mol = molecule.to_rdkit()
    n_atoms = molecule.n_atoms

    atomic_numbers = np.array([atom.atomic_number for atom in molecule.atoms])
    formal_charges = np.array([atom.formal_charge.m for atom in molecule.atoms])
    is_aromatic = np.array([atom.is_aromatic for atom in molecule.atoms], dtype=float)

    import warnings
    degree = np.array([atom.GetTotalDegree() for atom in rdkit_mol.GetAtoms()])
    # Suppress RDKit deprecation warning for GetTotalValence()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*GetValence.*")
        valence = np.array([atom.GetTotalValence() for atom in rdkit_mol.GetAtoms()])
    explicit_valence = np.array([atom.GetExplicitValence() for atom in rdkit_mol.GetAtoms()])

    hybridization = np.zeros((n_atoms, 5))
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        hyb = atom.GetHybridization()
        hyb_map = {
            0: 0,  # UNSPECIFIED
            1: 0,  # S
            2: 0,  # SP
            3: 1,  # SP2
            4: 2,  # SP3
            5: 3,  # SP3D
            6: 4   # SP3D2
        }
        hyb_idx = hyb_map.get(int(hyb), 0)
        if hyb_idx < 5:
            hybridization[i, hyb_idx] = 1.0

    ring_membership = np.zeros((n_atoms, 6))
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        for ring_size in range(3, 9):
            if atom.IsInRingSize(ring_size):
                ring_membership[i, ring_size - 3] = 1.0

    atomic_mass = np.array([atom.GetMass() for atom in rdkit_mol.GetAtoms()])

    return {
        'atomic_numbers': atomic_numbers,
        'formal_charges': formal_charges,
        'is_aromatic': is_aromatic,
        'degree': degree,
        'valence': valence,
        'explicit_valence': explicit_valence,
        'hybridization': hybridization,
        'ring_membership': ring_membership,
        'atomic_mass': atomic_mass
    }


def extract_graph_structure(molecule: Molecule) -> Dict[str, np.ndarray]:
    """Extract graph connectivity for GNN.

    Returns bidirectional edges (both i->j and j->i for each bond).

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule

    Returns
    -------
    dict
        Graph structure with edges and bond information
    """
    rdkit_mol = molecule.to_rdkit()

    bonds = list(rdkit_mol.GetBonds())
    n_bonds = len(bonds)

    edges_begin = []
    edges_end = []
    bond_orders = []

    for bond in bonds:
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_order = bond.GetBondType().real

        edges_begin.extend([begin_idx, end_idx])
        edges_end.extend([end_idx, begin_idx])
        bond_orders.extend([bond_order, bond_order])

    edges = np.array([edges_begin, edges_end], dtype=np.int64)
    bond_orders = np.array(bond_orders, dtype=np.float32)

    return {
        'edges': edges,
        'bond_orders': bond_orders,
        'n_bonds': n_bonds
    }


def extract_molecule_data(
    molecule: Molecule,
    mpfit_result: MPFITResult,
    molecule_id: Optional[str] = None
) -> MoleculeData:
    """Extract complete molecular data from MPFIT calculation.

    Parameters
    ----------
    molecule : Molecule
        OpenFF molecule
    mpfit_result : MPFITResult
        MPFIT calculation result
    molecule_id : str, optional
        Unique identifier (default: SMILES)

    Returns
    -------
    MoleculeData
        Complete molecular data for ML training

    Examples
    --------
    >>> from MMomentA.qm import MPFITCalculator
    >>> calculator = MPFITCalculator()
    >>> result = calculator.compute(molecule)
    >>> mol_data = extract_molecule_data(molecule, result)
    >>> mol_data.validate()
    """
    if not mpfit_result.success:
        raise ValueError(f"MPFIT calculation failed: {mpfit_result.error_message}")

    if molecule_id is None:
        molecule_id = mpfit_result.smiles

    conformer = mpfit_result.conformer
    multipole_moments = extract_multipole_features(mpfit_result.multipole_moments)
    atomic_features = extract_atomic_features(molecule)
    graph_structure = extract_graph_structure(molecule)
    target_charges = mpfit_result.charges

    # Get molecule total charge (default to 0 for neutral molecules)
    total_charge = molecule.total_charge.m if hasattr(molecule, 'total_charge') else 0

    qm_metadata = {
        **mpfit_result.metadata,
        'calculation_time': mpfit_result.time_seconds,
        'conformer_optimized': mpfit_result.metadata.get('minimize', True),
        'total_charge': total_charge
    }

    mol_data = MoleculeData(
        molecule_id=molecule_id,
        smiles=mpfit_result.smiles,
        conformer=conformer,
        multipole_moments=multipole_moments,
        atomic_features=atomic_features,
        graph_structure=graph_structure,
        target_charges=target_charges,
        qm_metadata=qm_metadata
    )

    mol_data.validate()

    return mol_data


def batch_extract_molecule_data(
    molecules: list,
    mpfit_results: list,
    molecule_ids: Optional[list] = None
) -> list:
    """Extract data from multiple molecules.

    Parameters
    ----------
    molecules : list of Molecule
        OpenFF molecules
    mpfit_results : list of MPFITResult
        MPFIT calculation results
    molecule_ids : list of str, optional
        Unique identifiers

    Returns
    -------
    list of MoleculeData
        Extracted molecular data

    Examples
    --------
    >>> from MMomentA.data import BatchProcessor
    >>> from MMomentA.qm import MPFITCalculator
    >>> calculator = MPFITCalculator()
    >>> processor = BatchProcessor({"MPFIT": calculator})
    >>> batch_results = processor.process(molecules)
    >>> # Extract successful results
    >>> mpfit_results = [r.results["MPFIT"] for r in batch_results if r.success]
    >>> mol_data_list = batch_extract_molecule_data(molecules, mpfit_results)
    """
    if molecule_ids is None:
        molecule_ids = [None] * len(molecules)

    if len(molecules) != len(mpfit_results):
        raise ValueError(
            f"Length mismatch: {len(molecules)} molecules vs "
            f"{len(mpfit_results)} results"
        )

    extracted_data = []

    for mol, result, mol_id in zip(molecules, mpfit_results, molecule_ids):
        if result.success:
            mol_data = extract_molecule_data(mol, result, mol_id)
            extracted_data.append(mol_data)

    return extracted_data
