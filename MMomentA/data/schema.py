"""Data schema for ML training datasets.

This module defines the data structure for storing molecular data
with multipole moments and charges for ML training.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class MoleculeData:
    """Complete molecular data for ML training.

    This schema matches the requirements from the project specification:
    {
        'molecule_id': str,
        'smiles': str,
        'conformer': np.ndarray,  # (n_atoms, 3)
        'multipole_moments': np.ndarray,  # (n_atoms, n_moments)
        'atomic_features': dict,  # atomic number, hybridization, etc.
        'graph_structure': dict,  # edges, bond orders
        'target_charges': np.ndarray,  # (n_atoms,) - from MPFIT
        'qm_metadata': dict,  # method, basis, energy, convergence info
    }

    Attributes
    ----------
    molecule_id : str
        Unique identifier for the molecule
    smiles : str
        SMILES representation
    conformer : np.ndarray
        3D coordinates (n_atoms, 3) in Angstrom
    multipole_moments : np.ndarray
        GDMA multipole moments (n_atoms, n_moments)
        Flattened multipole components per atom
    atomic_features : dict
        Atomic-level features for GNN:
        - atomic_numbers: np.ndarray (n_atoms,)
        - formal_charges: np.ndarray (n_atoms,)
        - hybridization: np.ndarray (n_atoms,)
        - aromaticity: np.ndarray (n_atoms,)
        - ring_membership: np.ndarray (n_atoms, max_ring_size)
        - degree: np.ndarray (n_atoms,)
        - valence: np.ndarray (n_atoms,)
    graph_structure : dict
        Graph connectivity:
        - edges: np.ndarray (2, n_edges) source/target indices
        - bond_orders: np.ndarray (n_edges,)
        - bond_types: np.ndarray (n_edges,)
    target_charges : np.ndarray
        MPFIT atomic charges (n_atoms,)
    qm_metadata : dict
        QM calculation metadata:
        - method: str (e.g., 'hf', 'pbe0')
        - basis: str (e.g., '6-31G*')
        - multipole_limit: int
        - energy: float (optional)
        - convergence: bool
        - calculation_time: float
    """
    molecule_id: str
    smiles: str
    conformer: np.ndarray
    multipole_moments: np.ndarray
    atomic_features: Dict[str, np.ndarray]
    graph_structure: Dict[str, np.ndarray]
    target_charges: np.ndarray
    qm_metadata: Dict[str, Any]

    def validate(self):
        """Validate data consistency."""
        n_atoms = len(self.conformer)

        assert self.conformer.shape == (n_atoms, 3), \
            f"Conformer shape mismatch: {self.conformer.shape} vs ({n_atoms}, 3)"

        assert len(self.target_charges) == n_atoms, \
            f"Charges length mismatch: {len(self.target_charges)} vs {n_atoms}"

        assert self.multipole_moments.shape[0] == n_atoms, \
            f"Multipole moments mismatch: {self.multipole_moments.shape[0]} vs {n_atoms}"

        assert len(self.atomic_features['atomic_numbers']) == n_atoms, \
            "Atomic features length mismatch"

        edges = self.graph_structure['edges']
        assert edges.shape[0] == 2, \
            f"Edges must be (2, n_edges), got {edges.shape}"

        assert np.abs(np.sum(self.target_charges)) < 0.01 or \
               'total_charge' in self.qm_metadata, \
            "Charges do not sum to zero (for neutral molecules)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'molecule_id': self.molecule_id,
            'smiles': self.smiles,
            'conformer': self.conformer.tolist(),
            'multipole_moments': self.multipole_moments.tolist(),
            'atomic_features': {
                k: v.tolist() for k, v in self.atomic_features.items()
            },
            'graph_structure': {
                k: v.tolist() for k, v in self.graph_structure.items()
            },
            'target_charges': self.target_charges.tolist(),
            'qm_metadata': self.qm_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoleculeData':
        """Load from dictionary."""
        return cls(
            molecule_id=data['molecule_id'],
            smiles=data['smiles'],
            conformer=np.array(data['conformer']),
            multipole_moments=np.array(data['multipole_moments']),
            atomic_features={
                k: np.array(v) for k, v in data['atomic_features'].items()
            },
            graph_structure={
                k: np.array(v) for k, v in data['graph_structure'].items()
            },
            target_charges=np.array(data['target_charges']),
            qm_metadata=data['qm_metadata']
        )


@dataclass
class DatasetMetadata:
    """Metadata for ML datasets.

    Attributes
    ----------
    dataset_name : str
        Name of dataset (e.g., 'SPICE', 'ZINC')
    n_molecules : int
        Total number of molecules
    qm_method : str
        QM method used
    qm_basis : str
        Basis set used
    multipole_limit : int
        Multipole expansion limit
    created_date : str
        Creation date
    split_strategy : str
        How train/val/test splits were made
    splits : dict
        Molecule IDs in each split
    """
    dataset_name: str
    n_molecules: int
    qm_method: str
    qm_basis: str
    multipole_limit: int
    created_date: str
    split_strategy: str = "scaffold"
    splits: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'n_molecules': self.n_molecules,
            'qm_method': self.qm_method,
            'qm_basis': self.qm_basis,
            'multipole_limit': self.multipole_limit,
            'created_date': self.created_date,
            'split_strategy': self.split_strategy,
            'splits': self.splits
        }
