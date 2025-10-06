"""Data storage utilities for molecular datasets.

Supports efficient storage and loading of molecular data using
HDF5 and Parquet formats.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from .schema import MoleculeData, DatasetMetadata


def save_dataset_json(
    molecule_data_list: List[MoleculeData],
    output_path: Union[str, Path],
    metadata: Optional[DatasetMetadata] = None
):
    """Save dataset to JSON format.

    JSON is human-readable but less efficient for large datasets.

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        Molecular data to save
    output_path : str or Path
        Output file path
    metadata : DatasetMetadata, optional
        Dataset metadata

    Examples
    --------
    >>> save_dataset_json(mol_data_list, "spice_mpfit.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_dicts = [mol_data.to_dict() for mol_data in molecule_data_list]

    output = {
        'molecules': data_dicts,
        'n_molecules': len(molecule_data_list)
    }

    if metadata is not None:
        output['metadata'] = metadata.to_dict()

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def load_dataset_json(
    input_path: Union[str, Path]
) -> tuple:
    """Load dataset from JSON format.

    Parameters
    ----------
    input_path : str or Path
        Input file path

    Returns
    -------
    molecule_data_list : list of MoleculeData
        Loaded molecular data
    metadata : dict or None
        Dataset metadata if present

    Examples
    --------
    >>> mol_data_list, metadata = load_dataset_json("spice_mpfit.json")
    """
    input_path = Path(input_path)

    with open(input_path, 'r') as f:
        data = json.load(f)

    molecule_data_list = [
        MoleculeData.from_dict(mol_dict)
        for mol_dict in data['molecules']
    ]

    metadata = data.get('metadata', None)

    return molecule_data_list, metadata


def save_dataset_hdf5(
    molecule_data_list: List[MoleculeData],
    output_path: Union[str, Path],
    metadata: Optional[DatasetMetadata] = None,
    compression: str = "gzip"
):
    """Save dataset to HDF5 format.

    HDF5 is efficient for large datasets and supports compression.

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        Molecular data to save
    output_path : str or Path
        Output file path
    metadata : DatasetMetadata, optional
        Dataset metadata
    compression : str
        HDF5 compression ('gzip', 'lzf', or None)

    Examples
    --------
    >>> save_dataset_hdf5(mol_data_list, "spice_mpfit.h5")
    """
    import h5py

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        for i, mol_data in enumerate(molecule_data_list):
            mol_group = f.create_group(f"molecule_{i}")

            mol_group.attrs['molecule_id'] = mol_data.molecule_id
            mol_group.attrs['smiles'] = mol_data.smiles

            mol_group.create_dataset(
                'conformer', data=mol_data.conformer, compression=compression
            )
            mol_group.create_dataset(
                'multipole_moments', data=mol_data.multipole_moments, compression=compression
            )
            mol_group.create_dataset(
                'target_charges', data=mol_data.target_charges, compression=compression
            )

            atomic_features_group = mol_group.create_group('atomic_features')
            for key, value in mol_data.atomic_features.items():
                atomic_features_group.create_dataset(
                    key, data=value, compression=compression
                )

            graph_structure_group = mol_group.create_group('graph_structure')
            for key, value in mol_data.graph_structure.items():
                # Don't use compression for scalar values
                if np.isscalar(value) or (hasattr(value, 'shape') and value.shape == ()):
                    graph_structure_group.create_dataset(key, data=value)
                else:
                    graph_structure_group.create_dataset(
                        key, data=value, compression=compression
                    )

            mol_group.attrs['qm_metadata'] = json.dumps(mol_data.qm_metadata)

        f.attrs['n_molecules'] = len(molecule_data_list)

        if metadata is not None:
            f.attrs['metadata'] = json.dumps(metadata.to_dict())


def load_dataset_hdf5(
    input_path: Union[str, Path]
) -> tuple:
    """Load dataset from HDF5 format.

    Parameters
    ----------
    input_path : str or Path
        Input file path

    Returns
    -------
    molecule_data_list : list of MoleculeData
        Loaded molecular data
    metadata : dict or None
        Dataset metadata if present

    Examples
    --------
    >>> mol_data_list, metadata = load_dataset_hdf5("spice_mpfit.h5")
    """
    import h5py

    input_path = Path(input_path)

    molecule_data_list = []

    with h5py.File(input_path, 'r') as f:
        n_molecules = f.attrs['n_molecules']

        for i in range(n_molecules):
            mol_group = f[f"molecule_{i}"]

            atomic_features = {
                key: mol_group['atomic_features'][key][:]
                for key in mol_group['atomic_features'].keys()
            }

            graph_structure = {}
            for key in mol_group['graph_structure'].keys():
                dataset = mol_group['graph_structure'][key]
                # Handle scalar vs array datasets
                if dataset.shape == ():
                    graph_structure[key] = dataset[()]
                else:
                    graph_structure[key] = dataset[:]

            mol_data = MoleculeData(
                molecule_id=mol_group.attrs['molecule_id'],
                smiles=mol_group.attrs['smiles'],
                conformer=mol_group['conformer'][:],
                multipole_moments=mol_group['multipole_moments'][:],
                atomic_features=atomic_features,
                graph_structure=graph_structure,
                target_charges=mol_group['target_charges'][:],
                qm_metadata=json.loads(mol_group.attrs['qm_metadata'])
            )

            molecule_data_list.append(mol_data)

        metadata_str = f.attrs.get('metadata', None)
        metadata = json.loads(metadata_str) if metadata_str else None

    return molecule_data_list, metadata


def save_dataset_pickle(
    molecule_data_list: List[MoleculeData],
    output_path: Union[str, Path],
    metadata: Optional[DatasetMetadata] = None
):
    """Save dataset to pickle format.

    Pickle is Python-specific but very convenient.

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        Molecular data to save
    output_path : str or Path
        Output file path
    metadata : DatasetMetadata, optional
        Dataset metadata

    Examples
    --------
    >>> save_dataset_pickle(mol_data_list, "spice_mpfit.pkl")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'molecules': molecule_data_list,
        'metadata': metadata
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def load_dataset_pickle(
    input_path: Union[str, Path]
) -> tuple:
    """Load dataset from pickle format.

    Parameters
    ----------
    input_path : str or Path
        Input file path

    Returns
    -------
    molecule_data_list : list of MoleculeData
        Loaded molecular data
    metadata : DatasetMetadata or None
        Dataset metadata if present

    Examples
    --------
    >>> mol_data_list, metadata = load_dataset_pickle("spice_mpfit.pkl")
    """
    input_path = Path(input_path)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    return data['molecules'], data.get('metadata', None)


def save_dataset(
    molecule_data_list: List[MoleculeData],
    output_path: Union[str, Path],
    metadata: Optional[DatasetMetadata] = None,
    format: str = "auto"
):
    """Save dataset with automatic format detection.

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        Molecular data to save
    output_path : str or Path
        Output file path
    metadata : DatasetMetadata, optional
        Dataset metadata
    format : str
        Format ('json', 'hdf5', 'pickle', or 'auto')

    Examples
    --------
    >>> save_dataset(mol_data_list, "spice_mpfit.h5")  # auto-detects HDF5
    """
    output_path = Path(output_path)

    if format == "auto":
        suffix = output_path.suffix.lower()
        if suffix in ['.h5', '.hdf5']:
            format = "hdf5"
        elif suffix in ['.pkl', '.pickle']:
            format = "pickle"
        elif suffix == '.json':
            format = "json"
        else:
            raise ValueError(f"Cannot auto-detect format from suffix: {suffix}")

    if format == "hdf5":
        save_dataset_hdf5(molecule_data_list, output_path, metadata)
    elif format == "pickle":
        save_dataset_pickle(molecule_data_list, output_path, metadata)
    elif format == "json":
        save_dataset_json(molecule_data_list, output_path, metadata)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_dataset(
    input_path: Union[str, Path],
    format: str = "auto"
) -> tuple:
    """Load dataset with automatic format detection.

    Parameters
    ----------
    input_path : str or Path
        Input file path
    format : str
        Format ('json', 'hdf5', 'pickle', or 'auto')

    Returns
    -------
    molecule_data_list : list of MoleculeData
        Loaded molecular data
    metadata : dict or DatasetMetadata or None
        Dataset metadata if present

    Examples
    --------
    >>> mol_data_list, metadata = load_dataset("spice_mpfit.h5")
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if format == "auto":
        suffix = input_path.suffix.lower()
        if suffix in ['.h5', '.hdf5']:
            format = "hdf5"
        elif suffix in ['.pkl', '.pickle']:
            format = "pickle"
        elif suffix == '.json':
            format = "json"
        else:
            raise ValueError(f"Cannot auto-detect format from suffix: {suffix}")

    if format == "hdf5":
        return load_dataset_hdf5(input_path)
    elif format == "pickle":
        return load_dataset_pickle(input_path)
    elif format == "json":
        return load_dataset_json(input_path)
    else:
        raise ValueError(f"Unknown format: {format}")
