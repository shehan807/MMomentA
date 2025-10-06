"""Data processing modules for MMomentA."""

from .batch import batch_process_molecules, BatchProcessor
from .loaders import load_molecules_from_file, load_zinc_molecules, load_spice_molecules
from .schema import MoleculeData, DatasetMetadata
from .extraction import extract_molecule_data, batch_extract_molecule_data
from .splitting import scaffold_split, random_split, stratified_split, split_by_names
from .storage import save_dataset, load_dataset

# Optional PyTorch/DGL imports
try:
    from .graph import molecule_data_to_dgl_graph, batch_to_dgl_graphs, get_feature_dimensions
    from .dataset import ChargeDataset, SplitDataset, create_split_datasets
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__all__ = [
    "batch_process_molecules",
    "BatchProcessor",
    "load_molecules_from_file",
    "load_zinc_molecules",
    "load_spice_molecules",
    "MoleculeData",
    "DatasetMetadata",
    "extract_molecule_data",
    "batch_extract_molecule_data",
    "scaffold_split",
    "random_split",
    "stratified_split",
    "split_by_names",
    "save_dataset",
    "load_dataset",
]

if _HAS_TORCH:
    __all__.extend([
        "molecule_data_to_dgl_graph",
        "batch_to_dgl_graphs",
        "get_feature_dimensions",
        "ChargeDataset",
        "SplitDataset",
        "create_split_datasets",
    ])
