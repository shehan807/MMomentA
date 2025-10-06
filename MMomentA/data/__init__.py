"""Data processing modules for MMomentA."""

from .schema import MoleculeData, DatasetMetadata
from .storage import save_dataset, load_dataset

# Optional OpenFF imports (require OpenFF)
try:
    from .batch import batch_process_molecules, BatchProcessor
    from .loaders import load_molecules_from_file, load_zinc_molecules, load_spice_molecules
    from .extraction import extract_molecule_data, batch_extract_molecule_data
    from .splitting import scaffold_split, random_split, stratified_split, split_by_names
    _HAS_OPENFF = True
except ImportError:
    _HAS_OPENFF = False

# Optional PyTorch/DGL imports
try:
    from .graph import molecule_data_to_dgl_graph, batch_to_dgl_graphs, get_feature_dimensions
    from .dataset import ChargeDataset, SplitDataset, create_split_datasets
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__all__ = [
    "MoleculeData",
    "DatasetMetadata",
    "save_dataset",
    "load_dataset",
]

if _HAS_OPENFF:
    __all__.extend([
        "batch_process_molecules",
        "BatchProcessor",
        "load_molecules_from_file",
        "load_zinc_molecules",
        "load_spice_molecules",
        "extract_molecule_data",
        "batch_extract_molecule_data",
        "scaffold_split",
        "random_split",
        "stratified_split",
        "split_by_names",
    ])

if _HAS_TORCH:
    __all__.extend([
        "molecule_data_to_dgl_graph",
        "batch_to_dgl_graphs",
        "get_feature_dimensions",
        "ChargeDataset",
        "SplitDataset",
        "create_split_datasets",
    ])
