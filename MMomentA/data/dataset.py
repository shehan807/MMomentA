"""PyTorch dataset classes for charge prediction."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable, Union

from .schema import MoleculeData
from .graph import molecule_data_to_dgl_graph


class ChargeDataset(torch.utils.data.Dataset):
    """PyTorch dataset for charge prediction with multipole moments.

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        List of molecular data
    include_multipoles : bool
        Include multipole moments in features
    use_conformer : bool
        Include 3D coordinates in features
    transform : callable, optional
        Optional transform to apply to graphs

    Examples
    --------
    >>> from MMomentA.data import extract_molecule_data
    >>> mol_data_list = [extract_molecule_data(mol, result) for mol, result in ...]
    >>> dataset = ChargeDataset(mol_data_list, include_multipoles=True)
    >>> print(len(dataset))
    >>> graph = dataset[0]
    """

    def __init__(
        self,
        molecule_data_list: List[MoleculeData],
        include_multipoles: bool = True,
        use_conformer: bool = False,
        transform: Optional[Callable] = None
    ):
        self.molecule_data_list = molecule_data_list
        self.include_multipoles = include_multipoles
        self.use_conformer = use_conformer
        self.transform = transform

        self.graphs = self._build_graphs()

    def _build_graphs(self):
        """Build DGL graphs from molecular data."""
        from .graph import molecule_data_to_dgl_graph

        graphs = []
        for mol_data in self.molecule_data_list:
            graph = molecule_data_to_dgl_graph(
                mol_data,
                include_multipoles=self.include_multipoles,
                use_conformer=self.use_conformer
            )
            graphs.append(graph)

        return graphs

    def __len__(self) -> int:
        """Return number of molecules in dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int):
        """Get graph at index."""
        graph = self.graphs[idx]

        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def get_molecule_data(self, idx: int) -> MoleculeData:
        """Get original molecular data at index."""
        return self.molecule_data_list[idx]

    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        return self.graphs[0].ndata['h0'].shape[1]


class SplitDataset:
    """Container for train/validation/test splits.

    Parameters
    ----------
    train_dataset : ChargeDataset
        Training dataset
    val_dataset : ChargeDataset
        Validation dataset
    test_dataset : ChargeDataset
        Test dataset

    Examples
    --------
    >>> split_data = SplitDataset(train_dataset, val_dataset, test_dataset)
    >>> train_loader = split_data.get_train_loader(batch_size=32)
    >>> val_loader = split_data.get_val_loader(batch_size=32)
    """

    def __init__(
        self,
        train_dataset: ChargeDataset,
        val_dataset: ChargeDataset,
        test_dataset: ChargeDataset
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """Get training data loader."""
        import dgl

        return dgl.dataloading.GraphDataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    def get_val_loader(self, batch_size: int = 32, shuffle: bool = False, **kwargs):
        """Get validation data loader."""
        import dgl

        return dgl.dataloading.GraphDataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False, **kwargs):
        """Get test data loader."""
        import dgl

        return dgl.dataloading.GraphDataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_dataset)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.val_dataset)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.test_dataset)

    def __repr__(self) -> str:
        return (
            f"SplitDataset(train={self.n_train}, "
            f"val={self.n_val}, test={self.n_test})"
        )


def create_split_datasets(
    molecule_data_list: List[MoleculeData],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    include_multipoles: bool = True,
    use_conformer: bool = False
) -> SplitDataset:
    """Create train/val/test datasets from split indices.

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        Complete list of molecular data
    train_indices : list of int
        Training set indices
    val_indices : list of int
        Validation set indices
    test_indices : list of int
        Test set indices
    include_multipoles : bool
        Include multipole moments in features
    use_conformer : bool
        Include 3D coordinates in features

    Returns
    -------
    SplitDataset
        Container with train/val/test datasets

    Examples
    --------
    >>> from MMomentA.data import scaffold_split, create_split_datasets
    >>> train_idx, val_idx, test_idx = scaffold_split(molecules)
    >>> split_data = create_split_datasets(
    ...     mol_data_list, train_idx, val_idx, test_idx
    ... )
    >>> print(split_data)
    """
    train_data = [molecule_data_list[i] for i in train_indices]
    val_data = [molecule_data_list[i] for i in val_indices]
    test_data = [molecule_data_list[i] for i in test_indices]

    train_dataset = ChargeDataset(
        train_data,
        include_multipoles=include_multipoles,
        use_conformer=use_conformer
    )
    val_dataset = ChargeDataset(
        val_data,
        include_multipoles=include_multipoles,
        use_conformer=use_conformer
    )
    test_dataset = ChargeDataset(
        test_data,
        include_multipoles=include_multipoles,
        use_conformer=use_conformer
    )

    return SplitDataset(train_dataset, val_dataset, test_dataset)
