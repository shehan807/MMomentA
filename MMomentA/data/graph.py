"""Graph construction for DGL-based GNN models.

This module converts molecular data to DGL graphs compatible with
espaloma-charge architecture.
"""

import torch
import numpy as np
from typing import Optional

from .schema import MoleculeData


def molecule_data_to_dgl_graph(
    mol_data: MoleculeData,
    include_multipoles: bool = True,
    use_conformer: bool = False,
    multipole_stats: Optional[dict] = None
):
    """Convert MoleculeData to DGL graph.

    Parameters
    ----------
    mol_data : MoleculeData
        Molecular data with features and targets
    include_multipoles : bool
        Include multipole moments in node features
    use_conformer : bool
        Include 3D coordinates in node features

    Returns
    -------
    dgl.DGLGraph
        Graph with node/edge features and target charges

    Examples
    --------
    >>> import dgl
    >>> from MMomentA.data import extract_molecule_data
    >>> mol_data = extract_molecule_data(molecule, mpfit_result)
    >>> graph = molecule_data_to_dgl_graph(mol_data)
    >>> print(graph.ndata.keys())  # ['h0', 'q_ref', 'type', ...]
    """
    import dgl

    n_atoms = len(mol_data.target_charges)

    g = dgl.graph(([], []))
    g.add_nodes(n_atoms)

    atomic_numbers = mol_data.atomic_features['atomic_numbers']
    g.ndata['type'] = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(-1)

    g.ndata['q_ref'] = torch.tensor(
        mol_data.target_charges, dtype=torch.float32
    ).unsqueeze(-1)

    h_v = torch.zeros(n_atoms, 100, dtype=torch.float32)
    h_v[torch.arange(n_atoms), atomic_numbers] = 1.0

    atomic_fp = construct_atomic_fingerprint(mol_data)

    if include_multipoles:
        multipole_features = torch.tensor(
            mol_data.multipole_moments, dtype=torch.float32
        )

        # Standardize multipole features if stats provided
        if multipole_stats is not None:
            mean = torch.tensor(multipole_stats['mean'], dtype=torch.float32)
            std = torch.tensor(multipole_stats['std'], dtype=torch.float32)
            multipole_features = (multipole_features - mean) / std

        h_v = torch.cat([h_v, atomic_fp, multipole_features], dim=-1)
    else:
        h_v = torch.cat([h_v, atomic_fp], dim=-1)

    if use_conformer:
        conformer_coords = torch.tensor(
            mol_data.conformer, dtype=torch.float32
        )
        h_v = torch.cat([h_v, conformer_coords], dim=-1)

    g.ndata['h0'] = h_v

    edges = mol_data.graph_structure['edges']
    g.add_edges(edges[0], edges[1])

    return g


def construct_atomic_fingerprint(mol_data: MoleculeData) -> torch.Tensor:
    """Construct atomic fingerprint matching espaloma-charge format.

    Features (17 dimensions matching espaloma-charge):
    - Total degree (1)
    - Total valence (1)
    - Explicit valence (1)
    - Is aromatic (1)
    - Atomic mass (1)
    - Ring membership 3-8 (6)
    - Hybridization one-hot (5)
    - Formal charge (1)

    Parameters
    ----------
    mol_data : MoleculeData
        Molecular data

    Returns
    -------
    torch.Tensor
        Atomic fingerprints (n_atoms, 17)
    """
    features = mol_data.atomic_features

    fp_components = [
        features['degree'].reshape(-1, 1),
        features['valence'].reshape(-1, 1),
        features['explicit_valence'].reshape(-1, 1),
        features['is_aromatic'].reshape(-1, 1),
        features['atomic_mass'].reshape(-1, 1),
        features['ring_membership'],
        features['hybridization'],
    ]

    atomic_fp = np.concatenate(fp_components, axis=1)

    return torch.tensor(atomic_fp, dtype=torch.float32)


def batch_to_dgl_graphs(
    molecule_data_list: list,
    include_multipoles: bool = True,
    use_conformer: bool = False
) -> list:
    """Convert batch of MoleculeData to DGL graphs.

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        List of molecular data
    include_multipoles : bool
        Include multipole moments in features
    use_conformer : bool
        Include 3D coordinates in features

    Returns
    -------
    list of dgl.DGLGraph
        DGL graphs ready for training

    Examples
    --------
    >>> graphs = batch_to_dgl_graphs(molecule_data_list)
    >>> from dgl.dataloading import GraphDataLoader
    >>> dataloader = GraphDataLoader(graphs, batch_size=32)
    """
    graphs = []

    for mol_data in molecule_data_list:
        graph = molecule_data_to_dgl_graph(
            mol_data,
            include_multipoles=include_multipoles,
            use_conformer=use_conformer
        )
        graphs.append(graph)

    return graphs


def get_feature_dimensions(
    include_multipoles: bool = True,
    use_conformer: bool = False,
    n_multipole_components: int = 81
) -> int:
    """Calculate total feature dimension for model initialization.

    Parameters
    ----------
    include_multipoles : bool
        Whether multipoles are included
    use_conformer : bool
        Whether conformer coordinates are included
    n_multipole_components : int
        Number of multipole moment components (default: 81 for limit=8)

    Returns
    -------
    int
        Total feature dimension

    Notes
    -----
    Base features:
    - One-hot element encoding: 100
    - Atomic fingerprint: 17
    Total base: 117 (matching espaloma-charge)

    Additional features:
    - Multipole moments: n_multipole_components (default 81)
    - Conformer: 3

    Examples
    --------
    >>> # Without multipoles (baseline)
    >>> get_feature_dimensions(include_multipoles=False)
    117
    >>> # With multipoles
    >>> get_feature_dimensions(include_multipoles=True)
    198
    """
    base_dim = 100 + 17

    if include_multipoles:
        base_dim += n_multipole_components

    if use_conformer:
        base_dim += 3

    return base_dim
