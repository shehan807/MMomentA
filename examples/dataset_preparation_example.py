#!/usr/bin/env python
"""Example: Prepare dataset for ML training.

This demonstrates the complete workflow for preparing ML training data:
1. Compute MPFIT charges
2. Extract features
3. Create dataset with splits
4. Save for training
"""

from openff.toolkit.topology import Molecule
from MMomentA.qm import MPFITCalculator, GDMAConfig
from MMomentA.data import (
    extract_molecule_data,
    scaffold_split,
    create_split_datasets,
    save_dataset,
    DatasetMetadata
)
from datetime import datetime

smiles_list = [
    "CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1", "CC(C)O",
    "c1cc[nH]n1", "CC(=O)NC", "c1ccc(O)cc1", "CCN(CC)CC", "c1cncnc1"
]

molecules = [Molecule.from_smiles(s) for s in smiles_list]

for mol in molecules:
    mol.generate_conformers(n_conformers=1)

print("Computing MPFIT charges...")
config = GDMAConfig(method="hf", basis="6-31G*", limit=8)
calculator = MPFITCalculator(config)

molecule_data_list = []

for mol in molecules:
    result = calculator.compute(mol)
    if result.success:
        mol_data = extract_molecule_data(mol, result)
        molecule_data_list.append(mol_data)

print(f"Extracted data for {len(molecule_data_list)} molecules")

print("Splitting dataset...")
train_idx, val_idx, test_idx = scaffold_split(molecules)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

split_data = create_split_datasets(
    molecule_data_list,
    train_idx,
    val_idx,
    test_idx,
    include_multipoles=True
)

print(split_data)

print("\nCreating data loaders...")
train_loader = split_data.get_train_loader(batch_size=4)
val_loader = split_data.get_val_loader(batch_size=4)

print(f"Feature dimension: {split_data.train_dataset.feature_dim}")

print("\nSaving dataset...")
metadata = DatasetMetadata(
    dataset_name="Example",
    n_molecules=len(molecule_data_list),
    qm_method="hf",
    qm_basis="6-31G*",
    multipole_limit=8,
    created_date=datetime.now().isoformat(),
    split_strategy="scaffold"
)

save_dataset(molecule_data_list, "example_dataset.h5", metadata)
print("Dataset saved to example_dataset.h5")

print("\nIterating through batches...")
for batch_idx, batch_graph in enumerate(train_loader):
    print(f"Batch {batch_idx}: {batch_graph.batch_size} molecules")
    print(f"  Node features shape: {batch_graph.ndata['h0'].shape}")
    print(f"  Target charges shape: {batch_graph.ndata['q_ref'].shape}")
    if batch_idx >= 1:
        break
