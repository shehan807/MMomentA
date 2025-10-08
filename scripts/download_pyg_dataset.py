#!/usr/bin/env python
"""Download molecular dataset using PyTorch Geometric."""

import argparse
import pickle
from pathlib import Path
import numpy as np


def download_qm9(output_path: str, max_molecules: int = None):
    """Download QM9 dataset and convert to OpenFF molecules."""
    from torch_geometric.datasets import QM9
    from openff.toolkit import Molecule
    from openff.units import unit

    print("Downloading QM9 dataset...")
    dataset = QM9(root='data/qm9_raw')

    molecules = []
    max_molecules = max_molecules or len(dataset)

    print(f"Converting {max_molecules} molecules to OpenFF format...")

    for i, data in enumerate(dataset[:max_molecules]):
        if i % 100 == 0:
            print(f"  Progress: {i}/{max_molecules}")

        try:
            # QM9 provides SMILES in newer versions
            if hasattr(data, 'smiles'):
                smiles = data.smiles
            elif hasattr(data, 'smile'):
                smiles = data.smile
            else:
                # Skip if no SMILES available
                print(f"  Skipping molecule {i}: no SMILES")
                continue

            # Create molecule from SMILES
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

            # Add 3D coordinates from QM9 dataset
            if hasattr(data, 'pos') and data.pos is not None:
                positions = data.pos.numpy()  # Shape: (n_atoms, 3) in Angstrom
                mol.add_conformer(positions * unit.angstrom)
            else:
                # Generate conformer if not provided
                mol.generate_conformers(n_conformers=1)

            molecules.append(mol)

        except Exception as e:
            print(f"  Failed on molecule {i}: {e}")
            continue

    # Save to pickle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(molecules, f)

    print(f"\n✓ Saved {len(molecules)} molecules to {output_path}")
    return len(molecules)


def download_zinc(output_path: str, max_molecules: int = None, split: str = 'train'):
    """Download ZINC dataset and convert to OpenFF molecules.

    Note: PyG ZINC dataset stores molecules as graphs, not SMILES.
    We reconstruct SMILES from atomic numbers and bonds.
    """
    from torch_geometric.datasets import ZINC
    from openff.toolkit import Molecule
    from openff.units import unit
    from rdkit import Chem

    print(f"Downloading ZINC dataset ({split} split)...")
    dataset = ZINC(root='data/zinc_raw', subset=True, split=split)

    molecules = []
    max_molecules = max_molecules or len(dataset)

    print(f"Converting {max_molecules} ZINC molecules from graphs to SMILES...")

    for i, data in enumerate(dataset[:max_molecules]):
        if i % 100 == 0:
            print(f"  Progress: {i}/{max_molecules}")

        try:
            # ZINC provides graph representation: atomic numbers (x) and edge indices
            atomic_nums = data.x[:, 0].long().numpy()  # First column is atomic number
            edge_index = data.edge_index.numpy()

            # Reconstruct RDKit molecule from graph
            mol_rdkit = Chem.RWMol()

            # Add atoms
            for atomic_num in atomic_nums:
                atom = Chem.Atom(int(atomic_num))
                mol_rdkit.AddAtom(atom)

            # Add bonds (edge_index has both directions, so only add once)
            added_bonds = set()
            for j in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, j]), int(edge_index[1, j])
                if (src, dst) not in added_bonds and (dst, src) not in added_bonds:
                    mol_rdkit.AddBond(src, dst, Chem.BondType.SINGLE)
                    added_bonds.add((src, dst))

            # Sanitize and get SMILES
            mol_rdkit = mol_rdkit.GetMol()
            Chem.SanitizeMol(mol_rdkit)
            smiles = Chem.MolToSmiles(mol_rdkit)

            # Print reconstructed SMILES periodically
            if i % 100 == 0:
                print(f"  Reconstructed SMILES: {smiles}")

            # Create OpenFF molecule from SMILES
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

            # Add 3D coordinates if available
            if hasattr(data, 'pos') and data.pos is not None:
                positions = data.pos.numpy()
                mol.add_conformer(positions * unit.angstrom)
            else:
                # Generate conformer
                mol.generate_conformers(n_conformers=1)

            molecules.append(mol)

        except Exception as e:
            print(f"  Failed on molecule {i}: {e}")
            continue

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(molecules, f)

    print(f"\n✓ Saved {len(molecules)} molecules to {output_path}")
    return len(molecules)


def main():
    parser = argparse.ArgumentParser(
        description="Download molecular datasets using PyTorch Geometric"
    )
    parser.add_argument("--dataset", choices=['qm9', 'zinc'], required=True,
                       help="Dataset to download")
    parser.add_argument("--output", type=str, required=True,
                       help="Output pickle file path")
    parser.add_argument("--max-molecules", type=int, default=1000,
                       help="Maximum number of molecules to download")
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='train',
                       help="ZINC split (ignored for QM9)")

    args = parser.parse_args()

    print("="*70)
    print("PyTorch Geometric Dataset Download")
    print("="*70)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Max molecules: {args.max_molecules}")
    if args.dataset == 'zinc':
        print(f"Split: {args.split}")
    print(f"Output: {args.output}")
    print("="*70)
    print()

    if args.dataset == 'qm9':
        download_qm9(args.output, args.max_molecules)
    elif args.dataset == 'zinc':
        download_zinc(args.output, args.max_molecules, args.split)

    print()
    print("="*70)
    print("Download Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print(f"1. python scripts/prepare_dataset_cached.py --input {args.output} --output data/{args.dataset}_mpfit.h5")
    print(f"2. python scripts/train_spice.py --dataset data/{args.dataset}_mpfit.h5 --output-dir runs/{args.dataset}")
    print()


if __name__ == "__main__":
    main()
