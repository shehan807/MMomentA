#!/usr/bin/env python
"""Convert SMILES file to OpenFF molecules with conformers."""

import pickle
from pathlib import Path
from openff.toolkit import Molecule

def read_smiles_file(smiles_file, max_molecules=None):
    """Read SMILES from file (one per line or space-separated)."""

    smiles_list = []

    with open(smiles_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Handle space-separated SMILES ID format
            parts = line.split()
            smiles = parts[0]

            smiles_list.append(smiles)

            if max_molecules and len(smiles_list) >= max_molecules:
                break

    return smiles_list

def create_molecules(smiles_list):
    """Create OpenFF molecules with conformers."""

    molecules = []

    print(f"Creating molecules from {len(smiles_list)} SMILES...")

    for i, smiles in enumerate(smiles_list):
        try:
            mol = Molecule.from_smiles(smiles)
            mol.generate_conformers(n_conformers=1)
            mol.name = f"mol_{i}_{smiles[:15]}"

            molecules.append(mol)

            if (i + 1) % 50 == 0:
                print(f"  Created {i + 1}/{len(smiles_list)} molecules")

        except Exception as e:
            print(f"  ✗ Failed on {smiles}: {e}")
            continue

    print(f"✓ Created {len(molecules)} molecules")

    return molecules

def save_molecules(molecules, output_path):
    """Save molecules as pickle."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(molecules, f)

    print(f"✓ Saved to {output_path}")

    # Summary
    n_atoms = sum(mol.n_atoms for mol in molecules)
    print(f"\nDataset summary:")
    print(f"  Molecules: {len(molecules)}")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Avg atoms/mol: {n_atoms / len(molecules):.1f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert SMILES to molecules")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input SMILES file (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output pickle file'
    )
    parser.add_argument(
        '--max-molecules',
        type=int,
        default=None,
        help='Maximum number of molecules to process'
    )

    args = parser.parse_args()

    smiles_list = read_smiles_file(args.input, args.max_molecules)
    molecules = create_molecules(smiles_list)
    save_molecules(molecules, args.output)

    print("\nNext step:")
    print(f"python scripts/prepare_dataset.py --input {args.output} --output {{output.h5}} --n-jobs 8")
