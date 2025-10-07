#!/usr/bin/env python
"""Create minimal test dataset from SMILES for ML pipeline testing."""

import pickle
from pathlib import Path
import numpy as np
from openff.toolkit import Molecule

# 50 diverse molecules
SMILES = [
    # Alkanes (5)
    'C', 'CC', 'CCC', 'CCCC', 'CC(C)C',
    # Alcohols (5)
    'CO', 'CCO', 'CCCO', 'CC(C)O', 'CC(C)(C)O',
    # Aromatics (5)
    'c1ccccc1', 'c1ccc(C)cc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1', 'c1ccc(F)cc1',
    # Carboxylic acids (5)
    'CC(=O)O', 'CCC(=O)O', 'c1ccc(C(=O)O)cc1', 'CC(C)C(=O)O', 'OC(=O)C(=O)O',
    # Amines (5)
    'CN', 'CCN', 'CCNC', 'c1ccc(N)cc1', 'CC(C)N',
    # Ethers (5)
    'COC', 'CCOC', 'CCOCC', 'c1ccc(OC)cc1', 'COc1ccccc1',
    # Ketones (5)
    'CC(=O)C', 'CCC(=O)C', 'CC(=O)CC', 'c1ccc(C(=O)C)cc1', 'CC(=O)c1ccccc1',
    # Aldehydes (3)
    'CC=O', 'CCC=O', 'c1ccc(C=O)cc1',
    # Esters (4)
    'CC(=O)OC', 'CCC(=O)OC', 'CC(=O)OCC', 'c1ccc(C(=O)OC)cc1',
    # Heterocycles (5)
    'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'C1CCNCC1', 'C1CCOCC1',
    # Nitriles (3)
    'CC#N', 'CCC#N', 'c1ccc(C#N)cc1',
]

def create_test_dataset():
    """Create test molecules with conformers."""

    molecules = []

    print(f"Creating {len(SMILES)} test molecules...")

    for i, smiles in enumerate(SMILES):
        try:
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            mol.generate_conformers(n_conformers=1)

            # Add molecule name for tracking
            mol.name = f"test_mol_{i}_{smiles[:10]}"

            molecules.append(mol)

            if (i + 1) % 10 == 0:
                print(f"  Created {i + 1}/{len(SMILES)} molecules")

        except Exception as e:
            print(f"  ✗ Failed on {smiles}: {e}")
            continue

    print(f"✓ Created {len(molecules)} molecules")

    return molecules

def save_dataset(molecules, output_path):
    """Save molecules as pickle."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(molecules, f)

    print(f"✓ Saved to {output_path}")

    # Print summary
    n_atoms = sum(mol.n_atoms for mol in molecules)
    print(f"\nDataset summary:")
    print(f"  Molecules: {len(molecules)}")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Avg atoms/mol: {n_atoms / len(molecules):.1f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test SMILES dataset")
    parser.add_argument(
        '--output',
        type=str,
        default='data/test_smiles.pkl',
        help='Output pickle file'
    )

    args = parser.parse_args()

    molecules = create_test_dataset()
    save_dataset(molecules, args.output)

    print("\nNext steps:")
    print("1. Activate mmomenta-data: conda activate mmomenta-data")
    print("2. Generate MPFIT dataset:")
    print(f"   python scripts/prepare_dataset.py --input {args.output} --output data/test_mpfit.h5 --n-jobs 8")
    print("3. Train baseline model:")
    print("   conda activate mmomenta-ml")
    print("   python scripts/train_spice.py --dataset data/test_mpfit.h5 --output-dir runs/test_baseline --no-multipoles --device cuda")
