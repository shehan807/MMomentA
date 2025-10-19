#!/usr/bin/env python
"""
Diagnose MPFIT charges in dataset to identify unit/scaling issues.

This script inspects the MPFIT charges stored in the dataset and compares
them to expected physical ranges for partial atomic charges.

Usage:
    python scripts/diagnose_mpfit_charges.py \
        --dataset data/zinc_100k_mpfit.h5 \
        --n-molecules 100

Expected charge ranges for partial charges:
    - Typical: -1.0 to +1.0 e
    - Maximum: -2.0 to +2.0 e (for highly charged species)
    - If outside this range → UNIT/SCALING BUG!
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MMomentA.data.storage import load_dataset_hdf5

# Atomic number to symbol mapping
ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


def main():
    parser = argparse.ArgumentParser(description="Diagnose MPFIT charges in dataset")

    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")
    parser.add_argument("--n-molecules", type=int, default=100,
                       help="Number of molecules to inspect (default: 100)")

    args = parser.parse_args()

    print("="*80)
    print("MPFIT Charge Diagnosis")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Inspecting first {args.n_molecules} molecules...")
    print("")

    # Load dataset
    print("Loading dataset...")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)

    print(f"Loaded {len(molecule_data_list)} molecules")
    print(f"Dataset: {metadata.get('dataset_name', 'Unknown')}")
    print(f"QM method: {metadata.get('qm_method', 'Unknown')}/{metadata.get('qm_basis', 'Unknown')}")
    print("")

    # Collect charges by element
    element_charges = defaultdict(list)
    all_charges = []

    n_inspect = min(args.n_molecules, len(molecule_data_list))

    for i in range(n_inspect):
        mol_data = molecule_data_list[i]

        charges = mol_data.target_charges
        atomic_numbers = mol_data.atomic_features['atomic_numbers']

        # Collect all charges
        all_charges.extend(charges)

        # Group by element
        for charge, z in zip(charges, atomic_numbers):
            element_symbol = ATOMIC_SYMBOLS.get(int(z), f'Z{int(z)}')
            element_charges[element_symbol].append(charge)

    # Convert to numpy arrays
    all_charges = np.array(all_charges)
    for element in element_charges:
        element_charges[element] = np.array(element_charges[element])

    print("="*80)
    print("OVERALL CHARGE STATISTICS")
    print("="*80)
    print(f"Total atoms inspected: {len(all_charges):,}")
    print(f"Charge range: [{all_charges.min():.4f}, {all_charges.max():.4f}] e")
    print(f"Mean charge: {all_charges.mean():.4f} e")
    print(f"Std charge: {all_charges.std():.4f} e")
    print("")

    # Check if charges are in reasonable range
    EXPECTED_MAX = 2.0  # Absolute max for partial charges
    TYPICAL_MAX = 1.0   # Typical max for most molecules

    if abs(all_charges).max() > EXPECTED_MAX:
        print("⚠️  WARNING: Charges exceed expected physical range!")
        print(f"   Expected max: ±{EXPECTED_MAX} e")
        print(f"   Actual max: {all_charges.max():.4f} e, min: {all_charges.min():.4f} e")
        print("")
        print("🔴 LIKELY ISSUE: Unit conversion or scaling error in MPFIT!")
        print("")
    elif abs(all_charges).max() > TYPICAL_MAX:
        print("⚠️  CAUTION: Charges are larger than typical range")
        print(f"   Typical max: ±{TYPICAL_MAX} e")
        print(f"   Actual max: {all_charges.max():.4f} e, min: {all_charges.min():.4f} e")
        print("   (This may be okay for highly charged species)")
        print("")
    else:
        print("✓ Charges are in reasonable physical range")
        print(f"  Max absolute charge: {abs(all_charges).max():.4f} e")
        print("")

    print("="*80)
    print("PER-ELEMENT CHARGE STATISTICS")
    print("="*80)
    print(f"{'Element':<10} {'Count':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'Status'}")
    print("-"*80)

    # Sort by element count
    elements_sorted = sorted(element_charges.keys(),
                            key=lambda e: len(element_charges[e]),
                            reverse=True)

    for element in elements_sorted:
        charges = element_charges[element]
        count = len(charges)
        min_q = charges.min()
        max_q = charges.max()
        mean_q = charges.mean()
        std_q = charges.std()

        # Determine status
        max_abs = max(abs(min_q), abs(max_q))
        if max_abs > EXPECTED_MAX:
            status = "🔴 TOO LARGE"
        elif max_abs > TYPICAL_MAX:
            status = "⚠️  LARGE"
        else:
            status = "✓ OK"

        print(f"{element:<10} {count:<10} {min_q:<12.4f} {max_q:<12.4f} {mean_q:<12.4f} {std_q:<12.4f} {status}")

    print("-"*80)
    print("")

    # Print first few molecules for detailed inspection
    print("="*80)
    print("SAMPLE MOLECULES (first 5)")
    print("="*80)

    for i in range(min(5, n_inspect)):
        mol_data = molecule_data_list[i]
        charges = mol_data.target_charges
        atomic_numbers = mol_data.atomic_features['atomic_numbers']
        smiles = mol_data.smiles

        print(f"\nMolecule {i}: {smiles}")
        print(f"  N atoms: {len(charges)}")
        print(f"  Total charge: {charges.sum():.4f} e")
        print(f"  Atom charges:")

        for j, (q, z) in enumerate(zip(charges, atomic_numbers)):
            element_symbol = ATOMIC_SYMBOLS.get(int(z), f'Z{int(z)}')
            print(f"    Atom {j:2d} ({element_symbol:2s}): {q:+8.4f} e")

    print("")
    print("="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("")

    # Final recommendation
    if abs(all_charges).max() > EXPECTED_MAX:
        print("RECOMMENDATION:")
        print("  1. Check MPFIT calculation in MMomentA/qm/mpfit.py")
        print("  2. Verify charge units from openff-recharge")
        print("  3. Check if multipole moments are being confused with charges")
        print("  4. Test with single molecule (use test_mpfit_single.py)")
        print("")
        print("NEXT STEPS:")
        print("  python scripts/test_mpfit_single.py")
    else:
        print("Charges appear to be correctly scaled!")
        print("If GNN training still has issues, check:")
        print("  - Model architecture")
        print("  - Training hyperparameters")
        print("  - Loss function")


if __name__ == "__main__":
    main()
