#!/usr/bin/env python
"""
Compare molecular property distributions across train/val/test splits.

This script analyzes whether the test set is significantly different
from the validation set, which could explain performance discrepancies.

Usage:
    python scripts/compare_split_distributions.py \
        --dataset data/zinc_100k_mpfit.h5
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


def analyze_split(molecules, split_name):
    """Analyze molecular properties in a split."""

    n_atoms_list = []
    n_heavy_atoms_list = []
    charge_ranges = []
    element_counts = defaultdict(int)
    charge_magnitudes = []

    for mol_data in molecules:
        charges = mol_data.target_charges
        atomic_numbers = mol_data.atomic_features['atomic_numbers']

        n_atoms = len(charges)
        n_heavy = np.sum(atomic_numbers > 1)

        n_atoms_list.append(n_atoms)
        n_heavy_atoms_list.append(n_heavy)
        charge_ranges.append(charges.max() - charges.min())
        charge_magnitudes.extend(np.abs(charges))

        # Count elements
        for z in atomic_numbers:
            element = ATOMIC_SYMBOLS.get(int(z), f'Z{int(z)}')
            element_counts[element] += 1

    return {
        'n_molecules': len(molecules),
        'n_atoms': {
            'mean': np.mean(n_atoms_list),
            'std': np.std(n_atoms_list),
            'min': np.min(n_atoms_list),
            'max': np.max(n_atoms_list),
            'median': np.median(n_atoms_list)
        },
        'n_heavy_atoms': {
            'mean': np.mean(n_heavy_atoms_list),
            'std': np.std(n_heavy_atoms_list),
            'min': np.min(n_heavy_atoms_list),
            'max': np.max(n_heavy_atoms_list),
            'median': np.median(n_heavy_atoms_list)
        },
        'charge_range': {
            'mean': np.mean(charge_ranges),
            'std': np.std(charge_ranges),
            'min': np.min(charge_ranges),
            'max': np.max(charge_ranges),
            'median': np.median(charge_ranges)
        },
        'charge_magnitude': {
            'mean': np.mean(charge_magnitudes),
            'std': np.std(charge_magnitudes),
            'max': np.max(charge_magnitudes),
            'median': np.median(charge_magnitudes)
        },
        'element_counts': dict(element_counts),
        'element_fractions': {
            elem: count / sum(element_counts.values())
            for elem, count in element_counts.items()
        }
    }


def print_comparison(train_stats, val_stats, test_stats):
    """Print comparison of split statistics."""

    print("="*100)
    print("MOLECULAR PROPERTY COMPARISON")
    print("="*100)
    print(f"{'Property':<30} {'Train':<20} {'Validation':<20} {'Test':<20}")
    print("-"*100)

    # Number of molecules
    print(f"{'N molecules':<30} {train_stats['n_molecules']:<20,} "
          f"{val_stats['n_molecules']:<20,} {test_stats['n_molecules']:<20,}")
    print("")

    # Number of atoms
    print("Number of atoms per molecule:")
    print(f"  {'Mean':<28} {train_stats['n_atoms']['mean']:<20.2f} "
          f"{val_stats['n_atoms']['mean']:<20.2f} {test_stats['n_atoms']['mean']:<20.2f}")
    print(f"  {'Std':<28} {train_stats['n_atoms']['std']:<20.2f} "
          f"{val_stats['n_atoms']['std']:<20.2f} {test_stats['n_atoms']['std']:<20.2f}")
    print(f"  {'Median':<28} {train_stats['n_atoms']['median']:<20.2f} "
          f"{val_stats['n_atoms']['median']:<20.2f} {test_stats['n_atoms']['median']:<20.2f}")
    print(f"  {'Min':<28} {train_stats['n_atoms']['min']:<20.0f} "
          f"{val_stats['n_atoms']['min']:<20.0f} {test_stats['n_atoms']['min']:<20.0f}")
    print(f"  {'Max':<28} {train_stats['n_atoms']['max']:<20.0f} "
          f"{val_stats['n_atoms']['max']:<20.0f} {test_stats['n_atoms']['max']:<20.0f}")
    print("")

    # Charge range
    print("Charge range per molecule (max - min):")
    print(f"  {'Mean':<28} {train_stats['charge_range']['mean']:<20.4f} "
          f"{val_stats['charge_range']['mean']:<20.4f} {test_stats['charge_range']['mean']:<20.4f}")
    print(f"  {'Std':<28} {train_stats['charge_range']['std']:<20.4f} "
          f"{val_stats['charge_range']['std']:<20.4f} {test_stats['charge_range']['std']:<20.4f}")
    print(f"  {'Max':<28} {train_stats['charge_range']['max']:<20.4f} "
          f"{val_stats['charge_range']['max']:<20.4f} {test_stats['charge_range']['max']:<20.4f}")
    print("")

    # Charge magnitude
    print("Charge magnitude (|q|):")
    print(f"  {'Mean':<28} {train_stats['charge_magnitude']['mean']:<20.4f} "
          f"{val_stats['charge_magnitude']['mean']:<20.4f} {test_stats['charge_magnitude']['mean']:<20.4f}")
    print(f"  {'Std':<28} {train_stats['charge_magnitude']['std']:<20.4f} "
          f"{val_stats['charge_magnitude']['std']:<20.4f} {test_stats['charge_magnitude']['std']:<20.4f}")
    print(f"  {'Max':<28} {train_stats['charge_magnitude']['max']:<20.4f} "
          f"{val_stats['charge_magnitude']['max']:<20.4f} {test_stats['charge_magnitude']['max']:<20.4f}")
    print("")

    print("-"*100)
    print("")

    # Element distribution
    print("="*100)
    print("ELEMENT DISTRIBUTION (% of total atoms)")
    print("="*100)

    all_elements = set(train_stats['element_fractions'].keys()) | \
                  set(val_stats['element_fractions'].keys()) | \
                  set(test_stats['element_fractions'].keys())

    # Sort by training frequency
    elements_sorted = sorted(all_elements,
                            key=lambda e: train_stats['element_fractions'].get(e, 0),
                            reverse=True)

    print(f"{'Element':<15} {'Train %':<15} {'Val %':<15} {'Test %':<15} {'Diff (Test-Val)':<15}")
    print("-"*100)

    for element in elements_sorted:
        train_frac = train_stats['element_fractions'].get(element, 0) * 100
        val_frac = val_stats['element_fractions'].get(element, 0) * 100
        test_frac = test_stats['element_fractions'].get(element, 0) * 100
        diff = test_frac - val_frac

        status = ""
        if abs(diff) > 2.0:
            status = " ⚠️"

        print(f"{element:<15} {train_frac:<15.2f} {val_frac:<15.2f} {test_frac:<15.2f} {diff:+<15.2f}{status}")

    print("-"*100)
    print("")

    # Analysis
    print("="*100)
    print("ANALYSIS")
    print("="*100)

    # Check for significant differences
    issues = []

    # Check molecule size difference
    val_mean_atoms = val_stats['n_atoms']['mean']
    test_mean_atoms = test_stats['n_atoms']['mean']
    atom_diff_pct = abs(test_mean_atoms - val_mean_atoms) / val_mean_atoms * 100

    if atom_diff_pct > 10:
        issues.append(f"⚠️  Test molecules are {atom_diff_pct:.1f}% different in size from validation")

    # Check charge range difference
    val_mean_range = val_stats['charge_range']['mean']
    test_mean_range = test_stats['charge_range']['mean']
    range_diff_pct = abs(test_mean_range - val_mean_range) / val_mean_range * 100

    if range_diff_pct > 15:
        issues.append(f"⚠️  Test molecules have {range_diff_pct:.1f}% different charge ranges")

    # Check element distribution shifts
    for element in elements_sorted[:5]:  # Top 5 elements
        val_frac = val_stats['element_fractions'].get(element, 0) * 100
        test_frac = test_stats['element_fractions'].get(element, 0) * 100
        diff = abs(test_frac - val_frac)

        if diff > 3.0:
            issues.append(f"⚠️  Element {element} fraction differs by {diff:.1f}% between val and test")

    if issues:
        print("Distribution differences detected:")
        for issue in issues:
            print(f"  {issue}")
        print("")
        print("🔴 CONCLUSION: Test set is significantly different from validation set")
        print("   This explains why model performs well on val (0.08 e) but poorly on test (0.3-0.7 e)")
        print("")
        print("RECOMMENDATION:")
        print("  1. Use random split instead of scaffold split")
        print("  2. Or use stratified sampling to balance molecular properties")
        print("  3. Or train on more diverse dataset (combine SPICE + ZINC)")
    else:
        print("✓ Validation and test sets have similar distributions")
        print("  Distribution mismatch is NOT the issue")
        print("  Look for other causes (feature extraction, model loading, etc.)")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Compare split distributions")

    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")

    args = parser.parse_args()

    print("="*100)
    print("Analyzing Train/Val/Test Split Distributions")
    print("="*100)
    print(f"Dataset: {args.dataset}")
    print("")

    # Load dataset
    print("Loading dataset...")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)
    print(f"  Loaded {len(molecule_data_list)} molecules")

    # Get split indices
    if 'splits' in metadata:
        splits = metadata['splits']

        # Convert molecule IDs to indices
        id_to_idx = {mol.molecule_id: idx for idx, mol in enumerate(molecule_data_list)}

        train_ids = splits.get('train', [])
        val_ids = splits.get('validation', [])
        test_ids = splits.get('test', [])

        train_indices = [id_to_idx[mol_id] for mol_id in train_ids if mol_id in id_to_idx]
        val_indices = [id_to_idx[mol_id] for mol_id in val_ids if mol_id in id_to_idx]
        test_indices = [id_to_idx[mol_id] for mol_id in test_ids if mol_id in id_to_idx]

        train_molecules = [molecule_data_list[i] for i in train_indices]
        val_molecules = [molecule_data_list[i] for i in val_indices]
        test_molecules = [molecule_data_list[i] for i in test_indices]

        print(f"  Train: {len(train_molecules)} molecules")
        print(f"  Val:   {len(val_molecules)} molecules")
        print(f"  Test:  {len(test_molecules)} molecules")
        print("")

        # Analyze each split
        print("Analyzing splits...")
        train_stats = analyze_split(train_molecules, "train")
        val_stats = analyze_split(val_molecules, "validation")
        test_stats = analyze_split(test_molecules, "test")

        print("")
        print_comparison(train_stats, val_stats, test_stats)

    else:
        print("ERROR: No split information in metadata!")
        sys.exit(1)


if __name__ == "__main__":
    main()
