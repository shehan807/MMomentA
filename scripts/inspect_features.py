#!/usr/bin/env python
"""
Inspect feature extraction to identify dimension mismatches.

This script analyzes the feature construction pipeline to verify:
1. Feature dimensions match expectations (117 baseline + 81 multipoles = 198)
2. Multipole moments are correctly extracted
3. No missing or extra features

Usage:
    python scripts/inspect_features.py \
        --dataset data/zinc_100k_mpfit.h5 \
        --n-molecules 10

Expected feature breakdown:
    - One-hot element encoding: 100
    - Atomic fingerprint: 17 (degree, valence, aromatic, mass, rings, hybrid, charge)
    - Multipole moments: 81 (limit=8 → 81 components)
    Total: 198 dimensions
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MMomentA.data.storage import load_dataset_hdf5
from MMomentA.data.graph import molecule_data_to_dgl_graph, get_feature_dimensions


def inspect_molecule_features(mol_data, include_multipoles=True):
    """Inspect features for a single molecule."""

    # Build graph
    graph = molecule_data_to_dgl_graph(
        mol_data,
        include_multipoles=include_multipoles,
        use_conformer=False,
        multipole_stats=None  # No normalization for inspection
    )

    h0 = graph.ndata['h0']

    # Extract feature components
    n_atoms = h0.shape[0]
    feature_dim = h0.shape[1]

    # One-hot encoding: first 100 dimensions
    one_hot = h0[:, :100]

    # Atomic fingerprint: next 17 dimensions
    atomic_fp = h0[:, 100:117]

    # Multipoles: remaining dimensions
    if include_multipoles:
        multipoles = h0[:, 117:]
        n_multipole_components = multipoles.shape[1]
    else:
        multipoles = None
        n_multipole_components = 0

    return {
        'n_atoms': n_atoms,
        'feature_dim': feature_dim,
        'one_hot': one_hot,
        'atomic_fp': atomic_fp,
        'multipoles': multipoles,
        'n_multipole_components': n_multipole_components
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect feature extraction")

    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")
    parser.add_argument("--n-molecules", type=int, default=10,
                       help="Number of molecules to inspect (default: 10)")

    args = parser.parse_args()

    print("="*80)
    print("Feature Extraction Inspection")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Inspecting first {args.n_molecules} molecules...")
    print("")

    # Load dataset
    print("Loading dataset...")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)

    print(f"Loaded {len(molecule_data_list)} molecules")
    print(f"Dataset: {metadata.get('dataset_name', 'Unknown')}")
    print("")

    # Expected dimensions
    expected_baseline_dim = get_feature_dimensions(include_multipoles=False)
    expected_multipole_dim = get_feature_dimensions(include_multipoles=True)

    print("="*80)
    print("EXPECTED FEATURE DIMENSIONS")
    print("="*80)
    print(f"Baseline (no multipoles): {expected_baseline_dim}")
    print(f"  - One-hot element encoding: 100")
    print(f"  - Atomic fingerprint: 17")
    print("")
    print(f"With multipoles: {expected_multipole_dim}")
    print(f"  - Baseline: {expected_baseline_dim}")
    print(f"  - Multipole components: {expected_multipole_dim - expected_baseline_dim}")
    print("")

    # Inspect molecules
    print("="*80)
    print("ACTUAL FEATURE DIMENSIONS")
    print("="*80)

    n_inspect = min(args.n_molecules, len(molecule_data_list))

    feature_dims = []
    multipole_component_counts = []

    for i in range(n_inspect):
        mol_data = molecule_data_list[i]

        # Inspect with multipoles
        features = inspect_molecule_features(mol_data, include_multipoles=True)

        feature_dims.append(features['feature_dim'])
        multipole_component_counts.append(features['n_multipole_components'])

        if i < 3:  # Detailed output for first 3
            print(f"\nMolecule {i}: {mol_data.smiles}")
            print(f"  N atoms: {features['n_atoms']}")
            print(f"  Total feature dim: {features['feature_dim']}")
            print(f"  One-hot dim: {features['one_hot'].shape[1]}")
            print(f"  Atomic FP dim: {features['atomic_fp'].shape[1]}")
            print(f"  Multipole dim: {features['n_multipole_components']}")

            # Show multipole moment shape from raw data
            print(f"  Multipole moments (raw): shape {mol_data.multipole_moments.shape}")

            # Sample atomic fingerprint
            print(f"  Sample atomic FP (atom 0): {features['atomic_fp'][0, :5].numpy()}...")

            # Sample multipole values (first atom, first 5 components)
            if features['multipoles'] is not None:
                print(f"  Sample multipoles (atom 0): {features['multipoles'][0, :5].numpy()}...")

    # Summary statistics
    feature_dims = np.array(feature_dims)
    multipole_component_counts = np.array(multipole_component_counts)

    print("")
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Feature dimension across {n_inspect} molecules:")
    print(f"  Min: {feature_dims.min()}")
    print(f"  Max: {feature_dims.max()}")
    print(f"  Unique values: {np.unique(feature_dims)}")
    print("")
    print(f"Multipole components across {n_inspect} molecules:")
    print(f"  Min: {multipole_component_counts.min()}")
    print(f"  Max: {multipole_component_counts.max()}")
    print(f"  Unique values: {np.unique(multipole_component_counts)}")
    print("")

    # Diagnosis
    print("="*80)
    print("DIAGNOSIS")
    print("="*80)

    actual_feature_dim = feature_dims[0]
    actual_multipole_dim = multipole_component_counts[0]

    if actual_feature_dim == expected_multipole_dim:
        print("✓ Feature dimensions match expected values!")
        print(f"  Expected: {expected_multipole_dim}, Actual: {actual_feature_dim}")
    else:
        print("⚠️  Feature dimension mismatch detected!")
        print(f"  Expected: {expected_multipole_dim}")
        print(f"  Actual: {actual_feature_dim}")
        print(f"  Difference: {actual_feature_dim - expected_multipole_dim}")
        print("")

        if actual_multipole_dim != 81:
            print(f"🔴 ISSUE: Multipole components = {actual_multipole_dim} (expected 81)")
            print("")
            print("Possible causes:")
            print("  1. GDMA multipole extraction limit != 8")
            print("  2. Multipole moments not fully extracted")
            print("  3. Bug in feature concatenation")
            print("")
            print("RECOMMENDATION:")
            print("  Check MMomentA/qm/mpfit.py - verify GDMA limit parameter")
            print("  Check MMomentA/data/graph.py:74 - verify multipole concatenation")
        else:
            print("Multipole components are correct (81)")
            print("Issue must be in baseline features or concatenation")

    print("")

    # Verify multipole moments are non-zero
    mol_data = molecule_data_list[0]
    features = inspect_molecule_features(mol_data, include_multipoles=True)

    if features['multipoles'] is not None:
        multipole_values = features['multipoles'].numpy()
        n_nonzero = np.count_nonzero(multipole_values)
        n_total = multipole_values.size

        print("="*80)
        print("MULTIPOLE VALUE CHECK")
        print("="*80)
        print(f"Non-zero multipole values: {n_nonzero}/{n_total} ({100*n_nonzero/n_total:.1f}%)")
        print(f"Multipole range: [{multipole_values.min():.4f}, {multipole_values.max():.4f}]")
        print(f"Multipole mean: {multipole_values.mean():.4f}")
        print(f"Multipole std: {multipole_values.std():.4f}")

        if n_nonzero == 0:
            print("")
            print("🔴 ERROR: All multipole moments are zero!")
            print("   This indicates a feature extraction bug")
        elif multipole_values.std() < 1e-6:
            print("")
            print("⚠️  WARNING: Multipole moments have very low variance")
            print("   This may indicate an extraction or normalization issue")
        else:
            print("")
            print("✓ Multipole moments appear valid (non-zero with variance)")

    print("")
    print("="*80)


if __name__ == "__main__":
    main()
