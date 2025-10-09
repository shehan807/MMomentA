#!/usr/bin/env python
"""Diagnostic script to analyze multipole feature statistics."""

import numpy as np
import h5py
from pathlib import Path
import sys

def analyze_multipole_statistics(dataset_path):
    """Analyze multipole moment statistics in dataset."""

    with h5py.File(dataset_path, 'r') as f:
        # Get all multipole moments
        all_multipoles = []

        for mol_id in f.keys():
            if mol_id == '_metadata':
                continue

            mol_group = f[mol_id]
            if 'multipole_moments' in mol_group:
                multipoles = mol_group['multipole_moments'][:]
                all_multipoles.append(multipoles)

        # Stack all multipole moments
        all_multipoles = np.vstack(all_multipoles)

        print("="*70)
        print("MULTIPOLE FEATURE ANALYSIS")
        print("="*70)
        print(f"Dataset: {dataset_path}")
        print(f"Total atoms: {all_multipoles.shape[0]}")
        print(f"Multipole features per atom: {all_multipoles.shape[1]}")
        print()

        # Compute statistics
        means = np.mean(all_multipoles, axis=0)
        stds = np.std(all_multipoles, axis=0)
        mins = np.min(all_multipoles, axis=0)
        maxs = np.max(all_multipoles, axis=0)

        # Component names (assuming limit=8 GDMA)
        component_names = [
            'Q00 (monopole)',
            'Q10 (dipole_z)', 'Q11c (dipole_x)', 'Q11s (dipole_y)',
            'Q20 (quad_zz)', 'Q21c (quad_xz)', 'Q21s (quad_yz)',
            'Q22c (quad_xx-yy)', 'Q22s (quad_xy)',
        ]

        # Add higher order if present
        for i in range(9, all_multipoles.shape[1]):
            component_names.append(f'Component_{i}')

        print(f"{'Component':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Range':>12}")
        print("-"*80)

        for i in range(all_multipoles.shape[1]):
            name = component_names[i] if i < len(component_names) else f"Q{i}"
            print(f"{name:<20} {means[i]:>12.6f} {stds[i]:>12.6f} "
                  f"{mins[i]:>12.6f} {maxs[i]:>12.6f} {maxs[i]-mins[i]:>12.6f}")

        print()
        print("SCALE ANALYSIS")
        print("-"*80)

        # Check if multipoles have wildly different scales
        scale_ratios = maxs / (stds + 1e-8)
        max_scale_ratio = np.max(scale_ratios)
        min_scale_ratio = np.min(scale_ratios)

        print(f"Max scale ratio (max/std): {max_scale_ratio:.2f}")
        print(f"Min scale ratio (max/std): {min_scale_ratio:.2f}")
        print(f"Scale range factor: {max_scale_ratio/min_scale_ratio:.2f}x")

        if max_scale_ratio / min_scale_ratio > 100:
            print()
            print("⚠️  WARNING: Multipole features have very different scales!")
            print("   This can cause training instability and poor performance.")
            print("   Standardization (z-score) is critical.")

        print()
        print("NORMALIZATION CHECK")
        print("-"*80)

        # Simulate standardization
        normalized = (all_multipoles - means) / (stds + 1e-8)
        norm_means = np.mean(normalized, axis=0)
        norm_stds = np.std(normalized, axis=0)

        print(f"After standardization:")
        print(f"  Mean of means: {np.mean(norm_means):.6e} (should be ~0)")
        print(f"  Mean of stds:  {np.mean(norm_stds):.6f} (should be ~1)")
        print(f"  Max |mean|:    {np.max(np.abs(norm_means)):.6e} (should be ~0)")
        print(f"  Max std:       {np.max(norm_stds):.6f} (should be ~1)")
        print(f"  Min std:       {np.min(norm_stds):.6f} (should be ~1)")

        # Check for potential issues
        print()
        print("POTENTIAL ISSUES")
        print("-"*80)

        issues_found = False

        # Check for zero variance
        zero_var = np.where(stds < 1e-6)[0]
        if len(zero_var) > 0:
            print(f"❌ Zero variance features: {zero_var}")
            print(f"   These features will cause division by zero!")
            issues_found = True

        # Check for extreme values
        extreme = np.where(np.abs(maxs) > 1000)[0]
        if len(extreme) > 0:
            print(f"⚠️  Extreme values (|max| > 1000): components {extreme}")
            print(f"   Max values: {maxs[extreme]}")
            issues_found = True

        # Check for skewed distributions
        skewness = (means - mins) / (maxs - mins + 1e-8)
        highly_skewed = np.where((skewness < 0.1) | (skewness > 0.9))[0]
        if len(highly_skewed) > 0:
            print(f"⚠️  Highly skewed distributions: components {highly_skewed}")
            issues_found = True

        if not issues_found:
            print("✓ No major issues detected")

        print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose multipole feature statistics")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")

    args = parser.parse_args()

    analyze_multipole_statistics(args.dataset)
