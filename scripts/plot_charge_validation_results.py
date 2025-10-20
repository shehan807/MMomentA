#!/usr/bin/env python
"""
Generate element-wise charge comparison plots from validation results.

Usage:
    python scripts/plot_charge_validation_results.py \
        --results figures/zinc_100k_charge_validation/charge_validation_test.json \
        --output-dir figures/zinc_100k_charge_validation/plots
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import plotting function
sys.path.insert(0, str(Path(__file__).parent.parent / 'figures'))
from plot_comparison import create_element_hexbin_plots


def main():
    parser = argparse.ArgumentParser(
        description="Plot element-wise charge comparison from validation results"
    )

    parser.add_argument("--results", type=str, required=True,
                       help="Path to validation results JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for plots")

    args = parser.parse_args()

    print("="*80)
    print("Generating Element-Wise Charge Comparison Plots")
    print("="*80)
    print(f"Results: {args.results}")
    print(f"Output: {args.output_dir}")
    print("")

    # Load results
    print("Loading results...")
    with open(args.results, 'r') as f:
        results = json.load(f)

    # Reconstruct element_charges dictionary
    print("Organizing charges by element...")

    mpfit_charges = np.array(results['charges']['mpfit'])
    gnn_charges = np.array(results['charges']['gnn'])
    element_symbols = np.array(results['charges']['element_symbols'])

    element_charges = defaultdict(lambda: {'mpfit': [], 'gnn': []})

    for mpfit_q, gnn_q, element in zip(mpfit_charges, gnn_charges, element_symbols):
        element_charges[element]['mpfit'].append(mpfit_q)
        element_charges[element]['gnn'].append(gnn_q)

    # Convert to numpy arrays
    for element in element_charges:
        element_charges[element]['mpfit'] = np.array(element_charges[element]['mpfit'])
        element_charges[element]['gnn'] = np.array(element_charges[element]['gnn'])

    print(f"  Found {len(element_charges)} elements")
    for element in sorted(element_charges.keys(),
                         key=lambda e: len(element_charges[e]['mpfit']),
                         reverse=True):
        count = len(element_charges[element]['mpfit'])
        print(f"    {element}: {count:,} atoms")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    try:
        create_element_hexbin_plots(
            element_charges,
            output_dir=str(output_dir),
            method_name="GNN"
        )

        print("\n" + "="*80)
        print("SUCCESS")
        print("="*80)
        print(f"Plots saved to: {output_dir}")
        print("")
        print("Generated files:")
        for plot_file in sorted(output_dir.glob("*.png")):
            print(f"  - {plot_file.name}")

    except Exception as e:
        print(f"\nERROR: Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
