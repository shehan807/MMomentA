#!/usr/bin/env python
"""
Regenerate ESP violin plots with 95% CI clipping.

Usage:
    python scripts/replot_esp_violins.py \
        --results figures/esp_validation/esp_validation_results.json \
        --output-dir figures/esp_validation_clipped
"""

import argparse
import json
import sys
from pathlib import Path

# Add figures directory to path
figures_dir = Path(__file__).parent.parent / 'figures'
sys.path.insert(0, str(figures_dir))

from plot_comparison import create_esp_violin_plot


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate ESP violin plots with 95% CI clipping"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to ESP validation results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for plots"
    )
    parser.add_argument(
        "--qm-method",
        type=str,
        default="hf",
        help="QM method used (default: hf)"
    )
    parser.add_argument(
        "--qm-basis",
        type=str,
        default="6-31G*",
        help="QM basis set used (default: 6-31G*)"
    )

    args = parser.parse_args()

    results_file = Path(args.results)
    output_dir = Path(args.output_dir)

    print("="*70)
    print("Regenerating ESP Violin Plots (95% CI Clipped)")
    print("="*70)
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}")
    print("")

    # Check file exists
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Try to extract esp_metrics (handle different file formats)
    if 'esp_metrics' in data:
        esp_metrics = data['esp_metrics']
    elif 'results' in data:
        esp_metrics = data['results']
    else:
        print(f"ERROR: Cannot find 'esp_metrics' or 'results' in JSON file")
        print(f"Available keys: {list(data.keys())}")
        sys.exit(1)

    print(f"Found methods: {list(esp_metrics.keys())}")

    # Check data structure and filter to only valid methods
    valid_metrics = {}
    for method in ['mpfit', 'am1bcc', 'resp', 'gnn']:
        if method in esp_metrics and esp_metrics[method]:
            if isinstance(esp_metrics[method], dict) and 'mae' in esp_metrics[method] and 'rmse' in esp_metrics[method]:
                n_mae = len(esp_metrics[method]['mae'])
                n_rmse = len(esp_metrics[method]['rmse'])
                print(f"  {method}: {n_mae} MAE values, {n_rmse} RMSE values")
                valid_metrics[method] = esp_metrics[method]
            else:
                print(f"  {method}: Skipped (missing 'mae' or 'rmse' data)")
        else:
            print(f"  {method}: Skipped (no data)")

    print("")

    if len(valid_metrics) == 0:
        print("ERROR: No valid method data found!")
        print("Expected format: {'method': {'mae': [...], 'rmse': [...]}}")
        sys.exit(1)

    # Generate plots with 95% CI clipping (only valid methods)
    print("Generating ESP violin plots (95% CI clipped)...")
    try:
        mae_file, rmse_file = create_esp_violin_plot(
            valid_metrics,
            output_dir=str(output_dir),
            qm_method=args.qm_method,
            qm_basis=args.qm_basis
        )

        print("")
        print("="*70)
        print("Success!")
        print("="*70)
        print(f"✓ ESP MAE plot:  {mae_file}")
        print(f"✓ ESP RMSE plot: {rmse_file}")
        print("")
        print("Plots now show data clipped to 95% CI (2.5th-97.5th percentile)")
        print("to remove extreme outliers for clearer visualization.")
        print("")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
