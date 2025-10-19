#!/usr/bin/env python
"""
Quick fix script to generate ESP violin plots from merged results.
Handles missing data gracefully.

Usage:
    cd /global/u1/p/parmar/MoML/MMomentA
    conda activate $SCRATCH/conda-envs/mmomenta-data
    python scripts/fix_and_plot_esp_violins.py
"""

import json
import sys
from pathlib import Path

# Add figures directory to path
figures_dir = Path(__file__).parent.parent / 'figures'
sys.path.insert(0, str(figures_dir))

from plot_comparison import create_esp_violin_plot

# Paths
results_file = Path('/global/u1/p/parmar/MoML/MMomentA/figures/zinc_100k_esp_validation/esp_validation_results.json')
output_dir = Path('/global/u1/p/parmar/MoML/MMomentA/figures/zinc_100k_esp_validation')

print("="*70)
print("Generating ESP Violin Plots")
print("="*70)
print(f"Results file: {results_file}")
print(f"Output directory: {output_dir}")
print("")

# Load merged results
print("Loading merged results...")
with open(results_file, 'r') as f:
    data = json.load(f)

# Extract esp_metrics
esp_metrics_raw = data['esp_metrics']

print(f"Found methods: {list(esp_metrics_raw.keys())}")
print("")

# Fix data structure - ensure all methods have consistent format
esp_metrics = {}
for method in ['mpfit', 'am1bcc', 'resp', 'gnn']:
    if method in esp_metrics_raw and esp_metrics_raw[method]:
        # Check if it has the expected structure
        if 'mae' in esp_metrics_raw[method] and 'rmse' in esp_metrics_raw[method]:
            esp_metrics[method] = esp_metrics_raw[method]
            print(f"  {method}: {len(esp_metrics[method]['mae'])} MAE values, {len(esp_metrics[method]['rmse'])} RMSE values")
        else:
            print(f"  {method}: Skipped (missing data structure)")
    else:
        print(f"  {method}: Skipped (no data)")

print("")

if len(esp_metrics) == 0:
    print("ERROR: No valid method data found!")
    sys.exit(1)

# Generate plots
print("Generating ESP violin plots...")
try:
    mae_file, rmse_file = create_esp_violin_plot(
        esp_metrics,
        output_dir=str(output_dir),
        qm_method='hf',
        qm_basis='6-31G*'
    )

    print("")
    print("="*70)
    print("Success!")
    print("="*70)
    print(f"✓ ESP MAE plot:  {mae_file}")
    print(f"✓ ESP RMSE plot: {rmse_file}")
    print("")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
