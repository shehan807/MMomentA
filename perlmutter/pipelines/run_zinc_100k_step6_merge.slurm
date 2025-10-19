#!/bin/bash
#SBATCH -J zinc_esp_merge
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 04:00:00
#SBATCH -A m1266
#SBATCH -o logs/zinc_100k_esp_merge_%j.out
#SBATCH -e logs/zinc_100k_esp_merge_%j.err

# Merge ESP Validation Results and Generate Visualizations
#
# This job runs after all chunk jobs complete (via SLURM dependency)
# It merges results, computes per-element statistics, and generates plots

# Set working directory
MMOMENTA_DIR="/global/u1/p/parmar/MoML/MMomentA"
cd "$MMOMENTA_DIR"

echo "================================================"
echo "ESP Validation Results Merge"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "================================================"
echo ""

# Configuration
OUTPUT_DIR="${MMOMENTA_DIR}/figures/zinc_100k_esp_validation"
N_CHUNKS=20

# Load conda
module load conda

# Activate data environment (has matplotlib, numpy, etc.)
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
conda activate "$DATA_ENV"

# Step 1: Merge chunk results
echo "Step 1: Merging chunk results..."
echo "  Output directory: $OUTPUT_DIR"
echo "  Expected chunks: $N_CHUNKS"
echo ""

python scripts/merge_esp_validation_results.py \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-chunks $N_CHUNKS

if [ $? -ne 0 ]; then
    echo "ERROR: Merge failed!"
    exit 1
fi

echo ""
echo "✓ Merge complete"
echo ""

# Step 2: Generate visualizations
echo "Step 2: Generating per-element visualizations..."
echo ""

python scripts/plot_element_charge_comparison.py \
    --data-file "${OUTPUT_DIR}/element_charge_data.npz" \
    --results-file "${OUTPUT_DIR}/esp_validation_results.json" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization generation failed!"
    exit 1
fi

echo ""
echo "✓ Visualizations complete"
echo ""

# Step 3: Generate ESP violin plots (from original pipeline)
echo "Step 3: Generating ESP validation violin plots..."
echo ""

# Load results and create ESP plots using inline Python
python << 'EOF'
import json
import sys
from pathlib import Path

# Add figures directory to path
sys.path.insert(0, str(Path.cwd() / 'figures'))

try:
    from plot_comparison import create_esp_violin_plot

    # Load merged results
    with open('figures/zinc_100k_esp_validation/esp_validation_results.json', 'r') as f:
        data = json.load(f)

    results = data['esp_metrics']
    output_dir = Path('figures/zinc_100k_esp_validation')

    # Generate ESP violin plots
    mae_file, rmse_file = create_esp_violin_plot(
        results,
        output_dir=output_dir,
        qm_method='hf',
        qm_basis='6-31G*'
    )

    print(f"✓ ESP MAE plot saved to {mae_file}")
    print(f"✓ ESP RMSE plot saved to {rmse_file}")

except ImportError as e:
    print(f"WARNING: Could not generate ESP violin plots: {e}")
    print("This is optional - main results are still available")

except Exception as e:
    print(f"WARNING: ESP violin plot generation failed: {e}")
    print("This is optional - main results are still available")
EOF

echo ""

# Print summary
echo "================================================"
echo "ESP Validation Results Merge Complete!"
echo "================================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Time: $SECONDS seconds ($((SECONDS/3600))h $((SECONDS%3600/60))m)"
echo ""
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "Main results:"
echo "  - Merged results:           esp_validation_results.json"
echo "  - Per-element charge data:  element_charge_data.npz"
echo ""
echo "Visualizations:"
echo "  - All-element hexbin:       element_wise_comparison/all_elements_hexbin.png"
echo "  - Element grid hexbin:      element_wise_comparison/element_grid_hexbin.png"
echo "  - RMSE/MAE bar chart:       element_wise_comparison/per_element_rmse_mae.png"
echo "  - Error distributions:      element_wise_comparison/element_error_distributions.png"
echo "  - Individual hexbins:       element_wise_comparison/individual/*.png"
echo "  - ESP validation plots:     esp_validation_mae.png, esp_validation_rmse.png"
echo ""
echo "Summary statistics:"
python << 'EOF'
import json

with open('figures/zinc_100k_esp_validation/esp_validation_results.json', 'r') as f:
    data = json.load(f)

config = data['config']
summary = data['summary']

print(f"  Total molecules:     {config['n_molecules']}")
print(f"  Successful:          {config['successful']}")
print(f"  Failed:              {config['failed']}")
print(f"")
print(f"  GNN ESP RMSE:        {summary['gnn']['rmse_mean']:.6e} ± {summary['gnn']['rmse_std']:.6e}")
print(f"  GNN ESP MAE:         {summary['gnn']['mae_mean']:.6e} ± {summary['gnn']['mae_std']:.6e}")
print(f"  GNN inference time:  {summary['gnn']['time_mean']:.3f} ± {summary['gnn']['time_std']:.3f} s/mol")

if 'mpfit' in summary and summary['mpfit']['time_mean']:
    speedup = summary['mpfit']['time_mean'] / summary['gnn']['time_mean']
    print(f"  Speedup over MPFIT:  {speedup:.1f}x")
EOF

echo ""
echo "Done!"
echo ""
