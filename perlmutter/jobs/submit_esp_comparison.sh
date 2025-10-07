#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -J mmomenta_esp
#SBATCH -o logs/esp_%j.out
#SBATCH -e logs/esp_%j.err

# ESP Validation Comparison for Perlmutter
# Compares all 4 methods: RESP, AM1-BCC, MPFIT, MPFIT-GNN

echo "================================================"
echo "ESP Validation Comparison (Perlmutter)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================"
echo ""

# Load conda module
module load conda

# Set up paths
MMOMENTA_DIR="${MMOMENTA_DIR:-$SLURM_SUBMIT_DIR}"
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

# Configuration
MOLECULES="${1:-data/test_smiles.pkl}"
MODEL="${2:-runs/test_multipoles/best_model.pt}"
OUTPUT_DIR="${3:-esp_comparison}"

cd "$MMOMENTA_DIR"
mkdir -p logs "$OUTPUT_DIR"

echo "Configuration:"
echo "  Molecules: $MOLECULES"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: MPFIT-GNN predictions (GPU)
echo "Step 1/3: Running MPFIT-GNN predictions..."
conda activate "$ML_ENV"

python scripts/validate_esp.py \
    --model "$MODEL" \
    --molecules "$MOLECULES" \
    --esp-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/mpfit_gnn_results.json" \
    --device cuda

# Step 2: QM methods (CPU - switch to data environment)
echo "Step 2/3: Running QM methods..."
conda activate "$DATA_ENV"

python scripts/compare_all_methods.py \
    --molecules "$MOLECULES" \
    --output-dir "$OUTPUT_DIR" \
    --mpfit-gnn-results "$OUTPUT_DIR/mpfit_gnn_results.json" \
    --qm-method hf \
    --qm-basis "6-31G*"

# Step 3: Generate plots (switch back to ML for lovelyplots)
echo "Step 3/3: Generating plots..."
conda activate "$ML_ENV"

python << EOF
import json
import sys
from pathlib import Path

sys.path.insert(0, 'figures')
from plot_comparison import create_violin_plots

# Load results
results_file = Path('$OUTPUT_DIR') / 'qm_methods_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Generate plots
create_violin_plots(
    results,
    output_dir='$OUTPUT_DIR',
    qm_method='hf',
    qm_basis='6-31G*'
)

print("\\n✓ ESP validation complete!")
print(f"Results in: $OUTPUT_DIR/")
EOF

echo ""
echo "ESP Comparison Complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "  - Violin plots: $OUTPUT_DIR/esp_validation_all_methods.png"
