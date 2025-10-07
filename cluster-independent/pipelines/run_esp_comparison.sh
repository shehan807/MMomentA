#!/bin/bash
# Complete ESP Validation Comparison Pipeline
# Compares 4 charge methods: RESP, AM1-BCC, MPFIT, MPFIT-GNN (MMomentA)
#
# This script handles environment switching between:
#   - mmomenta-data: QM methods (RESP, AM1-BCC, MPFIT)
#   - mmomenta-ml: ML method (MPFIT-GNN)

MMOMENTA_DIR="${MMOMENTA_DIR:-$(pwd)}"

echo "================================================"
echo "ESP Validation Comparison Pipeline"
echo "All 4 Methods: RESP, AM1-BCC, MPFIT, MPFIT-GNN"
echo "================================================"
echo ""

# Parse command-line arguments
MOLECULES_FILE="${1:-data/test_smiles.pkl}"
TRAINED_MODEL="${2:-runs/test_multipoles/best_model.pt}"
OUTPUT_DIR="${3:-esp_comparison}"
QM_METHOD="${QM_METHOD:-hf}"
QM_BASIS="${QM_BASIS:-6-31G*}"
ENABLE_RESP="${ENABLE_RESP:-true}"
ENABLE_AM1BCC="${ENABLE_AM1BCC:-true}"
ENABLE_MPFIT="${ENABLE_MPFIT:-true}"
ENABLE_MPFIT_GNN="${ENABLE_MPFIT_GNN:-true}"

echo "Configuration:"
echo "  Molecules:    $MOLECULES_FILE"
echo "  Trained model: $TRAINED_MODEL"
echo "  Output dir:    $OUTPUT_DIR"
echo "  QM method:     $QM_METHOD"
echo "  QM basis:      $QM_BASIS"
echo ""
echo "Enabled methods:"
echo "  RESP:        $ENABLE_RESP"
echo "  AM1-BCC:     $ENABLE_AM1BCC"
echo "  MPFIT:       $ENABLE_MPFIT"
echo "  MPFIT-GNN:   $ENABLE_MPFIT_GNN"
echo ""

# Check if files exist
if [ ! -f "$MOLECULES_FILE" ]; then
    echo "✗ Error: Molecules file not found: $MOLECULES_FILE"
    exit 1
fi

if [ "$ENABLE_MPFIT_GNN" = "true" ] && [ ! -f "$TRAINED_MODEL" ]; then
    echo "✗ Error: Trained model not found: $TRAINED_MODEL"
    echo "  Run training first or disable MPFIT-GNN with ENABLE_MPFIT_GNN=false"
    exit 1
fi

# Initialize conda for script
eval "$(conda shell.bash hook)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Run MPFIT-GNN (requires mmomenta-ml environment)
if [ "$ENABLE_MPFIT_GNN" = "true" ]; then
    echo "Step 1/3: Running MPFIT-GNN predictions..."
    echo "================================================"

    module load cuda 2>/dev/null || true
    conda activate mmomenta-ml

    cd "$MMOMENTA_DIR"

    python scripts/validate_esp.py \
        --model "$TRAINED_MODEL" \
        --molecules "$MOLECULES_FILE" \
        --esp-dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/mpfit_gnn_results.json" \
        --device cuda \
        ${VERBOSE:+--verbose}

    MPFIT_GNN_EXIT=$?

    if [ $MPFIT_GNN_EXIT -ne 0 ]; then
        echo "⚠ Warning: MPFIT-GNN prediction failed"
        MPFIT_GNN_RESULTS=""
    else
        echo "✓ MPFIT-GNN predictions complete"
        MPFIT_GNN_RESULTS="--mpfit-gnn-results $OUTPUT_DIR/mpfit_gnn_results.json"
    fi
    echo ""
else
    echo "Step 1/3: Skipping MPFIT-GNN (disabled)"
    echo ""
    MPFIT_GNN_RESULTS=""
fi

# Step 2: Run QM methods (requires mmomenta-data environment)
echo "Step 2/3: Running QM-based methods..."
echo "================================================"

module load anaconda3 2>/dev/null || true
conda activate mmomenta-data

cd "$MMOMENTA_DIR"

# Build command-line arguments
QM_ARGS=""
[ "$ENABLE_RESP" = "false" ] && QM_ARGS="$QM_ARGS --no-resp"
[ "$ENABLE_AM1BCC" = "false" ] && QM_ARGS="$QM_ARGS --no-am1bcc"
[ "$ENABLE_MPFIT" = "false" ] && QM_ARGS="$QM_ARGS --no-mpfit"
[ "$ENABLE_MPFIT_GNN" = "false" ] && QM_ARGS="$QM_ARGS --no-mpfit-gnn"
[ -n "$VERBOSE" ] && QM_ARGS="$QM_ARGS --verbose"

export QM_METHOD="$QM_METHOD"
export QM_BASIS="$QM_BASIS"

python scripts/compare_all_methods.py \
    --molecules "$MOLECULES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --qm-method "$QM_METHOD" \
    --qm-basis "$QM_BASIS" \
    $MPFIT_GNN_RESULTS \
    $QM_ARGS

QM_EXIT=$?

if [ $QM_EXIT -ne 0 ]; then
    echo "✗ Error: QM methods comparison failed"
    exit 1
fi

echo "✓ QM methods complete"
echo ""

# Step 3: Generate final comparison plots (requires lovelyplots in mmomenta-ml)
echo "Step 3/3: Generating publication-quality plots..."
echo "================================================"

conda activate mmomenta-ml

cd "$MMOMENTA_DIR"

python << 'EOF'
import json
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path('figures').resolve()))
from plot_comparison import create_violin_plots

# Load merged results
output_dir = Path('$OUTPUT_DIR')
results_file = output_dir / 'qm_methods_results.json'

if not results_file.exists():
    print(f"✗ Error: Results file not found: {results_file}")
    sys.exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

# Generate plots
plot_file = create_violin_plots(
    results,
    output_dir=str(output_dir),
    qm_method='$QM_METHOD',
    qm_basis='$QM_BASIS'
)

print(f"✓ Plots saved to {plot_file}")
EOF

PLOT_EXIT=$?

if [ $PLOT_EXIT -ne 0 ]; then
    echo "⚠ Warning: Plot generation failed"
else
    echo "✓ Plots generated"
fi
echo ""

# Summary
echo "================================================"
echo "ESP Validation Comparison Complete!"
echo "================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key files:"
echo "  - QM methods results: $OUTPUT_DIR/qm_methods_results.json"
if [ "$ENABLE_MPFIT_GNN" = "true" ] && [ $MPFIT_GNN_EXIT -eq 0 ]; then
echo "  - MPFIT-GNN results:  $OUTPUT_DIR/mpfit_gnn_results.json"
fi
echo "  - Violin plots:       $OUTPUT_DIR/esp_validation_all_methods.png"
echo ""

# Print summary statistics
echo "Summary Statistics:"
echo "-------------------"

python << 'EOF'
import json
import numpy as np
from pathlib import Path

results_file = Path('$OUTPUT_DIR') / 'qm_methods_results.json'

if results_file.exists():
    with open(results_file, 'r') as f:
        results = json.load(f)

    for method in results['mae'].keys():
        if results['mae'][method]:
            mae_values = results['mae'][method]
            rmse_values = results['rmse'][method]
            print(f"\n{method}:")
            print(f"  MAE:  {np.mean(mae_values):.6e} ± {np.std(mae_values):.6e} hartree/e")
            print(f"  RMSE: {np.mean(rmse_values):.6e} ± {np.std(rmse_values):.6e} hartree/e")
            print(f"  N = {len(mae_values)} molecules")
EOF

echo ""
echo "================================================"
echo ""
echo "Usage:"
echo "  Default: ./run_esp_comparison.sh"
echo "  Custom:  ./run_esp_comparison.sh <molecules.pkl> <model.pt> <output_dir>"
echo ""
echo "Environment variables:"
echo "  QM_METHOD=hf          # QM method for ESP (default: hf)"
echo "  QM_BASIS='6-31G*'     # QM basis set (default: 6-31G*)"
echo "  ENABLE_RESP=false     # Disable RESP"
echo "  ENABLE_AM1BCC=false   # Disable AM1-BCC"
echo "  ENABLE_MPFIT=false    # Disable MPFIT"
echo "  ENABLE_MPFIT_GNN=false # Disable MPFIT-GNN"
echo "  VERBOSE=1             # Enable verbose output"
echo ""
