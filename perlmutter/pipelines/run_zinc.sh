#!/bin/bash
# ZINC Dataset Pipeline for Perlmutter
# Same workflow as MVP, but with 2000 ZINC molecules

# Set working directory to MMomentA root
MMOMENTA_DIR="${MMOMENTA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "================================================"
echo "MMomentA ZINC 2k Pipeline (Perlmutter)"
echo "================================================"
echo "Working directory: $MMOMENTA_DIR"
echo ""

# Load conda
module load conda

# Set environment paths
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

cd "$MMOMENTA_DIR"

# Step 1: Download ZINC using PyTorch Geometric
echo "Step 1/6: Downloading ZINC dataset (PyTorch Geometric)..."
conda activate "$ML_ENV"

# Download ZINC using PyG (2000 molecules from train split)
python scripts/download_pyg_dataset.py \
    --dataset zinc \
    --output data/zinc_2k_molecules.pkl \
    --max-molecules 2000 \
    --split train

echo "✓ ZINC molecules downloaded"
echo ""

# Step 2: Generate MPFIT charges (with caching)
echo "Step 2/6: Computing MPFIT charges for ZINC (cached if previously run)..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset_cached.py \
    --input data/zinc_2k_molecules.pkl \
    --output data/zinc_2k_mpfit.h5 \
    --dataset-name "ZINC_2k" \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs -1

echo "✓ ZINC MPFIT dataset created"
echo ""

# Step 3: Train baseline model (no multipoles)
echo "Step 3/6: Training baseline model (5000 epochs, espaloma-charge settings)..."
conda activate "$ML_ENV"

if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_2k_baseline/checkpoints/best_model.pt" ]; then
    echo "✓ ZINC 2k baseline model already trained (found checkpoint), skipping..."
else
    # Baseline: espaloma-charge settings (width=32, input=128, 5000 epochs)
    python scripts/train_spice.py \
        --dataset data/zinc_2k_mpfit.h5 \
        --output-dir runs/zinc_2k_baseline \
        --no-multipoles \
        --n-epochs 5000 \
        --width 32 \
        --early-stopping-patience 2000 \
        --device cuda
    echo "✓ ZINC 2k baseline model trained"
fi
echo ""

# Step 4: Train multipole model (with automatic width scaling)
echo "Step 4/6: Training multipole model (5000 epochs, auto-scaled capacity)..."

if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_2k_multipoles/checkpoints/best_model.pt" ]; then
    echo "✓ ZINC 2k multipole model already trained (found checkpoint), skipping..."
else
    # Multipole: 198 features → auto-scales width from 32 to 54 (198/117 * 32)
    # This maintains the same compression ratio as baseline
    python scripts/train_spice.py \
        --dataset data/zinc_2k_mpfit.h5 \
        --output-dir runs/zinc_2k_multipoles \
        --n-epochs 5000 \
        --width 32 \
        --early-stopping-patience 2000 \
        --device cuda
    echo "✓ ZINC 2k multipole model trained"
fi
echo ""

# Step 5: Compare training results
echo "Step 5/6: Comparing training results..."

python << 'EOF'
import json

baseline = json.load(open('runs/zinc_2k_baseline/training_results.json'))
multipole = json.load(open('runs/zinc_2k_multipoles/training_results.json'))

baseline_test = baseline['results']['test_metrics']
multipole_test = multipole['results']['test_metrics']

print("\n" + "="*70)
print("TRAINING RESULTS COMPARISON")
print("="*70)

print("\nBaseline Model (no multipoles, width=32):")
print(f"  Test RMSE: {baseline_test['val_rmse']:.4f}")
print(f"  Test MAE:  {baseline_test['val_mae']:.4f}")

print("\nMultipole Model (auto-scaled width):")
print(f"  Test RMSE: {multipole_test['val_rmse']:.4f}")
print(f"  Test MAE:  {multipole_test['val_mae']:.4f}")

improvement = ((baseline_test['val_rmse'] - multipole_test['val_rmse']) /
               baseline_test['val_rmse'] * 100)

print(f"\nImprovement: {improvement:+.1f}%")
print("="*70)
EOF

echo ""

# Step 6: Validating ESP reproduction (MPFIT, AM1-BCC, RESP, MMomentA-GNN)
echo "Step 6/6: Validating ESP reproduction (using multipole model)..."
conda activate "$DATA_ENV"

mkdir -p figures/zinc_2k_esp_validation

python scripts/validate_esp_comparison.py \
    --dataset data/zinc_2k_mpfit.h5 \
    --model-dir runs/zinc_2k_multipoles \
    --output-dir figures/zinc_2k_esp_validation \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --device cpu \
    --n-jobs 32

echo ""
echo "================================================"
echo "ZINC 2k Pipeline Complete!"
echo "================================================"
echo "Training results:"
echo "  - Baseline:  runs/zinc_2k_baseline/"
echo "  - Multipole: runs/zinc_2k_multipoles/"
echo "ESP validation plots:"
echo "  - ESP MAE plot:  figures/zinc_2k_esp_validation/esp_validation_mae.png"
echo "  - ESP RMSE plot: figures/zinc_2k_esp_validation/esp_validation_rmse.png"
echo "  - Carbon hexbin: figures/zinc_2k_esp_validation/carbon_charge_hexbin.png"
echo "Validation metrics: figures/zinc_2k_esp_validation/esp_validation_results.json"
echo ""
