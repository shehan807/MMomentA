#!/bin/bash
# QM9 Dataset Pipeline for Perlmutter

# Set working directory to MMomentA root
MMOMENTA_DIR="${MMOMENTA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "================================================"
echo "MMomentA QM9 Pipeline (Perlmutter)"
echo "================================================"
echo "Working directory: $MMOMENTA_DIR"
echo ""

# Load conda
module load conda

# Set environment paths
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

cd "$MMOMENTA_DIR"

# Step 1: Download QM9 using PyTorch Geometric
echo "Step 1/5: Downloading QM9 dataset (PyTorch Geometric)..."
conda activate "$ML_ENV"

python scripts/download_pyg_dataset.py \
    --dataset qm9 \
    --output data/qm9_molecules.pkl \
    --max-molecules 1000

echo "✓ QM9 molecules downloaded"
echo ""

# Step 2: Generate MPFIT charges (with caching)
echo "Step 2/5: Computing MPFIT charges for QM9 (cached if previously run)..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset_cached.py \
    --input data/qm9_molecules.pkl \
    --output data/qm9_mpfit.h5 \
    --dataset-name "QM9" \
    --split-strategy scaffold \
    --train-frac 0.8 \
    --val-frac 0.1 \
    --test-frac 0.1 \
    --n-jobs -1

echo "✓ QM9 MPFIT dataset created"
echo ""

# Step 3: Train baseline model (no multipoles)
echo "Step 3/5: Training baseline model (1000 epochs)..."
conda activate "$ML_ENV"

python scripts/train_spice.py \
    --dataset data/qm9_mpfit.h5 \
    --output-dir runs/qm9_baseline \
    --no-multipoles \
    --n-epochs 1000 \
    --device cuda

echo "✓ QM9 baseline model trained"
echo ""

# Step 4: Train multipole model
echo "Step 4/5: Training multipole model (1000 epochs)..."

python scripts/train_spice.py \
    --dataset data/qm9_mpfit.h5 \
    --output-dir runs/qm9_multipoles \
    --n-epochs 1000 \
    --device cuda

echo "✓ QM9 multipole model trained"
echo ""

# Step 5: Compare results
echo "Step 5/6: Comparing training results..."

python << 'EOF'
import json

baseline = json.load(open('runs/qm9_baseline/training_results.json'))
multipole = json.load(open('runs/qm9_multipoles/training_results.json'))

baseline_test = baseline['results']['test_metrics']
multipole_test = multipole['results']['test_metrics']

print("\n" + "="*70)
print("TRAINING RESULTS COMPARISON")
print("="*70)

print("\nBaseline Model (no multipoles):")
print(f"  Test RMSE: {baseline_test['val_rmse']:.4f}")
print(f"  Test MAE:  {baseline_test['val_mae']:.4f}")

print("\nMultipole Model:")
print(f"  Test RMSE: {multipole_test['val_rmse']:.4f}")
print(f"  Test MAE:  {multipole_test['val_mae']:.4f}")

improvement = ((baseline_test['val_rmse'] - multipole_test['val_rmse']) /
               baseline_test['val_rmse'] * 100)

print(f"\nImprovement: {improvement:+.1f}%")
print("="*70)
EOF

echo ""

# Step 6: Validating ESP reproduction (MPFIT, AM1-BCC, RESP, MMomentA-GNN)
echo "Step 6/6: Validating ESP reproduction (MPFIT vs MMomentA-GNN)..."
conda activate "$DATA_ENV"

mkdir -p figures/qm9_esp_validation

python scripts/validate_esp_comparison.py \
    --dataset data/qm9_mpfit.h5 \
    --model-dir runs/qm9_multipoles \
    --output-dir figures/qm9_esp_validation \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --device cpu \
    --n-jobs 8

echo ""
echo "================================================"
echo "QM9 Pipeline Complete!"
echo "================================================"
echo "Training results: runs/qm9_{baseline,multipoles}/"
echo "ESP validation plots:"
echo "  - ESP violin plot: figures/qm9_esp_validation/esp_validation_all_methods.png"
echo "  - Carbon hexbin:   figures/qm9_esp_validation/carbon_charge_hexbin.png"
echo "Validation metrics: figures/qm9_esp_validation/esp_validation_results.json"
echo ""
