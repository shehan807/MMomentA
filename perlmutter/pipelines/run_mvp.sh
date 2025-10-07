#!/bin/bash
# MVP Pipeline for Perlmutter
# 150 test molecules: baseline vs multipole comparison

# Set working directory to MMomentA root
MMOMENTA_DIR="${MMOMENTA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "================================================"
echo "MMomentA MVP Pipeline (Perlmutter)"
echo "================================================"
echo "Working directory: $MMOMENTA_DIR"
echo ""

# Load conda
module load conda

# Set environment paths
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

cd "$MMOMENTA_DIR"

# Step 1: Create test SMILES dataset
echo "Step 1/5: Creating test SMILES dataset..."
conda activate "$DATA_ENV"

python scripts/create_test_smiles.py

echo "✓ Test dataset created"
echo ""

# Step 2: Generate MPFIT charges
echo "Step 2/5: Computing MPFIT charges (20-30 min)..."

python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 16

echo "✓ MPFIT charges computed"
echo ""

# Step 3: Train baseline model
echo "Step 3/5: Training baseline model (10-15 min)..."
conda activate "$ML_ENV"

python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_baseline \
    --no-multipoles \
    --n-epochs 100 \
    --device cuda

echo "✓ Baseline trained"
echo ""

# Step 4: Train multipole model
echo "Step 4/5: Training multipole model (10-15 min)..."

python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_multipoles \
    --n-epochs 100 \
    --device cuda

echo "✓ Multipole model trained"
echo ""

# Step 5: Compare results
echo "Step 5/5: Comparing results..."

python << 'EOF'
import json
from pathlib import Path
import numpy as np

baseline = json.load(open('runs/test_baseline/training_results.json'))
multipoles = json.load(open('runs/test_multipoles/training_results.json'))

baseline_test = baseline['results']['test_metrics']
multipole_test = multipoles['results']['test_metrics']

print("\n" + "="*70)
print("RESULTS COMPARISON")
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
echo "================================================"
echo "MVP Pipeline Complete!"
echo "================================================"
echo "Results: runs/test_{baseline,multipoles}/"
