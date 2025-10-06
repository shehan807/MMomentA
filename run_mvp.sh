#!/bin/bash
# Complete MVP Pipeline: Dataset → Training → Results
# Run this on Phoenix HPC cluster

set -e  # Exit on error

MMOMENTA_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA"

echo "================================================"
echo "MMomentA MVP Pipeline"
echo "================================================"
echo ""

# Step 1: Create test SMILES dataset
echo "Step 1/5: Creating test SMILES dataset (50 molecules)..."
module load anaconda3
conda activate mmomenta-ml

cd "$MMOMENTA_DIR"
python scripts/create_test_smiles.py --output data/test_smiles.pkl

echo "✓ Created test_smiles.pkl"
echo ""

# Step 2: Generate MPFIT charges
echo "Step 2/5: Computing MPFIT charges (20-30 min)..."
conda activate mmomenta-data

python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 8

echo "✓ Created test_mpfit.h5"
echo ""

# Step 3: Train baseline model (no multipoles)
echo "Step 3/5: Training baseline model (10-15 min)..."
module load cuda
conda activate mmomenta-ml

python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_baseline \
    --no-multipoles \
    --n-epochs 100 \
    --device cuda

echo "✓ Baseline model trained"
echo ""

# Step 4: Train with multipoles
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

baseline_results = json.load(open('runs/test_baseline/results.json'))
multipole_results = json.load(open('runs/test_multipoles/results.json'))

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print("\nBaseline Model (no multipoles):")
print(f"  Test RMSE: {baseline_results['test_rmse']:.4f}")
print(f"  Test MAE:  {baseline_results['test_mae']:.4f}")
print(f"  Best Val RMSE: {baseline_results['best_val_rmse']:.4f}")

print("\nMultipole Model:")
print(f"  Test RMSE: {multipole_results['test_rmse']:.4f}")
print(f"  Test MAE:  {multipole_results['test_mae']:.4f}")
print(f"  Best Val RMSE: {multipole_results['best_val_rmse']:.4f}")

improvement = ((baseline_results['test_rmse'] - multipole_results['test_rmse']) /
               baseline_results['test_rmse'] * 100)

print(f"\nImprovement: {improvement:+.1f}%")

if improvement > 0:
    print("✓ Multipoles improve charge prediction!")
else:
    print("✗ Multipoles did not improve performance")

print("\nModel files:")
print(f"  Baseline: runs/test_baseline/best_model.pt")
print(f"  Multipoles: runs/test_multipoles/best_model.pt")
print(f"\nTraining logs:")
print(f"  Baseline: runs/test_baseline/training.log")
print(f"  Multipoles: runs/test_multipoles/training.log")
print("="*70)
EOF

echo ""
echo "================================================"
echo "MVP Pipeline Complete!"
echo "================================================"
echo ""
echo "Results saved to:"
echo "  - runs/test_baseline/"
echo "  - runs/test_multipoles/"
echo ""
echo "Total time: ~40-60 minutes"
