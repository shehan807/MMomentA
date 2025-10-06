#!/bin/bash
# Complete MVP Pipeline: Dataset → Training → Results
# Run this on Phoenix HPC cluster

MMOMENTA_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA"

echo "================================================"
echo "MMomentA MVP Pipeline"
echo "================================================"
echo ""

# Initialize conda for script
eval "$(conda shell.bash hook)"

# Step 1: Create test SMILES dataset
echo "Step 1/5: Creating test SMILES dataset (50 molecules)..."
module load anaconda3
conda activate mmomenta-data  # Needs OpenFF for molecule creation

cd "$MMOMENTA_DIR"
python scripts/create_test_smiles.py --output data/test_smiles.pkl

echo "✓ Created test_smiles.pkl"
echo ""

# Step 2: Generate MPFIT charges (already in mmomenta-data)
echo "Step 2/5: Computing MPFIT charges (20-30 min)..."

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
import os

# Check what files exist
print("\nChecking for result files...")
baseline_dir = Path('runs/test_baseline')
multipole_dir = Path('runs/test_multipoles')

if baseline_dir.exists():
    print(f"\nBaseline directory contents:")
    for f in baseline_dir.iterdir():
        print(f"  {f.name}")
else:
    print("\n✗ Baseline directory doesn't exist - training may have failed")

if multipole_dir.exists():
    print(f"\nMultipole directory contents:")
    for f in multipole_dir.iterdir():
        print(f"  {f.name}")
else:
    print("\n✗ Multipole directory doesn't exist - training may have failed")

# Try to load results
baseline_results_file = baseline_dir / 'training_results.json'
multipole_results_file = multipole_dir / 'training_results.json'

if not baseline_results_file.exists():
    print(f"\n✗ Missing: {baseline_results_file}")
    print("Check training log: runs/test_baseline/training.log")
    exit(1)

if not multipole_results_file.exists():
    print(f"\n✗ Missing: {multipole_results_file}")
    print("Check training log: runs/test_multipoles/training.log")
    exit(1)

baseline_results = json.load(open(baseline_results_file))
multipole_results = json.load(open(multipole_results_file))

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
