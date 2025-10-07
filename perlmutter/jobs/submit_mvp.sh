#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -J mmomenta_mvp
#SBATCH -o logs/mvp_%j.out
#SBATCH -e logs/mvp_%j.err

# MVP Pipeline for Perlmutter
# Trains baseline and multipole models on 150 test molecules

echo "================================================"
echo "MMomentA MVP Pipeline (Perlmutter)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================"
echo ""

# Load conda module
module load conda

# Set up paths
MMOMENTA_DIR="${MMOMENTA_DIR:-$SLURM_SUBMIT_DIR}"
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

cd "$MMOMENTA_DIR"
mkdir -p logs

# Step 1: Create test dataset (CPU)
echo "Step 1/5: Creating test dataset..."
conda activate "$DATA_ENV"

python scripts/create_test_smiles.py

# Step 2: Generate MPFIT charges (CPU)
echo "Step 2/5: Computing MPFIT charges..."
python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 16

# Step 3: Train baseline model (GPU)
echo "Step 3/5: Training baseline model..."
conda activate "$ML_ENV"

python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_baseline \
    --no-multipoles \
    --n-epochs 100 \
    --device cuda

# Step 4: Train multipole model (GPU)
echo "Step 4/5: Training multipole model..."
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_multipoles \
    --n-epochs 100 \
    --device cuda

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
echo "MVP Pipeline Complete!"
echo "Results saved to runs/test_{baseline,multipoles}/"
