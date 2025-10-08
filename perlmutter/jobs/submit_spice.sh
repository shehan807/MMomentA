#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -J mmomenta_spice
#SBATCH -o logs/spice_%j.out
#SBATCH -e logs/spice_%j.err

# SPICE Pipeline for Perlmutter
# Extracts SPICE subset, trains baseline and multipole models

echo "================================================"
echo "MMomentA SPICE Pipeline (Perlmutter)"
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

# Step 1: Extract SPICE subset
echo "Step 1/4: Extracting SPICE subset..."
conda activate "$ML_ENV"

python scripts/extract_spice.py \
    --input data/SPICE-2.0.1.hdf5 \
    --output data/spice_100.pkl \
    --max-molecules 100

# Step 2: Generate MPFIT charges
echo "Step 2/4: Computing MPFIT charges..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset.py \
    --input data/spice_100.pkl \
    --output data/spice_mpfit.h5 \
    --dataset-name "SPICE" \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 16

# Step 3: Train baseline model
echo "Step 3/4: Training baseline model..."
conda activate "$ML_ENV"

python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/spice_baseline \
    --no-multipoles \
    --n-epochs 200 \
    --device cuda

# Step 4: Train multipole model
echo "Step 4/4: Training multipole model..."
python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/spice_multipoles \
    --n-epochs 200 \
    --device cuda

echo ""
echo "SPICE Pipeline Complete!"
echo "Results saved to runs/spice_{baseline,multipoles}/"
