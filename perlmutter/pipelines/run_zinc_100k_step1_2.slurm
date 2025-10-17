#!/bin/bash
#SBATCH -J zinc_100k_s12
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --mem=256GB
#SBATCH -t 48:00:00
#SBATCH -A m1266
#SBATCH -o logs/zinc_100k_step1_2_%j.out
#SBATCH -e logs/zinc_100k_step1_2_%j.err

# ZINC 100k Dataset Pipeline - Steps 1-2 Only
# Data preparation: Download and MPFIT charge generation
#
# Estimated time: ~42-45 hours on 1 CPU node (128 CPU cores)
# Time breakdown (100,000 molecules):
#   Step 1: Download - ~10-15 min
#   Step 2: MPFIT charges - ~42 hours (64 cores parallel, GDMA is expensive)
#   Total: ~42-45 hours
#
# Note: This only performs data preparation (steps 1-2).
#       Run separate training/validation jobs (steps 3-6) after this completes.

# Set working directory to MMomentA root
MMOMENTA_DIR="/global/u1/p/parmar/MoML/MMomentA"

cd "$MMOMENTA_DIR"
mkdir -p logs

echo "================================================"
echo "MMomentA ZINC 100k Pipeline - Steps 1-2"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "================================================"
echo "Working directory: $MMOMENTA_DIR"
echo ""

# Load conda
module load conda

# Set environment paths
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

# Step 1: Download ZINC using PyTorch Geometric
echo "Step 1/2: Downloading ZINC dataset (PyTorch Geometric)..."
conda activate "$ML_ENV"

# Download ZINC using PyG (100,000 molecules from train split)
python scripts/download_pyg_dataset.py \
    --dataset zinc \
    --output data/zinc_100k_molecules.pkl \
    --max-molecules 100000 \
    --split train

echo "✓ ZINC molecules downloaded"
echo ""

# Step 2: Generate MPFIT charges (with caching)
echo "Step 2/2: Computing MPFIT charges for ZINC (cached if previously run)..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset_cached.py \
    --input data/zinc_100k_molecules.pkl \
    --output data/zinc_100k_mpfit.h5 \
    --dataset-name "ZINC_100k" \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 64

echo "✓ ZINC 100k MPFIT dataset created"
echo ""

echo "================================================"
echo "ZINC 100k Steps 1-2 Complete!"
echo "================================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Total time: $SECONDS seconds ($((SECONDS/3600))h $((SECONDS%3600/60))m)"
echo ""
echo "Output:"
echo "  - Molecules: data/zinc_100k_molecules.pkl"
echo "  - MPFIT dataset: data/zinc_100k_mpfit.h5"
echo ""
echo "Next steps:"
echo "  - Run training pipeline (steps 3-6) using data/zinc_100k_mpfit.h5"
echo ""
