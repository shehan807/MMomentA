#!/bin/bash
#SBATCH -J zinc_10k
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 128
#SBATCH -t 48:00:00
#SBATCH -A m1266
#SBATCH -o logs/zinc_10k_%j.out
#SBATCH -e logs/zinc_10k_%j.err

# ZINC 10k Dataset Pipeline for Perlmutter
# Full-scale ZINC pipeline with 10,000 molecules
#
# Estimated time: 4-6 hours on 1 GPU node (128 CPU cores, 1 GPU)
# Time breakdown (10,000 molecules):
#   Step 1: Download - ~5 min
#   Step 2: MPFIT charges - ~70-80 min (128 cores parallel)
#   Step 3: Train baseline - ~30-60 min (GPU)
#   Step 4: Train multipole - ~30-60 min (GPU)
#   Step 5: Compare results - ~1 min
#   Step 6: ESP validation - ~105-150 min (test set = 1500 molecules, 128 cores)
#   Total: ~4-6 hours

# Set working directory to MMomentA root
# SLURM_SUBMIT_DIR is the directory where sbatch was called
MMOMENTA_DIR="${MMOMENTA_DIR:-$SLURM_SUBMIT_DIR}"

cd "$MMOMENTA_DIR"
mkdir -p logs

echo "================================================"
echo "MMomentA ZINC 10k Pipeline (Perlmutter)"
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
echo "Step 1/6: Downloading ZINC dataset (PyTorch Geometric)..."
conda activate "$ML_ENV"

# Download ZINC using PyG (10,000 molecules from train split)
python scripts/download_pyg_dataset.py \
    --dataset zinc \
    --output data/zinc_10k_molecules.pkl \
    --max-molecules 10000 \
    --split train

echo "✓ ZINC molecules downloaded"
echo ""

# Step 2: Generate MPFIT charges (with caching)
echo "Step 2/6: Computing MPFIT charges for ZINC (cached if previously run)..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset_cached.py \
    --input data/zinc_10k_molecules.pkl \
    --output data/zinc_10k_mpfit.h5 \
    --dataset-name "ZINC_10k" \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs -1

echo "✓ ZINC 10k MPFIT dataset created"
echo ""

# Step 3: Train baseline model (no multipoles)
echo "Step 3/6: Training baseline model (1000 epochs)..."
conda activate "$ML_ENV"

if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_10k_baseline/checkpoints/best_model.pt" ]; then
    echo "✓ ZINC 10k baseline model already trained (found checkpoint), skipping..."
else
    # Note: train_spice.py works with any HDF5 dataset (not just SPICE)
    python scripts/train_spice.py \
        --dataset data/zinc_10k_mpfit.h5 \
        --output-dir runs/zinc_10k_baseline \
        --no-multipoles \
        --n-epochs 1000 \
        --device cuda
    echo "✓ ZINC 10k baseline model trained"
fi
echo ""

# Step 4: Train multipole model
echo "Step 4/6: Training multipole model (1000 epochs)..."

if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_10k_multipoles/checkpoints/best_model.pt" ]; then
    echo "✓ ZINC 10k multipole model already trained (found checkpoint), skipping..."
else
    # Note: train_spice.py works with any HDF5 dataset (not just SPICE)
    python scripts/train_spice.py \
        --dataset data/zinc_10k_mpfit.h5 \
        --output-dir runs/zinc_10k_multipoles \
        --n-epochs 1000 \
        --device cuda
    echo "✓ ZINC 10k multipole model trained"
fi
echo ""

# Step 5: Compare training results
echo "Step 5/6: Comparing training results..."

python << 'EOF'
import json

baseline = json.load(open('runs/zinc_10k_baseline/training_results.json'))
multipole = json.load(open('runs/zinc_10k_multipoles/training_results.json'))

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

mkdir -p figures/zinc_10k_esp_validation

python scripts/validate_esp_comparison.py \
    --dataset data/zinc_10k_mpfit.h5 \
    --model-dir runs/zinc_10k_multipoles \
    --output-dir figures/zinc_10k_esp_validation \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --device cpu \
    --n-jobs 32

echo ""
echo "================================================"
echo "ZINC 10k Pipeline Complete!"
echo "================================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Total time: $SECONDS seconds ($((SECONDS/3600))h $((SECONDS%3600/60))m)"
echo ""
echo "Training results: runs/zinc_10k_{baseline,multipoles}/"
echo "ESP validation plots:"
echo "  - ESP violin plot: figures/zinc_10k_esp_validation/esp_validation_all_methods.png"
echo "  - Carbon hexbin:   figures/zinc_10k_esp_validation/carbon_charge_hexbin.png"
echo "Validation metrics: figures/zinc_10k_esp_validation/esp_validation_results.json"
echo ""
