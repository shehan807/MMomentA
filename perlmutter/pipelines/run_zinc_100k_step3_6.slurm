#!/bin/bash
#SBATCH -J zinc_100k_s36
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 128
#SBATCH --mem=256GB
#SBATCH -t 48:00:00
#SBATCH -A m1266
#SBATCH -o logs/zinc_100k_step3_6_%j.out
#SBATCH -e logs/zinc_100k_step3_6_%j.err

# ZINC 100k Dataset Pipeline - Steps 3-6 Only
# Model training and ESP validation
#
# Estimated time: ~30-40 hours on 1 GPU node (128 CPU cores, 1 GPU)
# Time breakdown (100,000 molecules):
#   Step 3: Train baseline - ~5-10 hours (GPU, 70k training molecules)
#   Step 4: Train multipole - ~5-10 hours (GPU, 70k training molecules)
#   Step 5: Compare results - ~1 min
#   Step 6: ESP validation - ~15-20 hours (15k test molecules, 32 cores)
#   Total: ~30-40 hours
#
# Prerequisites:
#   - data/zinc_100k_mpfit.h5 must exist (run run_zinc_100k_step1_2.slurm first)

# Set working directory to MMomentA root
# SLURM_SUBMIT_DIR is the directory where sbatch was called
MMOMENTA_DIR="${MMOMENTA_DIR:-$SLURM_SUBMIT_DIR}"

cd "$MMOMENTA_DIR"
mkdir -p logs

echo "================================================"
echo "MMomentA ZINC 100k Pipeline - Steps 3-6"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "================================================"
echo "Working directory: $MMOMENTA_DIR"
echo ""

# Check prerequisite
if [ ! -f "data/zinc_100k_mpfit.h5" ]; then
    echo "ERROR: data/zinc_100k_mpfit.h5 not found!"
    echo "Please run run_zinc_100k_step1_2.slurm first."
    exit 1
fi

# Load conda
module load conda

# Set environment paths
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

# Step 3: Train baseline model (no multipoles)
echo "Step 3/6: Training baseline model (5000 epochs, espaloma-charge settings)..."
conda activate "$ML_ENV"

if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_100k_baseline/checkpoints/best_model.pt" ]; then
    echo "✓ ZINC 100k baseline model already trained (found checkpoint), skipping..."
else
    python scripts/train_spice.py \
        --dataset data/zinc_100k_mpfit.h5 \
        --output-dir runs/zinc_100k_baseline \
        --no-multipoles \
        --n-epochs 5000 \
        --width 32 \
        --early-stopping-patience 2000 \
        --device cuda
    echo "✓ ZINC 100k baseline model trained"
fi
echo ""

# Step 4: Train multipole model (with automatic width scaling)
echo "Step 4/6: Training multipole model (5000 epochs, auto-scaled capacity)..."

if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_100k_multipoles/checkpoints/best_model.pt" ]; then
    echo "✓ ZINC 100k multipole model already trained (found checkpoint), skipping..."
else
    # Multipole: 198 features → auto-scales width from 32 to 54 (198/117 * 32)
    # This maintains the same compression ratio as baseline
    python scripts/train_spice.py \
        --dataset data/zinc_100k_mpfit.h5 \
        --output-dir runs/zinc_100k_multipoles \
        --n-epochs 5000 \
        --width 32 \
        --early-stopping-patience 2000 \
        --device cuda
    echo "✓ ZINC 100k multipole model trained"
fi
echo ""

# Step 5: Compare training results
echo "Step 5/6: Comparing training results..."

python << 'EOF'
import json

baseline = json.load(open('runs/zinc_100k_baseline/training_results.json'))
multipole = json.load(open('runs/zinc_100k_multipoles/training_results.json'))

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

mkdir -p figures/zinc_100k_esp_validation

python scripts/validate_esp_comparison.py \
    --dataset data/zinc_100k_mpfit.h5 \
    --model-dir runs/zinc_100k_multipoles \
    --output-dir figures/zinc_100k_esp_validation \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --device cpu \
    --n-jobs 32

echo ""
echo "================================================"
echo "ZINC 100k Pipeline Steps 3-6 Complete!"
echo "================================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Total time: $SECONDS seconds ($((SECONDS/3600))h $((SECONDS%3600/60))m)"
echo ""
echo "Training results:"
echo "  - Baseline:  runs/zinc_100k_baseline/"
echo "  - Multipole: runs/zinc_100k_multipoles/"
echo "ESP validation plots:"
echo "  - ESP MAE plot:  figures/zinc_100k_esp_validation/esp_validation_mae.png"
echo "  - ESP RMSE plot: figures/zinc_100k_esp_validation/esp_validation_rmse.png"
echo "  - Carbon hexbin: figures/zinc_100k_esp_validation/carbon_charge_hexbin.png"
echo "Validation metrics: figures/zinc_100k_esp_validation/esp_validation_results.json"
echo ""
