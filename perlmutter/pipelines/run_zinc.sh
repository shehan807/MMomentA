#!/bin/bash
# ZINC Dataset Pipeline for Perlmutter
# Same workflow as MVP, but with 1000 ZINC molecules

# Set working directory to MMomentA root
MMOMENTA_DIR="${MMOMENTA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "================================================"
echo "MMomentA ZINC Pipeline (Perlmutter)"
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

# Download ZINC using PyG (1000 molecules from train split)
python scripts/download_pyg_dataset.py \
    --dataset zinc \
    --output data/zinc_molecules.pkl \
    --max-molecules 1000 \
    --split train

echo "✓ ZINC molecules downloaded"
echo ""

# Step 2: Generate MPFIT charges (with caching)
echo "Step 2/6: Computing MPFIT charges for ZINC (cached if previously run)..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset_cached.py \
    --input data/zinc_molecules.pkl \
    --output data/zinc_mpfit.h5 \
    --dataset-name "ZINC" \
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

if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_baseline/checkpoints/best_model.pt" ]; then
    echo "✓ ZINC baseline model already trained (found checkpoint), skipping..."
else
    # Baseline: espaloma-charge settings (width=32, input=128, 5000 epochs)
    python scripts/train_spice.py \
        --dataset data/zinc_mpfit.h5 \
        --output-dir runs/zinc_baseline \
        --no-multipoles \
        --n-epochs 5000 \
        --width 32 \
        --device cuda
    echo "✓ ZINC baseline model trained"
fi
echo ""

# Step 4: Train multipole model (IMPROVED HYPERPARAMETERS)
# COMMENTED OUT - multipoles causing worse generalization
# echo "Step 4/6: Training multipole model (1000 epochs, optimized settings)..."
#
# if [ -f "/global/u1/p/parmar/MoML/MMomentA/runs/zinc_multipoles/checkpoints/best_model.pt" ]; then
#     echo "✓ ZINC multipole model already trained (found checkpoint), skipping..."
# else
#     # Multipole: 198 features → 256 input → 256 hidden (increased capacity)
#     # Added: batch norm, reduced weight decay, auto LR scheduling
#     python scripts/train_spice.py \
#         --dataset data/zinc_mpfit.h5 \
#         --output-dir runs/zinc_multipoles \
#         --n-epochs 1000 \
#         --width 256 \
#         --input-units 256 \
#         --batch-norm \
#         --weight-decay 1e-6 \
#         --device cuda
#     echo "✓ ZINC multipole model trained"
# fi
# echo ""

# Step 5: Compare training results
# COMMENTED OUT - multipole training disabled
# echo "Step 5/6: Comparing training results..."
#
# python << 'EOF'
# import json
#
# baseline = json.load(open('runs/zinc_baseline/training_results.json'))
# multipole = json.load(open('runs/zinc_multipoles/training_results.json'))
#
# baseline_test = baseline['results']['test_metrics']
# multipole_test = multipole['results']['test_metrics']
#
# print("\n" + "="*70)
# print("TRAINING RESULTS COMPARISON")
# print("="*70)
#
# print("\nBaseline Model (no multipoles):")
# print(f"  Test RMSE: {baseline_test['val_rmse']:.4f}")
# print(f"  Test MAE:  {baseline_test['val_mae']:.4f}")
#
# print("\nMultipole Model:")
# print(f"  Test RMSE: {multipole_test['val_rmse']:.4f}")
# print(f"  Test MAE:  {multipole_test['val_mae']:.4f}")
#
# improvement = ((baseline_test['val_rmse'] - multipole_test['val_rmse']) /
#                baseline_test['val_rmse'] * 100)
#
# print(f"\nImprovement: {improvement:+.1f}%")
# print("="*70)
# EOF
#
# echo ""

echo "Step 5/6: Skipping comparison (multipole training disabled)"
echo ""

# Step 6: Validating ESP reproduction (MPFIT, AM1-BCC, RESP, MMomentA-GNN)
echo "Step 6/6: Validating ESP reproduction (using baseline model)..."
conda activate "$DATA_ENV"

mkdir -p figures/zinc_esp_validation

python scripts/validate_esp_comparison.py \
    --dataset data/zinc_mpfit.h5 \
    --model-dir runs/zinc_baseline \
    --output-dir figures/zinc_esp_validation \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --device cpu \
    --n-jobs 64

echo ""
echo "================================================"
echo "ZINC Pipeline Complete!"
echo "================================================"
echo "Training results: runs/zinc_baseline/"
echo "ESP validation plots:"
echo "  - ESP MAE plot:  figures/zinc_esp_validation/esp_validation_mae.png"
echo "  - ESP RMSE plot: figures/zinc_esp_validation/esp_validation_rmse.png"
echo "  - Carbon hexbin: figures/zinc_esp_validation/carbon_charge_hexbin.png"
echo "Validation metrics: figures/zinc_esp_validation/esp_validation_results.json"
echo ""
