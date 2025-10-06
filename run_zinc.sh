#!/bin/bash
# ZINC Dataset Transfer Learning Pipeline
# Run this AFTER run_spice.sh (needs pretrained model)

MMOMENTA_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA"

echo "================================================"
echo "MMomentA ZINC Transfer Learning Pipeline"
echo "================================================"
echo ""

# Check if SPICE model exists
if [ ! -f "runs/spice_multipoles/best_model.pt" ]; then
    echo "✗ Error: SPICE pretrained model not found!"
    echo "  Please run run_spice.sh first to train the SPICE model."
    exit 1
fi

# Initialize conda for script
eval "$(conda shell.bash hook)"

cd "$MMOMENTA_DIR"

# Step 1: Download and convert ZINC
echo "Step 1/4: Downloading ZINC fragments..."
module load anaconda3
conda activate mmomenta-ml

mkdir -p data/geom
cd data/geom

if [ ! -f "zinc_fragments.smi" ]; then
    wget https://raw.githubusercontent.com/openforcefield/qca-dataset-submission/master/submissions/2022-12-13-OpenFF-ZINC-Fragments/zinc_fragments.smi
    echo "✓ Downloaded ZINC fragments"
else
    echo "✓ ZINC fragments already exist"
fi

cd ../..

python scripts/smiles_to_molecules.py \
    --input data/geom/zinc_fragments.smi \
    --output data/zinc_molecules.pkl \
    --max-molecules 100

echo "✓ ZINC molecules created"
echo ""

# Step 2: Generate MPFIT charges
echo "Step 2/4: Computing MPFIT charges for ZINC (30-45 min)..."
conda activate mmomenta-data

python scripts/prepare_dataset.py \
    --input data/zinc_molecules.pkl \
    --output data/zinc_mpfit.h5 \
    --dataset-name "ZINC" \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 16

echo "✓ ZINC MPFIT dataset created"
echo ""

# Step 3: Train from scratch on ZINC
echo "Step 3/4: Training ZINC from scratch (15-20 min)..."
module load cuda
conda activate mmomenta-ml

python scripts/train_spice.py \
    --dataset data/zinc_mpfit.h5 \
    --output-dir runs/zinc_from_scratch \
    --n-epochs 200 \
    --device cuda

echo "✓ ZINC from-scratch model trained"
echo ""

# Step 4: Transfer learning from SPICE
echo "Step 4/4: Transfer learning from SPICE (10-15 min)..."

python scripts/train_spice.py \
    --dataset data/zinc_mpfit.h5 \
    --output-dir runs/zinc_transfer \
    --pretrained runs/spice_multipoles/best_model.pt \
    --learning-rate 0.0001 \
    --n-epochs 100 \
    --device cuda

echo "✓ ZINC transfer learning complete"
echo ""

# Results
echo "================================================"
echo "ZINC Transfer Learning Results"
echo "================================================"
echo ""

python << 'EOF'
import json

def load_results(path):
    try:
        return json.load(open(path))
    except:
        return None

scratch = load_results('runs/zinc_from_scratch/training_results.json')
transfer = load_results('runs/zinc_transfer/training_results.json')

if scratch and transfer:
    print("Training from Scratch:")
    print(f"  Test RMSE: {scratch['test_rmse']:.4f}")
    print(f"  Test MAE:  {scratch['test_mae']:.4f}")

    print("\nTransfer Learning (pretrained on SPICE):")
    print(f"  Test RMSE: {transfer['test_rmse']:.4f}")
    print(f"  Test MAE:  {transfer['test_mae']:.4f}")

    improvement = ((scratch['test_rmse'] - transfer['test_rmse']) /
                   scratch['test_rmse'] * 100)

    print(f"\nTransfer Improvement: {improvement:+.1f}%")

    if improvement > 0:
        print("✓ Transfer learning improves performance!")
        print("  SPICE knowledge successfully transferred to ZINC")
    else:
        print("✗ Transfer learning did not improve performance")

    print("\nResults saved in:")
    print("  - runs/zinc_from_scratch/")
    print("  - runs/zinc_transfer/")
else:
    print("✗ Results not available - check training logs")

print("=" * 50)
EOF

echo ""
echo "================================================"
echo "ZINC Pipeline Complete!"
echo "================================================"
echo ""
echo "Total time: ~1-1.5 hours"
echo ""
echo "Complete workflow:"
echo "  1. SPICE (100 mols) → Train multipole model"
echo "  2. ZINC (100 mols)  → Fine-tune SPICE model"
echo "  3. Compare transfer vs from-scratch performance"
