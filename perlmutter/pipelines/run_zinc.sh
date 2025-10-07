#!/bin/bash
# ZINC Dataset Transfer Learning Pipeline for Perlmutter
# Run this AFTER run_spice.sh (needs pretrained model)

# Set working directory to MMomentA root
MMOMENTA_DIR="${MMOMENTA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "================================================"
echo "MMomentA ZINC Transfer Learning Pipeline (Perlmutter)"
echo "================================================"
echo "Working directory: $MMOMENTA_DIR"
echo ""

# Check if SPICE model exists
if [ ! -f "runs/spice_multipoles/best_model.pt" ]; then
    echo "✗ Error: SPICE pretrained model not found!"
    echo "  Please run run_spice.sh first to train the SPICE model."
    exit 1
fi

# Load conda
module load conda

# Set environment paths
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

cd "$MMOMENTA_DIR"

# Step 1: Download and convert ZINC
echo "Step 1/6: Downloading ZINC fragments..."
conda activate "$ML_ENV"

mkdir -p data/geom
cd data/geom

if [ ! -f "zinc_fragments.smi" ]; then
    wget https://raw.githubusercontent.com/openforcefield/qca-dataset-submission/master/submissions/2022-12-13-OpenFF-ZINC-Fragments/zinc_fragments.smi
    echo "✓ Downloaded ZINC fragments"
else
    echo "✓ ZINC fragments already exist"
fi

cd ../..

# Convert SMILES to molecules (take first 100)
python << 'EOF'
import pickle
from openff.toolkit import Molecule
from pathlib import Path

molecules = []
max_molecules = 100

print(f"Converting first {max_molecules} ZINC SMILES to molecules...")

with open('data/geom/zinc_fragments.smi', 'r') as f:
    for i, line in enumerate(f):
        if i >= max_molecules:
            break

        smiles = line.strip().split()[0]  # First column is SMILES

        try:
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            molecules.append(mol)

            if (i + 1) % 20 == 0:
                print(f"  Converted {i + 1}/{max_molecules} molecules")
        except Exception as e:
            print(f"  ✗ Failed on SMILES {i}: {e}")
            continue

Path('data').mkdir(exist_ok=True)
with open('data/zinc_molecules.pkl', 'wb') as f:
    pickle.dump(molecules, f)

print(f"✓ Saved {len(molecules)} molecules to data/zinc_molecules.pkl")
EOF

echo "✓ ZINC molecules created"
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
    --n-jobs 16

echo "✓ ZINC MPFIT dataset created"
echo ""

# Step 3: Train from scratch on ZINC
echo "Step 3/6: Training ZINC from scratch (1000 epochs)..."
conda activate "$ML_ENV"

python scripts/train_spice.py \
    --dataset data/zinc_mpfit.h5 \
    --output-dir runs/zinc_from_scratch \
    --n-epochs 1000 \
    --device cuda

echo "✓ ZINC from-scratch model trained"
echo ""

# Step 4: Transfer learning from SPICE
echo "Step 4/6: Transfer learning from SPICE (500 epochs)..."

python scripts/train_spice.py \
    --dataset data/zinc_mpfit.h5 \
    --output-dir runs/zinc_transfer \
    --pretrained runs/spice_multipoles/best_model.pt \
    --learning-rate 0.0001 \
    --n-epochs 500 \
    --device cuda

echo "✓ ZINC transfer learning complete"
echo ""

# Step 5: Compare transfer vs from-scratch results
echo "Step 5/6: Comparing transfer learning results..."

python << 'EOF'
import json

scratch = json.load(open('runs/zinc_from_scratch/training_results.json'))
transfer = json.load(open('runs/zinc_transfer/training_results.json'))

scratch_test = scratch['results']['test_metrics']
transfer_test = transfer['results']['test_metrics']

print("\n" + "="*70)
print("ZINC TRANSFER LEARNING RESULTS")
print("="*70)
print("\nTraining from Scratch:")
print(f"  Test RMSE: {scratch_test['val_rmse']:.4f}")
print(f"  Test MAE:  {scratch_test['val_mae']:.4f}")

print("\nTransfer Learning (pretrained on SPICE):")
print(f"  Test RMSE: {transfer_test['val_rmse']:.4f}")
print(f"  Test MAE:  {transfer_test['val_mae']:.4f}")

improvement = ((scratch_test['val_rmse'] - transfer_test['val_rmse']) /
               scratch_test['val_rmse'] * 100)

print(f"\nTransfer Improvement: {improvement:+.1f}%")

if improvement > 0:
    print("✓ Transfer learning improves performance!")
    print("  SPICE knowledge successfully transferred to ZINC")
else:
    print("✗ Transfer learning did not improve performance")

print("="*70)
EOF

echo ""

# Step 6: ESP Validation (RESP vs AM1-BCC vs MPFIT vs MMomentA-GNN)
echo "Step 6/6: ESP validation comparison (RESP, AM1-BCC, MPFIT, MMomentA-GNN)..."
conda activate "$DATA_ENV"
echo ""

# Use the transfer learning model for ESP validation
python scripts/compare_all_methods.py \
    --molecules data/zinc_molecules.pkl \
    --output-dir figures/zinc_esp_comparison \
    --mpfit-gnn-results runs/zinc_transfer/esp_validation_results.json \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --verbose

echo ""
echo "================================================"
echo "ZINC Pipeline Complete!"
echo "================================================"
echo ""
echo "Complete workflow:"
echo "  1. SPICE (100 mols) → Train multipole model"
echo "  2. ZINC (100 mols)  → Fine-tune SPICE model"
echo "  3. Compare transfer vs from-scratch performance"
echo ""
echo "Training results:"
echo "  - From scratch: runs/zinc_from_scratch/"
echo "  - Transfer learning: runs/zinc_transfer/"
echo ""
echo "ESP comparison: figures/zinc_esp_comparison/"
echo "  - Violin plots: figures/zinc_esp_comparison/esp_comparison_violin.png"
echo "  - Results JSON: figures/zinc_esp_comparison/qm_methods_results.json"
echo ""
