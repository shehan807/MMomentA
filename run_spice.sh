#!/bin/bash
# SPICE Dataset Training Pipeline
# Run this on Phoenix HPC cluster

MMOMENTA_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA"

echo "================================================"
echo "MMomentA SPICE Training Pipeline"
echo "================================================"
echo ""

# Initialize conda for script
eval "$(conda shell.bash hook)"

cd "$MMOMENTA_DIR"

# Step 1: Extract SPICE subset
echo "Step 1/4: Extracting SPICE subset (100 molecules)..."
module load anaconda3
conda activate mmomenta-ml

python << 'EOF'
import h5py
import pickle
from openff.toolkit import Molecule
from pathlib import Path

molecules = []
print("Extracting molecules from SPICE...")

with h5py.File('data/SPICE-2.0.1.hdf5', 'r') as f:
    mol_ids = list(f.keys())[:100]

    for i, mol_id in enumerate(mol_ids):
        try:
            grp = f[mol_id]

            if 'smiles' in grp.attrs:
                mol = Molecule.from_smiles(grp.attrs['smiles'])
                # Use first QM conformer
                coords = grp['conformations'][0]
                mol.add_conformer(coords)
                molecules.append(mol)

                if (i + 1) % 20 == 0:
                    print(f"  Extracted {i + 1}/{len(mol_ids)} molecules")
        except Exception as e:
            print(f"  ✗ Failed on {mol_id}: {e}")
            continue

Path('data').mkdir(exist_ok=True)
with open('data/spice_100.pkl', 'wb') as f:
    pickle.dump(molecules, f)

print(f"✓ Saved {len(molecules)} molecules to data/spice_100.pkl")
EOF

echo "✓ SPICE subset extracted"
echo ""

# Step 2: Generate MPFIT charges
echo "Step 2/4: Computing MPFIT charges for SPICE (30-45 min)..."
conda activate mmomenta-data

python scripts/prepare_dataset.py \
    --input data/spice_100.pkl \
    --output data/spice_mpfit.h5 \
    --dataset-name "SPICE" \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 16

echo "✓ SPICE MPFIT dataset created"
echo ""

# Step 3: Train baseline model
echo "Step 3/4: Training SPICE baseline model (15-20 min)..."
module load cuda
conda activate mmomenta-ml

python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/spice_baseline \
    --no-multipoles \
    --n-epochs 200 \
    --device cuda

echo "✓ SPICE baseline trained"
echo ""

# Step 4: Train with multipoles
echo "Step 4/4: Training SPICE multipole model (15-20 min)..."

python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/spice_multipoles \
    --n-epochs 200 \
    --device cuda

echo "✓ SPICE multipole model trained"
echo ""

# Results
echo "================================================"
echo "SPICE Results"
echo "================================================"
echo ""

python << 'EOF'
import json

def load_results(path):
    try:
        return json.load(open(path))
    except:
        return None

baseline = load_results('runs/spice_baseline/training_results.json')
multipoles = load_results('runs/spice_multipoles/training_results.json')

if baseline and multipoles:
    print("Baseline Model (no multipoles):")
    print(f"  Test RMSE: {baseline['test_rmse']:.4f}")
    print(f"  Test MAE:  {baseline['test_mae']:.4f}")

    print("\nMultipole Model:")
    print(f"  Test RMSE: {multipoles['test_rmse']:.4f}")
    print(f"  Test MAE:  {multipoles['test_mae']:.4f}")

    improvement = ((baseline['test_rmse'] - multipoles['test_rmse']) /
                   baseline['test_rmse'] * 100)

    print(f"\nImprovement: {improvement:+.1f}%")

    if improvement > 0:
        print("✓ Multipoles improve charge prediction!")
    else:
        print("✗ Multipoles did not improve performance")

    print("\nResults saved in:")
    print("  - runs/spice_baseline/")
    print("  - runs/spice_multipoles/")
else:
    print("✗ Results not available - check training logs")

print("=" * 50)
EOF

echo ""
echo "================================================"
echo "SPICE Pipeline Complete!"
echo "================================================"
echo ""
echo "Total time: ~1-1.5 hours"
