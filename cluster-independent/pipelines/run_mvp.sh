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
echo "Step 1/5: Creating test SMILES dataset (150 molecules)..."
module load anaconda3
conda activate mmomenta-data  # Needs OpenFF for molecule creation

cd "$MMOMENTA_DIR"
python << 'EOF'
from openff.toolkit import Molecule
import pickle
from pathlib import Path

# 150 diverse molecules (expand the original 50)
SMILES = [
    # Alkanes (15)
    'C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC',
    'CC(C)C', 'CC(C)CC', 'CC(C)(C)C', 'CCC(C)C',
    'CCCC(C)C', 'CC(C)C(C)C', 'CCC(C)(C)C', 'CCCCCCC', 'CCCCCCCC',

    # Alcohols (15)
    'CO', 'CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CC(C)(C)O',
    'CC(O)C', 'CCC(O)C', 'CCCC(O)C', 'C(O)CCO',
    'CC(O)CO', 'OCC(O)CO', 'OCCCO', 'OCCCCO', 'CC(C)CO',

    # Aromatics (20)
    'c1ccccc1', 'c1ccc(C)cc1', 'c1ccc(CC)cc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1',
    'c1ccc(F)cc1', 'c1ccc(Cl)cc1', 'c1ccc(Cc2ccccc2)cc1', 'c1ccc(C(C)C)cc1',
    'c1cc(C)ccc1C', 'c1cc(O)cc(O)c1', 'c1ccc(OC)cc1', 'c1ccc(N(C)C)cc1',
    'c1ccc2ccccc2c1', 'c1ccc2c(c1)ccc1ccccc12', 'c1ccc(c(c1)C)C',
    'c1cc(C)c(C)cc1', 'c1ccc(C#N)cc1', 'c1ccc([N+](=O)[O-])cc1', 'c1ccc(S)cc1',

    # Carboxylic acids (15)
    'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O', 'CC(C)C(=O)O', 'c1ccc(C(=O)O)cc1',
    'OC(=O)C(=O)O', 'CC(O)C(=O)O', 'CCC(O)C(=O)O', 'OC(=O)CC(=O)O',
    'c1cc(C(=O)O)ccc1C(=O)O', 'CC(N)C(=O)O', 'CCC(N)C(=O)O',
    'OC(=O)CCCC(=O)O', 'CC(CC(=O)O)C(=O)O', 'c1cc(O)c(C(=O)O)cc1',

    # Amines (15)
    'CN', 'CCN', 'CCCN', 'CC(C)N', 'CCNC', 'CCNCC', 'c1ccc(N)cc1',
    'CC(N)C', 'CCC(N)C', 'NCc1ccccc1', 'c1ccc(NC)cc1', 'c1ccc(N(C)C)cc1',
    'NCCN', 'NCCCN', 'c1ccc(CN)cc1',

    # Ethers (15)
    'COC', 'CCOC', 'CCOCC', 'CCCOCC', 'c1ccc(OC)cc1', 'COc1ccccc1',
    'CCOc1ccccc1', 'c1ccc(OCCC)cc1', 'COc1ccc(OC)cc1', 'c1cc(OC)ccc1OC',
    'COCCOC', 'CCOCCOC', 'c1ccc(Oc2ccccc2)cc1', 'COc1ccc(C)cc1', 'CCOc1ccccc1C',

    # Ketones (15)
    'CC(=O)C', 'CCC(=O)C', 'CC(=O)CC', 'CCC(=O)CC', 'CC(=O)CCC',
    'c1ccc(C(=O)C)cc1', 'CC(=O)c1ccccc1', 'CCC(=O)c1ccccc1', 'CC(=O)CC(=O)C',
    'c1ccc(C(=O)CC)cc1', 'CC(=O)CCCC(=O)C', 'CC(=O)c1ccc(C)cc1',
    'c1ccc(C(=O)c2ccccc2)cc1', 'CC(C)C(=O)C', 'CCC(C)C(=O)C',

    # Aldehydes (10)
    'CC=O', 'CCC=O', 'CCCC=O', 'c1ccc(C=O)cc1', 'CC(C)C=O',
    'c1cc(C=O)ccc1C', 'OCC=O', 'CC(O)C=O', 'c1ccc(CC=O)cc1', 'CCC(C)C=O',

    # Esters (15)
    'CC(=O)OC', 'CCC(=O)OC', 'CC(=O)OCC', 'CCC(=O)OCC', 'CCCC(=O)OC',
    'c1ccc(C(=O)OC)cc1', 'CC(=O)Oc1ccccc1', 'c1ccc(OC(=O)C)cc1',
    'COC(=O)c1ccccc1', 'CCOC(=O)c1ccccc1', 'CC(=O)OCC(=O)C',
    'c1ccc(C(=O)OCC)cc1', 'CC(=O)Oc1ccc(C)cc1', 'COC(=O)CC(=O)OC', 'CCC(=O)OCCC',

    # Heterocycles (15)
    'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'C1CCNCC1', 'C1CCOCC1',
    'c1cnc2ccccc2c1', 'c1ccc2ncccc2c1', 'C1CCC2CCCCC2C1',
    'c1ccc2[nH]ccc2c1', 'c1ccc2occc2c1', 'c1csc2ccccc12',
    'C1CNCCC1', 'c1cnccn1', 'c1cncnc1', 'c1coccn1',

    # Nitriles (10)
    'CC#N', 'CCC#N', 'CCCC#N', 'c1ccc(C#N)cc1', 'CC(C)C#N',
    'c1cc(C#N)ccc1C', 'NCC#N', 'c1ccc(CC#N)cc1', 'CCC(C)C#N', 'c1cc(C#N)cc(C#N)c1',
]

molecules = []
for i, smiles in enumerate(SMILES):
    try:
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        mol.name = f"mol_{i}_{smiles[:10]}"
        molecules.append(mol)

        if (i + 1) % 30 == 0:
            print(f"  Created {i + 1}/{len(SMILES)} molecules")
    except Exception as e:
        print(f"  ✗ Failed on {smiles}: {e}")

Path('data').mkdir(exist_ok=True)
with open('data/test_smiles.pkl', 'wb') as f:
    pickle.dump(molecules, f)

print(f"✓ Created {len(molecules)} molecules")
EOF

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

# Extract metrics from nested structure
baseline_test = baseline_results['results']['test_metrics']
multipole_test = multipole_results['results']['test_metrics']
baseline_best_val = baseline_results['results']['best_val_rmse']
multipole_best_val = multipole_results['results']['best_val_rmse']

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print("\nBaseline Model (no multipoles):")
print(f"  Test RMSE: {baseline_test['val_rmse']:.4f}")
print(f"  Test MAE:  {baseline_test['val_mae']:.4f}")
print(f"  Best Val RMSE: {baseline_best_val:.4f}")

print("\nMultipole Model:")
print(f"  Test RMSE: {multipole_test['val_rmse']:.4f}")
print(f"  Test MAE:  {multipole_test['val_mae']:.4f}")
print(f"  Best Val RMSE: {multipole_best_val:.4f}")

improvement = ((baseline_test['val_rmse'] - multipole_test['val_rmse']) /
               baseline_test['val_rmse'] * 100)

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
