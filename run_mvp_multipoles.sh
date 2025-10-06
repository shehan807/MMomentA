#!/bin/bash
# MVP Pipeline with Multipoles Only
# Run this on Phoenix HPC cluster

MMOMENTA_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA"

echo "================================================"
echo "MMomentA MVP Pipeline (Multipoles Only)"
echo "================================================"
echo ""

# Initialize conda for script
eval "$(conda shell.bash hook)"

# Step 1: Create test SMILES dataset
echo "Step 1/3: Creating test SMILES dataset (150 molecules)..."
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
echo "Step 2/3: Computing MPFIT charges (45-60 min)..."

python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 16

echo "✓ Created test_mpfit.h5"
echo ""

# Step 3: Train with multipoles
echo "Step 3/3: Training multipole model (20-30 min)..."
module load cuda
conda activate mmomenta-ml

python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_multipoles \
    --n-epochs 200 \
    --device cuda

echo "✓ Multipole model trained"
echo ""

# Results
echo "================================================"
echo "Training Results"
echo "================================================"
echo ""

python << 'EOF'
import json
from pathlib import Path

results_file = Path('runs/test_multipoles/training_results.json')

if results_file.exists():
    results = json.load(open(results_file))

    print("Multipole Model Results:")
    print(f"  Test RMSE: {results['test_rmse']:.4f}")
    print(f"  Test MAE:  {results['test_mae']:.4f}")
    print(f"  Best Val RMSE: {results['best_val_rmse']:.4f}")

    print("\nModel saved to:")
    print(f"  runs/test_multipoles/best_model.pt")
    print("\nTraining log:")
    print(f"  runs/test_multipoles/training.log")
else:
    print("✗ Results not available - check training log")

print("=" * 50)
EOF

echo ""
echo "================================================"
echo "MVP Pipeline Complete!"
echo "================================================"
echo ""
echo "Total time: ~1.5-2 hours"
