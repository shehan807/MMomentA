#!/bin/bash
# SPICE Dataset Training Pipeline for Perlmutter
# 100 molecules from SPICE dataset

# Set working directory to MMomentA root
MMOMENTA_DIR="${MMOMENTA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "================================================"
echo "MMomentA SPICE Training Pipeline (Perlmutter)"
echo "================================================"
echo "Working directory: $MMOMENTA_DIR"
echo ""

# Load conda
module load conda

# Set environment paths
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

cd "$MMOMENTA_DIR"

# Step 1: Extract SPICE subset
echo "Step 1/6: Extracting SPICE subset (100 molecules)..."
conda activate "$ML_ENV"

python << 'EOF'
import h5py
import pickle
from openff.toolkit import Molecule
from pathlib import Path
import numpy as np

molecules = []
print("Extracting molecules from SPICE...")

# Check if SPICE dataset exists
spice_path = Path('data/SPICE-2.0.1.hdf5')
if not spice_path.exists():
    print(f"✗ SPICE dataset not found at {spice_path}")
    print("  Please download SPICE-2.0.1.hdf5 first")
    exit(1)

with h5py.File('data/SPICE-2.0.1.hdf5', 'r') as f:
    mol_ids = list(f.keys())[:100]
    print(f"Found {len(f.keys())} total molecules in SPICE, extracting first 100")

    for i, mol_id in enumerate(mol_ids):
        try:
            grp = f[mol_id]

            # SPICE stores SMILES as dataset, not attribute
            smiles = None
            if 'smiles' in grp:
                smiles_data = grp['smiles'][()]
                if isinstance(smiles_data, bytes):
                    smiles = smiles_data.decode('utf-8')
                else:
                    smiles = str(smiles_data)
            elif 'smiles' in grp.attrs:
                smiles = grp.attrs['smiles']
                if isinstance(smiles, bytes):
                    smiles = smiles.decode('utf-8')

            if not smiles:
                print(f"  ✗ No SMILES for {mol_id}")
                continue

            # Create molecule from SMILES
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

            # Get conformations (should be in Angstroms in SPICE)
            if 'conformations' in grp:
                coords = grp['conformations'][0]  # Take first conformer
            elif 'geometry' in grp:
                coords = grp['geometry'][0]
            else:
                print(f"  ✗ No coordinates for {mol_id}")
                continue

            # Convert to numpy array and ensure correct shape
            coords = np.array(coords, dtype=float)
            if coords.shape[0] != mol.n_atoms:
                print(f"  ✗ Coordinate mismatch for {mol_id}: {coords.shape[0]} coords vs {mol.n_atoms} atoms")
                continue

            # Add conformer (OpenFF expects Quantity with units)
            from openff.units import unit
            mol.add_conformer(coords * unit.angstrom)

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

# Step 2: Generate MPFIT charges (with caching)
echo "Step 2/6: Computing MPFIT charges for SPICE (cached if previously run)..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset_cached.py \
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
echo "Step 3/6: Training SPICE baseline model (1000 epochs)..."
conda activate "$ML_ENV"

python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/spice_baseline \
    --no-multipoles \
    --n-epochs 1000 \
    --device cuda

echo "✓ SPICE baseline trained"
echo ""

# Step 4: Train with multipoles
echo "Step 4/6: Training SPICE multipole model (1000 epochs)..."

python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/spice_multipoles \
    --n-epochs 1000 \
    --device cuda

echo "✓ SPICE multipole model trained"
echo ""

# Step 5: Compare results
echo "Step 5/6: Comparing SPICE training results..."

python << 'EOF'
import json
from pathlib import Path

baseline = json.load(open('runs/spice_baseline/training_results.json'))
multipoles = json.load(open('runs/spice_multipoles/training_results.json'))

baseline_test = baseline['results']['test_metrics']
multipole_test = multipoles['results']['test_metrics']

print("\n" + "="*70)
print("SPICE TRAINING RESULTS COMPARISON")
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

if improvement > 0:
    print("✓ Multipoles improve charge prediction!")
else:
    print("✗ Multipoles did not improve performance")

print("="*70)
EOF

echo ""

# Step 6: ESP Validation (RESP vs AM1-BCC vs MPFIT vs MMomentA-GNN)
echo "Step 6/6: ESP validation comparison (RESP, AM1-BCC, MPFIT, MMomentA-GNN)..."
conda activate "$DATA_ENV"
echo ""

python scripts/compare_all_methods.py \
    --molecules data/spice_100.pkl \
    --output-dir figures/spice_esp_comparison \
    --mpfit-gnn-results runs/spice_multipoles/esp_validation_results.json \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --verbose

echo ""
echo "================================================"
echo "SPICE Pipeline Complete!"
echo "================================================"
echo "Training results: runs/spice_{baseline,multipoles}/"
echo "ESP comparison: figures/spice_esp_comparison/"
echo "  - Violin plots: figures/spice_esp_comparison/esp_comparison_violin.png"
echo "  - Results JSON: figures/spice_esp_comparison/qm_methods_results.json"
echo ""
echo "Next: Run ZINC transfer learning pipeline"
echo "  bash perlmutter/pipelines/run_zinc.sh"
echo ""
