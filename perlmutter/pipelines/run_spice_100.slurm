#!/bin/bash
#SBATCH -J spice_100
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 128
#SBATCH --mem=128GB
#SBATCH -t 08:00:00
#SBATCH -A m1266
#SBATCH -o logs/spice_100_%j.out
#SBATCH -e logs/spice_100_%j.err

# SPICE 100 Dataset Pipeline for Perlmutter
# Full 6-step pipeline with 100 molecules
#
# Estimated time: ~3-5 hours on 1 GPU node (128 CPU cores, 1 GPU)
# Time breakdown (100 molecules):
#   Step 1: Extract SPICE subset - ~2 min
#   Step 2: MPFIT charges - ~30-45 min (16 cores parallel)
#   Step 3: Train baseline - ~30-60 min (GPU)
#   Step 4: Train multipole - ~30-60 min (GPU)
#   Step 5: Compare results - ~1 min
#   Step 6: ESP validation - ~30-60 min (test set = 15 molecules, 16 cores)
#   Total: ~3-5 hours

# Set working directory to MMomentA root
MMOMENTA_DIR="/global/u1/p/parmar/MoML/MMomentA"

cd "$MMOMENTA_DIR"
mkdir -p logs

echo "================================================"
echo "MMomentA SPICE 100 Pipeline (Perlmutter)"
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

# Check if SPICE dataset exists (in $SCRATCH)
import os
scratch_dir = os.environ.get('SCRATCH', os.path.expanduser('~'))
spice_path = Path(scratch_dir) / 'SPICE-2.0.1.hdf5'
if not spice_path.exists():
    print(f"✗ SPICE dataset not found at {spice_path}")
    print("  Please download SPICE-2.0.1.hdf5 to $SCRATCH:")
    print(f"  cd {scratch_dir}")
    print("  wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5")
    exit(1)

print(f"Loading SPICE dataset from {spice_path}")
with h5py.File(str(spice_path), 'r') as f:
    mol_ids = list(f.keys())[:100]
    print(f"Found {len(f.keys())} total molecules in SPICE, extracting first 100")

    for i, mol_id in enumerate(mol_ids):
        try:
            grp = f[mol_id]

            # SPICE stores SMILES as dataset, not attribute
            smiles = None
            if 'smiles' in grp:
                smiles_data = grp['smiles'][()]
                # Handle different storage formats
                if isinstance(smiles_data, bytes):
                    smiles = smiles_data.decode('utf-8')
                elif isinstance(smiles_data, np.ndarray):
                    # Array of bytes - take first element and decode
                    smiles = smiles_data[0] if len(smiles_data) > 0 else None
                    if isinstance(smiles, bytes):
                        smiles = smiles.decode('utf-8')
                    elif smiles is not None:
                        smiles = str(smiles)
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

# Step 3: Train baseline model (no multipoles)
echo "Step 3/6: Training baseline model (1000 epochs, espaloma-charge settings)..."
conda activate "$ML_ENV"

if [ -f "$MMOMENTA_DIR/runs/spice_baseline/checkpoints/best_model.pt" ]; then
    echo "✓ SPICE baseline model already trained (found checkpoint), skipping..."
else
    # Baseline: espaloma-charge settings (width=32, input=128, 1000 epochs)
    python scripts/train_spice.py \
        --dataset data/spice_mpfit.h5 \
        --output-dir runs/spice_baseline \
        --no-multipoles \
        --n-epochs 1000 \
        --width 32 \
        --early-stopping-patience 500 \
        --device cuda
    echo "✓ SPICE baseline model trained"
fi
echo ""

# Step 4: Train multipole model (with automatic width scaling)
echo "Step 4/6: Training multipole model (1000 epochs, auto-scaled capacity)..."

if [ -f "$MMOMENTA_DIR/runs/spice_multipoles/checkpoints/best_model.pt" ]; then
    echo "✓ SPICE multipole model already trained (found checkpoint), skipping..."
else
    # Multipole: 198 features → auto-scales width from 32 to 54 (198/117 * 32)
    # This maintains the same compression ratio as baseline
    python scripts/train_spice.py \
        --dataset data/spice_mpfit.h5 \
        --output-dir runs/spice_multipoles \
        --n-epochs 1000 \
        --width 32 \
        --early-stopping-patience 500 \
        --device cuda
    echo "✓ SPICE multipole model trained"
fi
echo ""

# Step 5: Compare training results
echo "Step 5/6: Comparing training results..."

python << 'EOF'
import json

baseline = json.load(open('runs/spice_baseline/training_results.json'))
multipole = json.load(open('runs/spice_multipoles/training_results.json'))

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

mkdir -p figures/spice_esp_validation

python scripts/validate_esp_comparison.py \
    --dataset data/spice_mpfit.h5 \
    --model-dir runs/spice_multipoles \
    --output-dir figures/spice_esp_validation \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --device cpu \
    --n-jobs 16

echo ""
echo "================================================"
echo "SPICE 100 Pipeline Complete!"
echo "================================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Total time: $SECONDS seconds ($((SECONDS/3600))h $((SECONDS%3600/60))m)"
echo ""
echo "Training results:"
echo "  - Baseline:  runs/spice_baseline/"
echo "  - Multipole: runs/spice_multipoles/"
echo "ESP validation plots:"
echo "  - ESP MAE plot:  figures/spice_esp_validation/esp_validation_mae.png"
echo "  - ESP RMSE plot: figures/spice_esp_validation/esp_validation_rmse.png"
echo "  - Carbon hexbin: figures/spice_esp_validation/carbon_charge_hexbin.png"
echo "Validation metrics: figures/spice_esp_validation/esp_validation_results.json"
echo ""
