#!/bin/bash
#SBATCH -J spice_50k_s12
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --mem=256GB
#SBATCH -t 48:00:00
#SBATCH -A m1266
#SBATCH -o logs/spice_50k_step1_2_%j.out
#SBATCH -e logs/spice_50k_step1_2_%j.err

# SPICE 50k Dataset Pipeline - Steps 1-2 Only
# Data preparation: Extraction and MPFIT charge generation
#
# Estimated time: ~20-25 hours on 1 CPU node (128 CPU cores)
# Time breakdown (50,000 molecules):
#   Step 1: Extract SPICE subset - ~15 min
#   Step 2: MPFIT charges - ~20-24 hours (64 cores parallel, GDMA is expensive)
#   Total: ~20-25 hours
#
# Note: This only performs data preparation (steps 1-2).
#       Run separate training/validation jobs (steps 3-6) after this completes.

# Set working directory to MMomentA root
MMOMENTA_DIR="/global/u1/p/parmar/MoML/MMomentA"

cd "$MMOMENTA_DIR"
mkdir -p logs

echo "================================================"
echo "MMomentA SPICE 50k Pipeline - Steps 1-2"
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
echo "Step 1/2: Extracting SPICE subset (50,000 molecules)..."
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
    mol_ids = list(f.keys())[:50000]
    print(f"Found {len(f.keys())} total molecules in SPICE, extracting first 50,000")

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

            if (i + 1) % 5000 == 0:
                print(f"  Extracted {i + 1}/{len(mol_ids)} molecules")

        except Exception as e:
            print(f"  ✗ Failed on {mol_id}: {e}")
            continue

Path('data').mkdir(exist_ok=True)
with open('data/spice_50k.pkl', 'wb') as f:
    pickle.dump(molecules, f)

print(f"✓ Saved {len(molecules)} molecules to data/spice_50k.pkl")
EOF

echo "✓ SPICE subset extracted"
echo ""

# Step 2: Generate MPFIT charges (with caching)
echo "Step 2/2: Computing MPFIT charges for SPICE (cached if previously run)..."
conda activate "$DATA_ENV"

python scripts/prepare_dataset_cached.py \
    --input data/spice_50k.pkl \
    --output data/spice_50k_mpfit.h5 \
    --dataset-name "SPICE_50k" \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 64

echo "✓ SPICE 50k MPFIT dataset created"
echo ""

echo "================================================"
echo "SPICE 50k Steps 1-2 Complete!"
echo "================================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Total time: $SECONDS seconds ($((SECONDS/3600))h $((SECONDS%3600/60))m)"
echo ""
echo "Output:"
echo "  - Molecules: data/spice_50k.pkl"
echo "  - MPFIT dataset: data/spice_50k_mpfit.h5"
echo ""
echo "Next steps:"
echo "  - Run training pipeline (steps 3-6) using data/spice_50k_mpfit.h5"
echo "  - Submit: sbatch perlmutter/pipelines/run_spice_50k_step3_6.slurm"
echo ""
