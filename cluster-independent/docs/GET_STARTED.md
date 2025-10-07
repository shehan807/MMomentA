# Getting Started: Missing Files Setup

## Problem
The MMomentA repository contains code but not:
1. Training/test scripts (scripts/*.py)
2. Dataset files (data/*.h5)
3. Pre-trained models

## What You Need to Create

### 1. Download SPICE Dataset

**Option A: Full SPICE 2.0.1 (~30 GB)**
```bash
cd MMomentA
mkdir -p data
cd data
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5
mv SPICE-2.0.1.hdf5 spice_raw.hdf5
```

**Option B: Small subset for testing (recommended for MVP)**
```bash
# Download and extract just 50 molecules using Python
cd MMomentA
mkdir -p data

python << 'EOF'
import h5py
import numpy as np
from pathlib import Path

# Download full dataset first (or use existing)
print("Extracting 50-molecule subset from SPICE...")

# If you have full SPICE already
input_file = "data/SPICE-2.0.1.hdf5"
output_file = "data/spice_raw.hdf5"

# Extract first 50 molecules
# NOTE: This requires the full dataset downloaded first
# For true MVP, just download 50 molecules worth manually

print(f"Creating minimal dataset at {output_file}")
print("For MVP: Will be created by prepare_dataset.py from any molecular input")
EOF
```

**Option C: Use your own molecules**
```bash
# If you have OEB, SDF, or other molecular files
cd MMomentA/data
# Copy your molecule files here
# prepare_dataset.py can read: .oeb, .sdf, .mol2, .pdb
```

### 2. Create Missing Scripts

The repository is missing the actual training scripts. You need to create them:

**Critical missing files:**
- `scripts/prepare_dataset.py` - Dataset preparation with MPFIT
- `scripts/train_spice.py` - SPICE training script
- `scripts/train_zinc.py` - ZINC transfer learning
- `scripts/compare_charge_methods.py` - Method comparison

**These scripts use the MMomentA package code you already have** but provide CLI interfaces.

### 3. Quick Test Without Full Dataset

**For immediate testing (no dataset download needed):**

```bash
cd MMomentA

# Create a tiny test dataset from SMILES
python << 'EOF'
from openff.toolkit import Molecule
import pickle

# 5 test molecules
smiles = [
    'CC(=O)O',           # Acetic acid
    'c1ccccc1',          # Benzene
    'CCO',               # Ethanol
    'C1CCCCC1',          # Cyclohexane
    'CC(C)O'             # Isopropanol
]

molecules = [Molecule.from_smiles(s) for s in smiles]

# Generate 3D conformers
for mol in molecules:
    mol.generate_conformers(n_conformers=1)

# Save as pickle
with open('data/test_molecules.pkl', 'wb') as f:
    pickle.dump(molecules, f)

print("✓ Created data/test_molecules.pkl (5 molecules)")
EOF
```

Now you can test MPFIT calculations:
```bash
# This will work with the test molecules
python -c "
from mmomenta.qm import MPFITCalculator, MPFITConfig, QMSettings
from openff.toolkit import Molecule
import pickle

# Load test molecules
with open('data/test_molecules.pkl', 'rb') as f:
    molecules = pickle.load(f)

# Calculate MPFIT charges for first molecule
calc = MPFITCalculator(
    qc_settings=QMSettings(method='hf', basis='6-31G*'),
    config=MPFITConfig(limit=8)
)

result = calc.compute(molecules[0])
print(f'Charges: {result.charges}')
print(f'Multipole moments shape: {result.multipole_moments.shape}')
print('✓ MPFIT working!')
"
```

## Minimal Working Setup (30 minutes)

**For immediate MVP without waiting for full SPICE download:**

```bash
cd MMomentA

# 1. Create test molecules (5 molecules, instant)
python << 'EOF'
from openff.toolkit import Molecule
import pickle

smiles = ['CC(=O)O', 'c1ccccc1', 'CCO', 'C1CCCCC1', 'CC(C)O',
          'CCCC', 'c1cc(O)ccc1', 'CC(N)C', 'c1cnccc1', 'CCCCCC']
molecules = [Molecule.from_smiles(s) for s in smiles]
for mol in molecules:
    mol.generate_conformers(n_conformers=1)

with open('data/test_molecules_10.pkl', 'wb') as f:
    pickle.dump(molecules, f)
print("✓ Created 10 test molecules")
EOF

# 2. Test MPFIT on these molecules (should take ~2-3 minutes)
python -c "
from mmomenta.data import BatchProcessor
from mmomenta.qm import MPFITCalculator, MPFITConfig, QMSettings
import pickle

with open('data/test_molecules_10.pkl', 'rb') as f:
    molecules = pickle.load(f)

calculator = MPFITCalculator(
    qc_settings=QMSettings(method='hf', basis='6-31G*'),
    config=MPFITConfig(limit=8)
)

processor = BatchProcessor(calculator)
results = processor.process_batch(molecules, n_jobs=2)
print(f'✓ Processed {len(results)} molecules successfully')
"

# 3. Now you can test the full prepare_dataset.py script
# (once you create it - see next section)
```

## Creating the Missing Training Scripts

You have the MMomentA package code, but need CLI scripts. Here's the minimal version:

**Create `scripts/prepare_dataset.py`:**

```bash
mkdir -p scripts
cat > scripts/prepare_dataset.py << 'EOF'
#!/usr/bin/env python
"""Minimal dataset preparation script for MVP"""
import argparse
import pickle
from pathlib import Path
from openff.toolkit import Molecule
from mmomenta.qm import MPFITCalculator, MPFITConfig, QMSettings
from mmomenta.data import BatchProcessor, DatasetMetadata, save_dataset
from mmomenta.data.extraction import molecule_to_molecule_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input molecules (.pkl, .sdf, .oeb)')
    parser.add_argument('--output', required=True, help='Output HDF5 file')
    parser.add_argument('--max-molecules', type=int, default=50)
    parser.add_argument('--method', default='hf')
    parser.add_argument('--basis', default='6-31G*')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()

    # Load molecules
    print(f"Loading molecules from {args.input}")
    if args.input.endswith('.pkl'):
        with open(args.input, 'rb') as f:
            molecules = pickle.load(f)
    else:
        molecules = Molecule.from_file(args.input, allow_undefined_stereo=True)

    molecules = molecules[:args.max_molecules]
    print(f"Processing {len(molecules)} molecules")

    # Run MPFIT
    calculator = MPFITCalculator(
        qc_settings=QMSettings(method=args.method, basis=args.basis),
        config=MPFITConfig(limit=args.limit)
    )
    processor = BatchProcessor(calculator)
    results = processor.process_batch(molecules, n_jobs=args.n_jobs)

    # Convert to MoleculeData
    mol_data_list = []
    for mol, result in zip(molecules, results):
        if result is not None:
            mol_data = molecule_to_molecule_data(mol, result)
            mol_data_list.append(mol_data)

    # Create splits
    n = len(mol_data_list)
    train_idx = list(range(int(0.7*n)))
    val_idx = list(range(int(0.7*n), int(0.85*n)))
    test_idx = list(range(int(0.85*n), n))

    metadata = DatasetMetadata(
        dataset_name="MVP",
        n_molecules=n,
        splits={"train": train_idx, "val": val_idx, "test": test_idx}
    )

    # Save
    save_dataset(mol_data_list, metadata, args.output)
    print(f"✓ Saved {n} molecules to {args.output}")

if __name__ == '__main__':
    main()
EOF

chmod +x scripts/prepare_dataset.py
```

**Create `scripts/train_spice.py`:**

```bash
cat > scripts/train_spice.py << 'EOF'
#!/usr/bin/env python
"""Minimal training script for MVP"""
import argparse
import torch
from mmomenta.data import load_dataset, ChargeDataset
from mmomenta.models import ChargeModel, ModelConfig
from mmomenta.training import Trainer, TrainingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--feature-units', type=int, default=198)
    parser.add_argument('--no-multipoles', action='store_true')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # Load data
    mol_data, metadata = load_dataset(args.dataset)
    include_multipoles = not args.no_multipoles

    train_dataset = ChargeDataset(
        [mol_data[i] for i in metadata.splits['train']],
        include_multipoles=include_multipoles
    )
    val_dataset = ChargeDataset(
        [mol_data[i] for i in metadata.splits['val']],
        include_multipoles=include_multipoles
    )
    test_dataset = ChargeDataset(
        [mol_data[i] for i in metadata.splits['test']],
        include_multipoles=include_multipoles
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    model_config = ModelConfig(
        feature_units=args.feature_units,
        depth=args.depth,
        width=args.width
    )
    model = ChargeModel(model_config)

    # Train
    training_config = TrainingConfig(
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device
    )
    trainer = Trainer(model, training_config)
    trainer.train(train_loader, val_loader, test_loader)

    print(f"✓ Training complete: {args.output_dir}")

if __name__ == '__main__':
    main()
EOF

chmod +x scripts/train_spice.py
```

## Quick MVP Test (5 minutes)

```bash
cd MMomentA

# 1. Create test molecules (instant)
python -c "
from openff.toolkit import Molecule
import pickle
smiles = ['CC(=O)O', 'c1ccccc1', 'CCO', 'C1CCCCC1', 'CC(C)O']
molecules = [Molecule.from_smiles(s) for s in smiles]
for mol in molecules:
    mol.generate_conformers(n_conformers=1)
with open('data/test_5.pkl', 'wb') as f:
    pickle.dump(molecules, f)
"

# 2. Prepare dataset (2-3 minutes)
python scripts/prepare_dataset.py \
    --input data/test_5.pkl \
    --output data/test_5.h5 \
    --max-molecules 5 \
    --n-jobs 2

# 3. Train tiny model (1-2 minutes)
python scripts/train_spice.py \
    --dataset data/test_5.h5 \
    --output-dir runs/test \
    --feature-units 198 \
    --depth 2 \
    --width 32 \
    --n-epochs 10 \
    --batch-size 2 \
    --device cpu

# If all 3 steps work, you're ready for the full MVP!
```

## For the Full 2-Hour MVP

**Once environment is ready:**

```bash
# Option 1: Use test molecules scaled up
python -c "
from openff.toolkit import Molecule
import pickle
# 50 diverse drug-like SMILES
smiles = [
    'CC(=O)O', 'c1ccccc1', 'CCO', 'C1CCCCC1', 'CC(C)O',
    # Add 45 more diverse SMILES here...
]
molecules = [Molecule.from_smiles(s) for s in smiles]
for mol in molecules:
    mol.generate_conformers(n_conformers=1)
with open('data/spice_raw.hdf5', 'wb') as f:
    pickle.dump(molecules, f)
"

# Option 2: Download real SPICE subset
# (Takes longer but higher quality)
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5
# Then extract 50 molecules with Python

# Then run MVP
sbatch phoenix/submit_mvp_2hr.sh
```

## Summary

**What you DON'T have:**
1. Dataset files (need to download or create)
2. Training scripts (need to create minimal versions above)

**What you DO have:**
1. All MMomentA package code (qm/, data/, models/, training/)
2. Phoenix SLURM scripts
3. Environment configuration

**Fastest path to working MVP:**
1. Create test molecules (5 min) ← Do this now
2. Create training scripts (5 min) ← Copy from above
3. Test on 5 molecules (5 min) ← Verify it works
4. Scale to 50 molecules (2 hours) ← Submit MVP job

**You can start testing immediately without any downloads** using the 5-molecule test above.
