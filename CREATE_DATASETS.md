# How to Create All Three Datasets

You now have both environments working. Here's how to get your three datasets.

## Quick Reference

| Dataset | Purpose | Size | Time | Command |
|---------|---------|------|------|---------|
| Test SMILES | MVP testing | 50 mols | 5 min | `python scripts/create_test_smiles.py` |
| SPICE | Real training | 114K mols | 30 min download | `bash scripts/download_spice.sh` |
| ZINC | Transfer learning | 100-500 mols | 10 min | `bash scripts/download_zinc.sh` |

## Dataset 1: Test SMILES (START HERE)

**Best for immediate MVP testing - no download needed!**

```bash
# Activate ML environment (has OpenFF)
module load anaconda3
conda activate mmomenta-ml

# Create 50 diverse test molecules
python scripts/create_test_smiles.py --output data/test_smiles.pkl

# Output: data/test_smiles.pkl (50 molecules)
```

**Next: Generate MPFIT charges**

```bash
# Switch to data environment
conda activate mmomenta-data

# Compute MPFIT charges (20-30 min on 8 cores)
python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --n-jobs 8
```

**Next: Train baseline model**

```bash
# Switch to ML environment
conda activate mmomenta-ml

# Train (10 min on GPU)
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_baseline \
    --no-multipoles \
    --n-epochs 100 \
    --device cuda
```

## Dataset 2: SPICE (Real QM Data)

**8.5 GB download, 114K molecules**

```bash
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA

# Download (10-30 min depending on network)
bash scripts/download_spice.sh

# Output: data/SPICE-2.0.1.hdf5
```

**Option A: Extract subset (recommended for MVP)**

```bash
# Create subset extraction script
python << 'EOF'
import h5py
import pickle
from openff.toolkit import Molecule
from pathlib import Path

def extract_spice_subset(input_h5, output_pkl, n_molecules=100):
    """Extract first N molecules from SPICE."""

    molecules = []

    with h5py.File(input_h5, 'r') as f:
        mol_ids = list(f.keys())[:n_molecules]

        for mol_id in mol_ids:
            grp = f[mol_id]

            # Get SMILES or create from coordinates
            if 'smiles' in grp.attrs:
                smiles = grp.attrs['smiles']
                mol = Molecule.from_smiles(smiles)
            else:
                # Skip if no SMILES
                continue

            # Use QM geometry as conformer
            coords = grp['conformations'][0]  # Take first conformer
            mol.add_conformer(coords)

            molecules.append(mol)

    # Save
    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pkl, 'wb') as f:
        pickle.dump(molecules, f)

    print(f"✓ Extracted {len(molecules)} molecules to {output_pkl}")

# Run extraction
extract_spice_subset('data/SPICE-2.0.1.hdf5', 'data/spice_100.pkl', n_molecules=100)
EOF
```

**Option B: Use full dataset**

```bash
conda activate mmomenta-data

# This will take hours for full dataset
python scripts/prepare_dataset.py \
    --input data/SPICE-2.0.1.hdf5 \
    --output data/spice_full_mpfit.h5 \
    --n-jobs 16
```

## Dataset 3: ZINC (Transfer Learning)

**Option A: OpenFF ZINC Fragments (easiest)**

```bash
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA

# Download and convert
bash scripts/download_zinc.sh

# This downloads zinc_fragments.smi, then:
conda activate mmomenta-ml

python scripts/smiles_to_molecules.py \
    --input data/geom/zinc_fragments.smi \
    --output data/zinc_molecules.pkl \
    --max-molecules 100
```

**Option B: Sample ZINC SMILES (instant)**

The download script creates `data/geom/zinc_sample.smi` with 10 representative ZINC molecules. Use this for quick testing:

```bash
conda activate mmomenta-ml

python scripts/smiles_to_molecules.py \
    --input data/geom/zinc_sample.smi \
    --output data/zinc_molecules.pkl
```

**Next: Generate MPFIT charges**

```bash
conda activate mmomenta-data

python scripts/prepare_dataset.py \
    --input data/zinc_molecules.pkl \
    --output data/zinc_mpfit.h5 \
    --n-jobs 8
```

## Complete MVP Workflow (2 Hours)

**Step 1: Test Dataset (5 min)**
```bash
conda activate mmomenta-ml
python scripts/create_test_smiles.py --output data/test_smiles.pkl
```

**Step 2: MPFIT Charges (30 min)**
```bash
conda activate mmomenta-data
python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --n-jobs 8
```

**Step 3: Train Baseline (20 min)**
```bash
conda activate mmomenta-ml
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/baseline \
    --no-multipoles \
    --n-epochs 200 \
    --device cuda
```

**Step 4: Train with Multipoles (20 min)**
```bash
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/multipoles \
    --feature-units 198 \
    --n-epochs 200 \
    --device cuda
```

**Step 5: Compare Results (5 min)**
```bash
python scripts/compare_models.py \
    --baseline runs/baseline \
    --multipoles runs/multipoles \
    --output figures/
```

## Troubleshooting

**Issue: `create_test_smiles.py` fails with "No module named 'openff'"`**
- Solution: Make sure you're in mmomenta-ml environment: `conda activate mmomenta-ml`

**Issue: `prepare_dataset.py` fails with "No module named 'psi4'"`**
- Solution: Switch to data environment: `conda activate mmomenta-data`

**Issue: Download scripts fail with "command not found: wget"`**
- Solution: Use `curl -O <url>` instead of `wget <url>`

**Issue: SPICE download is slow****
- Solution: Start with test SMILES dataset first, download SPICE in background

## What You Should Have After This

```
data/
├── test_smiles.pkl          # 50 test molecules (instant)
├── test_mpfit.h5            # MPFIT charges for test set (30 min)
├── SPICE-2.0.1.hdf5         # Full SPICE dataset (optional)
├── spice_100.pkl            # SPICE subset (optional)
├── zinc_molecules.pkl       # ZINC molecules (10 min)
└── zinc_mpfit.h5            # MPFIT charges for ZINC (30 min)
```

## Next Steps After Datasets

Once you have `data/test_mpfit.h5`:

```bash
# Run complete MVP
bash phoenix/submit_mvp_2hr.sh

# Or manually train and compare
python scripts/train_spice.py --dataset data/test_mpfit.h5 --output-dir runs/baseline --no-multipoles
python scripts/train_spice.py --dataset data/test_mpfit.h5 --output-dir runs/multipoles
python scripts/compare_models.py --baseline runs/baseline --multipoles runs/multipoles
```

## Time Estimates

| Task | Time | Environment |
|------|------|-------------|
| Create test SMILES | 1 min | mmomenta-ml |
| MPFIT for 50 mols | 20-30 min | mmomenta-data |
| Train baseline | 10-20 min | mmomenta-ml |
| Train multipoles | 10-20 min | mmomenta-ml |
| Generate figures | 5 min | mmomenta-ml |
| **Total MVP** | **1-2 hours** | |

**Start with Dataset 1 (Test SMILES) to verify your pipeline works end-to-end!**
