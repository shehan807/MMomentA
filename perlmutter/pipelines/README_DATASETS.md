# Dataset Pipelines for MMomentA

## Available Datasets via PyTorch Geometric

### 1. QM9 (130K molecules, small organic)
```bash
bash perlmutter/pipelines/run_qm9.sh
```

**Dataset info:**
- 130,000 molecules (using 1000 by default)
- All molecules < 9 heavy atoms
- Elements: C, N, O, F
- Includes 3D coordinates
- Good for: method validation, quick tests

### 2. ZINC (Drug-like molecules)
```bash
bash perlmutter/pipelines/run_zinc.sh
```

**Dataset info:**
- Drug-like molecules from ZINC15
- Pre-split train/val/test
- Diverse functional groups
- Good for: transfer learning, real-world applications

### 3. MVP Test (Small test set)
```bash
bash perlmutter/pipelines/run_mvp.sh
```

**Dataset info:**
- 318 molecules (test set)
- Used for pipeline validation
- Fast to run

## Quick Start

### Download Dataset Only
```bash
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-ml

# QM9
python scripts/download_pyg_dataset.py \
    --dataset qm9 \
    --output data/qm9_molecules.pkl \
    --max-molecules 1000

# ZINC
python scripts/download_pyg_dataset.py \
    --dataset zinc \
    --output data/zinc_molecules.pkl \
    --max-molecules 1000 \
    --split train
```

### Prepare MPFIT Dataset
```bash
conda activate $SCRATCH/conda-envs/mmomenta-data

python scripts/prepare_dataset_cached.py \
    --input data/qm9_molecules.pkl \
    --output data/qm9_mpfit.h5 \
    --n-jobs 16
```

### Train Models
```bash
conda activate $SCRATCH/conda-envs/mmomenta-ml

# Baseline (no multipoles)
python scripts/train_spice.py \
    --dataset data/qm9_mpfit.h5 \
    --output-dir runs/qm9_baseline \
    --no-multipoles \
    --n-epochs 1000 \
    --device cuda

# Multipole model
python scripts/train_spice.py \
    --dataset data/qm9_mpfit.h5 \
    --output-dir runs/qm9_multipoles \
    --n-epochs 1000 \
    --device cuda
```

## Dataset Sizes

| Dataset | Molecules | Download Time | MPFIT Time (16 cores) |
|---------|-----------|---------------|----------------------|
| MVP Test | 318 | instant | ~5 min |
| QM9 (1K) | 1000 | ~2 min | ~15 min |
| ZINC (1K) | 1000 | ~2 min | ~15 min |
| QM9 (10K) | 10000 | ~5 min | ~2 hours |
| ZINC (10K) | 10000 | ~5 min | ~2 hours |

## Customizing Dataset Size

Edit the `--max-molecules` parameter in the pipeline scripts:

```bash
# For more molecules (e.g., 5000)
python scripts/download_pyg_dataset.py \
    --dataset zinc \
    --output data/zinc_5k.pkl \
    --max-molecules 5000
```

## Notes

- **Caching**: `prepare_dataset_cached.py` caches MPFIT results, so rerunning is fast
- **Parallel**: Use `--n-jobs 16` for parallel MPFIT computation
- **GPU**: Training requires GPU node (`--device cuda`)
- **Memory**: QM9/ZINC can run on login nodes for download, need GPU node for training
