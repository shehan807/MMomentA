# Two-Environment Workflow for MMomentA

## Overview

Split MMomentA into two separate environments to avoid dependency conflicts:

1. **mmomenta-data** - QM calculations and dataset preparation (CPU, no GPU)
2. **mmomenta-ml** - ML training and inference (GPU required)

## Why This Approach?

**Problems with single environment:**
- PyTorch CUDA conflicts with Psi4/OpenMM CUDA requirements
- Heavy QM packages not needed during training
- GPU nodes waste time on CPU-bound QM calculations

**Benefits of separation:**
- Each environment installs cleanly (no CUDA conflicts)
- Data generation on CPU nodes (cheaper, more available)
- ML training on GPU nodes (only when needed)
- Faster environment setup

## Architecture Compatibility

**Yes, the MMomentA package supports this!**

The codebase is modular:
- `mmomenta.qm.*` - Only needed in data environment
- `mmomenta.data.{extraction,graph,dataset}` - Shared (minimal deps)
- `mmomenta.models.*` - Only needed in ML environment
- `mmomenta.training.*` - Only needed in ML environment

**Data exchange via HDF5:**
- Data env generates: `data/prepared.h5`
- ML env consumes: `data/prepared.h5`
- No cross-environment dependencies

## Installation

### Environment 1: Data Generation (mmomenta-data)

```bash
# On Phoenix login node or CPU node
bash phoenix/install_data_only.sh
```

**Includes:**
- Psi4 + pygdma (QM calculations)
- OpenFF Toolkit (molecule handling)
- RDKit, openbabel (cheminformatics)
- No PyTorch, no DGL

**Use for:**
- MPFIT calculations
- Dataset preparation
- Charge method comparison

### Environment 2: ML Training (mmomenta-ml)

```bash
# On Phoenix login node
bash phoenix/install_ml_only.sh
```

**Includes:**
- PyTorch 2.4 + CUDA 12.1
- DGL (graph neural networks)
- NumPy, SciPy, H5py
- No Psi4, no OpenFF Toolkit

**Use for:**
- Training models
- Running predictions
- Evaluation and ablation studies

## Workflow

### Step 1: Generate Dataset (CPU Node)

```bash
# Request CPU node
salloc --nodes=1 --ntasks-per-node=32 --mem=64G --time=12:00:00

# Load data environment
module load anaconda3
conda activate mmomenta-data

# Prepare dataset
python scripts/prepare_dataset.py \
    --input data/molecules.pkl \
    --output data/spice_mpfit.h5 \
    --max-molecules 1000 \
    --n-jobs 32

exit  # Release CPU node
```

**Output:** `data/spice_mpfit.h5` (contains all features and targets)

### Step 2: Train Model (GPU Node)

```bash
# Request GPU node
salloc --gres=gpu:V100:1 --mem=32G -qinferno --time=8:00:00

# Load ML environment
module load anaconda3 cuda
conda activate mmomenta-ml

# Train model
python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/spice_multipoles \
    --feature-units 198 \
    --n-epochs 1000 \
    --device cuda

exit  # Release GPU node
```

**Output:** `runs/spice_multipoles/checkpoints/best_model.pt`

### Step 3: Transfer Learning (GPU Node)

```bash
# Same GPU environment
conda activate mmomenta-ml

python scripts/train_zinc.py \
    --dataset data/zinc_mpfit.h5 \
    --pretrained runs/spice_multipoles/checkpoints/best_model.pt \
    --output-dir runs/zinc_finetune \
    --device cuda
```

## SLURM Job Scripts

### Data Preparation Job

```bash
#!/bin/bash
#SBATCH --job-name=PREP_DATA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

module load anaconda3
conda activate mmomenta-data

python scripts/prepare_dataset.py \
    --input data/molecules.pkl \
    --output data/spice_mpfit.h5 \
    --n-jobs 32
```

### ML Training Job

```bash
#!/bin/bash
#SBATCH --job-name=TRAIN_ML
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH -qinferno
#SBATCH --time=8:00:00

module load anaconda3 cuda
conda activate mmomenta-ml

python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/model \
    --device cuda
```

## Environment Details

### mmomenta-data

**Size:** ~2-3 GB
**Setup time:** 20-40 min
**Python:** 3.11 (fixed)
**CUDA:** Not needed

**Key packages:**
- psi4 (QM)
- pygdma (multipole analysis)
- openff-toolkit (molecule handling)
- rdkit (cheminformatics)

### mmomenta-ml

**Size:** ~3-4 GB
**Setup time:** 5-10 min
**Python:** Auto-selected by PyTorch (likely 3.11 or 3.12)
**CUDA:** 12.1 (via pytorch-cuda)

**Key packages:**
- pytorch 2.4
- dgl (graph neural networks)
- numpy, scipy, pandas
- h5py (load datasets)

## Quick Start Commands

### Install Both Environments

```bash
# Install data environment (slower, CPU only)
bash phoenix/install_data_only.sh

# Install ML environment (faster, needs GPU)
bash phoenix/install_ml_only.sh
```

### Check Which Environment You're In

```bash
conda activate mmomenta-data
python -c "from mmomenta.qm import MPFITCalculator; print('Data environment')"

conda activate mmomenta-ml
python -c "from mmomenta.models import ChargeModel; print('ML environment')"
```

### Switch Between Environments

```bash
# Currently in data environment, need to train
conda deactivate
module load cuda  # ML needs CUDA
conda activate mmomenta-ml

# Currently in ML environment, need to prep data
conda deactivate
module unload cuda  # Data doesn't need CUDA
conda activate mmomenta-data
```

## Troubleshooting

### "Cannot import mmomenta.qm" in ML environment

**Expected!** The ML environment doesn't have Psi4/OpenFF.
- Switch to `mmomenta-data` for QM calculations
- Or use pre-computed datasets

### "Cannot import torch" in data environment

**Expected!** The data environment doesn't have PyTorch.
- Switch to `mmomenta-ml` for training
- Or generate datasets only

### HDF5 file compatibility

**Both environments use same HDF5 format**, so datasets are portable:
```bash
# Create in data env
conda activate mmomenta-data
python scripts/prepare_dataset.py --output data/dataset.h5

# Use in ML env
conda activate mmomenta-ml
python scripts/train_spice.py --dataset data/dataset.h5
```

## Advantages Summary

| Aspect | Single Env | Two Envs |
|--------|-----------|----------|
| Setup time | 30-60 min | 25-50 min (total) |
| Setup success rate | ~30% | ~80% |
| CUDA conflicts | Yes | No |
| Resource efficiency | Poor | Good |
| Debugging | Hard | Easy |
| Maintainability | Low | High |

**Recommendation: Use two-environment approach for production work.**

## Migration from Single Environment

If you have existing single environment:

```bash
# Export existing data
conda activate mmomenta
python scripts/prepare_dataset.py --output data/migration.h5

# Remove old environment
conda deactivate
conda env remove -n mmomenta

# Install new environments
bash phoenix/install_data_only.sh
bash phoenix/install_ml_only.sh

# Continue with training
conda activate mmomenta-ml
python scripts/train_spice.py --dataset data/migration.h5
```

Your data persists - only environment setup changes.
