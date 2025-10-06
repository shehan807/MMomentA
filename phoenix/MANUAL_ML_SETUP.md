# Manual ML Environment Setup (Step-by-Step)

## This Approach Works!

These are the exact commands that successfully created the ML environment.

## Setup Commands

```bash
# 1. Load modules
module load anaconda3
module load cuda

# 2. Create base environment
conda create -n mmomenta-ml python=3.11 -y

# 3. Activate it
conda activate mmomenta-ml

# 4. Install PyTorch with CUDA
conda install -y pytorch=2.4 pytorch-cuda=12.1 -c pytorch -c nvidia

# 5. Install DGL
conda install -y -c dglteam/label/th24_cu121 dgl

# 6. Install scientific packages
conda install -c conda-forge -y numpy scipy pandas matplotlib h5py joblib

# 7. Install MMomentA
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA
pip install -e .

# 8. Verify installation
python -c "import torch, dgl; print(f'PyTorch: {torch.__version__}'); print(f'DGL: {dgl.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## What Each Step Does

1. **Load modules** - Ensures conda and CUDA are available
2. **Create environment** - Fresh Python 3.11 environment
3. **Activate** - Switch to new environment
4. **PyTorch** - Install PyTorch 2.4 with CUDA 12.1 support
5. **DGL** - Install Deep Graph Library for GNN models
6. **Scientific** - NumPy, SciPy, plotting, HDF5 support
7. **MMomentA** - Install your package in editable mode
8. **Verify** - Check everything works

## Verification

After setup, test it:

```bash
conda activate mmomenta-ml

python << 'EOF'
import torch
import dgl
from mmomenta.models import ChargeModel
from mmomenta.training import Trainer

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ DGL:", dgl.__version__)
print("✓ MMomentA ML modules loaded")
print("\nEnvironment ready for training!")
EOF
```

## Quick Reference

**Activate environment:**
```bash
module load anaconda3 cuda
conda activate mmomenta-ml
```

**Deactivate:**
```bash
conda deactivate
```

**Check what's installed:**
```bash
conda list
```

## If You Need to Recreate

```bash
conda deactivate
conda env remove -n mmomenta-ml
# Then run all setup commands again
```
