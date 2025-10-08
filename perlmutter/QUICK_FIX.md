# Quick Fix for Perlmutter Issues

## Four Issues to Fix

### Issue 1: Psi4 Pickle Error
**Temporary Workaround**: Use sequential processing (no parallel)
**Status**: ✅ FIXED in batch.py (already applied)

### Issue 2: MMomentA Not Installed in ML Environment
**Fix**: Install MMomentA package in both environments
**Status**: ⚠️ NEEDS TO BE RUN

### Issue 3: Pint Version Incompatibility in ML Environment
**Error**: `TypeError: cannot inherit frozen dataclass from a non-frozen one`
**Fix**: Downgrade pint to <0.24
**Status**: ⚠️ NEEDS TO BE RUN

### Issue 4: DGL 2.x Requires Deprecated torchdata
**Error**: `ModuleNotFoundError: No module named 'torchdata'`
**Root Cause**: DGL 2.0+ requires torchdata (deprecated), conflicts with PyTorch 2.5+
**Fix**: Downgrade to DGL 1.1.x (stable, no torchdata needed)
**Status**: ⚠️ NEEDS TO BE RUN

## Quick Fix Steps

### 1. Fix Pint Version (ML Environment)

```bash
cd /global/homes/p/parmar/MoML/MMomentA

# Run the fix script
bash perlmutter/setup/FIX_PINT_ERROR.sh
```

This downgrades pint to <0.24 which is compatible with openff-units.

### 2. Fix DGL Version (ML Environment)

```bash
cd /global/homes/p/parmar/MoML/MMomentA

# Run the fix script
bash perlmutter/setup/FIX_DGL_VERSION.sh
```

This downgrades DGL to 1.1.x which doesn't require torchdata (deprecated package).

### 3. Install MMomentA Package

```bash
cd /global/homes/p/parmar/MoML/MMomentA

# Install in data environment
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-data
pip install -e .

# Install in ML environment
conda activate $SCRATCH/conda-envs/mmomenta-ml
pip install -e .
```

### 4. Use Updated Pipeline

The pipeline now uses `--n-jobs 16` for parallel processing with smart caching:

```bash
bash perlmutter/pipelines/run_mvp.sh
```

## Alternative: Manual Run (Testing)

If you want to test step-by-step:

```bash
# Step 1: Create dataset
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-data
python scripts/create_test_smiles.py

# Step 2: MPFIT charges (sequential to avoid pickle error)
python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --split-strategy random \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --n-jobs 1

# Step 3: Train baseline
conda activate $SCRATCH/conda-envs/mmomenta-ml
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_baseline \
    --no-multipoles \
    --n-epochs 100 \
    --device cuda

# Step 4: Train multipole
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_multipoles \
    --n-epochs 100 \
    --device cuda
```

## Why This Happens

1. **Pickle Error**: Psi4 errors in the old code couldn't serialize across processes
   - **Workaround**: Use `--n-jobs 1` (sequential)
   - **Proper Fix**: Update `batch.py` with exception handling (already done in repo)

2. **MMomentA Not Found**: Package not installed in environments
   - **Fix**: `pip install -e .` in both environments

## Performance Impact

Using `--n-jobs 1`:
- **Slower**: ~30-45 min instead of ~20 min for MPFIT charges (159 molecules)
- **Works**: No pickle errors!

Once you update `batch.py` (or pull from repo), you can use `--n-jobs 16` again.
