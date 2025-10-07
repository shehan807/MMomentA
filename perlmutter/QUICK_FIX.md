# Quick Fix for Perlmutter Issues

## Two Issues to Fix

### Issue 1: Psi4 Pickle Error
**Temporary Workaround**: Use sequential processing (no parallel)

### Issue 2: MMomentA Not Installed in ML Environment
**Fix**: Install MMomentA package in both environments

## Quick Fix Steps

### 1. Install MMomentA Package

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

### 2. Pull Updated batch.py (For Parallel Support Later)

```bash
# Copy the fixed batch.py from your local machine
# Or manually update MMomentA/data/batch.py to wrap calculator calls in try-except
```

### 3. Use Updated Pipeline

The pipeline now uses `--n-jobs 1` to avoid pickle issues:

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
