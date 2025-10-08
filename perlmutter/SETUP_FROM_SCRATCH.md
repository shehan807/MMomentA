# Perlmutter Setup From Scratch

This guide shows how to set up MMomentA on Perlmutter from a completely fresh start. **All known issues are now fixed in the main setup scripts** - no manual fixes needed!

## Prerequisites

- Access to Perlmutter
- MMomentA repository cloned to `/global/homes/p/parmar/MoML/MMomentA`

## One-Time Setup (15-20 minutes)

### Step 1: Configure Conda

First time only - set up conda to use your SCRATCH directory:

```bash
cd /global/homes/p/parmar/MoML/MMomentA
bash perlmutter/setup/configure_conda.sh
```

This creates `~/.condarc` with proper package cache locations.

### Step 2: Create Data Environment

```bash
bash perlmutter/setup/MANUAL_SETUP_DATA.sh
```

This installs (~10 minutes):
- ✅ Psi4 (QM calculations)
- ✅ PyGDMA (MPFIT charges)
- ✅ OpenFF toolkit
- ✅ Scientific packages (numpy, scipy, h5py)
- ✅ Plotting libraries (matplotlib, seaborn)
- ✅ MMomentA package

### Step 3: Create ML Environment

```bash
bash perlmutter/setup/MANUAL_SETUP_ML.sh
```

This installs (~10 minutes):
- ✅ PyTorch with CUDA support
- ✅ torchdata (DGL dependency)
- ✅ DGL (Deep Graph Library)
- ✅ OpenFF toolkit
- ✅ pint <0.24 (compatibility fix)
- ✅ lovelyplots (publication figures)
- ✅ MMomentA package

### Step 4: Verify Installation

Both setup scripts automatically test the installation. You should see:

**Data environment**:
```
✓ Psi4 installed
✓ OpenFF toolkit installed
✓ HDF5 support installed
✓ mmomenta-data environment ready!
```

**ML environment**:
```
✓ PyTorch 2.x installed
  CUDA available: True
  CUDA version: 12.1
✓ DGL x.x installed
✓ OpenFF toolkit installed
✓ lovelyplots installed
✓ mmomenta-ml environment ready!
```

## Running the Pipeline

```bash
cd /global/homes/p/parmar/MoML/MMomentA
bash perlmutter/pipelines/run_mvp.sh
```

**Expected timeline**:
- Step 1: Create dataset (~5 min)
- Step 2: MPFIT charges (~5-8 min first run, ~5 sec if cached)
- Step 3: Train baseline (~10-15 min)
- Step 4: Train multipole (~10-15 min)
- Step 5: Compare results (~1 min)
- Step 6: ESP validation (~10-15 min)

**Total**: ~45-60 minutes first run

## What's Been Fixed

All these issues are **automatically fixed** in the setup scripts:

### 1. ✅ Pint Version Incompatibility
- **Issue**: `TypeError: cannot inherit frozen dataclass from a non-frozen one`
- **Fix**: `pip install "pint<0.24"` in ML setup script (line 88)

### 2. ✅ DGL Version Compatibility
- **Issue**: `ModuleNotFoundError: No module named 'torchdata'`
- **Root Cause**: DGL 2.0+ requires torchdata (deprecated)
- **Fix**: `conda install "dgl<2.0"` in ML setup script (line 74)

### 3. ✅ Missing PyGDMA
- **Issue**: `ModuleNotFoundError: Python module gdma not found`
- **Fix**: `conda install pygdma` in data setup script (line 38)

### 4. ✅ MMomentA Not Installed
- **Issue**: `ModuleNotFoundError: No module named 'MMomentA'`
- **Fix**: `pip install -e .` in both setup scripts (lines 62/109)

### 5. ✅ Parallel Processing Enabled
- **Was**: `--n-jobs 1` (sequential, slow)
- **Now**: `--n-jobs 16` (parallel, 4x faster)
- **Why**: Pickle error fix already in `batch.py`

### 6. ✅ Smart Caching Implemented
- **Was**: Recompute all molecules every run
- **Now**: Cache MPFIT results, only compute new molecules
- **Impact**: Subsequent runs ~instant instead of 20-30 min

## No Manual Fixes Needed!

The old `FIX_*.sh` scripts are **no longer needed** if you create environments from scratch using the updated setup scripts.

However, if you already have environments and don't want to recreate them, you can still use:
- `FIX_PINT_ERROR.sh` - Fix existing ML environment
- `FIX_TORCHDATA.sh` - Fix existing ML environment

## Recreating Environments

If you encounter issues with existing environments, recreate them:

```bash
# Delete old environments
rm -rf $SCRATCH/conda-envs/mmomenta-data
rm -rf $SCRATCH/conda-envs/mmomenta-ml

# Recreate from scratch
bash perlmutter/setup/MANUAL_SETUP_DATA.sh
bash perlmutter/setup/MANUAL_SETUP_ML.sh

# Run pipeline
bash perlmutter/pipelines/run_mvp.sh
```

## Troubleshooting

### "Conda package cache not writable"

Run: `bash perlmutter/setup/configure_conda.sh`

This is only needed once per user account.

### "Module load conda not found"

You're not on Perlmutter, or need to update your environment. On Perlmutter:
```bash
module avail conda  # Check available modules
module load conda   # Load conda module
```

### Environment creation fails

Check disk quota:
```bash
myquota           # Check SCRATCH quota
df -h $SCRATCH    # Check available space
```

MMomentA environments need ~5-10 GB total.

### Pipeline fails with import errors

Verify MMomentA is installed:
```bash
conda activate $SCRATCH/conda-envs/mmomenta-ml
python -c "from MMomentA import ChargeModel; print('✓ Works')"
```

If not installed:
```bash
cd /global/homes/p/parmar/MoML/MMomentA
pip install -e .
```

## Summary

**Old way** (required manual fixes):
1. Run setup scripts
2. Run FIX_PINT_ERROR.sh
3. Run FIX_TORCHDATA.sh
4. Manually install MMomentA
5. Hope everything works

**New way** (just works):
1. bash perlmutter/setup/configure_conda.sh (once)
2. bash perlmutter/setup/MANUAL_SETUP_DATA.sh
3. bash perlmutter/setup/MANUAL_SETUP_ML.sh
4. bash perlmutter/pipelines/run_mvp.sh ✨

## Files Modified

Setup scripts now include all fixes:
- `perlmutter/setup/MANUAL_SETUP_DATA.sh` - Added pygdma, MMomentA install
- `perlmutter/setup/MANUAL_SETUP_ML.sh` - Added torchdata, pint fix, MMomentA install
- `perlmutter/pipelines/run_mvp.sh` - Uses parallel processing + caching

Legacy fix scripts (kept for existing environments):
- `perlmutter/setup/FIX_PINT_ERROR.sh`
- `perlmutter/setup/FIX_TORCHDATA.sh`

## Next Steps

After successful MVP run:
1. Check results in `runs/test_{baseline,multipoles}/`
2. View ESP validation: `figures/esp_validation_mpfit_vs_gnn.png`
3. Scale up to larger datasets (SPICE, ZINC)

### SPICE Pipeline (100 molecules)

```bash
bash perlmutter/pipelines/run_spice.sh
```

**What it does**:
1. Extracts 100 molecules from SPICE-2.0.1.hdf5
2. Computes MPFIT charges (5-8 min with caching)
3. Trains baseline model (no multipoles)
4. Trains multipole model
5. Compares baseline vs multipole performance
6. **ESP validation**: Compares RESP vs AM1-BCC vs MPFIT vs MMomentA-GNN

**Total time**: ~1-1.5 hours

**Results**:
- Training: `runs/spice_{baseline,multipoles}/`
- ESP comparison: `figures/spice_esp_comparison/esp_comparison_violin.png`

### ZINC Pipeline (100 molecules + Transfer Learning)

**Prerequisites**: Run SPICE pipeline first (needs pretrained model)

```bash
bash perlmutter/pipelines/run_zinc.sh
```

**What it does**:
1. Downloads ZINC fragments dataset
2. Converts 100 SMILES to molecules
3. Computes MPFIT charges (5-8 min with caching)
4. Trains from scratch on ZINC
5. Fine-tunes SPICE model on ZINC (transfer learning)
6. Compares transfer vs from-scratch performance
7. **ESP validation**: Compares RESP vs AM1-BCC vs MPFIT vs MMomentA-GNN

**Total time**: ~1-1.5 hours

**Results**:
- From scratch: `runs/zinc_from_scratch/`
- Transfer learning: `runs/zinc_transfer/`
- ESP comparison: `figures/zinc_esp_comparison/esp_comparison_violin.png`

### Complete Scientific Workflow

```bash
# Step 1: MVP (150 test molecules)
bash perlmutter/pipelines/run_mvp.sh

# Step 2: SPICE (100 molecules from SPICE dataset)
bash perlmutter/pipelines/run_spice.sh

# Step 3: ZINC (100 molecules + transfer learning)
bash perlmutter/pipelines/run_zinc.sh
```

**Total time**: ~3-4 hours for complete workflow

Enjoy your working MMomentA setup! 🎉
