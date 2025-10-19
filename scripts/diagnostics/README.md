# Diagnostic Scripts for MMomentA

This directory contains diagnostic and debugging scripts for analyzing model performance, data quality, and feature extraction.

## Scripts Overview

### 1. Model Evaluation

#### `evaluate_saved_model.py`
**Purpose:** Comprehensive evaluation of trained model checkpoints on train/val/test splits.

**Usage:**
```bash
# Validate checkpoint loads correctly (should match training log: 0.08049 e)
python scripts/diagnostics/evaluate_saved_model.py \
    --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
    --dataset data/zinc_100k_mpfit.h5 \
    --split validation

# Check test set performance
python scripts/diagnostics/evaluate_saved_model.py \
    --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
    --dataset data/zinc_100k_mpfit.h5 \
    --split test

# Full evaluation across all splits
python scripts/diagnostics/evaluate_saved_model.py \
    --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
    --dataset data/zinc_100k_mpfit.h5 \
    --split all
```

**Outputs:**
- Overall RMSE, MAE, max error per split
- Per-element breakdown (H, C, N, O, F, S, Cl, Br, P)
- Comparison to training log performance
- Train/test gap analysis

---

#### `quick_model_test.py`
**Purpose:** Quick sanity check for model loading and basic inference.

**Usage:**
```bash
python scripts/diagnostics/quick_model_test.py \
    --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
    --dataset data/zinc_100k_mpfit.h5 \
    --n-molecules 5
```

**What it checks:**
- Model loads without errors
- Predictions are non-zero and non-constant
- Charge conservation works
- Per-atom predictions have variance

---

### 2. Data Quality

#### `compare_split_distributions.py`
**Purpose:** Analyze molecular property distributions across train/val/test splits to identify distribution mismatch.

**Usage:**
```bash
python scripts/diagnostics/compare_split_distributions.py \
    --dataset data/zinc_100k_mpfit.h5
```

**What it checks:**
- Number of atoms per molecule (mean, std, median, min, max)
- Heavy atom counts
- Charge ranges (max - min per molecule)
- Charge magnitudes (|q|)
- Element fractions (% of H, C, N, O, etc.)

**Thresholds:**
- Molecule size: >10% difference → flag
- Charge range: >15% difference → flag
- Element fractions: >3% difference → flag

**Interpretation:**
- If significant differences detected → Distribution mismatch explains poor test performance
- If distributions similar → Look for other causes

---

#### `diagnose_mpfit_charges.py`
**Purpose:** Inspect MPFIT charges in dataset to identify unit/scaling issues.

**Usage:**
```bash
python scripts/diagnostics/diagnose_mpfit_charges.py \
    --dataset data/zinc_100k_mpfit.h5 \
    --n-molecules 100
```

**Expected ranges:**
- Typical: -1.0 to +1.0 e
- Maximum: -2.0 to +2.0 e
- If outside → UNIT/SCALING BUG

**Outputs:**
- Overall charge statistics
- Per-element charge distributions
- Sample molecules with detailed charges
- Diagnostic recommendations

---

#### `test_mpfit_single.py`
**Purpose:** Test MPFIT calculation on a single molecule (ethanol) to verify correctness.

**Usage:**
```bash
python scripts/diagnostics/test_mpfit_single.py
```

**What it does:**
- Computes MPFIT charges for ethanol
- Compares to AM1-BCC reference
- Checks for unit conversion bugs
- Inspects raw multipole moments

**Expected ethanol charges:**
- C: ~-0.1 to +0.1 e
- H: ~0.0 to +0.1 e
- O: ~-0.6 to -0.7 e

---

### 3. Feature Engineering

#### `inspect_features.py`
**Purpose:** Verify feature extraction matches expected dimensions and values.

**Usage:**
```bash
python scripts/diagnostics/inspect_features.py \
    --dataset data/zinc_100k_mpfit.h5 \
    --n-molecules 10
```

**Expected feature breakdown:**
- One-hot element encoding: 100
- Atomic fingerprint: 17
- Multipole moments: 81
- **Total: 198 dimensions**

**What it checks:**
- Actual vs expected feature dimensions
- Multipole component count (should be 81)
- Multipole value ranges (non-zero, reasonable variance)
- Feature concatenation correctness

**Known issue:** Training log showed 197 features (not 198)

---

## Recommended Diagnostic Workflow

Run these scripts in order to diagnose model performance issues:

### Step 1: Quick Sanity Check (fast)
```bash
python scripts/diagnostics/quick_model_test.py \
    --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
    --dataset data/zinc_100k_mpfit.h5
```

### Step 2: Feature Inspection (fast)
```bash
python scripts/diagnostics/inspect_features.py \
    --dataset data/zinc_100k_mpfit.h5
```

### Step 3: Distribution Comparison (fast)
```bash
python scripts/diagnostics/compare_split_distributions.py \
    --dataset data/zinc_100k_mpfit.h5
```

### Step 4: Full Model Evaluation (slower)
```bash
# Validation (should be ~0.08 e)
python scripts/diagnostics/evaluate_saved_model.py \
    --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
    --dataset data/zinc_100k_mpfit.h5 \
    --split validation

# Test (check if matches 0.3-0.7 e)
python scripts/diagnostics/evaluate_saved_model.py \
    --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
    --dataset data/zinc_100k_mpfit.h5 \
    --split test
```

---

## Common Issues and Solutions

### Issue 1: Large Val/Test Performance Gap
**Symptoms:**
- Val RMSE = 0.08 e
- Test RMSE = 0.3-0.7 e

**Diagnosis:**
Run `compare_split_distributions.py` to check for distribution mismatch.

**Solutions:**
- Retrain with random split instead of scaffold split
- Use stratified sampling to balance molecular properties
- Combine datasets (SPICE + ZINC) for more diversity

---

### Issue 2: MPFIT Charges Too Large
**Symptoms:**
- Charges > 2.0 e
- Unrealistic charge distributions

**Diagnosis:**
Run `diagnose_mpfit_charges.py` and `test_mpfit_single.py`.

**Solutions:**
- Check unit conversion in `MMomentA/qm/mpfit.py`
- Verify GDMA output units
- Compare to AM1-BCC reference

---

### Issue 3: Feature Dimension Mismatch
**Symptoms:**
- Expected 198 features, got 197
- Training log shows different feature count

**Diagnosis:**
Run `inspect_features.py` to identify missing features.

**Solutions:**
- Check GDMA limit parameter (should be 8 → 81 components)
- Verify multipole extraction in `MMomentA/qm/mpfit.py`
- Fix feature concatenation in `MMomentA/data/graph.py`

---

### Issue 4: Model Not Loading Correctly
**Symptoms:**
- Predictions all zero or constant
- Val RMSE ≠ training log

**Diagnosis:**
Run `quick_model_test.py` for rapid debugging.

**Solutions:**
- Check checkpoint file integrity
- Verify model architecture matches training config
- Ensure feature extraction matches training

---

## Additional Notes

- All scripts support `--help` flag for detailed usage
- Scripts are designed to work on both local machines and HPC clusters
- Output is formatted for easy copy/paste into documentation
- Per-element analysis covers: H, C, N, O, F, S, Cl, Br, P

---

## Related Files

- Training pipeline: `perlmutter/pipelines/run_zinc_100k_step3_6.slurm`
- ESP validation: `scripts/validate_esp_comparison_chunk.py`
- Feature extraction: `MMomentA/data/graph.py`
- MPFIT calculator: `MMomentA/qm/mpfit.py`
- Model architecture: `MMomentA/models/charge_model.py`
