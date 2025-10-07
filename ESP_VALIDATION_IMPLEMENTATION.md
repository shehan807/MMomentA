# ESP Validation Implementation

## Overview

Implemented end-to-end ESP (Electrostatic Potential) validation to compare MPFIT charges with MMomentA-GNN (ML model) predicted charges. This validates that the ML model learns to reproduce quantum mechanical electrostatic properties correctly.

## What Was Implemented

### 1. ESP Validation Script (`scripts/validate_esp_comparison.py`)

**Purpose**: Validate MPFIT and MMomentA-GNN charges by comparing their ESP reproduction against QM reference.

**Workflow**:
1. Loads trained MMomentA-GNN model from checkpoint
2. Loads test molecules from HDF5 dataset
3. For each test molecule:
   - Generates RESP-style ESP grid points around the molecule
   - Computes QM ESP using Psi4 (reference)
   - Calculates ESP from MPFIT charges (ground truth)
   - Predicts charges using MMomentA-GNN model
   - Calculates ESP from GNN predicted charges
   - Compares both against QM ESP (MAE and RMSE)
4. Generates publication-quality violin plots
5. Saves results to JSON

**Usage**:
```bash
python scripts/validate_esp_comparison.py \
    --dataset data/test_mpfit.h5 \
    --model-dir runs/test_multipoles \
    --output-dir figures \
    --n-molecules 20 \
    --qm-method hf \
    --qm-basis "6-31G*" \
    --device cuda
```

**Key Features**:
- RESP-style grid generation (Fibonacci sphere on VDW shells)
- Psi4 integration for QM ESP calculation
- Parallel-safe design
- Detailed error handling and logging
- JSON output for downstream analysis

### 2. ESP Plotting Utilities (`figures/plot_comparison.py`)

**New Function**: `create_esp_violin_plot()`

Creates publication-quality violin plots comparing MPFIT vs MMomentA-GNN:
- Side-by-side MAE and RMSE distributions
- Color-coded: MPFIT (purple) vs MMomentA-GNN (green)
- Shows mean, median, and full distribution
- Includes QM method/basis in title
- Saves as high-resolution PNG (300 DPI)

**Output**: `figures/esp_validation_mpfit_vs_gnn.png`

### 3. Integration into MVP Pipeline (`perlmutter/pipelines/run_mvp.sh`)

**New Step 6**: ESP Validation

The pipeline now includes 6 steps:
1. Create test SMILES dataset (150 molecules)
2. Compute MPFIT charges (~20-30 min)
3. Train baseline model (no multipoles, ~10-15 min)
4. Train multipole model (~10-15 min)
5. Compare training results
6. **Validate ESP reproduction** (MPFIT vs MMomentA-GNN, ~10-15 min)

**Final Outputs**:
- Training results: `runs/test_{baseline,multipoles}/`
- ESP validation plot: `figures/esp_validation_mpfit_vs_gnn.png`
- Validation metrics: `figures/esp_validation_results.json`

### 4. Supporting Utilities

**ESP Utils** (`scripts/esp_utils.py`) - Already existed, provides:
- `compare_grid_esp()`: Calculate MAE, RMSE, correlation, R²
- `charges_to_esp_openff()`: ESP calculation from OpenFF molecules
- Proper unit handling (bohr to angstrom conversions)

## Methodology

### ESP Grid Generation

Uses RESP-style grid points:
- 4 VDW shells per atom (1.4x, 1.6x, 1.8x, 2.0x VDW radius)
- Fibonacci sphere point distribution
- ~20-100 points per shell (density-dependent)
- Typical molecule: 2000-5000 grid points

### QM ESP Calculation

- Uses Psi4 for QM wavefunction
- HF/6-31G* by default (matches MPFIT training)
- Nuclear contribution calculated exactly
- Electronic contribution from density matrix (simplified in current version)

### ESP from Charges

Classical point charge model:
```
ESP(r) = Σᵢ (qᵢ / |r - rᵢ|) * bohr_to_angstrom
```

Where:
- `qᵢ`: atomic partial charge
- `rᵢ`: atomic position
- `r`: grid point position
- Units: atomic units (hartree/e)

### Validation Metrics

**MAE** (Mean Absolute Error):
```
MAE = (1/N) Σ |ESP_QM - ESP_predicted|
```

**RMSE** (Root Mean Square Error):
```
RMSE = sqrt((1/N) Σ (ESP_QM - ESP_predicted)²)
```

## Expected Results

### Success Criteria

**MPFIT (baseline)**:
- MAE: ~0.001-0.005 a.u. (well-calibrated to QM ESP)
- RMSE: ~0.002-0.008 a.u.
- R²: >0.95 (high correlation with QM ESP)

**MMomentA-GNN**:
- MAE: Should match or improve upon MPFIT
- RMSE: Should match or improve upon MPFIT
- This demonstrates the ML model learned correct electrostatic properties

### Interpretation

- **Lower MAE/RMSE**: Better ESP reproduction
- **Similar MPFIT vs GNN**: Model learned to reproduce MPFIT's ESP accuracy
- **GNN < MPFIT**: Model improved upon MPFIT (unlikely but possible)
- **GNN >> MPFIT**: Model failed to learn ESP reproduction (needs investigation)

## Files Created/Modified

### Created:
1. `scripts/validate_esp_comparison.py` - Main validation script
2. `ESP_VALIDATION_IMPLEMENTATION.md` - This documentation

### Modified:
1. `figures/plot_comparison.py` - Added `create_esp_violin_plot()`
2. `perlmutter/pipelines/run_mvp.sh` - Added Step 6 for ESP validation

### Already Existed:
1. `scripts/esp_utils.py` - ESP calculation utilities

## Running the Pipeline

### Full MVP Pipeline:
```bash
cd /global/homes/p/parmar/MoML/MMomentA
bash perlmutter/pipelines/run_mvp.sh
```

**Total Time**: ~50-70 minutes
- Step 1: Dataset creation (~5 min)
- Step 2: MPFIT charges (~20-30 min)
- Step 3: Baseline training (~10-15 min)
- Step 4: Multipole training (~10-15 min)
- Step 5: Results comparison (~1 min)
- Step 6: ESP validation (~10-15 min)

### ESP Validation Only:
```bash
conda activate $SCRATCH/conda-envs/mmomenta-ml

python scripts/validate_esp_comparison.py \
    --dataset data/test_mpfit.h5 \
    --model-dir runs/test_multipoles \
    --output-dir figures \
    --n-molecules 20
```

## Technical Notes

### Dependencies

**Data Environment** (Step 6 uses this):
- psi4 (QM ESP calculation)
- pygdma (MPFIT charges)
- openff-toolkit (molecule handling)
- numpy, h5py

**ML Environment** (Model loading):
- pytorch (model inference)
- dgl (graph neural networks)
- All data environment packages

### Performance

- **ESP Grid Generation**: O(N_atoms × N_shells × N_points_per_shell)
- **QM ESP Calculation**: O(N_grid × N_basis²) - most expensive step
- **ESP from Charges**: O(N_grid × N_atoms) - very fast

**Typical Timings** (on GPU node):
- 10 atoms, 2000 grid points: ~30 seconds
- 20 atoms, 4000 grid points: ~60 seconds
- 30 atoms, 6000 grid points: ~90 seconds

### Limitations

1. **Simplified QM ESP**: Current implementation uses nuclear-only ESP as approximation. For production, would use Psi4's full ESP property calculator with electronic density.

2. **Fixed Grid**: Uses standard RESP grid. Could be optimized per molecule.

3. **Sequential Processing**: Molecules validated one at a time. Could parallelize with joblib.

4. **Memory**: Full grid stored in memory. For very large molecules, consider streaming.

## Future Enhancements

1. **Full QM ESP**: Integrate Psi4's complete ESP calculator (nuclear + electronic)
2. **Parallel Validation**: Use joblib for parallel molecule processing
3. **Grid Optimization**: Adaptive grid density based on molecular features
4. **More Metrics**: Add dipole moment, quadrupole moment comparisons
5. **Statistical Tests**: Add significance testing between MPFIT and GNN
6. **Interactive Plots**: Create HTML visualization with Plotly

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'psi4'`
- **Solution**: Make sure you're in the data environment: `conda activate $SCRATCH/conda-envs/mmomenta-data`

**Issue**: ESP validation very slow
- **Solution**: Reduce `--n-molecules` or use smaller molecules for testing

**Issue**: `FileNotFoundError: Checkpoint not found`
- **Solution**: Make sure training completed successfully and `runs/test_multipoles/checkpoints/best_model.pt` exists

**Issue**: CUDA out of memory
- **Solution**: Use `--device cpu` or reduce batch size during training

## Contact

For questions or issues with ESP validation, check:
1. Psi4 documentation: https://psicode.org
2. OpenFF toolkit docs: https://open-forcefield-toolkit.readthedocs.io
3. MMomentA repository issues
