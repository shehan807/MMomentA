# MMomentA Quick Reference

## Complete Pipeline Overview

```
1. Dataset Creation → 2. Training → 3. ESP Validation
```

## 1. Dataset Creation (CREATE_DATASETS.md)

```bash
# Option A: Test SMILES (150 molecules, quick)
python scripts/create_test_smiles.py

# Option B: SPICE subset (100 molecules from QM data)
bash scripts/download_spice.sh

# Option C: ZINC fragments (100 drug-like molecules)
bash scripts/download_zinc.sh
```

## 2. Training Pipelines

### MVP Pipeline (150 molecules, ~1-2 hours)
```bash
./run_mvp.sh              # Baseline + Multipoles comparison
./run_mvp_multipoles.sh   # Multipoles only (faster)
```

### SPICE Pipeline (100 molecules, ~1.5 hours)
```bash
./run_spice.sh            # Extract SPICE → Train models → Compare
```

### ZINC Pipeline (100 molecules, ~1.5 hours)
```bash
./run_zinc.sh             # Requires SPICE model first (transfer learning)
```

## 3. ESP Validation (ESP_VALIDATION.md)

### Quick Start
```bash
# All 4 methods (RESP, AM1-BCC, MPFIT, MPFIT-GNN)
./run_esp_comparison.sh
```

### Custom Usage
```bash
# Custom dataset and model
./run_esp_comparison.sh data/spice_100.pkl runs/spice_multipoles/best_model.pt results

# Select specific methods
ENABLE_RESP=false ENABLE_AM1BCC=false ./run_esp_comparison.sh  # Only MPFIT vs MPFIT-GNN

# Custom QM settings
QM_METHOD=mp2 QM_BASIS=cc-pVDZ ./run_esp_comparison.sh
```

## Directory Structure

```
MMomentA/
├── run_mvp.sh                    # MVP: baseline vs multipoles
├── run_mvp_multipoles.sh         # MVP: multipoles only
├── run_spice.sh                  # SPICE: 100 molecules
├── run_zinc.sh                   # ZINC: transfer learning
├── run_esp_comparison.sh         # ESP validation (all 4 methods)
│
├── scripts/
│   ├── create_test_smiles.py     # Generate test molecules
│   ├── download_spice.sh         # Download SPICE dataset
│   ├── download_zinc.sh          # Download ZINC dataset
│   ├── prepare_dataset.py        # Compute MPFIT charges
│   ├── train_spice.py            # Train GNN model
│   ├── validate_esp.py           # MPFIT-GNN ESP validation
│   ├── compare_all_methods.py    # Run all 4 methods comparison
│   └── esp_utils.py              # ESP calculation utilities
│
├── figures/
│   └── plot_comparison.py        # Publication-quality plots
│
├── data/
│   ├── test_smiles.pkl           # 150 test molecules
│   ├── test_mpfit.h5             # MPFIT charges for test set
│   ├── spice_100.pkl             # 100 SPICE molecules
│   ├── spice_mpfit.h5            # MPFIT charges for SPICE
│   ├── zinc_molecules.pkl        # 100 ZINC molecules
│   └── zinc_mpfit.h5             # MPFIT charges for ZINC
│
├── runs/
│   ├── test_baseline/            # MVP baseline results
│   ├── test_multipoles/          # MVP multipole results
│   ├── spice_baseline/           # SPICE baseline results
│   ├── spice_multipoles/         # SPICE multipole results
│   ├── zinc_from_scratch/        # ZINC from-scratch results
│   └── zinc_transfer/            # ZINC transfer learning results
│
├── esp_comparison/               # ESP validation results
│   ├── qm_methods_results.json
│   ├── mpfit_gnn_results.json
│   ├── esp_validation_all_methods.png
│   └── mol_*/                    # Per-molecule ESP files
│
├── CREATE_DATASETS.md            # Dataset creation guide
├── ESP_VALIDATION.md             # ESP validation documentation
├── ESP_FRAMEWORK_SUMMARY.md      # Implementation summary
└── QUICK_REFERENCE.md            # This file
```

## Common Workflows

### 1. Quick Test (1-2 hours total)
```bash
# Create test data + train + validate
./run_mvp_multipoles.sh           # Train model
./run_esp_comparison.sh           # Validate ESP
```

### 2. SPICE Workflow (3-4 hours total)
```bash
# Download SPICE + train both models + compare
./run_spice.sh                    # Train baseline + multipoles
./run_esp_comparison.sh \
    data/spice_100.pkl \
    runs/spice_multipoles/best_model.pt \
    spice_esp
```

### 3. Transfer Learning (2-3 hours total)
```bash
# Train on SPICE, transfer to ZINC
./run_spice.sh                    # Pretrain on SPICE
./run_zinc.sh                     # Transfer to ZINC
```

### 4. Publication Figures (30-60 min)
```bash
# Use existing trained model
./run_esp_comparison.sh \
    data/test_smiles.pkl \
    runs/test_multipoles/best_model.pt \
    publication_results

# High-res output in: publication_results/esp_validation_all_methods.png
```

## Environments

### mmomenta-data (QM calculations)
- Psi4, OpenFF toolkit
- Used for: Dataset creation, MPFIT charges, QM ESP validation
- Activate: `conda activate mmomenta-data`

### mmomenta-ml (ML training)
- PyTorch, DGL
- Used for: Model training, MPFIT-GNN predictions, plotting
- Activate: `conda activate mmomenta-ml`

## Output Files

### Training Results
```json
{
  "results": {
    "test_metrics": {
      "val_rmse": 0.0234,
      "val_mae": 0.0156
    },
    "best_val_rmse": 0.0198
  }
}
```

### ESP Validation Results
```json
{
  "mae": {
    "RESP": [0.001, 0.002, ...],
    "AM1-BCC": [0.005, 0.006, ...],
    "MPFIT": [0.002, 0.003, ...],
    "MPFIT-GNN": [0.002, 0.003, ...]
  },
  "rmse": { ... }
}
```

## Key Metrics

### Training
- **RMSE**: Charge prediction error (e)
- **MAE**: Mean absolute error (e)
- Lower = better prediction accuracy

### ESP Validation
- **MAE**: ESP reproduction error (hartree/e)
- **RMSE**: RMS ESP error (hartree/e)
- Lower = more physically accurate charges

## Troubleshooting

### "Dataset not found"
```bash
# Create test dataset
python scripts/create_test_smiles.py
```

### "Model checkpoint not found"
```bash
# Train a model first
./run_mvp_multipoles.sh
```

### "Environment not found"
```bash
# Create environments (see main README)
conda env create -f environment-data.yml
conda env create -f environment-ml.yml
```

### "ESP files missing"
```bash
# QM methods must run first to generate ESP grids
# Run full pipeline or QM methods before MPFIT-GNN
./run_esp_comparison.sh  # Runs in correct order
```

## Performance Expectations

### Training Time
- MVP (150 mol): 20-30 min per model
- SPICE (100 mol): 30-45 min per model
- ZINC (100 mol): 30-45 min per model

### ESP Validation Time
- QM methods: ~2-5 min per molecule
- MPFIT-GNN: ~0.1 sec per molecule (1000x faster!)

### Accuracy Targets
- Baseline RMSE: ~0.03-0.05 e
- Multipole RMSE: ~0.02-0.03 e (better)
- ESP MAE: ~0.002-0.003 hartree/e (MPFIT-GNN ≈ MPFIT)

## Next Steps

1. **Read**: `ESP_VALIDATION.md` for detailed usage
2. **Train**: `./run_mvp_multipoles.sh` for quick test
3. **Validate**: `./run_esp_comparison.sh` for ESP comparison
4. **Customize**: Use environment variables for selective methods
5. **Publish**: Generate high-res plots for papers

## Help

- Issues: Check troubleshooting sections in each .md file
- Questions: Review architecture diagrams in ESP_VALIDATION.md
- Customization: See "Advanced Usage" sections
