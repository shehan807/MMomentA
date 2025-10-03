# MMomentA

**Fast, Transferable Multipole Moment-based Charge Assignment**

A machine learning framework that learns multipole moment-based charge fitting (MPFIT) for polarizable force fields, combining quantum chemistry calculations (Psi4/GDMA) with graph neural network architectures.

## Overview

MMomentA provides a complete pipeline for:
1. **QM Charge Calculation** (Phase 1): Clean, modular MPFIT/RESP/AM1-BCC computation
2. **ML Data Preparation** (Phase 2): Feature extraction and dataset creation
3. **Model Training** (Phase 3): Graph neural networks for charge prediction

### Key Features

- ✅ **Modular architecture** separating QM, data processing, and ML
- ✅ **Parallel processing** with proper Psi4 process isolation
- ✅ **Multipole moment integration** for improved accuracy
- ✅ **Scaffold-based splitting** for transferability testing
- ✅ **Comprehensive metrics** and evaluation tools
- ✅ **Simple, interpretable code** with no try/except masking failures

## Installation

```bash
cd MMomentA
pip install -e .
```

### Dependencies

- OpenFF Toolkit
- OpenFF Recharge (Psi4, GDMA)
- PyTorch
- DGL (Deep Graph Library)
- NumPy, Pandas, Joblib

## Quick Start

### 1. Compute MPFIT Charges

```python
from openff.toolkit.topology import Molecule
from MMomentA.qm import MPFITCalculator, GDMAConfig

molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)

calculator = MPFITCalculator(GDMAConfig(method="hf", basis="6-31G*"))
result = calculator.compute(molecule)

print(f"Charges: {result.charges}")
print(f"Multipole moments shape: {result.multipole_moments.shape}")
```

### 2. Prepare ML Dataset

```bash
python scripts/prepare_dataset.py \
    --input spice.oeb \
    --output spice_mpfit.h5 \
    --max-molecules 1000 \
    --method hf \
    --basis "6-31G*" \
    --n-jobs 8
```

### 3. Train Model

```bash
# Baseline model (no multipoles)
python scripts/train_spice.py \
    --dataset spice_mpfit.h5 \
    --output-dir runs/baseline \
    --feature-units 117 \
    --no-multipoles

# Model with multipole moments
python scripts/train_spice.py \
    --dataset spice_mpfit.h5 \
    --output-dir runs/multipoles \
    --feature-units 198 \
    --n-epochs 1000
```

## Project Structure

```
MMomentA/
├── mmomenta/
│   ├── qm/              # QM calculators (MPFIT, RESP, AM1-BCC)
│   ├── data/            # Data processing and loaders
│   ├── comparison/      # Analysis and benchmarking
│   ├── models/          # Neural network architectures
│   ├── training/        # Training loops and evaluation
│   └── utils/           # Utilities (logging, I/O)
├── scripts/             # Command-line scripts
│   ├── compute_mpfit_charges.py
│   ├── compare_charge_methods.py
│   ├── prepare_dataset.py
│   ├── train_spice.py
│   └── train_zinc.py
├── examples/            # Usage examples
└── README_PHASE*.md     # Detailed documentation
```

## Documentation

Detailed documentation for each phase:
- **[Phase 1: MPFIT Data Generation](README_PHASE1.md)** - QM calculations and comparison framework
- **[Phase 2: ML Data Preparation](README_PHASE2.md)** - Feature extraction and dataset creation
- **[Phase 3: Model Training](README_PHASE3.md)** - Neural network training and evaluation

## Examples

### Batch Processing

```python
from MMomentA.qm import MPFITCalculator, RESPCalculator
from MMomentA.data import batch_process_molecules

calculators = {
    "MPFIT": MPFITCalculator(),
    "RESP": RESPCalculator()
}

results = batch_process_molecules(
    molecules,
    calculators=calculators,
    n_jobs=4
)
```

### Model Training

```python
from MMomentA.data import load_dataset, create_split_datasets
from MMomentA.models import ChargeModel, ModelConfig
from MMomentA.training import Trainer, TrainingConfig

# Load dataset
molecule_data_list, metadata = load_dataset("spice_mpfit.h5")

# Create splits
split_data = create_split_datasets(
    molecule_data_list, train_idx, val_idx, test_idx,
    include_multipoles=True
)

# Train model
model = ChargeModel(ModelConfig(feature_units=198, depth=4, width=128))
trainer = Trainer(model, TrainingConfig(n_epochs=1000))

results = trainer.train(
    split_data.get_train_loader(128),
    split_data.get_val_loader(128),
    split_data.get_test_loader(128)
)

print(f"Test RMSE: {results['test_metrics']['val_rmse']:.5f}")
```

## Performance

Expected performance on SPICE dataset:

| Model | Features | Test RMSE (e) | Speedup vs QM |
|-------|----------|---------------|---------------|
| Baseline | 117 | ~0.045 | 100-1000x |
| + Multipoles | 198 | **<0.040** | 100-1000x |

## Design Principles

Following your requirements:
- **No try/except statements** - code fails explicitly when behavior is incorrect
- **No hard-coded constants** - all parameters configurable
- **Simple, interpretable** - easy for users to understand
- **Modular architecture** - clean separation of concerns
- **Comprehensive logging** - structured logging instead of print statements

## Citation

If you use MMomentA in your research, please cite:

```bibtex
@software{mmomenta2025,
  author = {Parmar, Shehan M.},
  title = {MMomentA: Multipole Moment-based Charge Assignment},
  year = {2025},
  url = {https://github.com/...}
}
```

## References

- OpenFF Recharge: https://github.com/openforcefield/openff-recharge
- espaloma-charge: https://github.com/choderalab/espaloma_charge
- DGL: https://www.dgl.ai/

## Copyright

Copyright (c) 2025, Shehan M. Parmar

## Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
