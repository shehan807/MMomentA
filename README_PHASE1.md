# MMomentA - Phase 1: Clean MPFIT Data Generation Pipeline

## Overview

Phase 1 delivers a clean, modular implementation of MPFIT charge calculations with:
- **Modular architecture** separating QM calculations, data processing, and analysis
- **Robust parallel processing** with proper Psi4 process isolation
- **Comparison framework** for benchmarking MPFIT vs RESP vs AM1-BCC
- **Configuration management** and comprehensive logging

## Installation

```bash
cd /Users/shehanparmar/Desktop/Code/MIT_MoML/MMomentA
pip install -e .
```

## Package Structure

```
MMomentA/
├── mmomenta/
│   ├── qm/                  # Quantum chemistry calculators
│   │   ├── mpfit.py         # MPFIT calculator
│   │   ├── resp.py          # RESP calculator
│   │   ├── am1bcc.py        # AM1-BCC calculator
│   │   └── settings.py      # Configuration classes
│   ├── data/                # Data processing
│   │   ├── batch.py         # Batch parallel processing
│   │   └── loaders.py       # Dataset loading utilities
│   ├── comparison/          # Analysis and benchmarking
│   │   ├── analysis.py      # Charge comparison utilities
│   │   └── esp_validation.py  # ESP reproduction metrics
│   └── utils/               # Utilities
│       ├── logging_config.py
│       └── io.py
├── scripts/
│   ├── compute_mpfit_charges.py     # MPFIT batch computation
│   └── compare_charge_methods.py    # Method comparison
└── examples/
    ├── simple_mpfit_example.py
    └── batch_processing_example.py
```

## Quick Start

### 1. Single Molecule MPFIT Calculation

```python
from openff.toolkit.topology import Molecule
from MMomentA.qm import MPFITCalculator, GDMAConfig

# Load molecule
molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)

# Configure MPFIT
config = GDMAConfig(method="hf", basis="6-31G*", limit=8)
calculator = MPFITCalculator(config)

# Compute charges
result = calculator.compute(molecule)
print(f"Charges: {result.charges}")
print(f"Multipole moments: {result.multipole_moments.shape}")
```

### 2. Batch Processing with Multiple Methods

```python
from MMomentA.qm import MPFITCalculator, RESPCalculator, AM1BCCCalculator
from MMomentA.data import batch_process_molecules

# Define calculators
calculators = {
    "AM1-BCC": AM1BCCCalculator(),
    "RESP": RESPCalculator(),
    "MPFIT": MPFITCalculator()
}

# Process molecules in parallel
results = batch_process_molecules(
    molecules,
    calculators=calculators,
    n_jobs=4,
    backend="loky"
)
```

### 3. Command-Line Scripts

```bash
# Compute MPFIT charges for ZINC dataset
python scripts/compute_mpfit_charges.py \
    --input zinc_molecules.pkl \
    --output zinc_mpfit.json \
    --max-molecules 100 \
    --method hf \
    --basis "6-31G*" \
    --n-jobs 8

# Compare all three methods
python scripts/compare_charge_methods.py \
    --input spice.oeb \
    --output comparison_results.json \
    --max-molecules 50 \
    --n-jobs 4
```

## Key Features

### Modular QM Calculators

All calculators follow a consistent interface:

```python
result = calculator.compute(molecule)

# Result attributes:
result.charges           # np.ndarray of charges
result.time_seconds      # computation time
result.success           # bool
result.error_message     # if failed
result.metadata          # calculation settings
```

### Robust Error Handling

- No try/except statements - code fails explicitly when behavior is incorrect
- Proper Psi4 process isolation prevents file conflicts in parallel execution
- Detailed error messages and logging

### Configuration Management

Type-safe configuration classes:

```python
from MMomentA.qm import GDMAConfig, ESPConfig, AM1BCCConfig

mpfit_config = GDMAConfig(
    method="pbe0",
    basis="def2-SVP",
    limit=8,
    svd_threshold=1.0e-4
)

resp_config = ESPConfig(
    method="hf",
    basis="6-31G*",
    restraint_weight=0.0005
)
```

### Parallel Processing

Joblib-based parallel processing with proper Psi4 isolation:

```python
from MMomentA.data import BatchProcessor

processor = BatchProcessor(
    calculators=calculators,
    n_jobs=-1,              # use all CPUs
    backend="loky",         # process isolation
    verbose=1
)

results = processor.process(molecules)
```

### Comparison Framework

Built-in charge comparison and ESP validation:

```python
from MMomentA.comparison import compare_charges, validate_esp_reproduction

# Compare two sets of charges
comparison = compare_charges(charges_mpfit, charges_resp)
print(f"RMSD: {comparison.rmsd:.4f}")
print(f"Correlation: {comparison.correlation:.4f}")

# Validate ESP reproduction
esp_validation = validate_esp_reproduction(
    molecule, charges, esp_record
)
print(f"ESP RMSE: {esp_validation.rmse:.4f} kcal/mol/e")
```

## Improvements Over Original Code

### From `run_comparison_multi.py` to MMomentA:

1. **Modular architecture**: Separated concerns (I/O, QM, analysis)
2. **Type-safe configuration**: Dataclass configs instead of environment variables
3. **Proper error handling**: No try/except masking failures
4. **Clean interfaces**: Consistent API across all calculators
5. **Comprehensive logging**: Structured logging instead of print statements
6. **Reusable components**: Library functions instead of scripts
7. **Clear documentation**: Docstrings and examples

### Code Quality

- **No hard-coded constants**: All parameters configurable
- **No try/except statements**: Explicit failure modes
- **Simple, interpretable**: Easy for users to understand
- **Well-tested interfaces**: Consistent Result dataclasses

## Usage Examples

See `examples/` directory:
- `simple_mpfit_example.py` - Basic MPFIT calculation
- `batch_processing_example.py` - Parallel batch processing

## Next Steps (Phase 2)

Phase 2 will focus on ML training data preparation:

1. **Data extraction**: Parse GDMA multipole moments for ML features
2. **Feature engineering**: Graph features compatible with DGL/PyTorch Geometric
3. **Dataset creation**: Structured HDF5/Parquet storage with train/val/test splits
4. **Data loaders**: PyTorch-compatible loaders for espaloma-charge integration

## References

- Original benchmark code: `openff-PyMPFIT/examples/gdma-charges/comparison/`
- espaloma-charge: Reference ML architecture
- OpenFF Recharge: QM calculation backend
