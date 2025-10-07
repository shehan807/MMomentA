# MMomentA: Multipole Moment Analysis for Molecular Charge Prediction

Machine learning framework for predicting atomic charges using multipole moment-based QM fitting (MPFIT) as training data.

## Repository Structure

```
MMomentA/
├── README.md                          # This file
│
├── cluster-independent/               # General scripts (any cluster)
│   ├── pipelines/                     # Training/validation workflows
│   │   ├── run_mvp.sh                 # Quick test (150 molecules)
│   │   ├── run_spice.sh               # SPICE dataset (100 molecules)
│   │   ├── run_zinc.sh                # Transfer learning
│   │   └── run_esp_comparison.sh      # ESP validation (4 methods)
│   │
│   └── docs/                          # Documentation
│       ├── GET_STARTED.md             # Quick start guide
│       ├── QUICK_REFERENCE.md         # Command reference
│       ├── ESP_VALIDATION.md          # ESP comparison framework
│       └── CREATE_DATASETS.md         # Dataset creation guide
│
├── phoenix/                           # Georgia Tech Phoenix HPC
│   ├── setup/                         # Environment setup scripts
│   ├── jobs/                          # SLURM job scripts
│   └── README.md                      # Phoenix-specific guide
│
├── perlmutter/                        # NERSC Perlmutter
│   ├── setup/                         # Environment setup scripts
│   ├── jobs/                          # SLURM job scripts
│   └── README.md                      # Perlmutter-specific guide
│
├── scripts/                           # Python scripts
│   ├── create_test_smiles.py
│   ├── prepare_dataset.py             # Generate MPFIT charges
│   ├── train_spice.py                 # Train GNN model
│   ├── validate_esp.py                # MPFIT-GNN ESP validation
│   └── compare_all_methods.py         # Compare all 4 methods
│
├── figures/                           # Plotting utilities
│   └── plot_comparison.py
│
└── MMomentA/                          # Python package
    ├── data/                          # Data processing
    ├── models/                        # GNN architectures
    └── training/                      # Training utilities
```

## Quick Start

### 1. Choose Your Cluster

**Phoenix (Georgia Tech)**:
```bash
cd phoenix
bash setup/MANUAL_SETUP_DATA.sh      # CPU environment
bash setup/MANUAL_SETUP_ML.sh        # GPU environment
```

**Perlmutter (NERSC)**:
```bash
cd perlmutter
bash setup/MANUAL_SETUP_DATA.sh      # CPU environment
# Request GPU node first
salloc -C gpu -q interactive -t 01:00:00 -A <account>
bash setup/MANUAL_SETUP_ML.sh        # GPU environment
```

### 2. Run a Pipeline

**Interactive (any cluster)**:
```bash
# Activate environment
conda activate mmomenta-ml  # or use cluster-specific path

# Run MVP pipeline (1-2 hours)
cd cluster-independent/pipelines
./run_mvp.sh
```

**Batch Job (Phoenix)**:
```bash
sbatch phoenix/jobs/submit_mvp_2hr.sh
```

**Batch Job (Perlmutter)**:
```bash
# Edit to add your account
sbatch perlmutter/jobs/submit_mvp.sh
```

### 3. ESP Validation

Compare 4 charge methods (RESP, AM1-BCC, MPFIT, MPFIT-GNN):

```bash
cd cluster-independent/pipelines
./run_esp_comparison.sh
```

## Workflows

### MVP Workflow (Quick Test)
**Time**: ~1-2 hours | **Size**: 150 molecules

```bash
# Creates test molecules → Trains models → Compares results
./cluster-independent/pipelines/run_mvp.sh
```

### SPICE Workflow
**Time**: ~3-4 hours | **Size**: 100 molecules from QM data

```bash
# Extracts SPICE → Trains models → Compares results
./cluster-independent/pipelines/run_spice.sh
```

### Transfer Learning
**Time**: ~2-3 hours | **Requires**: SPICE model first

```bash
./cluster-independent/pipelines/run_spice.sh   # Pretrain
./cluster-independent/pipelines/run_zinc.sh    # Transfer to ZINC
```

### ESP Validation
**Time**: ~1 hour | **Requires**: Trained model

```bash
# All 4 methods enabled by default
./cluster-independent/pipelines/run_esp_comparison.sh

# Selective methods
ENABLE_RESP=false ENABLE_AM1BCC=false \
./cluster-independent/pipelines/run_esp_comparison.sh
```

## Documentation

### Getting Started
- **[GET_STARTED.md](cluster-independent/docs/GET_STARTED.md)** - First-time setup
- **[QUICK_REFERENCE.md](cluster-independent/docs/QUICK_REFERENCE.md)** - Command cheat sheet

### Cluster-Specific
- **[phoenix/README.md](phoenix/README.md)** - Phoenix HPC guide
- **[perlmutter/README.md](perlmutter/README.md)** - Perlmutter guide

### Advanced Topics
- **[ESP_VALIDATION.md](cluster-independent/docs/ESP_VALIDATION.md)** - ESP comparison framework
- **[CREATE_DATASETS.md](cluster-independent/docs/CREATE_DATASETS.md)** - Custom datasets

## Key Features

✅ **Two-environment architecture**: Separate QM and ML dependencies
✅ **Multi-cluster support**: Phoenix, Perlmutter, local
✅ **Complete pipelines**: Dataset → Training → Validation
✅ **ESP validation**: Compare 4 charge methods with publication plots
✅ **Transfer learning**: Pretrain on SPICE, transfer to ZINC

## Charge Methods Comparison

| Method | Type | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| **RESP** | QM | Slow | Reference | Gold standard |
| **AM1-BCC** | Empirical | Fast | Good | Quick baseline |
| **MPFIT** | QM | Slow | Excellent | Ground truth |
| **MPFIT-GNN** | ML | **Fast** | **Excellent** | Production |

MPFIT-GNN achieves MPFIT accuracy at 1000x speed!

## Output Files

### Training Results
```
runs/
├── test_multipoles/
│   ├── best_model.pt              # Trained model
│   ├── training_results.json      # Metrics
│   └── training.log               # Detailed log
```

### ESP Validation
```
esp_comparison/
├── esp_validation_all_methods.png # Publication plot
├── qm_methods_results.json        # All methods results
└── mol_*/                          # Per-molecule ESP grids
```

## Citation

If you use this framework, please cite:
- MMomentA (this work)
- openff-PyMPFIT (MPFIT implementation)
- GDMA (Stone, J. Chem. Theory Comput. 2005)

## Support

- **Issues**: Check cluster-specific README troubleshooting
- **Questions**: See documentation in `cluster-independent/docs/`
- **Bugs**: Open GitHub issue with reproducible example

## License

[Add your license here]
