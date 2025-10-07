# Repository Reorganization Plan

## New Structure

```
MMomentA/
├── README.md                          # Main README
│
├── cluster-independent/               # General scripts and docs
│   ├── pipelines/                     # Training/validation pipelines
│   │   ├── run_mvp.sh                 # MVP pipeline (150 molecules)
│   │   ├── run_mvp_multipoles.sh      # MVP multipoles only
│   │   ├── run_spice.sh               # SPICE pipeline (100 molecules)
│   │   ├── run_zinc.sh                # ZINC transfer learning
│   │   └── run_esp_comparison.sh      # ESP validation (all 4 methods)
│   │
│   └── docs/                          # General documentation
│       ├── CREATE_DATASETS.md
│       ├── ESP_VALIDATION.md
│       ├── ESP_FRAMEWORK_SUMMARY.md
│       ├── GET_STARTED.md
│       ├── QUICK_REFERENCE.md
│       └── ...
│
├── phoenix/                           # Phoenix HPC specific
│   ├── setup/                         # Environment setup
│   │   ├── MANUAL_DATA_SETUP.md
│   │   ├── MANUAL_ML_SETUP.md
│   │   ├── install_data_only.sh
│   │   └── install_ml_only.sh
│   │
│   ├── jobs/                          # SLURM job scripts
│   │   ├── submit_mvp_2hr.sh
│   │   ├── submit_train_baseline.sh
│   │   ├── submit_train_multipoles.sh
│   │   └── ...
│   │
│   └── README.md                      # Phoenix-specific guide
│
├── perlmutter/                        # Perlmutter specific
│   ├── setup/                         # Environment setup
│   │   ├── MANUAL_SETUP_DATA.sh
│   │   ├── MANUAL_SETUP_ML.sh
│   │   └── verify_installation.sh
│   │
│   ├── jobs/                          # SLURM job scripts (to be created)
│   │   ├── submit_mvp.sh
│   │   ├── submit_spice.sh
│   │   └── submit_esp_comparison.sh
│   │
│   └── README.md                      # Perlmutter-specific guide
│
├── scripts/                           # Python scripts (cluster-independent)
│   ├── create_test_smiles.py
│   ├── prepare_dataset.py
│   ├── train_spice.py
│   ├── extract_spice.py
│   ├── validate_esp.py
│   ├── compare_all_methods.py
│   └── ...
│
├── figures/                           # Plotting scripts
│   └── plot_comparison.py
│
└── MMomentA/                          # Python package
    ├── __init__.py
    ├── data/
    ├── models/
    ├── training/
    └── ...
```

## Migration Steps

1. Move cluster-independent pipelines to `cluster-independent/pipelines/`
2. Move general docs to `cluster-independent/docs/`
3. Organize Phoenix files in `phoenix/setup/` and `phoenix/jobs/`
4. Move Perlmutter files to `perlmutter/setup/`
5. Create Perlmutter job scripts in `perlmutter/jobs/`
6. Update README with new structure

## Files to Move

### To cluster-independent/pipelines/
- run_mvp.sh
- run_mvp_multipoles.sh
- run_spice.sh
- run_zinc.sh
- run_esp_comparison.sh

### To cluster-independent/docs/
- CREATE_DATASETS.md
- ESP_VALIDATION.md
- ESP_FRAMEWORK_SUMMARY.md
- GET_STARTED.md
- MVP_QUICKSTART.md
- QUICK_REFERENCE.md
- DOWNLOAD_DATASETS.md

### To perlmutter/setup/
- MANUAL_SETUP_PERLMUTTER_DATA.sh → MANUAL_SETUP_DATA.sh
- MANUAL_SETUP_PERLMUTTER_ML.sh → MANUAL_SETUP_ML.sh
- PERLMUTTER_GUIDE.md → README.md

### Keep in root:
- README.md (main)
- CODE_OF_CONDUCT.md
- LICENSE (if exists)
