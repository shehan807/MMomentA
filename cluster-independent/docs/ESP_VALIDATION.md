# ESP Validation Comparison Framework

Compare electrostatic potential (ESP) reproduction for 4 charge fitting methods:
- **RESP**: QM-based ESP fitting (reference method)
- **AM1-BCC**: Empirical bond charge correction
- **MPFIT**: Multipole moment-based QM fitting
- **MPFIT-GNN**: ML-based prediction using MMomentA (this work)

## Overview

This framework validates charge methods by comparing their ability to reproduce QM electrostatic potential. Each method's charges are used to calculate ESP at a grid of points around molecules, then compared against reference QM ESP values.

### Validation Metrics

- **MAE**: Mean absolute error in ESP reproduction (hartree/e)
- **RMSE**: Root mean square error in ESP reproduction (hartree/e)

Lower values indicate better ESP reproduction and more physically accurate charges.

## Quick Start

### Prerequisites

1. **Two conda environments** (see main README):
   - `mmomenta-data`: Psi4, OpenFF, openff-PyMPFIT
   - `mmomenta-ml`: PyTorch, DGL, lovelyplots

2. **Trained MMomentA model**:
   ```bash
   # Train a model first if you haven't already
   ./run_mvp_multipoles.sh
   ```

3. **Test molecules**:
   ```bash
   # Use existing test dataset or create new one
   python scripts/create_test_smiles.py
   ```

### Run Complete Comparison

```bash
# Default: uses data/test_smiles.pkl and runs/test_multipoles/best_model.pt
./run_esp_comparison.sh

# Custom dataset and model
./run_esp_comparison.sh data/spice_100.pkl runs/spice_multipoles/best_model.pt esp_results
```

### Outputs

All results saved to output directory (default: `esp_comparison/`):

```
esp_comparison/
├── qm_methods_results.json          # Full results for all methods
├── mpfit_gnn_results.json            # MPFIT-GNN predictions
├── esp_validation_all_methods.png    # Publication-quality violin plots
└── mol_*/                            # Per-molecule ESP grid files
    ├── grid.dat                      # ESP grid points
    └── grid_esp.dat                  # QM ESP values
```

## Customization

### Select Methods

Disable methods using environment variables (all enabled by default):

```bash
# Only compare MPFIT vs MPFIT-GNN
ENABLE_RESP=false ENABLE_AM1BCC=false ./run_esp_comparison.sh

# Skip ML method (QM-only comparison)
ENABLE_MPFIT_GNN=false ./run_esp_comparison.sh
```

### QM Settings

Control QM method and basis set for ESP calculation:

```bash
# Use MP2 with larger basis
QM_METHOD=mp2 QM_BASIS=cc-pVDZ ./run_esp_comparison.sh

# Default: HF/6-31G*
```

### Verbose Output

```bash
# Print detailed ESP metrics for each molecule
VERBOSE=1 ./run_esp_comparison.sh
```

## Architecture

### Pipeline Stages

**Stage 1: MPFIT-GNN Predictions** (`mmomenta-ml` environment)
- Load trained MMomentA model
- Predict charges for test molecules
- Save results to JSON

**Stage 2: QM Methods** (`mmomenta-data` environment)
- Run RESP, AM1-BCC, MPFIT charge fitting
- Generate QM ESP grids for each molecule
- Validate ESP reproduction for all methods
- Merge MPFIT-GNN results

**Stage 3: Visualization** (`mmomenta-ml` environment)
- Generate publication-quality violin plots
- Show MAE and RMSE distributions
- Use lovelyplots styling

### Key Components

#### `scripts/validate_esp.py`
MPFIT-GNN wrapper that:
- Loads trained MMomentA model
- Predicts charges for molecules
- Validates against QM ESP grids
- Saves results to JSON

Usage:
```bash
conda activate mmomenta-ml
python scripts/validate_esp.py \
    --model runs/test_multipoles/best_model.pt \
    --molecules data/test_smiles.pkl \
    --esp-dir esp_comparison \
    --output mpfit_gnn_results.json \
    --device cuda
```

#### `scripts/compare_all_methods.py`
Unified comparison script that:
- Runs QM methods (RESP, AM1-BCC, MPFIT)
- Generates ESP grids
- Validates ESP reproduction
- Merges MPFIT-GNN results
- Creates summary plots

Usage:
```bash
conda activate mmomenta-data
python scripts/compare_all_methods.py \
    --molecules data/test_smiles.pkl \
    --output-dir esp_comparison \
    --mpfit-gnn-results esp_comparison/mpfit_gnn_results.json \
    --qm-method hf \
    --qm-basis "6-31G*"
```

#### `scripts/esp_utils.py`
ESP calculation utilities:
- `charges_to_esp()`: Calculate ESP from charges (Psi4 molecules)
- `charges_to_esp_openff()`: Calculate ESP from charges (OpenFF molecules)
- `compare_grid_esp()`: Compare ESP arrays and calculate metrics

#### `figures/plot_comparison.py`
Publication-quality plotting:
- `create_violin_plots()`: ESP validation violin plots (MAE, RMSE)
- `create_correlation_plot()`: Charge-charge correlation plots
- `get_method_colors()`: Consistent color scheme

**Color scheme:**
- RESP: Blue (`#2E86AB`) - QM reference
- AM1-BCC: Golden (`#E8A317`) - Empirical
- MPFIT: Purple (`#A23B72`) - Multipole QM
- MPFIT-GNN: Green (`#00C853`) - ML method

## Manual Usage

### Step 1: Train MMomentA Model

```bash
# Quick test (150 molecules, ~1-2 hours)
./run_mvp_multipoles.sh

# SPICE dataset (100 molecules, ~1.5 hours)
./run_spice.sh
```

### Step 2: Run MPFIT-GNN Validation

```bash
conda activate mmomenta-ml

python scripts/validate_esp.py \
    --model runs/test_multipoles/best_model.pt \
    --molecules data/test_smiles.pkl \
    --esp-dir esp_comparison \
    --output esp_comparison/mpfit_gnn_results.json \
    --device cuda \
    --verbose
```

### Step 3: Run QM Methods Comparison

```bash
conda activate mmomenta-data

export QM_METHOD=hf
export QM_BASIS="6-31G*"

python scripts/compare_all_methods.py \
    --molecules data/test_smiles.pkl \
    --output-dir esp_comparison \
    --mpfit-gnn-results esp_comparison/mpfit_gnn_results.json \
    --verbose
```

### Step 4: Generate Plots

```bash
conda activate mmomenta-ml

python << 'EOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, 'figures')
from plot_comparison import create_violin_plots

# Load results
with open('esp_comparison/qm_methods_results.json', 'r') as f:
    results = json.load(f)

# Generate plots
create_violin_plots(
    results,
    output_dir='esp_comparison',
    qm_method='hf',
    qm_basis='6-31G*'
)
EOF
```

## Expected Results

### Typical Performance

Based on literature and preliminary results:

| Method | MAE (hartree/e) | RMSE (hartree/e) | Speed |
|--------|-----------------|------------------|-------|
| RESP | ~1e-3 | ~2e-3 | Slow (QM) |
| AM1-BCC | ~5e-3 | ~8e-3 | Fast (empirical) |
| MPFIT | ~2e-3 | ~4e-3 | Slow (QM) |
| MPFIT-GNN | ~2e-3 | ~4e-3 | **Fast (ML)** |

**Key insight**: MPFIT-GNN should match MPFIT accuracy at a fraction of the computational cost.

### Interpreting Results

- **Lower MAE/RMSE** = Better ESP reproduction
- **MPFIT-GNN near MPFIT** = Successful ML surrogate
- **MPFIT-GNN < AM1-BCC** = ML improves over empirical methods
- **MPFIT-GNN ≈ RESP** = Excellent performance

## Troubleshooting

### Missing openff-PyMPFIT imports

```bash
# Add openff-PyMPFIT to PYTHONPATH
export PYTHONPATH="/path/to/openff-PyMPFIT:$PYTHONPATH"

# Or install as package
cd /path/to/openff-PyMPFIT
pip install -e .
```

### ESP files not found

The script automatically saves ESP grids during QM fitting. If missing:
1. Ensure RESP or MPFIT ran successfully
2. Check for `grid.dat` and `grid_esp.dat` in molecule directories
3. Run QM methods before MPFIT-GNN validation

### CUDA errors

```bash
# Use CPU for inference (slower but more compatible)
./run_esp_comparison.sh
# Edit script to change --device cuda to --device cpu
```

### Environment activation fails

```bash
# Initialize conda in your shell
conda init bash
source ~/.bashrc

# Or use full path
source /path/to/conda/etc/profile.d/conda.sh
conda activate mmomenta-ml
```

## Publication-Quality Figures

The framework generates publication-ready violin plots using `lovelyplots`:

### Figure Components

1. **Two subplots**: MAE (left), RMSE (right)
2. **Four violins**: One per method (RESP, AM1-BCC, MPFIT, MPFIT-GNN)
3. **Statistical markers**: Mean (solid line), median (dashed line)
4. **Color-coded**: Each method has distinct color
5. **High resolution**: 300 DPI for publication

### Customizing Plots

Edit `figures/plot_comparison.py` to customize:

```python
# Change figure size
def create_violin_plots(metrics, figsize=(14, 6)):  # Default
def create_violin_plots(metrics, figsize=(20, 8)):  # Larger

# Change colors
def get_method_colors():
    return {
        'MPFIT-GNN': '#00C853',  # Green
        # Change to your preferred color
    }

# Change DPI
plt.savefig(output_file, dpi=300)  # Default
plt.savefig(output_file, dpi=600)  # Higher resolution
```

## Advanced Usage

### Custom Molecules

```bash
# Create custom dataset
python << 'EOF'
from openff.toolkit import Molecule
import pickle

# Your molecules
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
molecules = []

for smi in smiles_list:
    mol = Molecule.from_smiles(smi)
    mol.generate_conformers(n_conformers=1)
    molecules.append(mol)

# Save
with open('my_molecules.pkl', 'wb') as f:
    pickle.dump(molecules, f)
EOF

# Run comparison
./run_esp_comparison.sh my_molecules.pkl runs/test_multipoles/best_model.pt my_results
```

### Batch Processing

For large datasets, run stages separately to checkpoint progress:

```bash
# Stage 1: QM methods (slow, checkpoint here)
conda activate mmomenta-data
python scripts/compare_all_methods.py \
    --molecules large_dataset.pkl \
    --output-dir large_esp \
    --no-mpfit-gnn

# Stage 2: MPFIT-GNN (fast, can retry if needed)
conda activate mmomenta-ml
python scripts/validate_esp.py \
    --model runs/model.pt \
    --molecules large_dataset.pkl \
    --esp-dir large_esp \
    --output large_esp/mpfit_gnn_results.json

# Stage 3: Merge and plot
conda activate mmomenta-data
python scripts/compare_all_methods.py \
    --molecules large_dataset.pkl \
    --output-dir large_esp \
    --mpfit-gnn-results large_esp/mpfit_gnn_results.json \
    --no-resp --no-am1bcc --no-mpfit  # Skip QM, just merge
```

## Integration with Existing Workflows

This framework is designed to integrate with the existing openff-PyMPFIT comparison framework:

```
openff-PyMPFIT/                    MMomentA/
├── compare_charges.py             ├── scripts/compare_all_methods.py
│   (RESP, AM1-BCC, MPFIT)         │   (imports from openff-PyMPFIT + MPFIT-GNN)
├── esp_utils.py         ────────> ├── scripts/esp_utils.py (copied)
├── plot_utils.py        ────────> └── figures/plot_comparison.py (extended)
└── run_comparison.py                  + MPFIT-GNN wrapper
```

You can continue using openff-PyMPFIT for QM-only comparisons, and use MMomentA when you want to include ML predictions.

## References

- **RESP**: Bayly et al., J. Phys. Chem. 1993
- **AM1-BCC**: Jakalian et al., J. Comput. Chem. 2002
- **MPFIT**: openff-PyMPFIT (this work's implementation)
- **MMomentA**: This work - ML surrogate for MPFIT

## Citation

If you use this framework, please cite:
- MMomentA framework (this work)
- openff-PyMPFIT (MPFIT implementation)
- Original GDMA paper (Stone, J. Chem. Theory Comput. 2005)
