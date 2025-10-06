# Manual Data Environment Setup (Step-by-Step)

## This is the Working Approach (Same as ML)

These commands avoid the conda env YAML file that causes OOM errors.

## Setup Commands

```bash
# 1. Load modules (no CUDA needed for data generation)
module load anaconda3

# 2. Create base environment
conda create -n mmomenta-data python=3.11 -y

# 3. Activate it
conda activate mmomenta-data

# 4. Install QM packages (SLOWEST - 10-20 min)
conda install -c conda-forge -c psi4 -y psi4 pygdma

# 5. Install chemistry tools
conda install -c conda-forge -y rdkit openbabel

# 6. Install scientific packages
conda install -c conda-forge -y numpy scipy pandas h5py pytables joblib

# 7. Install OpenFF dependencies
conda install -c conda-forge -y \
    packaging networkx xmltodict pymongo \
    python-constraint cachetools typing_extensions

# 8. Install OpenFF Toolkit from source
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML
git clone https://github.com/openforcefield/openff-toolkit.git
cd openff-toolkit
git checkout 0.17.1
pip install .

# 9. Install OpenFF Interchange from source
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML
git clone https://github.com/openforcefield/openff-interchange.git
cd openff-interchange
git checkout 0.4.0
pip install .

# 10. Install OpenFF Recharge
pip install openff-recharge

# 11. Install MMomentA
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA
pip install -e .

# 12. Verify installation
python -c "import psi4; from openff.toolkit import Molecule; from mmomenta.qm import MPFITCalculator; print('✓ Data environment ready')"
```

## What Each Step Does

1. **Load anaconda** - No CUDA needed (CPU-only calculations)
2. **Create environment** - Fresh Python 3.11 environment
3. **Activate** - Switch to new environment
4. **QM packages** - Psi4 for quantum chemistry, pygdma for multipoles
5. **Chemistry tools** - RDKit and openbabel for molecule handling
6. **Scientific** - NumPy, SciPy, HDF5 for data processing
7. **OpenFF deps** - Dependencies for OpenFF Toolkit
8. **OpenFF Toolkit** - Molecule handling (from source)
9. **OpenFF Interchange** - Additional OpenFF utilities (from source)
10. **OpenFF Recharge** - MPFIT charge calculations
11. **MMomentA** - Your package in editable mode
12. **Verify** - Check everything works

## Time Estimates

| Step | Time |
|------|------|
| 1-3  | 1 min |
| 4    | 10-20 min (Psi4 is large) |
| 5    | 5-10 min |
| 6-7  | 2-5 min |
| 8-9  | 3-5 min |
| 10-11| 1-2 min |
| 12   | 30 sec |
| **Total** | **25-45 min** |

## Verification

After setup, test QM calculations:

```bash
conda activate mmomenta-data

python << 'EOF'
import psi4
from openff.toolkit import Molecule
from mmomenta.qm import MPFITCalculator, MPFITConfig, QMSettings

print("✓ Psi4:", psi4.__version__)

# Test MPFIT on simple molecule
mol = Molecule.from_smiles("C")
mol.generate_conformers(n_conformers=1)

calc = MPFITCalculator(
    qc_settings=QMSettings(method="hf", basis="sto-3g"),
    config=MPFITConfig(limit=2)
)

print("Computing MPFIT charges for CH4...")
result = calc.compute(mol)
print("✓ Charges:", result.charges)
print("✓ Multipoles shape:", result.multipole_moments.shape)
print("\nData environment ready!")
EOF
```

## Quick Reference

**Activate environment:**
```bash
module load anaconda3
conda activate mmomenta-data
```

**Use for:**
- MPFIT calculations
- Dataset preparation
- Charge method comparisons

**Cannot do:**
- ML training (use mmomenta-ml for that)

## If You Need to Recreate

```bash
conda deactivate
conda env remove -n mmomenta-data

# Then run all setup commands again
# Or skip OpenFF git clones if you already have them
```

## Notes

- **No GPU needed** - This environment runs on CPU nodes
- **No PyTorch** - Data generation doesn't need ML libraries
- **Slow install** - Psi4 is ~500MB, takes time
- **From source** - OpenFF packages installed from git for compatibility

## After Setup

You can:

1. **Prepare datasets:**
```bash
python scripts/prepare_dataset.py \
    --input data/molecules.pkl \
    --output data/dataset.h5 \
    --n-jobs 8
```

2. **Run MPFIT calculations:**
```bash
python scripts/compute_mpfit_charges.py \
    --input data/molecules.pkl \
    --output results/charges.json
```

3. **Compare charge methods:**
```bash
python scripts/compare_charge_methods.py \
    --input data/test_molecules.pkl \
    --output results/comparison.json
```

Then switch to `mmomenta-ml` environment to train on the generated datasets.
