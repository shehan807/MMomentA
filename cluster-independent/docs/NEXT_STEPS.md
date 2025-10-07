# What To Do Next with mmomenta-ml Environment

## You Now Have

✓ **mmomenta-ml environment** - Ready for ML training
- PyTorch 2.4 with CUDA
- DGL for graph neural networks
- Scientific Python stack
- MMomentA package installed

## What You Still Need

### 1. A Dataset

**You need a prepared HDF5 dataset with:**
- Molecular structures
- MPFIT charges (targets)
- Multipole moments (features)
- Graph structures

**Options:**

**Option A: Download SPICE (easiest for testing)**
```bash
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA/data
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5
```

**Option B: Create test dataset from SMILES (fastest for MVP)**
```bash
conda activate mmomenta-ml
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA

python << 'EOF'
from openff.toolkit import Molecule
import numpy as np
import h5py

# Create 10 test molecules
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCC', 'c1cccnc1',
          'CCCC', 'c1cc(O)ccc1', 'CC(C)O', 'CCCCC', 'c1ccc(N)cc1']

# Generate dummy data for testing ML pipeline
with h5py.File('data/test_ml.h5', 'w') as f:
    for i, s in enumerate(smiles):
        mol = Molecule.from_smiles(s)
        mol.generate_conformers(n_conformers=1)
        n_atoms = mol.n_atoms

        grp = f.create_group(f'mol_{i}')
        grp.create_dataset('charges', data=np.random.randn(n_atoms) * 0.1)
        grp.create_dataset('multipoles', data=np.random.randn(n_atoms, 81))
        # Add minimal metadata
        grp.attrs['smiles'] = s
        grp.attrs['n_atoms'] = n_atoms

print("✓ Created data/test_ml.h5 for testing")
EOF
```

**Option C: Set up mmomenta-data environment and generate real dataset**
(See below for data environment setup)

### 2. Training Scripts

**You need the training scripts created earlier. Check if they exist:**
```bash
ls scripts/train_spice.py
ls scripts/prepare_dataset.py
```

**If missing, you need to create them** (they were outlined in GET_STARTED.md)

### 3. Run a Test Training

**Once you have a dataset, test the ML pipeline:**

```bash
# Activate environment
module load anaconda3 cuda
conda activate mmomenta-ml

# Test training (CPU first to verify pipeline works)
python scripts/train_spice.py \
    --dataset data/test_ml.h5 \
    --output-dir runs/test \
    --feature-units 198 \
    --depth 2 \
    --width 32 \
    --n-epochs 10 \
    --batch-size 5 \
    --device cpu

# If that works, try GPU
python scripts/train_spice.py \
    --dataset data/test_ml.h5 \
    --output-dir runs/test_gpu \
    --feature-units 198 \
    --n-epochs 50 \
    --device cuda
```

## Complete Workflow (Next Steps)

### Step 1: Set Up Data Environment (If Not Done)

The ML environment **cannot** generate datasets. You need a separate data environment for QM calculations.

**Manual setup for mmomenta-data:**
```bash
# 1. Create environment
conda create -n mmomenta-data python=3.11 -y
conda activate mmomenta-data

# 2. Install QM packages
conda install -c conda-forge -c psi4 -y psi4 pygdma

# 3. Install chemistry tools
conda install -c conda-forge -y rdkit openbabel

# 4. Install dependencies
conda install -c conda-forge -y \
    numpy scipy pandas h5py joblib \
    packaging networkx xmltodict pymongo \
    python-constraint cachetools typing_extensions

# 5. Install OpenFF (you already did this via git clone)
cd /storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA
# Assuming you have openff-toolkit and openff-interchange cloned
cd ../openff-toolkit && pip install .
cd ../openff-interchange && pip install .
pip install openff-recharge

# 6. Install MMomentA
cd ../MMomentA
pip install -e .
```

### Step 2: Generate Dataset (mmomenta-data environment)

```bash
# Switch to data environment
conda activate mmomenta-data

# Prepare dataset
python scripts/prepare_dataset.py \
    --input data/molecules.pkl \
    --output data/spice_mpfit.h5 \
    --max-molecules 100 \
    --n-jobs 8
```

### Step 3: Train Model (mmomenta-ml environment)

```bash
# Switch to ML environment
conda activate mmomenta-ml

# Train baseline
python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/baseline \
    --feature-units 117 \
    --no-multipoles \
    --device cuda

# Train with multipoles
python scripts/train_spice.py \
    --dataset data/spice_mpfit.h5 \
    --output-dir runs/multipoles \
    --feature-units 198 \
    --device cuda
```

### Step 4: Compare Results

```bash
# Check results
cat runs/baseline/training_results.json | python -m json.tool
cat runs/multipoles/training_results.json | python -m json.tool
```

## Immediate Next Action

**Choose one:**

### Path A: Quick Test (30 min)
1. Create dummy test dataset (Option B above)
2. Verify training script exists or create minimal version
3. Run test training on CPU to verify pipeline
4. Run on GPU if available

### Path B: Real MVP (2-4 hours)
1. Set up mmomenta-data environment
2. Download SPICE or prepare 50-100 molecules
3. Generate MPFIT dataset
4. Train baseline + multipole models
5. Generate comparison figures

### Path C: Full Production (1-2 days)
1. Set up both environments completely
2. Download full SPICE dataset
3. Generate complete dataset (1000 molecules)
4. Train production models
5. Run ablation studies
6. Transfer learning on ZINC

## Checking Your Current Status

```bash
# What environments do you have?
conda env list

# What's in mmomenta-ml?
conda activate mmomenta-ml
conda list | grep -E 'torch|dgl|numpy'

# Do you have training scripts?
ls -la scripts/

# Do you have any datasets?
ls -la data/

# Do you have the data environment?
conda activate mmomenta-data  # Will fail if not created
```

## Most Likely Next Steps for You

Based on your 2-hour deadline mentioned earlier:

```bash
# 1. Create quick test dataset
conda activate mmomenta-ml
python [create test dataset script from Option B above]

# 2. Create minimal training script (if missing)
# See GET_STARTED.md for the script

# 3. Run quick test
python scripts/train_spice.py \
    --dataset data/test_ml.h5 \
    --output-dir runs/quick_test \
    --n-epochs 10 \
    --device cpu

# 4. If it works, you have a functioning pipeline!
```

**Tell me:**
1. Do you have `scripts/train_spice.py`? (ls scripts/)
2. Do you have any dataset files? (ls data/)
3. What's your immediate goal - quick test or real training?

Then I can give you the exact next command to run.
