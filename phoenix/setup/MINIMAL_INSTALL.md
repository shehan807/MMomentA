# Minimal Installation Strategy (Most Reliable)

## The Problem

You've experienced:
- Python version conflicts (3.13 vs 3.11)
- CUDA library mismatches
- Conda solver getting killed (OOM)
- Missing dependencies (cachetools, etc.)
- OpenFF packages requiring source installation

**The reality:** Getting all these packages to work together on HPC is hard.

## Minimal Viable Approach

**Goal:** Get JUST enough working to run your MVP, not a perfect environment.

### Step 1: Start with Conda Basics Only

```bash
# Clean slate
conda deactivate
conda env remove -n mmomenta

# Minimal base - ONLY what conda handles well
conda create -n mmomenta python=3.11 -y
conda activate mmomenta

# Install mamba
conda install mamba -c conda-forge -y

# ONLY essential scientific packages via conda
mamba install -c conda-forge -y \
    numpy scipy pandas matplotlib \
    h5py joblib networkx

# Chemistry basics
mamba install -c conda-forge -y rdkit

# Stop here with conda - everything else via pip
```

### Step 2: PyTorch via Pip (More Reliable than Conda)

```bash
# Load CUDA first
module load cuda/11.8

# Install PyTorch via pip (includes CUDA, no system dependency issues)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: DGL via Pip

```bash
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
```

### Step 4: Skip Psi4 for Now - Use Pre-computed Data

**Critical decision:** Psi4 is causing issues. For MVP:

**Option A: Skip Psi4, use pre-computed dataset**
```bash
# Don't install Psi4 at all
# Download pre-computed MPFIT results (if available)
# OR use test molecules with dummy multipoles
```

**Option B: Minimal Psi4 (if you MUST have it)**
```bash
mamba install -c psi4 psi4 -y
# This might take 15-30 min, might fail
```

### Step 5: OpenFF - Minimal Dependencies

```bash
# Install minimal OpenFF dependencies
pip install \
    packaging \
    cachetools \
    typing_extensions \
    networkx \
    xmltodict \
    pymongo \
    python-constraint

# Try pip install first (easier)
pip install openff-toolkit openff-units openff-utilities

# If that fails, THEN do source install
cd ~/
git clone https://github.com/openforcefield/openff-toolkit.git
cd openff-toolkit
pip install .
```

### Step 6: Install MMomentA

```bash
cd ~/MMomentA
pip install -e .
```

### Step 7: Test What Works

```bash
python << 'EOF'
import sys

# Test each component
components = []

try:
    import numpy
    components.append(('NumPy', True))
except:
    components.append(('NumPy', False))

try:
    import torch
    components.append(('PyTorch', True))
    components.append(('CUDA', torch.cuda.is_available()))
except:
    components.append(('PyTorch', False))
    components.append(('CUDA', False))

try:
    import dgl
    components.append(('DGL', True))
except:
    components.append(('DGL', False))

try:
    import rdkit
    components.append(('RDKit', True))
except:
    components.append(('RDKit', False))

try:
    from openff.toolkit import Molecule
    components.append(('OpenFF', True))
except:
    components.append(('OpenFF', False))

try:
    import psi4
    components.append(('Psi4', True))
except:
    components.append(('Psi4', False))

try:
    from mmomenta.models import ChargeModel
    components.append(('MMomentA', True))
except:
    components.append(('MMomentA', False))

print("\nEnvironment Status:")
print("=" * 40)
for name, status in components:
    symbol = '✓' if status else '✗'
    print(f"{symbol} {name}")

# Determine what you can do
can_train = all([components[i][1] for i in [0, 1, 2, 6]])  # numpy, torch, dgl, mmomenta
can_qm = components[5][1]  # psi4

print("\n" + "=" * 40)
if can_train:
    print("✓ Can train ML models")
else:
    print("✗ Cannot train ML models")

if can_qm:
    print("✓ Can run QM calculations")
else:
    print("✗ Cannot run QM calculations")
    print("  → Use pre-computed dataset instead")
EOF
```

## Alternative: Use Docker/Singularity Container

**If the above still fails, consider containerization:**

```bash
# On Phoenix, use Singularity (HPC-friendly Docker)
module load singularity

# Pull a container with everything pre-installed
singularity pull docker://condaforge/miniforge3

# Run your code in container
singularity exec miniforge3.sif python scripts/train_spice.py ...
```

## Pragmatic MVP Strategy

**For your 2-hour deadline:**

1. **Don't fight the environment** - Use what installs easily
2. **Skip Psi4 QM calculations** - Use pre-generated test data
3. **Focus on ML training** - That's the novel part anyway
4. **Test molecules from SMILES** - No dataset download needed

**Minimal test:**

```bash
# Create test dataset WITHOUT QM (just random multipoles for testing)
python << 'EOF'
from openff.toolkit import Molecule
import numpy as np
import pickle

# 10 test molecules
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCC', 'c1cccnc1',
          'CCCC', 'c1cc(O)ccc1', 'CC(C)O', 'CCCCC', 'c1ccc(N)cc1']

molecules = []
for s in smiles:
    mol = Molecule.from_smiles(s)
    mol.generate_conformers(n_conformers=1)
    molecules.append(mol)

# Create dummy "MPFIT" results (for testing ML only)
mol_data = []
for mol in molecules:
    n_atoms = mol.n_atoms
    data = {
        'molecule': mol,
        'charges': np.random.randn(n_atoms) * 0.1,  # Random charges
        'multipoles': np.random.randn(n_atoms, 81),  # Random multipoles
    }
    mol_data.append(data)

with open('data/test_ml_only.pkl', 'wb') as f:
    pickle.dump(mol_data, f)

print("✓ Created test data for ML training (no QM needed)")
EOF

# Train a tiny model to test ML pipeline
python scripts/train_spice.py \
    --dataset data/test_ml_only.pkl \
    --output-dir runs/test \
    --n-epochs 10 \
    --device cpu
```

**If this works, you have a functioning ML pipeline.** The QM part can be fixed later.

## My Honest Assessment

**What will definitely work:**
- NumPy, SciPy, pandas (via conda)
- RDKit (via conda)
- PyTorch + CUDA (via pip)
- DGL (via pip)

**What might work:**
- OpenFF Toolkit (via pip or source)
- Psi4 (via conda, but slow/fragile)

**What you can skip for MVP:**
- Full QM calculations (use test data)
- OpenFF Interchange (not needed for MPFIT)
- AmberTools (not needed for MVP)

## Recommended Action Plan

**Right now:**

1. **Test what you have**:
   ```bash
   conda activate mmomenta
   python -c "import torch, dgl; from openff.toolkit import Molecule; print('Core packages work')"
   ```

2. **If Psi4 is broken, skip it**:
   ```bash
   # Use dummy data or download pre-computed dataset
   ```

3. **Focus on getting ML to run**:
   ```bash
   # Train on ANY dataset, even random data
   # Proves your ML code works
   ```

4. **Get figures generated**:
   ```bash
   # Even with dummy results, you can test plotting
   ```

**For paper deadline:**
- Get ML pipeline working (proven concept)
- Use literature values for QM comparison
- Note "QM calculations in progress" if needed

**After deadline:**
- Fix Psi4 installation properly
- Run real QM calculations
- Update figures with real data

The perfect is the enemy of the good. A working ML pipeline with test data is better than a broken environment with nothing running.
