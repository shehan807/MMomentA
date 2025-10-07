# Manual Environment Installation (Step-by-Step)

## Why Manual Installation?

The full `environment.yml` often gets stuck because conda tries to solve all dependencies at once. Breaking it into chunks is much faster.

## Strategy

Install packages in groups:
1. **Base Python + conda/mamba** (fast)
2. **Scientific computing** (numpy, scipy) (medium)
3. **Chemistry tools** (rdkit, openbabel) (slow)
4. **QM packages** (psi4, pygdma) (very slow)
5. **ML frameworks** (pytorch, dgl) (very slow)
6. **Pip packages** (openff-toolkit) (fast)
7. **MMomentA** (fast)

## Step-by-Step Installation

### Step 1: Create Base Environment (1 min)

```bash
module load anaconda3/2022.05

# Create minimal environment
conda create -n mmomenta python=3.10 pip -y

# Activate
conda activate mmomenta
```

### Step 2: Install Mamba (1-2 min)

```bash
# Install mamba FIRST to speed up everything else
conda install mamba -c conda-forge -y

# Verify
mamba --version
```

### Step 3: Install Scientific Stack (2-3 min)

```bash
mamba install -c conda-forge -y \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    h5py \
    pytables \
    joblib
```

### Step 4: Install Chemistry Tools (5-10 min)

```bash
mamba install -c conda-forge -y \
    rdkit \
    openbabel
```

### Step 5: Install QM Packages (10-15 min - SLOWEST)

```bash
# Add psi4 channel and install
mamba install -c conda-forge -c psi4 -y \
    psi4 \
    pygdma

# This is the bottleneck - psi4 is huge (~500MB)
# You can monitor with: watch -n 1 du -sh ~/.conda/envs/mmomenta
```

**If Psi4 hangs or fails:**
```bash
# Try installing separately
mamba install -c psi4 psi4 -y
mamba install -c conda-forge pygdma -y
```

### Step 6: Install PyTorch with CUDA (5-10 min)

```bash
# Check CUDA version on node
module load cuda/11.8

# Install PyTorch with matching CUDA
mamba install -c conda-forge -y \
    pytorch \
    pytorch-gpu \
    cudatoolkit=11.8

# Verify GPU support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**If PyTorch CUDA fails:**
```bash
# Alternative: Use pip with specific CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 7: Install DGL (3-5 min)

```bash
# DGL needs to match PyTorch version
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu118/repo.html

# Verify
python -c "import dgl; print(dgl.__version__)"
```

### Step 8: Install OpenFF Packages (2-3 min)

```bash
pip install \
    openff-toolkit \
    openff-recharge \
    openff-units
```

### Step 9: Install Jupyter (Optional, 1 min)

```bash
mamba install -c conda-forge jupyter -y
```

### Step 10: Install MMomentA (1 min)

```bash
cd ~/MMomentA
pip install -e .

# Verify
python -c "from mmomenta.qm import MPFITCalculator; print('✓ MMomentA installed')"
```

## Complete Manual Script

Save this as `phoenix/manual_install.sh`:

```bash
#!/bin/bash
# Manual step-by-step installation

echo "================================================"
echo "MMomentA Manual Installation"
echo "================================================"

module load anaconda3/2022.05

# Step 1: Base environment
echo ""
echo "Step 1/10: Creating base environment..."
conda create -n mmomenta python=3.10 pip -y
source activate mmomenta
echo "✓ Base environment created"

# Step 2: Mamba
echo ""
echo "Step 2/10: Installing mamba..."
conda install mamba -c conda-forge -y
echo "✓ Mamba installed"

# Step 3: Scientific stack
echo ""
echo "Step 3/10: Installing scientific packages..."
mamba install -c conda-forge -y \
    numpy scipy pandas matplotlib seaborn h5py pytables joblib
echo "✓ Scientific packages installed"

# Step 4: Chemistry tools
echo ""
echo "Step 4/10: Installing chemistry tools..."
mamba install -c conda-forge -y rdkit openbabel
echo "✓ Chemistry tools installed"

# Step 5: QM packages
echo ""
echo "Step 5/10: Installing Psi4 and GDMA (this may take 10-15 min)..."
mamba install -c conda-forge -c psi4 -y psi4 pygdma
if [ $? -ne 0 ]; then
    echo "Trying Psi4 separately..."
    mamba install -c psi4 psi4 -y
    mamba install -c conda-forge pygdma -y
fi
echo "✓ QM packages installed"

# Step 6: PyTorch
echo ""
echo "Step 6/10: Installing PyTorch with CUDA..."
module load cuda/11.8
mamba install -c conda-forge -y pytorch pytorch-gpu cudatoolkit=11.8
echo "✓ PyTorch installed"

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Step 7: DGL
echo ""
echo "Step 7/10: Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu118/repo.html
echo "✓ DGL installed"

# Step 8: OpenFF
echo ""
echo "Step 8/10: Installing OpenFF packages..."
pip install openff-toolkit openff-recharge openff-units
echo "✓ OpenFF packages installed"

# Step 9: Jupyter (optional)
echo ""
echo "Step 9/10: Installing Jupyter..."
mamba install -c conda-forge jupyter -y
echo "✓ Jupyter installed"

# Step 10: MMomentA
echo ""
echo "Step 10/10: Installing MMomentA..."
cd ~/MMomentA
pip install -e .
echo "✓ MMomentA installed"

# Final verification
echo ""
echo "================================================"
echo "Verifying Installation"
echo "================================================"

python << 'EOF'
import sys

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {name}: {version}')
        return True
    except ImportError:
        print(f'✗ {name}: NOT FOUND')
        return False

all_ok = True
all_ok &= check_package('NumPy', 'numpy')
all_ok &= check_package('PyTorch', 'torch')
all_ok &= check_package('DGL', 'dgl')
all_ok &= check_package('Psi4', 'psi4')
all_ok &= check_package('RDKit', 'rdkit')
all_ok &= check_package('OpenFF Toolkit', 'openff.toolkit')
all_ok &= check_package('MMomentA', 'mmomenta')

import torch
print(f'\nCUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')

if all_ok:
    print('\n✓ All packages installed successfully!')
    sys.exit(0)
else:
    print('\n✗ Some packages failed to install')
    sys.exit(1)
EOF

echo "================================================"
echo "Installation Complete!"
echo "================================================"
```

## Interactive Installation (Recommended)

**Instead of running a script, install interactively so you can see what's slow:**

```bash
# SSH to Phoenix
ssh parmar@login-phoenix.pace.gatech.edu

# Load module
module load anaconda3/2022.05

# Create environment
conda create -n mmomenta python=3.10 pip -y
conda activate mmomenta

# Install mamba first
conda install mamba -c conda-forge -y

# Now install groups one at a time
# Watch for errors after each step

# Group 1: Fast packages
mamba install -c conda-forge -y numpy scipy pandas matplotlib seaborn h5py pytables joblib

# Group 2: Chemistry (slower)
mamba install -c conda-forge -y rdkit openbabel

# Group 3: QM (SLOWEST - may take 15 min)
mamba install -c conda-forge -c psi4 -y psi4 pygdma

# Group 4: PyTorch (slow)
module load cuda/11.8
mamba install -c conda-forge -y pytorch pytorch-gpu cudatoolkit=11.8

# Group 5: DGL
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu118/repo.html

# Group 6: OpenFF
pip install openff-toolkit openff-recharge openff-units

# Group 7: MMomentA
cd ~/MMomentA
pip install -e .
```

## Troubleshooting Individual Steps

### If Psi4 fails or hangs:

**Option 1: Different channel**
```bash
conda install -c conda-forge/label/cf202003 psi4 -y
```

**Option 2: Specific version**
```bash
mamba install -c psi4 psi4=1.8 -y
```

**Option 3: Skip for now, install later**
```bash
# Continue without Psi4, install it last
# Some functionality will work without it for testing
```

### If PyTorch CUDA fails:

**Option 1: CPU-only first**
```bash
mamba install pytorch cpuonly -c pytorch -y
# Later upgrade to GPU version
```

**Option 2: Direct pip**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### If DGL fails:

**Check PyTorch version first:**
```bash
python -c "import torch; print(torch.__version__)"

# Then install matching DGL
# For PyTorch 2.6.x:
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu118/repo.html

# For PyTorch 2.5.x:
pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/cu118/repo.html
```

### If mamba itself fails:

**Use conda throughout:**
```bash
# Just use conda instead of mamba
# It's slower but more reliable
conda install -c conda-forge <package>
```

## Minimal Working Environment (Skip Heavy Packages)

If you just want to test the code structure without QM:

```bash
conda create -n mmomenta python=3.10 -y
conda activate mmomenta

# Install only essential packages
mamba install -c conda-forge -y \
    numpy scipy pandas matplotlib h5py joblib rdkit

pip install torch dgl openff-toolkit

cd ~/MMomentA
pip install -e .
```

**This won't run QM calculations but lets you:**
- Test data structures
- Load/save datasets
- Test ML models
- Debug code

## Checking Installation Progress

**Monitor disk usage (see if packages are downloading):**
```bash
watch -n 5 du -sh ~/.conda/envs/mmomenta
```

**Check what's installed so far:**
```bash
conda list
```

**Check if a specific package is available:**
```bash
mamba search psi4 -c psi4
```

## Time Estimates Per Step

| Step | Package | Time | Bottleneck |
|------|---------|------|------------|
| 1 | Base Python | 1 min | - |
| 2 | Mamba | 1-2 min | - |
| 3 | Scientific | 2-3 min | - |
| 4 | Chemistry | 5-10 min | RDKit compilation |
| 5 | Psi4/GDMA | 10-20 min | **SLOWEST** (large download) |
| 6 | PyTorch | 5-10 min | CUDA version resolution |
| 7 | DGL | 3-5 min | PyTorch compatibility |
| 8 | OpenFF | 2-3 min | - |
| 9 | Jupyter | 1-2 min | - |
| 10 | MMomentA | 1 min | - |

**Total: 30-60 minutes** (vs 2+ hours with full environment.yml)

## Resume After Failure

If installation fails partway through:

```bash
# Environment already exists, just activate
conda activate mmomenta

# Check what's installed
conda list

# Continue from where it failed
# Example: If failed at Psi4, skip to step 5
mamba install -c conda-forge -c psi4 -y psi4 pygdma
```

Environment persists, so you can retry individual steps without starting over.
