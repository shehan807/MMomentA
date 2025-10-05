#!/bin/bash
# Manual step-by-step installation (conda only, no mamba)
# Run interactively or watch output carefully

echo "================================================"
echo "MMomentA Manual Installation (Step-by-Step)"
echo "================================================"

module load anaconda3/2022.05

# Step 1: Base environment
echo ""
echo "Step 1/9: Creating base environment..."
conda create -n mmomenta python=3.11 pip -y
if [ $? -ne 0 ]; then
    echo "✗ Failed to create base environment"
    exit 1
fi
source activate mmomenta
echo "✓ Base environment created"

# Step 2: Scientific stack
echo ""
echo "Step 2/9: Installing scientific packages..."
conda install -c conda-forge -y \
    numpy scipy pandas matplotlib seaborn h5py pytables joblib
if [ $? -ne 0 ]; then
    echo "✗ Scientific packages failed"
    exit 1
fi
echo "✓ Scientific packages installed"

# Step 3: Chemistry tools
echo ""
echo "Step 3/9: Installing chemistry tools (may take 5-10 min)..."
conda install -c conda-forge -y rdkit openbabel
if [ $? -ne 0 ]; then
    echo "✗ Chemistry tools failed"
    exit 1
fi
echo "✓ Chemistry tools installed"

# Step 4: QM packages
echo ""
echo "Step 4/9: Installing Psi4 and GDMA (may take 10-20 min)..."
echo "This is the slowest step - be patient..."
conda install -c conda-forge -c psi4 -y psi4 pygdma
if [ $? -ne 0 ]; then
    echo "Psi4 installation failed, trying separately..."
    conda install -c psi4 psi4 -y
    if [ $? -ne 0 ]; then
        echo "✗ Psi4 installation failed"
        echo "You can continue and install Psi4 manually later"
        read -p "Continue without Psi4? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        conda install -c conda-forge pygdma -y
    fi
else
    echo "✓ QM packages installed"
fi

# Step 5: PyTorch
echo ""
echo "Step 5/9: Installing PyTorch with CUDA..."
module load cuda/11.8

# Use conda with pytorch and nvidia channels
echo "Installing PyTorch with CUDA 11.8..."
conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
if [ $? -ne 0 ]; then
    echo "CUDA 11.8 failed, trying CUDA 12.1..."
    conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    if [ $? -ne 0 ]; then
        echo "✗ PyTorch installation failed"
        exit 1
    fi
fi
echo "✓ PyTorch installed"

# Verify CUDA
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}')"

# Step 6: DGL
echo ""
echo "Step 6/9: Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu118/repo.html
if [ $? -ne 0 ]; then
    echo "✗ DGL installation failed"
    exit 1
fi
echo "✓ DGL installed"

# Step 7: OpenFF dependencies
echo ""
echo "Step 7/9: Installing OpenFF dependencies..."

# Install OpenFF dependencies from conda
conda install -c conda-forge -y \
    packaging \
    openff-forcefields \
    openff-amber-ff-ports \
    openff-units \
    openff-utilities \
    networkx \
    xmltodict \
    pymongo \
    python-constraint \
    cachetools \
    typing_extensions \
    openmm \
    mdtraj \
    ambertools

if [ $? -ne 0 ]; then
    echo "✗ OpenFF dependencies failed"
    exit 1
fi
echo "✓ OpenFF dependencies installed"

# Install OpenFF Toolkit from source
echo ""
echo "Installing OpenFF Toolkit from source..."
cd ~/
git clone https://github.com/openforcefield/openff-toolkit.git
cd openff-toolkit
git checkout 0.17.1
python -m pip install .
if [ $? -ne 0 ]; then
    echo "✗ OpenFF Toolkit installation failed"
    exit 1
fi
echo "✓ OpenFF Toolkit installed from source"

# Install OpenFF Interchange from source
echo ""
echo "Installing OpenFF Interchange from source..."
cd ~/
git clone https://github.com/openforcefield/openff-interchange.git
cd openff-interchange
git checkout 0.4.0
python -m pip install .
if [ $? -ne 0 ]; then
    echo "✗ OpenFF Interchange installation failed"
    exit 1
fi
echo "✓ OpenFF Interchange installed from source"

# Install OpenFF Recharge (can use pip)
echo ""
echo "Installing OpenFF Recharge..."
pip install openff-recharge
if [ $? -ne 0 ]; then
    echo "✗ OpenFF Recharge failed"
    exit 1
fi
echo "✓ OpenFF Recharge installed"

# Step 8: Jupyter (optional)
echo ""
echo "Step 8/9: Installing Jupyter..."
conda install -c conda-forge jupyter -y
if [ $? -ne 0 ]; then
    echo "⚠ Jupyter installation failed (optional, continuing...)"
else
    echo "✓ Jupyter installed"
fi

# Step 9: MMomentA
echo ""
echo "Step 9/9: Installing MMomentA..."
cd ~/MMomentA
pip install -e .
if [ $? -ne 0 ]; then
    echo "✗ MMomentA installation failed"
    exit 1
fi
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
    print('\n⚠ Some packages missing but installation may still be usable')
    sys.exit(0)
EOF

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "To activate environment:"
echo "  module load anaconda3/2022.05"
echo "  conda activate mmomenta"
echo ""
