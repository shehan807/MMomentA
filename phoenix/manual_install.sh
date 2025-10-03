#!/bin/bash
# Manual step-by-step installation
# Run interactively or watch output carefully

echo "================================================"
echo "MMomentA Manual Installation (Step-by-Step)"
echo "================================================"

module load anaconda3/2022.05

# Step 1: Base environment
echo ""
echo "Step 1/10: Creating base environment..."
# Use Python 3.11 (better openbabel compatibility than 3.10)
conda create -n mmomenta python=3.11 pip -y
if [ $? -ne 0 ]; then
    echo "✗ Failed to create base environment"
    exit 1
fi
source activate mmomenta
echo "✓ Base environment created"

# Step 2: Mamba
echo ""
echo "Step 2/10: Installing mamba..."
conda install mamba -c conda-forge -y
if [ $? -ne 0 ]; then
    echo "✗ Mamba installation failed, continuing with conda..."
    alias mamba=conda
else
    echo "✓ Mamba installed"
fi

# Step 3: Scientific stack
echo ""
echo "Step 3/10: Installing scientific packages..."
mamba install -c conda-forge -y \
    numpy scipy pandas matplotlib seaborn h5py pytables joblib
if [ $? -ne 0 ]; then
    echo "✗ Scientific packages failed"
    exit 1
fi
echo "✓ Scientific packages installed"

# Step 4: Chemistry tools
echo ""
echo "Step 4/10: Installing chemistry tools (may take 5-10 min)..."
mamba install -c conda-forge -y rdkit openbabel
if [ $? -ne 0 ]; then
    echo "✗ Chemistry tools failed"
    exit 1
fi
echo "✓ Chemistry tools installed"

# Step 5: QM packages
echo ""
echo "Step 5/10: Installing Psi4 and GDMA (may take 10-20 min)..."
echo "This is the slowest step - be patient..."
mamba install -c conda-forge -c psi4 -y psi4 pygdma
if [ $? -ne 0 ]; then
    echo "Psi4 installation via mamba failed, trying separately..."
    mamba install -c psi4 psi4 -y
    if [ $? -ne 0 ]; then
        echo "✗ Psi4 installation failed"
        echo "You can continue and install Psi4 manually later"
        read -p "Continue without Psi4? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        mamba install -c conda-forge pygdma -y
    fi
else
    echo "✓ QM packages installed"
fi

# Step 6: PyTorch
echo ""
echo "Step 6/10: Installing PyTorch with CUDA..."
module load cuda/11.8
mamba install -c conda-forge -y pytorch pytorch-gpu cudatoolkit=11.8
if [ $? -ne 0 ]; then
    echo "✗ PyTorch installation failed"
    exit 1
fi
echo "✓ PyTorch installed"

# Verify CUDA
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Step 7: DGL
echo ""
echo "Step 7/10: Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu118/repo.html
if [ $? -ne 0 ]; then
    echo "✗ DGL installation failed"
    exit 1
fi
echo "✓ DGL installed"

# Step 8: OpenFF
echo ""
echo "Step 8/10: Installing OpenFF packages..."
pip install openff-toolkit openff-recharge openff-units
if [ $? -ne 0 ]; then
    echo "✗ OpenFF packages failed"
    exit 1
fi
echo "✓ OpenFF packages installed"

# Step 9: Jupyter (optional)
echo ""
echo "Step 9/10: Installing Jupyter..."
mamba install -c conda-forge jupyter -y
if [ $? -ne 0 ]; then
    echo "⚠ Jupyter installation failed (optional, continuing...)"
else
    echo "✓ Jupyter installed"
fi

# Step 10: MMomentA
echo ""
echo "Step 10/10: Installing MMomentA..."
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
