#!/bin/bash
# Setup conda environment for MMomentA on Phoenix cluster

echo "================================================"
echo "MMomentA Environment Setup for Phoenix Cluster"
echo "================================================"

# Load conda module (Phoenix-specific)
echo "Loading anaconda3 module..."
module load anaconda3/2022.05
if [ $? -ne 0 ]; then
    echo "✗ Failed to load anaconda3 module"
    exit 1
fi
echo "✓ Anaconda3 module loaded"

# Create conda environment from YAML
echo ""
echo "Creating conda environment 'mmomenta'..."
echo "This may take 10-15 minutes..."

conda env create -f environment.yml
if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Failed to create conda environment"
    echo "Check the error messages above for details"
    exit 1
fi
echo "✓ Conda environment created"

echo ""
echo "Activating environment..."
source activate mmomenta
if [ $? -ne 0 ]; then
    echo "✗ Failed to activate environment"
    exit 1
fi
echo "✓ Environment activated"

# Install MMomentA package in development mode
echo ""
echo "Installing MMomentA package..."
cd ..
pip install -e .
if [ $? -ne 0 ]; then
    echo "✗ Failed to install MMomentA package"
    exit 1
fi
echo "✓ MMomentA package installed"

# Verify installations
echo ""
echo "================================================"
echo "Verifying Installations"
echo "================================================"

echo ""
echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}')"
if [ $? -ne 0 ]; then
    echo "✗ PyTorch check failed"
else
    echo "✓ PyTorch OK"
fi

echo ""
echo "Checking DGL..."
python -c "import dgl; print(f'DGL version: {dgl.__version__}')"
if [ $? -ne 0 ]; then
    echo "✗ DGL check failed"
else
    echo "✓ DGL OK"
fi

echo ""
echo "Checking OpenFF Toolkit..."
python -c "from openff.toolkit import Molecule; print('OpenFF Toolkit OK')"
if [ $? -ne 0 ]; then
    echo "✗ OpenFF Toolkit check failed"
else
    echo "✓ OpenFF Toolkit OK"
fi

echo ""
echo "Checking Psi4..."
python -c "import psi4; print(f'Psi4 version: {psi4.__version__}')"
if [ $? -ne 0 ]; then
    echo "✗ Psi4 check failed"
else
    echo "✓ Psi4 OK"
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  module load anaconda3/2022.05"
echo "  conda activate mmomenta"
echo ""
echo "To run comprehensive tests:"
echo "  bash phoenix/test_installation.sh"
echo ""
