#!/bin/bash
# Setup conda environment for MMomentA on Phoenix cluster

set -e  # Exit on error

echo "================================================"
echo "MMomentA Environment Setup for Phoenix Cluster"
echo "================================================"

# Load conda module (Phoenix-specific)
module load anaconda3/2022.05

# Create conda environment from YAML
echo ""
echo "Creating conda environment 'mmomenta'..."
echo "This may take 10-15 minutes..."

conda env create -f environment.yml

echo ""
echo "Activating environment..."
source activate mmomenta

# Install MMomentA package in development mode
echo ""
echo "Installing MMomentA package..."
cd ..
pip install -e .

# Verify GPU support (if available)
echo ""
echo "Checking PyTorch GPU support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}')"

# Verify DGL
echo ""
echo "Checking DGL installation..."
python -c "import dgl; print(f'DGL version: {dgl.__version__}')"

# Verify OpenFF
echo ""
echo "Checking OpenFF Toolkit..."
python -c "from openff.toolkit import Molecule; print('OpenFF Toolkit: OK')"

# Verify Psi4
echo ""
echo "Checking Psi4..."
python -c "import psi4; print(f'Psi4 version: {psi4.__version__}')"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate mmomenta"
echo ""
echo "To verify installation, run:"
echo "  bash phoenix/test_installation.sh"
echo ""
