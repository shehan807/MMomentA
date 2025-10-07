#!/bin/bash
# Manual setup for mmomenta-data environment on Perlmutter
# Run this on a Perlmutter login node

echo "================================================"
echo "MMomentA Data Environment Setup (Perlmutter)"
echo "================================================"
echo ""

# Load conda module (Perlmutter-specific)
module load conda

# Configure conda to use SCRATCH for packages and cache
export CONDA_PKGS_DIRS="$SCRATCH/.conda/pkgs"
export CONDA_ENVS_DIRS="$SCRATCH/conda-envs"

# Create directories if they don't exist
mkdir -p "$CONDA_PKGS_DIRS"
mkdir -p "$CONDA_ENVS_DIRS"

# Set environment path in SCRATCH
ENV_PATH="$SCRATCH/conda-envs/mmomenta-data"

echo "Conda package cache: $CONDA_PKGS_DIRS"
echo "Conda environments: $CONDA_ENVS_DIRS"
echo "Creating environment at: $ENV_PATH"
echo ""

# Create conda environment in SCRATCH
conda create -p "$ENV_PATH" python=3.10 -y

echo ""
echo "Activating environment..."
conda activate "$ENV_PATH"

echo ""
echo "Installing Psi4 and PyGDMA (QM engine and MPFIT support)..."
conda install -c conda-forge -c psi4 -y psi4 pygdma

echo ""
echo "Installing OpenFF toolkit..."
conda install -c conda-forge openff-toolkit -y

echo ""
echo "Installing RDKit..."
conda install -c conda-forge rdkit -y

echo ""
echo "Installing scientific computing packages..."
conda install -c conda-forge numpy scipy h5py -y

echo ""
echo "Installing plotting libraries..."
conda install -c conda-forge matplotlib seaborn -y

echo ""
echo "Installing utilities..."
conda install -c conda-forge tqdm joblib -y

echo ""
echo "Installing PyTorch (CPU-only, for ESP validation with GNN models)..."
echo "Using pip to avoid Intel VTune symbol conflicts..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing DGL (CPU-only, for graph neural networks)..."
conda install -c dglteam "dgl<2.0" -y

echo ""
echo "Installing MMomentA package..."
cd "${MMOMENTA_DIR:-$(dirname $(dirname $(dirname ${BASH_SOURCE[0]})))}"
pip install -e .

echo ""
echo "================================================"
echo "Environment created successfully!"
echo "================================================"
echo ""
echo "Environment location: $ENV_PATH"
echo ""
echo "To activate this environment in future sessions:"
echo "  module load conda"
echo "  conda activate $ENV_PATH"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  export MMOMENTA_DATA_ENV=\"$ENV_PATH\""
echo "  alias mmdata='module load conda && conda activate \$MMOMENTA_DATA_ENV'"
echo ""

# Test installation
echo "Testing installation..."
python << 'EOF'
try:
    import psi4
    print("✓ Psi4 installed")
except ImportError as e:
    print(f"✗ Psi4 not available: {e}")

try:
    import openff.toolkit
    print("✓ OpenFF toolkit installed")
except ImportError as e:
    print(f"✗ OpenFF toolkit not available: {e}")

try:
    import h5py
    print("✓ HDF5 support installed")
except ImportError as e:
    print(f"✗ HDF5 not available: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed (CPU-only)")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")

try:
    import dgl
    print(f"✓ DGL {dgl.__version__} installed (CPU-only)")
except ImportError as e:
    print(f"✗ DGL not available: {e}")

print("\n✓ mmomenta-data environment ready!")
print("  This environment can now run ESP validation with GNN models!")
EOF

echo ""
echo "Setup complete!"
