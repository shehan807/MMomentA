#!/bin/bash
# Manual setup for mmomenta-ml environment on Perlmutter
# Run this on a Perlmutter GPU node (use salloc for interactive session)

echo "================================================"
echo "MMomentA ML Environment Setup (Perlmutter)"
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

# Perlmutter uses CUDA 12.x, check available modules
echo "Checking available CUDA modules..."
module avail cuda 2>&1 | grep -i cuda || echo "No CUDA modules found via 'module avail'"

# Set environment path in SCRATCH
ENV_PATH="$SCRATCH/conda-envs/mmomenta-ml"

echo ""
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
echo "Installing PyTorch with CUDA support..."
echo "NOTE: Perlmutter typically uses CUDA 12.x"
echo ""

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "nvcc not found, using default CUDA 12.1"
    CUDA_VERSION="12.1"
fi

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == 12.* ]]; then
    echo "Installing PyTorch for CUDA 12.x..."
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
    DGL_CHANNEL="cu121"
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
    DGL_CHANNEL="cu118"
else
    echo "Unknown CUDA version, installing CPU-only PyTorch"
    conda install pytorch cpuonly -c pytorch -y
    DGL_CHANNEL="cpu"
fi

echo ""
echo "Installing DGL (Deep Graph Library)..."
echo "Note: Using DGL <2.0 to avoid torchdata dependency"
if [[ "$DGL_CHANNEL" != "cpu" ]]; then
    conda install -c dglteam/label/$DGL_CHANNEL "dgl<2.0" -y
else
    conda install -c dglteam "dgl<2.0" -y
fi

echo ""
echo "Installing OpenFF toolkit (for molecule handling)..."
conda install -c conda-forge openff-toolkit -y

echo ""
echo "Fixing pint version (openff-units compatibility)..."
pip install "pint<0.24"

echo ""
echo "Installing scientific computing packages..."
conda install -c conda-forge numpy scipy h5py -y

echo ""
echo "Installing plotting libraries..."
conda install -c conda-forge matplotlib seaborn -y

echo ""
echo "Installing lovelyplots (publication-quality figures)..."
pip install lovelyplots

echo ""
echo "Installing utilities..."
conda install -c conda-forge tqdm joblib -y

echo ""
echo "Installing PyTorch Geometric (for dataset download)..."
pip install torch-geometric

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
echo "  export MMOMENTA_ML_ENV=\"$ENV_PATH\""
echo "  alias mmml='module load conda && conda activate \$MMOMENTA_ML_ENV'"
echo ""

# Test installation
echo "Testing installation..."
python << 'EOF'
import sys

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")

try:
    import dgl
    print(f"✓ DGL {dgl.__version__} installed")
except ImportError as e:
    print(f"✗ DGL not available: {e}")

try:
    import openff.toolkit
    print("✓ OpenFF toolkit installed")
except ImportError as e:
    print(f"✗ OpenFF toolkit not available: {e}")

try:
    import lovelyplots
    print("✓ lovelyplots installed")
except ImportError as e:
    print(f"✗ lovelyplots not available: {e}")

try:
    import torch_geometric
    print(f"✓ PyTorch Geometric {torch_geometric.__version__} installed")
except ImportError as e:
    print(f"✗ PyTorch Geometric not available: {e}")

print("\n✓ mmomenta-ml environment ready!")
EOF

echo ""
echo "Setup complete!"
echo ""
echo "IMPORTANT: To use GPU on Perlmutter, run jobs with:"
echo "  salloc -C gpu -q interactive -t 01:00:00 -A <your_account>"
echo "  or submit batch jobs to GPU partition"
