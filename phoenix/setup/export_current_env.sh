#!/bin/bash
# Export current conda environment for reproducibility

echo "================================================"
echo "Exporting Current Conda Environment"
echo "================================================"

# Get current environment name
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')

if [ "$CURRENT_ENV" == "base" ]; then
    echo "Error: You are in the base environment."
    echo "Please activate your MMomentA environment first:"
    echo "  conda activate mmomenta"
    exit 1
fi

echo ""
echo "Current environment: $CURRENT_ENV"
echo ""

# Export environment.yml (cross-platform)
echo "Exporting environment.yml (cross-platform)..."
conda env export --from-history > phoenix/environment_export.yml

# Export full environment with exact versions (platform-specific)
echo "Exporting environment_full.yml (exact versions)..."
conda env export > phoenix/environment_full.yml

# Export pip requirements
echo "Exporting pip requirements..."
pip list --format=freeze > phoenix/requirements_export.txt

# Export detailed package list
echo "Exporting detailed package list..."
conda list > phoenix/conda_list.txt

echo ""
echo "================================================"
echo "Export Complete!"
echo "================================================"
echo ""
echo "Files created in phoenix/:"
echo "  environment_export.yml     - Cross-platform (recommended)"
echo "  environment_full.yml       - Exact versions (platform-specific)"
echo "  requirements_export.txt    - Pip packages"
echo "  conda_list.txt             - Full package list"
echo ""
echo "To recreate environment on Phoenix:"
echo "  conda env create -f phoenix/environment_export.yml"
echo ""

# Print important package versions
echo "Key Package Versions:"
echo "================================================"
python << 'EOF'
import sys
try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except ImportError:
    print("NumPy: NOT INSTALLED")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import dgl
    print(f"DGL: {dgl.__version__}")
except ImportError:
    print("DGL: NOT INSTALLED")

try:
    from openff.toolkit import __version__ as off_version
    print(f"OpenFF Toolkit: {off_version}")
except ImportError:
    print("OpenFF Toolkit: NOT INSTALLED")

try:
    import psi4
    print(f"Psi4: {psi4.__version__}")
except ImportError:
    print("Psi4: NOT INSTALLED")

print(f"Python: {sys.version.split()[0]}")
EOF

echo "================================================"
