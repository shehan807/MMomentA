#!/bin/bash
# Install data generation environment (QM packages, no GPU)
# This is for preparing datasets with MPFIT calculations

echo "================================================"
echo "MMomentA Data Generation Environment Setup"
echo "================================================"

MMOMENTA_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML"

module load anaconda3

# Remove existing environment
echo ""
echo "Removing any existing mmomenta-data environment..."
conda env remove -n mmomenta-data -y 2>/dev/null || true

echo ""
echo "Creating mmomenta-data environment..."
cd "$MMOMENTA_DIR"
conda env create -f phoenix/environment_data.yml
if [ $? -ne 0 ]; then
    echo "✗ Failed to create environment"
    exit 1
fi

echo ""
echo "Activating environment..."
source activate mmomenta-data

# Install OpenFF Toolkit from source
echo ""
echo "Installing OpenFF Toolkit from source..."
cd "$MMOMENTA_DIR"
if [ -d "openff-toolkit" ]; then
    rm -rf openff-toolkit
fi
git clone https://github.com/openforcefield/openff-toolkit.git
cd openff-toolkit
git checkout 0.17.1
pip install .

# Install OpenFF Interchange from source
echo ""
echo "Installing OpenFF Interchange from source..."
cd "$MMOMENTA_DIR"
if [ -d "openff-interchange" ]; then
    rm -rf openff-interchange
fi
git clone https://github.com/openforcefield/openff-interchange.git
cd openff-interchange
git checkout 0.4.0
pip install .

echo ""
echo "Installing MMomentA package..."
cd "$MMOMENTA_DIR/MMomentA"
pip install -e .

echo ""
echo "================================================"
echo "Testing Data Environment"
echo "================================================"

python << 'EOF'
import sys

try:
    import psi4
    print(f'✓ Psi4: {psi4.__version__}')
except ImportError as e:
    print(f'✗ Psi4: {e}')
    sys.exit(1)

try:
    from openff.toolkit import Molecule
    print('✓ OpenFF Toolkit')
except ImportError as e:
    print(f'✗ OpenFF Toolkit: {e}')
    sys.exit(1)

try:
    from openff.recharge.esp import ESPSettings
    print('✓ OpenFF Recharge')
except ImportError as e:
    print(f'✗ OpenFF Recharge: {e}')
    sys.exit(1)

try:
    from mmomenta.qm import MPFITCalculator
    from mmomenta.data import BatchProcessor
    print('✓ MMomentA QM modules')
except ImportError as e:
    print(f'✗ MMomentA QM: {e}')
    sys.exit(1)

print('\n✓ Data generation environment ready!')
print('\nYou can now:')
print('  - Run MPFIT calculations')
print('  - Prepare datasets')
print('  - Compare charge methods')
print('\nNote: This environment cannot train ML models.')
print('Use mmomenta-ml environment for training.')
EOF

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To use this environment:"
echo "  module load anaconda3"
echo "  conda activate mmomenta-data"
echo ""
echo "To prepare datasets:"
echo "  python scripts/prepare_dataset.py --input molecules.pkl ..."
echo ""
