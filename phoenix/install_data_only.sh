#!/bin/bash
# Install data generation environment (QM packages, no GPU)
# This is for preparing datasets with MPFIT calculations

echo "================================================"
echo "MMomentA Data Generation Environment Setup"
echo "================================================"

module load anaconda3

echo ""
echo "Creating mmomenta-data environment..."
conda env create -f phoenix/environment_data.yml

echo ""
echo "Activating environment..."
conda activate mmomenta-data

# Install OpenFF Toolkit from source
echo ""
echo "Installing OpenFF Toolkit from source..."
cd ~/
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
cd ~/
if [ -d "openff-interchange" ]; then
    rm -rf openff-interchange
fi
git clone https://github.com/openforcefield/openff-interchange.git
cd openff-interchange
git checkout 0.4.0
pip install .

echo ""
echo "Installing MMomentA package..."
cd ~/MMomentA
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
