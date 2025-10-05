#!/bin/bash
# Install OpenFF Toolkit and Interchange from source
# Based on conda recipe requirements

set -e

echo "================================================"
echo "Installing OpenFF Packages from Source"
echo "================================================"

# Ensure environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Error: mmomenta environment not activated"
    echo "Run: conda activate mmomenta"
    exit 1
fi

echo "Active environment: $CONDA_DEFAULT_ENV"

# Install all dependencies first
echo ""
echo "Installing OpenFF dependencies via conda..."
mamba install -c conda-forge -y \
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

echo "✓ Dependencies installed"

# Create source directory
mkdir -p ~/openff_source
cd ~/openff_source

# Install OpenFF Toolkit from source
echo ""
echo "================================================"
echo "Installing OpenFF Toolkit 0.17.1 from source"
echo "================================================"

if [ -d "openff-toolkit" ]; then
    rm -rf openff-toolkit
fi

git clone https://github.com/openforcefield/openff-toolkit.git
cd openff-toolkit
git checkout 0.17.1
git log -1 --oneline

echo ""
echo "Installing openff-toolkit..."
python -m pip install . --no-deps
if [ $? -ne 0 ]; then
    echo "✗ OpenFF Toolkit installation failed"
    exit 1
fi

echo "✓ OpenFF Toolkit installed"

# Test import
python -c "from openff.toolkit import Molecule; print(f'OpenFF Toolkit version: {Molecule.__module__}')"

# Install OpenFF Interchange from source
echo ""
echo "================================================"
echo "Installing OpenFF Interchange 0.4.0 from source"
echo "================================================"

cd ~/openff_source

if [ -d "openff-interchange" ]; then
    rm -rf openff-interchange
fi

git clone https://github.com/openforcefield/openff-interchange.git
cd openff-interchange
git checkout 0.4.0
git log -1 --oneline

echo ""
echo "Installing openff-interchange..."
python -m pip install . --no-deps
if [ $? -ne 0 ]; then
    echo "✗ OpenFF Interchange installation failed"
    exit 1
fi

echo "✓ OpenFF Interchange installed"

# Test import
python -c "from openff.interchange import Interchange; print('OpenFF Interchange imported successfully')"

# Install OpenFF Recharge (standard pip)
echo ""
echo "================================================"
echo "Installing OpenFF Recharge"
echo "================================================"

pip install openff-recharge
echo "✓ OpenFF Recharge installed"

# Final verification
echo ""
echo "================================================"
echo "Verifying OpenFF Installation"
echo "================================================"

python << 'EOF'
import sys

try:
    from openff.toolkit import Molecule
    print("✓ OpenFF Toolkit")
except ImportError as e:
    print(f"✗ OpenFF Toolkit: {e}")
    sys.exit(1)

try:
    from openff.interchange import Interchange
    print("✓ OpenFF Interchange")
except ImportError as e:
    print(f"✗ OpenFF Interchange: {e}")
    sys.exit(1)

try:
    from openff.recharge.esp import ESPSettings
    print("✓ OpenFF Recharge")
except ImportError as e:
    print(f"✗ OpenFF Recharge: {e}")
    sys.exit(1)

print("\n✓ All OpenFF packages installed successfully!")
EOF

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Source repositories cloned to: ~/openff_source/"
echo ""
